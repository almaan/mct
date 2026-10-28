"""Library-form wrapper for the in-silico DepMap gene-KO scripts.

Consolidates the near-identical adata and cellarr variants (see
``run_gene_ko_effect_on_tumor_cells_{adata,cellarr}.py``) behind one entry
point. The CLI scripts remain and now call into this function.

The computation is byte-identical to the previous scripts: baseline forward
pass -> for each KO gene present in the batch, drop the token and rerun the
forward pass -> collect ``gex_reconstruction`` at the tumor cell-type slot.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from cell_cell_interactions.api.model import MCTModel

# Public disease-code -> canonical-string map used by the released
# DepMap CRISPR gene table shipped by the paper.
DEPMAP_DISEASE_MAP = {
    "lung_adenocarcinoma": "lung adenocarcinoma",
    "squamous_lung_carcinoma": "squamous cell lung carcinoma",
    "non_small_lung_carcinoma": "non-small cell lung carcinoma",
}


def run_gene_ko_effect(
    model: MCTModel,
    dataloader,
    tumor_cell_type: str,
    genes_to_knock_out: Iterable[str],
    name2idx: dict[str, int],
    output_dir: str | os.PathLike,
    max_steps: int | None = None,
    save_every_n_batches: int = 10,
    collect_all_gex: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Iterate ``dataloader`` and score in-silico KO of each gene on tumor cells.

    Parameters
    ----------
    model:
        Loaded :class:`MCTModel`.
    dataloader:
        Batches produced by a datamodule with ``agg_cell_type_gex=True``.
        The batch's cell-type-slot layout is read from
        ``dataloader.dataset`` / the underlying datamodule; the caller is
        responsible for passing a dataloader configured that way.
    tumor_cell_type:
        Name of the tumor cell type in the datamodule's
        ``cell_type_input_order``. Used to pick the row of
        ``gex_reconstruction`` to score.
    genes_to_knock_out:
        Iterable of gene names. Only those present in ``name2idx`` are used.
    name2idx:
        Mapping from gene name to token index.
    output_dir:
        Directory that will receive ``gene_ko_expression.pt`` and
        ``gene_ko_relative_expression.pt``.
    max_steps:
        Optional cap on iterations.
    save_every_n_batches:
        Checkpoint cadence.
    collect_all_gex:
        If True, keep the full ``gex_reconstruction`` sum-over-slots vector
        for both KO and baseline. If False (default), keep only the tumor
        cell-type entry.

    Returns
    -------
    (expression, relative_expression):
        Two dicts mapping gene name to a stacked tensor over batches.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = getattr(dataloader, "datamodule", None) or _try_infer_datamodule(dataloader)
    if dm is None or not hasattr(dm, "cell_type_input_order"):
        raise ValueError(
            "Cannot determine cell_type_input_order from the dataloader. "
            "Pass a dataloader produced by a MCT datamodule."
        )
    ct2idx = {ct: idx for idx, ct in enumerate(dm.cell_type_input_order)}
    if tumor_cell_type not in ct2idx:
        raise KeyError(
            f"tumor cell type '{tumor_cell_type}' not in datamodule.cell_type_input_order. "
            f"Available: {list(ct2idx)}"
        )
    tumor_ct_idx = ct2idx[tumor_cell_type]

    kos = [g for g in genes_to_knock_out if g in name2idx]

    output_expression: dict[str, list[torch.Tensor]] = defaultdict(list)
    output_rel_expression: dict[str, list[torch.Tensor]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if max_steps is not None and i >= max_steps:
                break
            baseline = model.embed_batch(batch)
            baseline_rel_gex = (
                torch.nn.functional.softmax(baseline.gex_reconstruction, dim=-1)
                .sum(dim=1)
                .squeeze(0)
                .cpu()
            )
            baseline_gex = baseline.gex_reconstruction.sum(dim=1).squeeze(0).cpu()

            for gene in kos:
                gene_idx = name2idx[gene]
                if gene_idx not in batch.gene_ids:
                    continue

                idx = torch.nonzero(batch.gene_ids == gene_idx, as_tuple=True)[-1][0]

                new_expression = torch.cat(
                    (batch.expression_matrix[:, :idx, :], batch.expression_matrix[:, idx + 1 :, :]),
                    dim=1,
                )
                new_gene_ids = torch.cat((batch.gene_ids[:, :idx], batch.gene_ids[:, idx + 1 :]), dim=1)
                new_masked_gex = torch.cat(
                    (batch.is_masked_gene_expression[:, :idx], batch.is_masked_gene_expression[:, idx + 1 :]),
                    dim=1,
                )
                new_padding_mask = torch.cat(
                    (batch.padding_mask[:, :idx], batch.padding_mask[:, idx + 1 :]), dim=1
                )

                ko_batch = type(batch)(
                    expression_matrix=new_expression,
                    gene_ids=new_gene_ids,
                    is_masked_gene_expression=new_masked_gex,
                    padding_mask=new_padding_mask,
                    tissue_idx=batch.tissue_idx,
                )
                knock_out = model.embed_batch(ko_batch)

                ko_rel_gex = (
                    torch.nn.functional.softmax(knock_out.gex_reconstruction, dim=-1)
                    .sum(dim=1)
                    .squeeze(0)
                    .cpu()
                )
                ko_gex = knock_out.gex_reconstruction.sum(dim=1).squeeze(0).cpu()

                if not collect_all_gex:
                    output_rel_expression[gene].append(
                        torch.tensor([ko_rel_gex[tumor_ct_idx], baseline_rel_gex[tumor_ct_idx]])
                    )
                    output_expression[gene].append(
                        torch.tensor([ko_gex[tumor_ct_idx], baseline_gex[tumor_ct_idx]])
                    )
                else:
                    output_rel_expression[gene].append(torch.stack([ko_rel_gex, baseline_rel_gex]))
                    output_expression[gene].append(torch.stack([ko_gex, baseline_gex]))

            if (i + 1) % save_every_n_batches == 0:
                _save_ko_dicts(output_expression, output_rel_expression, output_dir)

    _save_ko_dicts(output_expression, output_rel_expression, output_dir)

    return (
        {gene: torch.stack(expr) for gene, expr in output_expression.items()},
        {gene: torch.stack(expr) for gene, expr in output_rel_expression.items()},
    )


def _save_ko_dicts(
    expr: dict[str, list[torch.Tensor]],
    rel_expr: dict[str, list[torch.Tensor]],
    output_dir: Path,
) -> None:
    torch.save(
        {gene: torch.stack(v) for gene, v in expr.items()},
        output_dir / "gene_ko_expression.pt",
    )
    torch.save(
        {gene: torch.stack(v) for gene, v in rel_expr.items()},
        output_dir / "gene_ko_relative_expression.pt",
    )


def _try_infer_datamodule(dataloader):
    """Best-effort recovery of a datamodule reference from a dataloader."""
    ds = getattr(dataloader, "dataset", None)
    if ds is None:
        return None
    return getattr(ds, "datamodule", None)
