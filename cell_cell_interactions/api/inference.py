"""High-level inference helpers.

These wrap the raw ``model.model(...)`` forward pass and the per-cell
gene-embedding demux that every paper tutorial reimplements. They do not
change any underlying computation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from cell_cell_interactions.api.model import MCTIndex, MCTModel
from cell_cell_interactions.modeling.data_types import MulticellularTransformerOutput
from cell_cell_interactions.pp.data_types import BagOfCellsBatch


def get_context_aware_gene_embeddings(
    model: MCTModel | None,
    output: MulticellularTransformerOutput,
    batch: BagOfCellsBatch,
    idx2gene: dict[int, str] | None = None,
    skip_gene_ids: Iterable[int] = (0, 1),
) -> list[dict[str, np.ndarray]]:
    """Demux ``last_hidden_state`` back to a per-cell ``gene_symbol -> vector`` map.

    Every paper tutorial contained its own copy of this loop; this is the
    canonical version.

    Parameters
    ----------
    model:
        The :class:`MCTModel` the output came from. Used only as a fallback
        source of ``idx2gene``; may be ``None`` if ``idx2gene`` is supplied.
    output:
        The output of :meth:`MCTModel.embed_batch`.
    batch:
        The batch that produced ``output``.
    idx2gene:
        Mapping from token index back to gene symbol. Defaults to
        ``model.index.idx2gene``.
    skip_gene_ids:
        Token IDs to drop from the per-cell dict (default: PAD=0, MASK=1).
    """
    if idx2gene is None:
        if model is None or model.index.idx2gene is None:
            raise ValueError(
                "idx2gene is required. Either pass it explicitly or provide an MCTModel "
                "loaded with a gene index."
            )
        idx2gene = model.index.idx2gene

    hidden = output.last_hidden_state.detach().cpu().numpy()
    gene_ids = batch.gene_ids.detach().cpu().numpy()
    skip = set(skip_gene_ids)

    per_cell: list[dict[str, np.ndarray]] = []
    for cell_idx in range(gene_ids.shape[0]):
        cell_map: dict[str, np.ndarray] = {}
        for pos, gid in enumerate(gene_ids[cell_idx]):
            gid = int(gid)
            if gid in skip:
                continue
            symbol = idx2gene.get(gid)
            if symbol is None:
                continue
            cell_map[symbol] = hidden[cell_idx, pos]
        per_cell.append(cell_map)
    return per_cell


def embed_dataloader(
    model: MCTModel,
    dataloader,
    return_batches: bool = False,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Iterate a DataLoader, run the model on each batch, and collect outputs.

    Returns a list of dicts, one per batch, each containing:

    - ``last_hidden_state``: ``np.ndarray`` of shape ``(B, S, gene_emb_dim)``
    - ``gene_ids``: ``np.ndarray`` of shape ``(B, S)``
    - ``sample_label``: list of sample identifiers if the batch carries them
    - ``batch``: the original ``BagOfCellsBatch`` if ``return_batches=True``
    """
    model.eval()
    results: list[dict[str, Any]] = []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        out = model.embed_batch(batch)
        entry: dict[str, Any] = {
            "last_hidden_state": out.last_hidden_state.detach().cpu().numpy(),
            "gene_ids": batch.gene_ids.detach().cpu().numpy(),
            "sample_label": getattr(batch, "sample_label", None),
        }
        if return_batches:
            entry["batch"] = batch
        results.append(entry)
    return results


def embed_adata(
    model: MCTModel,
    preset,
    return_batches: bool = False,
    loader: str = "val",
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Build a preset's datamodule and run inference over the chosen loader.

    ``preset`` should be an :class:`AdataInferencePreset`,
    :class:`CellarrInferencePreset`, or :class:`SpatialInferencePreset`.
    ``loader`` selects among ``"train"``, ``"val"``, ``"test"``.
    """
    datamodule = preset.build()
    dl = _select_loader(datamodule, loader)
    return embed_dataloader(model, dl, return_batches=return_batches, max_batches=max_batches)


def _select_loader(datamodule, loader: str):
    if loader == "train":
        return datamodule.train_dataloader()
    if loader == "val":
        return datamodule.val_dataloader()
    if loader == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"unknown loader kind '{loader}'; expected one of 'train', 'val', 'test'.")
