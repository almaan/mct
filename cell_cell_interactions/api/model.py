"""User-facing model wrapper.

Loads a pretrained Multicellular Transformer checkpoint together with its
paired index files and exposes a small scanpy-style surface. The intent is
that a user with a ``model_dir`` never has to touch ``modeling/``,
``pp/*_datamodule.py``, or Lightning APIs directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch

from cell_cell_interactions.modeling.data_types import (
    MulticellularGraphConfig,
    MulticellularTransformerOutput,
)
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel

# Names the loader will accept when auto-discovering index files.
_CANDIDATE_INDEX_NAMES = {
    "gene2idx": ("gene2idx.json", "gene_symbol2idx.json", "ensemble2idx.json"),
    "cell_type_input_order": ("cell_type_input_order.json",),
    "tissue2idx": ("tissue2idx.json",),
    "cell_type_map": ("cell_type_map.json",),
}
# Names of directories under a model_dir that may hold the checkpoint.
_CANDIDATE_CKPT_DIRS = ("", "checkpoints", "ckpts")
# Names of checkpoint files, most-specific first.
_CANDIDATE_CKPT_NAMES = (
    "model.ckpt",
    "multicellular_transformer.ckpt",
    "best.ckpt",
    "last.ckpt",
)

# Special-token ids reserved at the start of the gene vocabulary. Frozen by
# the checkpoint contract.
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
N_SPECIAL_TOKENS = 2


@dataclass
class MCTIndex:
    """Metadata files paired with a Multicellular Transformer checkpoint."""

    gene2idx: dict[str, int] | None = None
    idx2gene: dict[int, str] | None = None
    cell_type_input_order: dict[str, tuple[int, int]] | None = None
    tissue2idx: dict[str, int] | None = None
    cell_type_map: dict[str, str] | None = None

    @classmethod
    def from_dir(cls, index_dir: str | os.PathLike) -> "MCTIndex":
        """Load whichever of the canonical index files are present in ``index_dir``."""
        p = Path(index_dir)
        if not p.exists():
            raise FileNotFoundError(f"index dir {p} does not exist")

        payload: dict[str, Any] = {}

        gene_path = _first_existing(p, _CANDIDATE_INDEX_NAMES["gene2idx"])
        if gene_path is not None:
            with gene_path.open() as fh:
                gene2idx = json.load(fh)
            payload["gene2idx"] = gene2idx
            payload["idx2gene"] = {int(v): k for k, v in gene2idx.items()}

        ct_path = _first_existing(p, _CANDIDATE_INDEX_NAMES["cell_type_input_order"])
        if ct_path is not None:
            with ct_path.open() as fh:
                raw = json.load(fh)
            payload["cell_type_input_order"] = {k: tuple(v) for k, v in raw.items()}

        tissue_path = _first_existing(p, _CANDIDATE_INDEX_NAMES["tissue2idx"])
        if tissue_path is not None:
            with tissue_path.open() as fh:
                payload["tissue2idx"] = json.load(fh)

        ctmap_path = _first_existing(p, _CANDIDATE_INDEX_NAMES["cell_type_map"])
        if ctmap_path is not None:
            with ctmap_path.open() as fh:
                payload["cell_type_map"] = json.load(fh)

        return cls(**payload)


class MCTModel:
    """Loaded Multicellular Transformer.

    Construct with :func:`load`. The class is intentionally small: three
    methods for the three things almost every downstream task needs.

    Attributes
    ----------
    model_dir:
        Root directory the model was loaded from (``None`` when loaded from
        a bare checkpoint path).
    checkpoint_path, index:
        Locations of the underlying files.
    config:
        ``MulticellularGraphConfig`` recorded with the checkpoint.

    Examples
    --------
    >>> import cell_cell_interactions as cci
    >>> mct = cci.load("/models/mct_v1")            # dir with model.ckpt + index JSONs
    >>> emb = mct.gene_embeddings()                 # DataFrame (genes x dims)
    >>> emb.loc[["EGFR", "TP53"]].shape
    (2, 64)
    >>> sample_embs = mct.embed(adata,             # AnnData in
    ...                         cell_type_key="cell_type",
    ...                         sample_key="sample")
    >>> list(sample_embs)[:3]                       # dict[sample_id -> DataFrame]
    """

    def __init__(
        self,
        pl_model: MulticellularTransformerPLModel,
        index: MCTIndex,
        checkpoint_path: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        self._pl_model = pl_model
        self.index = index
        self.checkpoint_path = checkpoint_path
        self.model_dir = model_dir

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> MulticellularGraphConfig:
        return self._pl_model.config

    @property
    def device(self) -> torch.device:
        return next(self._pl_model.parameters()).device

    @property
    def module(self):
        """Underlying ``MulticellularTransformerModule`` — advanced use."""
        return self._pl_model.model

    @property
    def lightning_module(self) -> MulticellularTransformerPLModel:
        return self._pl_model

    @property
    def gene2idx(self) -> dict[str, int]:
        if self.index.gene2idx is None:
            raise RuntimeError(
                "This model has no gene2idx mapping loaded. Provide one via "
                "load(index_dir=...) or set model.index.gene2idx manually."
            )
        return self.index.gene2idx

    @property
    def idx2gene(self) -> dict[int, str]:
        if self.index.idx2gene is None:
            raise RuntimeError("No idx2gene available; ensure gene2idx was loaded.")
        return self.index.idx2gene

    @property
    def cell_type_input_order(self) -> dict[str, tuple[int, int]]:
        if self.index.cell_type_input_order is None:
            raise RuntimeError(
                "This model has no cell_type_input_order loaded. Provide it via "
                "load(index_dir=...)."
            )
        return self.index.cell_type_input_order

    def to(self, device: str | torch.device) -> "MCTModel":
        self._pl_model.to(device)
        return self

    def eval(self) -> "MCTModel":
        self._pl_model.eval()
        return self

    # ------------------------------------------------------------------ #
    # Zero-shot gene embeddings
    # ------------------------------------------------------------------ #

    def gene_embeddings(
        self,
        genes: Iterable[str] | None = None,
        drop_special_tokens: bool = True,
    ) -> pd.DataFrame:
        """Return the model's intrinsic gene embeddings.

        These are the rows of the gene-ID classifier's decoder weight (which
        is tied to the gene-ID input embedding by construction of the head).
        Users get to skip the raw
        ``state_dict["model.gene_id_classifier.decoder.weight"]`` slice.

        Parameters
        ----------
        genes:
            If given, return only rows for these symbols. Missing symbols
            are dropped silently. If ``None``, return all vocabulary
            entries.
        drop_special_tokens:
            If True (default), drop the ``[PAD]``/``[MASK]`` rows.

        Returns
        -------
        pandas.DataFrame
            Rows are gene symbols, columns are ``dim_0 .. dim_{D-1}``.

        Examples
        --------
        >>> mct.gene_embeddings().shape        # (n_genes, gene_emb_dim)
        (12000, 64)
        >>> mct.gene_embeddings(["EGFR", "TP53", "MYC"]).index.tolist()
        ['EGFR', 'TP53', 'MYC']
        """
        weight = self.module.gene_id_classifier.decoder.weight.detach().cpu().numpy()
        idx2gene = self.idx2gene
        names = [idx2gene.get(i, f"__idx_{i}__") for i in range(weight.shape[0])]
        columns = [f"dim_{j}" for j in range(weight.shape[1])]
        df = pd.DataFrame(weight, index=names, columns=columns)
        if drop_special_tokens:
            df = df.iloc[N_SPECIAL_TOKENS:]
        if genes is not None:
            keep = [g for g in genes if g in df.index]
            df = df.loc[keep]
        return df

    # ------------------------------------------------------------------ #
    # Context-aware embeddings — AnnData in, DataFrame out
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def embed(
        self,
        adata: ad.AnnData,
        cell_type_key: str = "cell_type",
        sample_key: str = "sample",
        tissue_key: str | None = None,
        target_sum: float = 1e4,
    ) -> dict[str, pd.DataFrame]:
        """Compute context-aware gene embeddings per sample from an AnnData.

        The AnnData is grouped by ``sample_key`` and, within each sample,
        pseudobulked by ``cell_type_key`` using the cell-type ordering the
        model was trained with. Counts are normalized to a
        ``target_sum``-scaled log1p and then passed through the model.
        The returned tensor for each sample has shape ``(n_genes,
        gene_emb_dim)`` — one context-aware vector per gene present in that
        sample, indexed by gene symbol.

        Parameters
        ----------
        adata:
            AnnData with raw or normalized counts in ``.X``, gene symbols
            in ``.var_names``, and ``sample_key``/``cell_type_key`` columns
            in ``.obs``.
        cell_type_key:
            Column in ``adata.obs`` with cell-type labels. Labels not in
            the model's ``cell_type_input_order`` are silently dropped for
            that sample.
        sample_key:
            Column in ``adata.obs`` identifying samples. The output dict is
            keyed by unique values here.
        tissue_key:
            Optional column in ``adata.obs`` with a tissue label per cell.
            When provided and the model has a tissue vocabulary, that
            tissue's token is passed into the forward.
        target_sum:
            Library-size target for log-normalization. Default 1e4 matches
            the paper's training-time preprocessing; changing this WILL
            shift embeddings.

        Returns
        -------
        dict[str, pandas.DataFrame]
            ``{sample_id: DataFrame}`` where each DataFrame is indexed by
            gene symbol and has one column per model embedding dim.

        Examples
        --------
        >>> per_sample = mct.embed(adata,
        ...                        cell_type_key="cell_type",
        ...                        sample_key="donor_id")
        >>> emb = per_sample["donor_42"]         # DataFrame (n_genes x 64)
        >>> emb.loc["EGFR"]                      # context-aware EGFR vector
        """
        _require_columns(adata, [cell_type_key, sample_key])
        gene2idx = self.gene2idx
        cell_type_input_order = self.cell_type_input_order
        device = self.device

        result: dict[str, pd.DataFrame] = {}
        for sample_id in adata.obs[sample_key].unique().tolist():
            sub = adata[adata.obs[sample_key] == sample_id]
            sub = sub[:, [g for g in sub.var_names if g in gene2idx]]
            if sub.n_obs == 0 or sub.n_vars == 0:
                continue

            pseudobulk, gene_ids = _anndata_to_pseudobulk(
                sub, cell_type_key=cell_type_key, gene2idx=gene2idx
            )
            pseudobulk = _normalize_log1p(pseudobulk, target_sum=target_sum)
            input_matrix = _align_to_cell_type_order(pseudobulk, cell_type_input_order)

            expression = torch.tensor(input_matrix, dtype=torch.float32, device=device).unsqueeze(0)
            gene_id_tensor = torch.tensor(gene_ids, dtype=torch.long, device=device).unsqueeze(0)
            is_masked = torch.zeros_like(gene_id_tensor)
            padding_mask = torch.zeros_like(gene_id_tensor, dtype=torch.bool)

            tissue_tensor = _resolve_tissue_tensor(
                sub, tissue_key=tissue_key, tissue2idx=self.index.tissue2idx, device=device
            )

            output: MulticellularTransformerOutput = self.module(
                expression_matrix=expression,
                gene_ids=gene_id_tensor,
                is_masked_gene_expression=is_masked,
                padding_mask=padding_mask,
                tissue=tissue_tensor,
            )
            hidden = output.last_hidden_state.squeeze(0).detach().cpu().numpy()
            columns = [f"dim_{j}" for j in range(hidden.shape[1])]
            result[str(sample_id)] = pd.DataFrame(
                hidden, index=pseudobulk.index.tolist(), columns=columns
            )

        return result


# ---------------------------------------------------------------------- #
# Loader
# ---------------------------------------------------------------------- #


def load(
    model_dir: str | os.PathLike | None = None,
    *,
    checkpoint_path: str | os.PathLike | None = None,
    index_dir: str | os.PathLike | None = None,
    device: str | torch.device | None = None,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> MCTModel:
    """Load a pretrained Multicellular Transformer.

    Two calling styles:

    - ``load("/models/mct_v1")`` — point at a directory with the canonical
      layout: a ``checkpoints/`` (or ``ckpts/``) subdirectory holding one
      ``.ckpt`` file, plus the paired ``gene2idx.json``,
      ``cell_type_input_order.json``, and optionally ``tissue2idx.json``
      alongside it.
    - ``load(checkpoint_path=..., index_dir=...)`` — bypass the directory
      layout and point directly at the individual files.

    Parameters
    ----------
    model_dir:
        Directory to auto-discover. Ignored if both ``checkpoint_path`` and
        ``index_dir`` are given.
    checkpoint_path:
        Explicit ``.ckpt`` path.
    index_dir:
        Explicit directory to load index JSONs from.
    device:
        If given, move the model to this device after loading. Otherwise
        stays on ``map_location``.
    map_location:
        Passed through to ``torch.load`` while reading the checkpoint.
    eval_mode:
        If True (default) set eval mode.

    Returns
    -------
    MCTModel

    Examples
    --------
    >>> import cell_cell_interactions as cci
    >>> mct = cci.load("/models/mct_v1", device="cuda")
    >>> mct.config.gene_emb_dim
    64
    """
    ckpt_path, resolved_index_dir, resolved_model_dir = _resolve_paths(
        model_dir=model_dir, checkpoint_path=checkpoint_path, index_dir=index_dir
    )

    pl_model = MulticellularTransformerPLModel.load_from_checkpoint(
        str(ckpt_path), map_location=map_location
    )
    index = (
        MCTIndex.from_dir(resolved_index_dir)
        if resolved_index_dir is not None
        else MCTIndex()
    )

    model = MCTModel(
        pl_model=pl_model,
        index=index,
        checkpoint_path=str(ckpt_path),
        model_dir=str(resolved_model_dir) if resolved_model_dir is not None else None,
    )
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model


# ---------------------------------------------------------------------- #
# Path resolution / auto-discovery
# ---------------------------------------------------------------------- #


def _resolve_paths(
    model_dir: str | os.PathLike | None,
    checkpoint_path: str | os.PathLike | None,
    index_dir: str | os.PathLike | None,
) -> tuple[Path, Path | None, Path | None]:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint {ckpt} does not exist")
        idx = Path(index_dir) if index_dir is not None else _guess_index_dir(ckpt)
        return ckpt, idx, None

    if model_dir is None:
        raise ValueError(
            "load() requires either a model_dir or both checkpoint_path and index_dir."
        )

    mdir = Path(model_dir)
    if not mdir.exists():
        raise FileNotFoundError(f"model_dir {mdir} does not exist")

    ckpt = _find_checkpoint(mdir)
    idx = Path(index_dir) if index_dir is not None else _guess_index_dir(ckpt, model_dir=mdir)
    return ckpt, idx, mdir


def _find_checkpoint(model_dir: Path) -> Path:
    """Locate a checkpoint under ``model_dir`` using a small search policy.

    Order: prefer files in ``checkpoints/`` (or ``ckpts/``) matching the
    canonical name list; then any ``*.ckpt`` in those dirs; then any
    ``*.ckpt`` at the top level.
    """
    for sub in _CANDIDATE_CKPT_DIRS:
        d = model_dir / sub if sub else model_dir
        if not d.is_dir():
            continue
        for name in _CANDIDATE_CKPT_NAMES:
            cand = d / name
            if cand.exists():
                return cand
        globbed = sorted(d.glob("*.ckpt"))
        if globbed:
            return globbed[0]

    raise FileNotFoundError(
        f"Could not find a .ckpt file under {model_dir}. "
        f"Expected one of {_CANDIDATE_CKPT_NAMES} in {_CANDIDATE_CKPT_DIRS}, "
        "or a single *.ckpt file. Pass checkpoint_path=... to override."
    )


def _guess_index_dir(ckpt_path: Path, model_dir: Path | None = None) -> Path | None:
    """Search for index JSONs alongside the checkpoint or model directory."""
    seen: set[Path] = set()
    candidates: list[Path] = []
    for cand in (ckpt_path.parent, ckpt_path.parent.parent, model_dir):
        if cand is None or cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)

    for cand in candidates:
        if not cand.exists():
            continue
        if _first_existing(cand, _CANDIDATE_INDEX_NAMES["gene2idx"]) is not None:
            return cand
    return None


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------- #
# AnnData -> model input helpers
# ---------------------------------------------------------------------- #


def _require_columns(adata: ad.AnnData, keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"adata.obs is missing required column(s): {missing}. "
            f"Available: {list(adata.obs.columns)}"
        )


def _anndata_to_pseudobulk(
    adata: ad.AnnData,
    cell_type_key: str,
    gene2idx: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sum counts per cell type. Returns (genes x cell_types, gene_ids)."""
    df = adata.to_df().copy()
    df[cell_type_key] = adata.obs[cell_type_key].values
    df = df.loc[df[cell_type_key].notna()]
    grouped = df.groupby([cell_type_key], observed=False).agg("sum")
    grouped = grouped.loc[:, [c for c in grouped.columns if c in gene2idx]]
    gene_ids = np.array([gene2idx[g] for g in grouped.columns], dtype=int)
    return grouped.T, gene_ids


def _normalize_log1p(df: pd.DataFrame, target_sum: float) -> pd.DataFrame:
    total = df.values.sum()
    if total <= 0:
        return df * 0.0
    return np.log1p(df / total * target_sum)


def _align_to_cell_type_order(
    pseudobulk: pd.DataFrame,
    cell_type_input_order: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Place pseudobulk columns into the model's fixed cell-type slot layout."""
    n_slots = len(cell_type_input_order)
    out = np.zeros((pseudobulk.shape[0], n_slots), dtype=np.float32)
    order_index = {ct: i for i, ct in enumerate(cell_type_input_order.keys())}
    for ct in pseudobulk.columns:
        slot = order_index.get(ct)
        if slot is None:
            continue
        out[:, slot] = pseudobulk[ct].values
    return out


def _resolve_tissue_tensor(
    adata: ad.AnnData,
    tissue_key: str | None,
    tissue2idx: dict[str, int] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tissue_key is None or tissue2idx is None:
        return None
    if tissue_key not in adata.obs.columns:
        return None
    tissues = adata.obs[tissue_key].dropna().unique().tolist()
    if not tissues:
        return None
    tissue = tissues[0]
    idx = tissue2idx.get(str(tissue))
    if idx is None:
        return None
    return torch.tensor([idx], dtype=torch.long, device=device)
