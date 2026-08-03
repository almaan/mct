"""User-facing model wrapper.

Loads a pretrained Multicellular Transformer checkpoint together with the
paired gene / cell-type / tissue index files it was trained under, and exposes
a small set of methods that replace the low-level access patterns previously
required in the tutorials.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cell_cell_interactions.modeling.data_types import (
    MulticellularGraphConfig,
    MulticellularTransformerOutput,
)
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel

DEFAULT_INDEX_FILE_NAMES = {
    "gene2idx": ("gene2idx.json", "gene_symbol2idx.json", "ensemble2idx.json"),
    "cell_type_input_order": ("cell_type_input_order.json",),
    "tissue2idx": ("tissue2idx.json",),
    "cell_type_map": ("cell_type_map.json",),
}


@dataclass
class MCTIndex:
    """Metadata files that pair with a Multicellular Transformer checkpoint.

    Fields are all optional because different downstream tasks require different
    subsets. ``gene2idx`` is required for anything that consumes gene
    embeddings by symbol.
    """

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

        def _first_existing(names: tuple[str, ...]) -> Path | None:
            for n in names:
                cand = p / n
                if cand.exists():
                    return cand
            return None

        payload: dict[str, Any] = {}

        gene_path = _first_existing(DEFAULT_INDEX_FILE_NAMES["gene2idx"])
        if gene_path is not None:
            with gene_path.open() as fh:
                gene2idx = json.load(fh)
            payload["gene2idx"] = gene2idx
            payload["idx2gene"] = {int(v): k for k, v in gene2idx.items()}

        ct_path = _first_existing(DEFAULT_INDEX_FILE_NAMES["cell_type_input_order"])
        if ct_path is not None:
            with ct_path.open() as fh:
                raw = json.load(fh)
            payload["cell_type_input_order"] = {k: tuple(v) for k, v in raw.items()}

        tissue_path = _first_existing(DEFAULT_INDEX_FILE_NAMES["tissue2idx"])
        if tissue_path is not None:
            with tissue_path.open() as fh:
                payload["tissue2idx"] = json.load(fh)

        ctmap_path = _first_existing(DEFAULT_INDEX_FILE_NAMES["cell_type_map"])
        if ctmap_path is not None:
            with ctmap_path.open() as fh:
                payload["cell_type_map"] = json.load(fh)

        return cls(**payload)


class MCTModel:
    """User-facing wrapper around a loaded Multicellular Transformer.

    Do not construct directly; use :func:`load_model`.
    """

    def __init__(
        self,
        pl_model: MulticellularTransformerPLModel,
        index: MCTIndex,
        checkpoint_path: str | None = None,
    ) -> None:
        self._pl_model = pl_model
        self.index = index
        self.checkpoint_path = checkpoint_path

    @property
    def config(self) -> MulticellularGraphConfig:
        """The ``MulticellularGraphConfig`` recorded with this checkpoint."""
        return self._pl_model.config

    @property
    def device(self) -> torch.device:
        return next(self._pl_model.parameters()).device

    @property
    def module(self):
        """Underlying ``MulticellularTransformerModule`` (advanced use)."""
        return self._pl_model.model

    @property
    def lightning_module(self) -> MulticellularTransformerPLModel:
        """Underlying Lightning wrapper (advanced use)."""
        return self._pl_model

    def to(self, device: str | torch.device) -> "MCTModel":
        """Move the model to ``device`` and return ``self`` for chaining."""
        self._pl_model.to(device)
        return self

    def eval(self) -> "MCTModel":
        self._pl_model.eval()
        return self

    def gene_embeddings(
        self,
        genes: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return zero-shot gene embeddings from the model's gene-ID classifier head.

        This replaces the raw ``state_dict['model.gene_id_classifier.decoder.weight']``
        access shown in the paper tutorials.

        Parameters
        ----------
        genes:
            If given, only return embeddings for these gene symbols. Symbols
            not found in the model's ``gene2idx`` are skipped. If ``None``,
            all genes in the vocabulary are returned.
        """
        if self.index.gene2idx is None:
            raise RuntimeError(
                "MCTIndex has no gene2idx mapping. Pass index_dir=... to load_model "
                "or set model.index.gene2idx manually."
            )
        weight = self.module.gene_id_classifier.decoder.weight.detach().cpu().numpy()
        if genes is None:
            return {gene: weight[idx] for gene, idx in self.index.gene2idx.items()}
        out: dict[str, np.ndarray] = {}
        for gene in genes:
            idx = self.index.gene2idx.get(gene)
            if idx is None:
                continue
            out[gene] = weight[idx]
        return out

    @torch.no_grad()
    def embed_batch(self, batch) -> MulticellularTransformerOutput:
        """Run a forward pass on a ``BagOfCellsBatch`` and return the raw output.

        Handles moving every input tensor to the model's device. The user is
        responsible for iterating a DataLoader; see :func:`embed_dataloader`
        for the batched equivalent.
        """
        device = self.device

        def _to(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        return self.module(
            expression_matrix=_to(batch.expression_matrix),
            gene_ids=_to(batch.gene_ids),
            is_masked_gene_expression=_to(batch.is_masked_gene_expression),
            padding_mask=_to(batch.padding_mask),
            tissue=_to(batch.tissue_idx) if batch.tissue_idx is not None else None,
        )


def load_model(
    checkpoint_path: str | os.PathLike,
    index_dir: str | os.PathLike | None = None,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> MCTModel:
    """Load a pretrained Multicellular Transformer as an :class:`MCTModel`.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.ckpt`` file saved by ``MulticellularTransformerPLModel``.
    index_dir:
        Directory containing ``gene2idx.json`` (or ``gene_symbol2idx.json``),
        ``cell_type_input_order.json``, ``tissue2idx.json``,
        ``cell_type_map.json``. Any subset may be present. If ``None``, the
        loader looks in the directory of the checkpoint and then in a
        ``tutorials/data`` directory relative to it.
    map_location:
        Passed through to ``torch.load``.
    eval_mode:
        If True, put the model in eval mode after loading.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} does not exist")

    pl_model = MulticellularTransformerPLModel.load_from_checkpoint(
        str(ckpt_path), map_location=map_location
    )

    if index_dir is None:
        index = _autodiscover_index(ckpt_path)
    else:
        index = MCTIndex.from_dir(index_dir)

    model = MCTModel(pl_model, index, checkpoint_path=str(ckpt_path))
    if eval_mode:
        model.eval()
    return model


def _autodiscover_index(ckpt_path: Path) -> MCTIndex:
    """Best-effort search for index files near the checkpoint."""
    candidates = [
        ckpt_path.parent,
        ckpt_path.parent / "index",
        ckpt_path.parent.parent / "tutorials" / "data",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                idx = MCTIndex.from_dir(cand)
                if idx.gene2idx is not None:
                    return idx
            except FileNotFoundError:
                continue
    return MCTIndex()
