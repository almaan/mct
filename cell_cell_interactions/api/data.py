"""Inference-time presets for the three datamodule kinds.

Each preset is a small dataclass with defaults that match how the paper
tutorials build their datamodules for inference. ``build()`` returns the
underlying Lightning datamodule; the user can still pass any advanced kwarg
via ``extra=``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AdataInferencePreset:
    """Preset for :class:`AdataRnaseqDataModule` used in inference.

    Only the fields likely to change per-user are surfaced; anything else can
    be supplied via ``extra``. Defaults mirror the values used across the
    inference notebooks.
    """

    adata_file_path: str
    gene2idx: dict[str, int]
    sample_size: int = 512
    label_column: str = "cell_type"
    sample_column: str = "sample"
    tissue_column: str = "tissue"
    batch_size: int = 1
    num_workers: int = 0
    max_seq_len: int = 12000
    cell_type_mapping: dict[str, str] | None = None
    restrict_to_conditions: dict[str, Any] | None = None
    val_set_proportion: float = 1.0  # inference: send everything through the val loader
    mask_prob: float = 0.0
    mask_tissue_prob: float = 0.0
    random_seed: int = 32
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self):
        """Instantiate the underlying :class:`AdataRnaseqDataModule`."""
        from cell_cell_interactions.pp.adata_rnaseq_datamodule import AdataRnaseqDataModule

        kwargs = {
            "adata_file_path": self.adata_file_path,
            "gene2idx": self.gene2idx,
            "sample_size": self.sample_size,
            "label_column": self.label_column,
            "sample_column": self.sample_column,
            "tissue_column": self.tissue_column,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_seq_len": self.max_seq_len,
            "cell_type_mapping": self.cell_type_mapping,
            "restrict_to_conditions": self.restrict_to_conditions,
            "val_set_proportion": self.val_set_proportion,
            "mask_prob": self.mask_prob,
            "mask_tissue_prob": self.mask_tissue_prob,
            "random_seed": self.random_seed,
        }
        kwargs.update(self.extra)
        return AdataRnaseqDataModule(**kwargs)


@dataclass
class CellarrInferencePreset:
    """Preset for :class:`CellarrDataModule` used in inference."""

    cell_tdb_uri: str
    gene_tdb_uri: str
    counts_tdb_uri: str
    sample_size: int = 512
    label_column: str = "cellTypeName"
    study_column: str = "study"
    sample_column: str = "sampleID"
    tissue_column: str = "tissue"
    batch_size: int = 1
    num_workers: int = 0
    max_seq_len: int = 12000
    cell_type_mapping: dict[str, str] | None = None
    restrict_to_conditions: dict[str, Any] | None = None
    lognorm: bool = True
    target_sum: float = 1e4
    val_set_proportion: float = 1.0
    mask_prob: float = 0.0
    mask_tissue_prob: float = 0.0
    random_seed: int = 32
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self):
        from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule

        kwargs = {
            "cell_tdb_uri": self.cell_tdb_uri,
            "gene_tdb_uri": self.gene_tdb_uri,
            "counts_tdb_uri": self.counts_tdb_uri,
            "sample_size": self.sample_size,
            "label_column": self.label_column,
            "study_column": self.study_column,
            "sample_column": self.sample_column,
            "tissue_column": self.tissue_column,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_seq_len": self.max_seq_len,
            "cell_type_mapping": self.cell_type_mapping,
            "restrict_to_conditions": self.restrict_to_conditions,
            "lognorm": self.lognorm,
            "target_sum": self.target_sum,
            "val_set_proportion": self.val_set_proportion,
            "mask_prob": self.mask_prob,
            "mask_tissue_prob": self.mask_tissue_prob,
            "random_seed": self.random_seed,
        }
        kwargs.update(self.extra)
        return CellarrDataModule(**kwargs)


@dataclass
class SpatialInferencePreset:
    """Preset for :class:`SpatialDataModule` used in inference."""

    adata_file_path: str
    gene2idx: dict[str, int]
    sample_size: int = 512
    label_column: Literal["leiden"] | str = "leiden"
    batch_size: int = 1
    num_workers: int = 0
    max_seq_len: int = 2500
    val_set_proportion: float = 1.0
    mask_prob: float = 0.0
    random_seed: int = 42
    restrict_to_conditions: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self):
        from cell_cell_interactions.pp.spatial_datamodule import SpatialDataModule

        kwargs = {
            "adata_file_path": self.adata_file_path,
            "gene2idx": self.gene2idx,
            "sample_size": self.sample_size,
            "label_column": self.label_column,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "max_seq_len": self.max_seq_len,
            "val_set_proportion": self.val_set_proportion,
            "mask_prob": self.mask_prob,
            "random_seed": self.random_seed,
            "restrict_to_conditions": self.restrict_to_conditions,
        }
        kwargs.update(self.extra)
        return SpatialDataModule(**kwargs)
