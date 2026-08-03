from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from torch.utils.data import DataLoader


@dataclass_json
@dataclass
class DataReaderOutput:
    """Dataclass for the output of the data reader."""

    train_dataloader: DataLoader
    val_dataloader: DataLoader
    cell_type_counts: dict[str, int] = None
    cell_type_input_order: list[str] = None
    test_dataloader: DataLoader = None
    train_dataset_len: int = None
    n_cell_types: int = None
    gene_to_idx: dict[str, int] = None


@dataclass_json
@dataclass
class BagOfCellsSample:
    """Dataclass for a sample in the BagOfCellsDataset."""

    expression_matrix: torch.Tensor
    gene_ids: torch.Tensor = None
    labels: torch.Tensor = None
    is_masked_gene_id: torch.Tensor = None
    is_masked_gene_expression: torch.Tensor = None
    tissue_idx: int = None
    cell_type_indices: dict[str, list[int]] = None
    study_label: str = None
    sample_label: str = None
    n_expressed_genes: int = None
    disease: str = None
    cell_indices: torch.Tensor = None
    cell_idx: int = None


@dataclass_json
@dataclass
class BagOfCellsDatasetOutput:
    """Dataclass for a sample in the BagOfCellsDataset."""

    cell_indices: list[int]
    study_label: str
    sample_label: str
    tissue_idx: int
    cell_type_indices: dict[str, list[int]] = None
    cell_pad_mask: list[int] = None
    disease: str = None


@dataclass_json
@dataclass
class BagOfCellsBatch:
    """Dataclass for a sample in the BagOfCellsDataset."""

    expression_matrix: torch.Tensor
    gene_ids: torch.Tensor = None
    labels: torch.Tensor = None
    is_masked_gene_id: torch.Tensor = None
    is_masked_gene_expression: torch.Tensor = None
    tissue_labels: torch.Tensor = None
    padding_mask: torch.Tensor = None
    tissue_idx: torch.Tensor = None
    cell_type_indices: list[dict[str, list[int]]] = None
    study_label: list[str] = None
    sample_label: list[str] = None
    disease: list[str] = None
    n_expressed_genes: torch.Tensor = None
    cell_indices: torch.Tensor = None
    cell_idx: torch.Tensor = None
