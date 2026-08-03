import random
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from cell_cell_interactions.pp.cellarr_datamodule import GENE_MASKING_STRATEGY, TISSUE_MASKING_STRATEGY
from cell_cell_interactions.pp.data_types import BagOfCellsBatch, BagOfCellsSample
from cell_cell_interactions.pp.utils import get_cell_type_order, group_cell_types_by_sample


def _build_sample_index(data_df: pd.DataFrame, sample_column: str, min_sample_size: int | None) -> tuple[dict, dict]:
    grouped = data_df.groupby(sample_column)
    sample2cells = {sample: cells.tolist() for sample, cells in grouped.groups.items()}
    if min_sample_size is not None:
        sample2cells = {sample: cells for sample, cells in sample2cells.items() if len(cells) >= min_sample_size}
    int2sample = dict(enumerate(sample2cells.keys()))
    return int2sample, sample2cells


class AdataRnaseqDataset(torch.utils.data.Dataset):
    """A class to load and process spatial data for training and evaluation."""

    def __init__(
        self,
        adata: sc.AnnData,
        data_df: pd.DataFrame,
        gene2idx: dict[str, int],  # global gene2idx mapping from pretraining
        sample2cells: dict,
        int2sample: dict,
        tissue2idx: dict,
        sample_size: int,
        label_column: str = "cell_type_major",
        sample_column: str = "sample",
        tissue_column: str = "tissue",
        agg_cell_type_gex: bool = False,
        sample_sampling: Literal["random", "weighted"] = "weighted",
        cell_sampling: Literal["random", "stratified", "all_cells"] = "stratified",  # sampling strategy for cells
        cell_type_input_order: dict[str, tuple[int, int]] = None,
        mask_prob: float = 0.15,
        cell_pad_idx: int = 0,
        min_n_expressed_cells_per_gene: int = 3,
        disease_column: str = "disease",
        max_seq_len: int = 12000,
    ):
        super().__init__()
        self.adata = adata
        self.data_df = data_df
        self.gene2idx = gene2idx
        self.sample2cells = sample2cells
        self.int2sample = int2sample
        self.tissue2idx = tissue2idx
        self.sample_size = sample_size
        self.label_column = label_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column
        self.agg_cell_type_gex = agg_cell_type_gex
        self.sample_sampling = sample_sampling
        self.cell_sampling = cell_sampling
        self.cell_type_input_order = cell_type_input_order
        self.mask_prob = mask_prob
        self.cell_pad_idx = cell_pad_idx
        self.disease_column = disease_column
        self.min_n_expressed_cells_per_gene = min_n_expressed_cells_per_gene
        self.max_seq_len = max_seq_len

        self.samples = list(sample2cells.keys())
        self.cells_per_sample = [len(sample) for sample in sample2cells.values()]
        self.n_cells = len(data_df)

        self.cell_types = list(cell_type_input_order.keys())
        self.cell_types_grouped_by_sample = group_cell_types_by_sample(
            data_df, self.sample_column, self.label_column, self.cell_types
        )

        self.n_genes = adata.shape[1]

        self.global_gene_ids = torch.tensor([self.gene2idx[g] for g in self.adata.var.index], dtype=torch.long)

        if self.cell_sampling == "all_cells":
            assert (
                self.agg_cell_type_gex is True
            ), "If using `cell_sampling=True`, `agg_cell_type_gex` must also be set to True, it is currently set to False."

    def __len__(self) -> int:
        # if sampling by sample we randomly pick samples
        if self.sample_sampling == "random":
            return len(self.int2sample)
        # if sampling is weighted we select from larger samples more often
        return self.n_cells

    def __getitem__(self, idx) -> BagOfCellsSample:
        # fetch the sample
        if self.sample_sampling == "random":
            sample = self.int2sample[idx]
        else:
            sample = random.choices(self.samples, weights=self.cells_per_sample, k=1)[0]

        sample_cell_indices = self.sample2cells[sample]
        if self.cell_sampling != "all_cells":
            if len(sample_cell_indices) >= self.sample_size:
                cell_idx = random.sample(sample_cell_indices, self.sample_size)
            else:
                cell_idx = random.choices(sample_cell_indices, k=self.sample_size)
        else:
            # sample all the cells, used only with aggregation the information
            # by cell type, to be used for inference
            cell_idx = sample_cell_indices

        # fetch label data
        sample_label = self.data_df.loc[cell_idx, self.sample_column].iloc[0]

        tissue_label = None
        if self.tissue_column in self.data_df.columns:
            tissue_label = self.data_df.loc[cell_idx, self.tissue_column].iloc[0]

        disease = None
        if self.disease_column in self.data_df.columns:
            disease = self.data_df.loc[cell_idx, self.disease_column].iloc[0]

        if self.cell_sampling == "random":
            # fetch the cell indices
            labels = self.data_df.loc[cell_idx, self.label_column]
            # get cell type indices
            label_indices = defaultdict(list)
            for idx, ct in zip(cell_idx, labels):
                label_indices[ct].append(idx)
            cell_idx, cell_pad_mask = self.order_cells_in_input(label_indices)
        elif self.cell_sampling == "stratified":
            # fetch the cell indices per each cell type in a sample
            sample_data = self.cell_types_grouped_by_sample.loc[sample]
            cell_idx, cell_pad_mask = self.order_cells_in_input(sample_data)
        else:
            cell_pad_mask = [0.0] * len(cell_idx)

        # fetch gene expression data
        X = self.adata.X[cell_idx, :].toarray()
        # zero out the items which were padded
        X = torch.tensor(np.where(np.array(cell_pad_mask)[:, None], 0.0, X), dtype=torch.float32)

        expressed_gene_indices = ((X > 0).sum(0) > self.min_n_expressed_cells_per_gene).nonzero().squeeze(-1)
        X = X[:, expressed_gene_indices]
        n_expressed_genes = len(expressed_gene_indices)

        if X.shape[1] > self.max_seq_len:
            selected_genes = torch.randperm(X.shape[1])[: self.max_seq_len]
            X = X[:, selected_genes]
            expressed_gene_indices = expressed_gene_indices[selected_genes]

        if self.agg_cell_type_gex and self.cell_sampling != "all_cells":
            output = [X[start:end].sum(0) for _, (start, end) in self.cell_type_input_order.items()]
            X = torch.stack(output)
        elif self.cell_sampling == "all_cells":
            labels = self.data_df.loc[cell_idx, self.label_column]
            X = self.order_all_cells_in_input(X=X, labels=labels)

        # map local gene indices to global gene indices (pretrained)
        expressed_gene_indices = self.global_gene_ids[expressed_gene_indices]

        # do the masking
        expressed_gene_indices_w_mask, is_masked_gene_id, is_masked_gene_expression = self.mask(expressed_gene_indices)

        return BagOfCellsSample(
            expression_matrix=X.T,
            gene_ids=expressed_gene_indices_w_mask,
            labels=expressed_gene_indices,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            tissue_idx=self.tissue2idx.get(tissue_label, None),
            sample_label=sample_label,
            n_expressed_genes=n_expressed_genes,
            disease=disease,
        )

    def mask(self, gene_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask gene indices."""
        # Step 1: Calculate the number of tokens to mask
        gene_indices_w_mask = gene_indices.clone()
        seq_len = gene_indices_w_mask.size(0)
        num_masked_genes = int(seq_len * self.mask_prob)

        # Step 2: Randomly select indices for masking
        masked_indices = torch.randperm(seq_len)[:num_masked_genes]

        # Step 3: Create a mask of the same size as the sequence
        is_masked = torch.zeros_like(gene_indices_w_mask).type_as(gene_indices_w_mask)
        is_masked[masked_indices] = 1

        # Step 4: Randomly assign the masking strategy
        probs = torch.rand(num_masked_genes)

        mask_token = self.gene2idx["[MASK]"]
        # 70% Replace with mask token
        mask_gene_prob = GENE_MASKING_STRATEGY["mask_gene_id"]
        gene_indices_w_mask[masked_indices[probs < mask_gene_prob]] = mask_token

        # 10% replace with random gene
        random_gene_prob = GENE_MASKING_STRATEGY["random"] + mask_gene_prob
        random_indices = masked_indices[(probs >= mask_gene_prob) & (probs < random_gene_prob)]
        gene_indices_w_mask[random_indices] = torch.randint(2, self.n_genes, (len(random_indices),))

        # 10% mask gene expression
        mask_gene_expression_prob = GENE_MASKING_STRATEGY["mask_gene_expression"] + random_gene_prob
        mask_gene_expression = masked_indices[(probs >= random_gene_prob) & (probs < mask_gene_expression_prob)]
        is_masked_gene_expression = torch.zeros_like(gene_indices_w_mask).type_as(gene_indices_w_mask)
        is_masked_gene_expression[mask_gene_expression] = 1

        # 10% we do not mask so we skip it

        return gene_indices_w_mask, is_masked, is_masked_gene_expression

    def order_cells_in_input(self, sample_data: pd.Series | dict) -> tuple[list[int], list[int]]:
        """Stratified sampling of cell types."""
        cell_indices = []
        cell_pad_mask = []
        for ct, (start, end) in self.cell_type_input_order.items():
            n_ct_cells = end - start
            ct_data = sample_data.get(ct, [])
            if len(ct_data) >= n_ct_cells:
                # sample cells
                ct_indices = random.sample(ct_data, n_ct_cells)
                cell_indices += ct_indices
                cell_pad_mask += [0] * n_ct_cells
            elif len(ct_data) > 0:
                n_to_fill = n_ct_cells - len(ct_data)
                random.shuffle(ct_data)
                cell_indices += ct_data + [self.cell_pad_idx] * n_to_fill
                cell_pad_mask += [0] * len(ct_data) + [1] * n_to_fill
            else:
                cell_indices += [self.cell_pad_idx] * n_ct_cells
                cell_pad_mask += [1] * n_ct_cells

        return cell_indices, cell_pad_mask

    def order_all_cells_in_input(self, X: np.ndarray, labels: list[int]) -> np.ndarray:
        """Order all cells in the input without sampling."""
        # get cell type indices
        label_indices = defaultdict(list)
        for idx, ct in enumerate(labels):
            label_indices[ct].append(idx)
        output = []
        for label in self.cell_type_input_order.keys():
            if label in label_indices:
                output.append(X[label_indices[label], :].sum(0))
            else:
                output.append(torch.zeros(X.shape[1]))
        X = torch.stack(output)
        return X


class AdataRnaseqDataModule(LightningDataModule):
    """A class to load and process adata object for training and evaluation."""

    def __init__(
        self,
        adata_file_path: str,
        gene2idx: dict[str, int],  # ENSEMBL gene ID to index mapping
        sample_size: int = 512,
        val_samples: list[str] | None = None,
        val_set_proportion: float = 0.2,
        label_column: str = "cell_type_major",
        sample_column: str = "sample",
        tissue_column: str = "tissue",
        batch_size: int = 1,
        num_workers: int = 0,
        agg_cell_type_gex: bool = False,
        min_sample_size: int | None = None,
        min_cell_type_size: float | None = 0.001,
        sample_sampling: Literal["random", "weighted"] = "weighted",
        cell_sampling: Literal["random", "stratified"] = "stratified",  # sampling strategy for cells
        pad_value_expression: float = 0.0,
        mask_prob: float = 0.15,
        mask_tissue_prob: float = 0.0,
        max_seq_len: int = 12000,
        random_seed: int = 32,
        perc_min_expressed_cells_per_gene: float = 0.01,
        backed: bool = False,
        cell_type_mapping: dict | None = None,
        restrict_to_conditions: dict | None = None,  # restrict to specific conditions, e.g. {"study": "study1"}
    ):
        super().__init__()
        self.adata_file_path = adata_file_path
        self.gene2idx = gene2idx
        self.sample_size = sample_size
        self.val_samples = val_samples
        self.val_set_proportion = val_set_proportion
        self.label_column = label_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column
        self.min_sample_size = min_sample_size
        self.min_cell_type_size = min_cell_type_size
        self.sample_sampling = sample_sampling
        self.cell_sampling = cell_sampling
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.agg_cell_type_gex = agg_cell_type_gex
        self.pad_value_expression = pad_value_expression
        self.mask_prob = mask_prob
        self.mask_tissue_prob = mask_tissue_prob
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.restrict_to_conditions = restrict_to_conditions
        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)
        random.seed(random_seed)

        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)

        # load the data
        self.adata = sc.read_h5ad(adata_file_path, backed="r" if backed else None)
        self.data_df = self.adata.obs.copy().reset_index()

        assert (
            self.label_column in self.data_df.columns
        ), f"label_column: {self.label_column} not present in obs columns, make sure it is present!"

        # make the columns not categorical which results in error later
        self.data_df[self.label_column] = self.data_df[self.label_column].astype(str)
        self.data_df[self.sample_column] = self.data_df[self.sample_column].astype(str)

        if cell_type_mapping is not None:
            self.data_df[self.label_column] = self.data_df[self.label_column].map(cell_type_mapping)

        # filter out cell types with less than min_cell_type_size cells
        if self.min_cell_type_size is not None:
            self.cell_type_proportions = self.data_df[self.label_column].value_counts(normalize=True)
            cell_types = [ct for ct, p in self.cell_type_proportions.items() if p >= self.min_cell_type_size]
            self.data_df = self.data_df[self.data_df[self.label_column].isin(cell_types)]

        # filter out sample without enough cells
        if self.min_sample_size is not None:
            sample_sizes = self.data_df[self.sample_column].value_counts()
            samples = sample_sizes[sample_sizes >= self.min_sample_size].index
            self.data_df = self.data_df[self.data_df[self.sample_column].isin(samples)]

        self.cell_type_sampled = dict(self.data_df[self.label_column].value_counts().items())

        s = sum(self.cell_type_sampled.values())
        self.cell_type_proportions = {ct: c / s for ct, c in self.cell_type_sampled.items()}
        self.cell_type_input_order = get_cell_type_order(self.cell_type_proportions, self.sample_size)

        # TODO: add support for using pretrained tissue embeddings
        self.tissue2idx = {"[PAD]": 0, "[MASK]": 1}
        if self.tissue_column in self.data_df.columns:
            self.tissue2idx.update({t: idx + 2 for idx, t in enumerate(self.data_df[self.tissue_column].unique())})

        # restrict to specific conditions
        if restrict_to_conditions is not None:
            for key, value in restrict_to_conditions.items():
                self.data_df = self.data_df[self.data_df[key] == value]

        self.val_df = None
        if self.val_samples is not None:
            # split out validation studies
            self.val_df = self.data_df[self.data_df[self.sample_column].isin(self.val_samples)]
            self.data_df = self.data_df[~self.data_df[self.sample_column].isin(self.val_samples)]
            # limit validation celltypes to those in the training data
            self.val_df = self.val_df[self.val_df[self.label_column].isin(self.data_df[self.label_column].unique())]
        else:
            # split out validation samples
            all_samples = self.data_df[self.sample_column].unique().tolist()
            random.shuffle(all_samples)
            val_samples = all_samples[: int(val_set_proportion * len(all_samples))]
            self.val_df = self.data_df[self.data_df[self.sample_column].isin(val_samples)]
            self.data_df = self.data_df[~self.data_df[self.sample_column].isin(val_samples)]

        print(f"Training data size: {self.data_df.shape}")
        if self.val_df is not None:
            print(f"Validation data size: {self.val_df.shape}")

        self.class_names = sorted(set(self.data_df[self.label_column].values))
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        self.n_genes = self.adata.shape[1]

        self.train_int2sample, self.train_sample2cells = _build_sample_index(
            self.data_df, self.sample_column, self.min_sample_size
        )
        self.train_dataset = AdataRnaseqDataset(
            adata=self.adata,
            data_df=self.data_df,
            gene2idx=self.gene2idx,
            sample2cells=self.train_sample2cells,
            int2sample=self.train_int2sample,
            tissue2idx=self.tissue2idx,
            sample_size=self.sample_size,
            label_column=self.label_column,
            mask_prob=self.mask_prob,
            sample_column=self.sample_column,
            tissue_column=self.tissue_column,
            agg_cell_type_gex=self.agg_cell_type_gex,
            sample_sampling=sample_sampling,
            cell_sampling=cell_sampling,
            cell_type_input_order=self.cell_type_input_order,
            max_seq_len=self.max_seq_len,
            min_n_expressed_cells_per_gene=self.min_n_expressed_cells_per_gene,
            disease_column="disease",
        )

        self.val_dataset = None
        if len(self.val_df) > 0:
            self.val_int2sample, self.val_sample2cells = _build_sample_index(
                self.val_df, self.sample_column, self.min_sample_size
            )
            self.val_dataset = AdataRnaseqDataset(
                adata=self.adata,
                data_df=self.val_df,
                gene2idx=self.gene2idx,
                sample2cells=self.val_sample2cells,
                int2sample=self.val_int2sample,
                tissue2idx=self.tissue2idx,
                sample_size=self.sample_size,
                label_column=self.label_column,
                sample_column=self.sample_column,
                mask_prob=self.mask_prob,
                tissue_column=self.tissue_column,
                agg_cell_type_gex=self.agg_cell_type_gex,
                sample_sampling=sample_sampling,
                cell_sampling=cell_sampling,
                cell_type_input_order=self.cell_type_input_order,
                max_seq_len=self.max_seq_len,
                min_n_expressed_cells_per_gene=self.min_n_expressed_cells_per_gene,
                disease_column="disease",
            )

    def collate(self, batch: list[BagOfCellsSample]) -> BagOfCellsBatch:
        """Collate tensors"""
        gex = pad_sequence([x.expression_matrix for x in batch], batch_first=True, padding_value=0.0)

        # pad sequences
        is_masked_gene_id = pad_sequence([x.is_masked_gene_id for x in batch], batch_first=True, padding_value=0.0)
        is_masked_gene_expression = pad_sequence(
            [x.is_masked_gene_expression for x in batch], batch_first=True, padding_value=0.0
        )

        # 0 = padding_token_idx
        gene_ids = pad_sequence([x.gene_ids for x in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([x.labels for x in batch], batch_first=True, padding_value=-100)
        # fetch number of genes which were not filtered out in each sample
        n_expressed_genes = torch.Tensor([x.n_expressed_genes for x in batch])

        padding_mask = [torch.zeros_like(x.is_masked_gene_id) for x in batch]
        padding_mask = pad_sequence(padding_mask, padding_value=1.0, batch_first=True).bool()

        tissue_idx = [x.tissue_idx for x in batch]
        if None not in tissue_idx:
            tissue_idx = torch.tensor(tissue_idx, dtype=torch.long)
            tissue_idx, tissue_labels = self.mask_tissue(tissue_idx)
        else:
            tissue_idx = None
            tissue_labels = None

        return BagOfCellsBatch(
            expression_matrix=gex,
            gene_ids=gene_ids,
            labels=labels,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            padding_mask=padding_mask,
            n_expressed_genes=n_expressed_genes,
            disease=[x.disease for x in batch],
            sample_label=[x.sample_label for x in batch],
            tissue_idx=tissue_idx,
            tissue_labels=tissue_labels,
        )

    def mask_tissue(self, tissue_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask tissue indices."""
        # Step 1: Calculate the number of tokens to mask
        tissue_labels = torch.ones_like(tissue_indices) * -100  # -100 is the ignore label
        seq_len = tissue_indices.size(0)
        num_masked_tissues = int(seq_len * self.mask_tissue_prob)
        if num_masked_tissues == 0:
            return tissue_indices, tissue_labels

        # Step 2: Randomly select indices for masking
        masked_indices = torch.randperm(seq_len)[:num_masked_tissues]
        # assign the tissue labels
        tissue_labels[masked_indices] = tissue_indices[masked_indices]

        # Step 3: Randomly assign the masking strategy
        probs = torch.rand(num_masked_tissues)

        mask_token = self.tissue2idx["[MASK]"]
        # 80% Replace with mask token
        mask_tissue_prob = TISSUE_MASKING_STRATEGY["mask"]
        tissue_indices[masked_indices[probs < mask_tissue_prob]] = mask_token

        # 10% replace with random tissue
        random_gene_prob = TISSUE_MASKING_STRATEGY["random"] + mask_tissue_prob
        random_indices = masked_indices[(probs >= mask_tissue_prob) & (probs < random_gene_prob)]
        tissue_indices[random_indices] = torch.randint(2, len(self.tissue2idx), (len(random_indices),))

        # 10% we do not mask so we skip it
        return tissue_indices, tissue_labels

    def train_dataloader(self) -> DataLoader:
        """Load the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Load the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the validation dataset.
        """
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """
        return self.val_dataloader()


class AdataRnaseqDataModuleForInference(LightningDataModule):
    """A class to load and process adata object for inference.

    This datamodule uses existing cell type input order and does not adapt it to a dataset
    in contrary to AdataRnaseqDataModule and CellarrDataModule.
    The datamodule should be used for running inference on datasets
    which are too small for finetuning.
    """

    def __init__(
        self,
        adata_file_path: str,
        gene2idx: dict[str, int],  # ENSEMBL gene ID to index mapping
        pretrained_label_input_order: dict[str, int],  # cell_type_input_order from a pretrained model
        sample_size: int = 16,  # significantly smaller than for training
        label_column: str = "cell_type_major",
        sample_column: str = "sample",
        tissue_column: str = "tissue",
        batch_size: int = 1,
        num_workers: int = 0,
        agg_cell_type_gex: bool = False,
        min_sample_size: int | None = None,
        sample_sampling: Literal["random", "weighted"] = "weighted",
        cell_sampling: Literal["random", "stratified", "all_cells"] = "stratified",  # sampling strategy for cells
        pad_value_expression: float = 0.0,
        mask_prob: float = 0.0,
        mask_tissue_prob: float = 0.0,
        max_seq_len: int = 12000,
        random_seed: int = 32,
        perc_min_expressed_cells_per_gene: float = 0.001,
        backed: bool = False,
        label_mapping: dict | None = None,
        restrict_to_conditions: dict | None = None,  # restrict to specific conditions, e.g. {"study": "study1"}
    ):
        super().__init__()
        self.adata_file_path = adata_file_path
        self.gene2idx = gene2idx
        self.pretrained_label_input_order = pretrained_label_input_order
        self.sample_size = sample_size
        self.label_column = label_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column
        self.min_sample_size = min_sample_size
        self.sample_sampling = sample_sampling
        self.cell_sampling = cell_sampling
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.agg_cell_type_gex = agg_cell_type_gex
        self.pad_value_expression = pad_value_expression
        self.mask_prob = mask_prob
        self.mask_tissue_prob = mask_tissue_prob
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.restrict_to_conditions = restrict_to_conditions
        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)
        random.seed(random_seed)

        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)

        # load the data
        self.adata = sc.read_h5ad(adata_file_path, backed="r" if backed else None)
        self.data_df = self.adata.obs.copy().reset_index()

        # make the columns not categorical which results in error later
        self.data_df[self.label_column] = self.data_df[self.label_column].astype(str)
        self.data_df[self.sample_column] = self.data_df[self.sample_column].astype(str)

        # map labels
        if label_mapping is not None:
            self.data_df[self.label_column] = self.data_df[self.label_column].map(label_mapping)

        # get labels in the AnnData
        adata_labels = self.data_df[label_column].unique().tolist()
        # get labels present in the AnnData but not in the pretrained_cell_type_input_order
        adata_labels_not_present_in_pretrained_cts = [
            i for i in adata_labels if i not in pretrained_label_input_order.keys()
        ]
        # ensure there is overlap between pretrained_label_input_order and labels present in AnnData
        if len(adata_labels_not_present_in_pretrained_cts) == len(adata_labels):
            raise ValueError(
                "No overlap between labels present in the AnnData and pretrained label input order."
                "Please make sure there is at least one overlapping label. Shutting down."
            )
        if len(adata_labels_not_present_in_pretrained_cts) > 0:
            print(
                f"WARNING! There are {len(adata_labels_not_present_in_pretrained_cts)} cell types in AnnData which do not match the pretrained labels."
                f"List of unmatched labels: {adata_labels_not_present_in_pretrained_cts}"
            )

        assert (
            self.label_column in self.data_df.columns
        ), f"label_column: {self.label_column} not present in obs columns, make sure it is present!"

        # make the columns not categorical which results in error later
        self.data_df[self.label_column] = self.data_df[self.label_column].astype(str)
        self.data_df[self.sample_column] = self.data_df[self.sample_column].astype(str)

        # TODO: add support for using pretrained tissue embeddings
        self.tissue2idx = {"[PAD]": 0, "[MASK]": 1}
        if self.tissue_column in self.data_df.columns:
            self.tissue2idx.update({t: idx + 2 for idx, t in enumerate(self.data_df[self.tissue_column].unique())})

        # restrict to specific conditions
        if restrict_to_conditions is not None:
            for key, value in restrict_to_conditions.items():
                self.data_df = self.data_df[self.data_df[key] == value]

        # filter out sample without enough cells
        if self.min_sample_size is not None:
            sample_sizes = self.data_df[self.sample_column].value_counts()
            samples = sample_sizes[sample_sizes >= self.min_sample_size].index
            self.data_df = self.data_df[self.data_df[self.sample_column].isin(samples)]

        self.class_names = sorted(set(self.data_df[self.label_column].values))
        self.label2int = {label: i for i, label in enumerate(self.class_names)}
        self.int2label = {value: key for key, value in self.label2int.items()}

        self.n_genes = self.adata.shape[1]

        self.int2sample, self.sample2cells = _build_sample_index(
            self.data_df, self.sample_column, self.min_sample_size
        )

        print(f"Nr of cells available: {sum([len(cells) for cells in self.sample2cells.values()])}")
        print(f"Nr of samples available: {len(self.sample2cells)}")

        self.dataset = AdataRnaseqDataset(
            adata=self.adata,
            data_df=self.data_df,
            gene2idx=self.gene2idx,
            sample2cells=self.sample2cells,
            int2sample=self.int2sample,
            tissue2idx=self.tissue2idx,
            sample_size=self.sample_size,
            label_column=self.label_column,
            mask_prob=self.mask_prob,
            sample_column=self.sample_column,
            tissue_column=self.tissue_column,
            agg_cell_type_gex=self.agg_cell_type_gex,
            sample_sampling=sample_sampling,
            cell_sampling=cell_sampling,
            cell_type_input_order=self.pretrained_label_input_order,
            max_seq_len=self.max_seq_len,
            min_n_expressed_cells_per_gene=self.min_n_expressed_cells_per_gene,
            disease_column="disease",
        )

    def collate(self, batch: list[BagOfCellsSample]) -> BagOfCellsBatch:
        """Collate tensors"""
        gex = pad_sequence([x.expression_matrix for x in batch], batch_first=True, padding_value=0.0)

        # pad sequences
        is_masked_gene_id = pad_sequence([x.is_masked_gene_id for x in batch], batch_first=True, padding_value=0.0)
        is_masked_gene_expression = pad_sequence(
            [x.is_masked_gene_expression for x in batch], batch_first=True, padding_value=0.0
        )

        # 0 = padding_token_idx
        gene_ids = pad_sequence([x.gene_ids for x in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([x.labels for x in batch], batch_first=True, padding_value=-100)
        # fetch number of genes which were not filtered out in each sample
        n_expressed_genes = torch.Tensor([x.n_expressed_genes for x in batch])

        padding_mask = [torch.zeros_like(x.is_masked_gene_id) for x in batch]
        padding_mask = pad_sequence(padding_mask, padding_value=1.0, batch_first=True).bool()

        tissue_idx = [x.tissue_idx for x in batch]
        if None not in tissue_idx:
            tissue_idx = torch.tensor(tissue_idx, dtype=torch.long)
            tissue_labels = None
        else:
            tissue_idx = None
            tissue_labels = None

        return BagOfCellsBatch(
            expression_matrix=gex,
            gene_ids=gene_ids,
            labels=labels,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            padding_mask=padding_mask,
            n_expressed_genes=n_expressed_genes,
            disease=[x.disease for x in batch],
            sample_label=[x.sample_label for x in batch],
            tissue_idx=tissue_idx,
            tissue_labels=tissue_labels,
        )

    def train_dataloader(self) -> DataLoader:
        """Load the train dataset"""
        raise ValueError("Train dataloader not available for the inference dataset, please use validation dataloader.")

    def val_dataloader(self) -> DataLoader:
        """Load the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the validation dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """
        return self.val_dataloader()
