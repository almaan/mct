import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from cell_cell_interactions.pp.cellarr_datamodule import GENE_MASKING_STRATEGY
from cell_cell_interactions.pp.data_types import BagOfCellsBatch, BagOfCellsSample
from cell_cell_interactions.pp.utils import get_cell_type_order


class SpatialDataset(torch.utils.data.Dataset):
    """A class to load and process spatial data for training and evaluation."""

    def __init__(
        self,
        adata: sc.AnnData,
        data_df: pd.DataFrame,
        sample_size: int,
        gene2idx: dict[str, int],
        label_column: str,
        agg_cell_type_gex: bool,
        cell_type_input_order: dict[str, tuple[int, int]],
        mask_prob: float = 0.15,
        max_seq_len: int = 12000,
        min_n_expressed_cells_per_gene: float = 0.01,
    ):
        super().__init__()
        self.adata = adata
        self.data_df = data_df
        self.sample_size = sample_size
        self.gene2idx = gene2idx
        self.label_column = label_column
        self.agg_cell_type_gex = agg_cell_type_gex
        self.cell_type_input_order = cell_type_input_order
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.min_n_expressed_cells_per_gene = min_n_expressed_cells_per_gene

        self.n_genes = adata.shape[1]

        self.global_gene_ids = torch.tensor([self.gene2idx[g] for g in self.adata.var["ensemble_id"]], dtype=torch.long)
        self.cell2label = dict(enumerate(adata.obs[label_column].tolist()))
        # get labels ordered by largest to smallest
        self.labels = list(dict(adata.obs[label_column].value_counts()).keys())

        assert "spatial_connectivities" in adata.obsp, "spatial_connectivities not found in adata.obsp"

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> BagOfCellsSample:
        # get neighbouring cells and their labels
        idx = int(self.data_df.iloc[idx].name)
        neighbors = self.adata.obsp["spatial_connectivities"][idx].nonzero()[1]
        # if there are more cells in the neighborhood - sample them
        if len(neighbors) > self.sample_size:
            neighbors = np.random.choice(neighbors, self.sample_size - 1, replace=False)
        cell_indices = np.concatenate([[idx], neighbors])

        label_cell_indices = defaultdict(list)
        for cell_idx in cell_indices:
            label_cell_indices[self.cell2label[cell_idx]].append(cell_idx)

        cell_indices = []
        X = np.zeros((self.sample_size, self.n_genes), dtype=np.float32)
        for ct, (start, end) in self.cell_type_input_order.items():
            n_ct_cells = end - start
            ct_data = label_cell_indices.get(ct, [])
            if len(ct_data) >= n_ct_cells:
                # sample cells
                label_indices = random.sample(ct_data, n_ct_cells)
                X[start:end, :] = self.adata.X[label_indices, :].toarray()
                cell_indices += label_indices
            elif len(ct_data) > 0:
                random.shuffle(ct_data)
                X[start : start + len(ct_data), :] = self.adata.X[ct_data, :].toarray()
                cell_indices += ct_data
                cell_indices += [-1] * (n_ct_cells - len(ct_data))
            else:
                cell_indices += [-1] * n_ct_cells

        # fetch gene expression
        X = torch.tensor(X, dtype=torch.float32)

        # filter genes which are present in less than % of cells
        expressed_gene_indices = ((X > 0).sum(0) > self.min_n_expressed_cells_per_gene).nonzero().squeeze(-1)
        X = X[:, expressed_gene_indices]
        n_expressed_genes = len(expressed_gene_indices)

        if X.shape[1] > self.max_seq_len:
            selected_genes = torch.randperm(X.shape[1])[: self.max_seq_len]
            X = X[:, selected_genes]
            expressed_gene_indices = expressed_gene_indices[selected_genes]

        expressed_gene_indices = self.global_gene_ids[expressed_gene_indices]

        # do the masking
        expressed_gene_indices_w_mask, is_masked_gene_id, is_masked_gene_expression = self.mask(expressed_gene_indices)
        return BagOfCellsSample(
            expression_matrix=X.T,
            gene_ids=expressed_gene_indices_w_mask,
            labels=expressed_gene_indices,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            n_expressed_genes=n_expressed_genes,
            cell_indices=torch.tensor(cell_indices, dtype=torch.long),
            cell_idx=idx,
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


class SpatialDataModule(LightningDataModule):
    """A class to load and process spatial data for training and evaluation."""

    def __init__(
        self,
        adata_file_path: str,
        gene2idx: dict[str, int],  # ENSEMBL gene ID to index mapping
        sample_size: int = 512,
        val_set_proportion: float = 0.2,
        label_column: str = "leiden",
        batch_size: int = 1,
        num_workers: int = 0,
        agg_cell_type_gex: bool = False,
        pad_value_expression: float = 0.0,
        mask_prob: float = 0.0,
        max_seq_len: int = 2500,
        random_seed: int = 42,
        restrict_to_conditions: dict | None = None,  # restrict to specific conditions, e.g. {"study": "study1"}
        perc_min_expressed_cells_per_gene: float = 0.01,
    ):
        super().__init__()
        self.adata_file_path = adata_file_path
        self.gene2idx = gene2idx
        self.sample_size = sample_size
        self.val_set_proportion = val_set_proportion
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.agg_cell_type_gex = agg_cell_type_gex
        self.pad_value_expression = pad_value_expression
        self.mask_prob = mask_prob
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.restrict_to_conditions = restrict_to_conditions
        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)

        # load the data
        self.adata = sc.read_h5ad(adata_file_path)

        # restrict to specific conditions
        if restrict_to_conditions is not None:
            for key, value in restrict_to_conditions.items():
                self.adata = self.adata[self.adata.obs[key] == value]
            self.adata.obs = self.adata.obs.reset_index(drop=True)

        self.data_df = self.adata.obs.copy()

        self.cell_type_sampled = dict(self.data_df[self.label_column].value_counts().items())

        s = sum(self.cell_type_sampled.values())
        self.cell_type_proportions = {ct: c / s for ct, c in self.cell_type_sampled.items()}
        self.cell_type_input_order = get_cell_type_order(self.cell_type_proportions, self.sample_size)

        # do the split
        np.random.seed(self.random_seed)
        self.data_df["is_val"] = np.random.rand(len(self.data_df)) < self.val_set_proportion
        self.train_df = self.data_df[~self.data_df["is_val"]]
        self.val_df = self.data_df[self.data_df["is_val"]]

        self.n_genes = self.adata.shape[1]

        self.train_dataset = SpatialDataset(
            adata=self.adata,
            data_df=self.train_df,
            sample_size=self.sample_size,
            gene2idx=self.gene2idx,
            label_column=self.label_column,
            agg_cell_type_gex=self.agg_cell_type_gex,
            cell_type_input_order=self.cell_type_input_order,
            mask_prob=self.mask_prob,
            max_seq_len=self.max_seq_len,
            min_n_expressed_cells_per_gene=self.min_n_expressed_cells_per_gene,
        )

        self.val_dataset = None
        if len(self.val_df) > 0:
            self.val_dataset = SpatialDataset(
                adata=self.adata,
                data_df=self.val_df,
                sample_size=self.sample_size,
                gene2idx=self.gene2idx,
                label_column=self.label_column,
                agg_cell_type_gex=self.agg_cell_type_gex,
                cell_type_input_order=self.cell_type_input_order,
                mask_prob=self.mask_prob,
                max_seq_len=self.max_seq_len,
                min_n_expressed_cells_per_gene=self.min_n_expressed_cells_per_gene,
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

        cell_indices = pad_sequence([x.cell_indices for x in batch], batch_first=True, padding_value=-1)

        return BagOfCellsBatch(
            expression_matrix=gex,
            gene_ids=gene_ids,
            labels=labels,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            padding_mask=padding_mask,
            n_expressed_genes=n_expressed_genes,
            cell_indices=cell_indices,
            cell_idx=torch.tensor([x.cell_idx for x in batch]),
        )

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
