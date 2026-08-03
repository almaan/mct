import itertools
import random
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import tiledb
import torch
from lightning import LightningDataModule
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from cell_cell_interactions.pp.data_types import BagOfCellsBatch, BagOfCellsDatasetOutput, BagOfCellsSample
from cell_cell_interactions.pp.utils import get_cell_type_order, group_cell_types_by_sample

# Turn off multithreading to allow multiple pytorch dataloader workers
config = tiledb.Config()
config["sm.compute_concurrency_level"] = 1
config["sm.io_concurrency_level"] = 1
config["sm.num_async_threads"] = 1
config["sm.num_reader_threads"] = 1
config["sm.num_tbb_threads"] = 1
config["sm.num_writer_threads"] = 1
config["vfs.num_threads"] = 1

# define the masking strategy
GENE_MASKING_STRATEGY = {
    "mask_gene_id": 0.7,
    "mask_gene_expression": 0.1,
    "random": 0.1,
    "no_mask": 0.1,
}

TISSUE_MASKING_STRATEGY = {
    "mask": 0.7,
    "random": 0.1,
    "no_mask": 0.2,
}


class CellarrDataset(Dataset):
    """A class that represent a collection of sample cells in TileDB."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        sample2cells: dict,
        int2sample: dict,
        tissue2idx: dict,
        sample_size: int,
        label_column: str = "cellTypeName",
        study_column: str = "study",
        sample_column: str = "sample",
        tissue_column: str = "tissue",
        agg_cell_type_gex: bool = False,
        sample_sampling: Literal["random", "weighted"] = "weighted",
        cell_sampling: Literal["random", "stratified"] = "stratified",  # sampling strategy for cells
        cell_type_input_order: dict[str, tuple[int, int]] = None,
        cell_pad_idx: int = 0,
        disease_column: str = "disease",
    ):
        """Constructor.

        Parameters
        ----------
        data_df: pandas.DataFrame
            Pandas dataframe of valid cells.
        counts_tdb_uri: str
            Counts TileDB URI.
        sample_size: int
            Number of sample cells.
        label_column: str, default: "cellTypeName"
            Label column name.
        study_column: str, default: "study"
            Study column name.
        sample_column: str, default: "sample"
            Sample column name.
        sparse: bool, default: False
            Return a sparse matrix.
        agg_cell_type_gex: bool, default: False
            Aggregate gene expression by cell type.
        sample_sampling: Literal["random", "weighted"], default: "weighted"
            Sample sampling method.
        cell_type_input_order: dict[str, tuple[int, int]], default: None
            Cell type input order.
        cell_pad_idx: int, default: 0
            Cell pad index.
        """
        self.data_df = data_df
        self.sample2cells = sample2cells
        self.int2sample = int2sample
        self.tissue2idx = tissue2idx
        self.sample_size = sample_size
        self.label_column = label_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column
        self.agg_cell_type_gex = agg_cell_type_gex
        self.sample_sampling = sample_sampling
        self.cell_sampling = cell_sampling
        self.cell_type_input_order = cell_type_input_order
        self.cell_pad_idx = cell_pad_idx
        self.disease_column = disease_column

        self.sampleID = "study::::sample"

        self.samples = list(sample2cells.keys())
        self.cells_per_sample = [len(sample) for sample in sample2cells.values()]
        self.n_cells = sum([len(sample) for sample in sample2cells.values()])

        self.cell_types = list(cell_type_input_order.keys())
        self.cell_types_grouped_by_sample = group_cell_types_by_sample(
            data_df, self.sampleID, self.label_column, self.cell_types
        )

    def __len__(self) -> int:
        # if sampling by sample we randomly pick samples
        if self.sample_sampling == "random":
            return len(self.int2sample)
        # if sampling is weighted we select from larger samples more often
        return self.n_cells

    def __getitem__(self, idx) -> BagOfCellsDatasetOutput:
        # fetch the sample
        if self.sample_sampling == "random":
            sample = self.int2sample[idx]
        else:
            sample = random.choices(self.samples, weights=self.cells_per_sample, k=1)[0]

        sample_cell_indices = self.sample2cells[sample]
        if len(sample_cell_indices) >= self.sample_size:
            cell_idx = random.sample(sample_cell_indices, self.sample_size)
        else:
            cell_idx = random.choices(sample_cell_indices, k=self.sample_size)

        # fetch label data
        study_label = self.data_df.loc[cell_idx, self.study_column].iloc[0]
        sample_label = self.data_df.loc[cell_idx, self.sample_column].iloc[0]

        tissue_label = None
        if self.tissue_column in self.data_df.columns:
            tissue_label = self.data_df.loc[cell_idx, self.tissue_column].iloc[0]

        disease = None
        if self.disease_column in self.data_df.columns:
            disease = self.data_df.loc[cell_idx, self.disease_column].iloc[0]

        if self.cell_sampling == "random":
            # fetch the cell indices
            cell_type_labels = self.data_df.loc[cell_idx, self.label_column]
            # get cell type indices
            ct_indices = defaultdict(list)
            for idx, ct in zip(cell_idx, cell_type_labels):
                ct_indices[ct].append(idx)
            cell_idx, cell_pad_mask = self.order_cells_in_input(ct_indices)
        else:
            # fetch the cell indices per each cell type in a sample
            sample_data = self.cell_types_grouped_by_sample.loc[sample]
            cell_idx, cell_pad_mask = self.order_cells_in_input(sample_data)

        return BagOfCellsDatasetOutput(
            cell_indices=cell_idx,
            study_label=study_label,
            sample_label=sample_label,
            tissue_idx=self.tissue2idx.get(tissue_label, None),
            cell_pad_mask=cell_pad_mask,
            disease=disease,
        )

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


class CellarrDataModule(LightningDataModule):
    """A class to encapsulate sample cells in TileDB to train the model."""

    def __init__(
        self,
        cell_tdb_uri: str,
        gene_tdb_uri: str,
        counts_tdb_uri: str,
        val_samples: list[str] | None = None,
        val_set_proportion: float = 0.1,
        label_column: str = "cell_type",
        study_column: str = "study",
        sample_column: str = "sample",
        tissue_column: str = "tissue",
        batch_size: int = 1,
        sample_size: int = 512,
        num_workers: int = 1,
        min_sample_size: int | None = None,
        min_cell_type_size: float | None = 0.001,
        agg_cell_type_gex: bool = False,
        sample_sampling: Literal["random", "weighted"] = "weighted",
        cell_sampling: Literal["random", "stratified"] = "stratified",  # sampling strategy for cells
        pad_value_expression: float = 0.0,
        mask_prob: float = 0.0,
        mask_tissue_prob: float = 0.0,
        cell_type_sqrt_sampling: bool = False,
        perc_min_expressed_cells_per_gene: float = 0.05,
        max_seq_len: int = 2500,
        lognorm: bool = True,
        target_sum: float = 1e4,
        random_seed: int = 32,
        cell_type_mapping: dict | None = None,
        restrict_to_conditions: dict | None = None,  # restrict to specific conditions, e.g. {"study": "study1"}
    ):
        """Constructor.

        Parameters
        ----------
        cell_tdb_uri: str
            Cell metadata TileDB storage URI.
        gene_tdb_uri: str
            Gene metadata TileDB storage URI.
        counts_tdb_uri: str
            Counts TileDB storage URI.
        val_samples: List[str], optional, default: None
            List of studies to use as validation and test.
        label_column: str, default: "cell_type"
            Cell type column name.
        study_column: str, default: "dataset"
            Study column name.
        sample_column: str, default: "sampleID"
            Sample column name.
        batch_size: int, default: 1
            Batch size.
        sample_size: int, default: 100
            Sample size
        num_workers: int, default: 1
            The number of worker threads for dataloaders
        min_sample_size: int, optional, default: None
            Set a minimum number of cells in a sample for it to be valid.
        agg_cell_type_gex: bool, default: False
            Aggregate gene expression by cell type.
        sample_sampling: Literal["random", "weighted"], default: "weighted"
            Sample sampling method.
        pad_value_expression: float, default: 0.0
            Value to pad the output with if there are not enough cells of a given type.
        mask_prob: float, default: 0.0
            Probability of masking a gene ID.
        mask_prob_gene_expression: float, default: 0.0
            Probability of masking a gene expression.
        cell_type_sqrt_sampling: bool, default: False
            Sample cell types proportionally to the square root of the number of cells.
        perc_min_expressed_cells_per_gene: float, default: 0.10
            Minimum percentage of cells expressing a gene for it to be included in the input.
        max_seq_len: int, default: 2500
            Maximum sequence length.
        lognorm: bool, default: True
            Log normalize the input.
        target_sum: float, default: 1e4
            Target sum for log normalization.
        random_seed: int, default: 42
            Random seed.
        cell_type_mapping: dict, optional, default: None
            Dict which maps initial cell types to harmonised cell types across studies.

        Examples
        --------
        >>> datamodule = MetricLearningZarrDataModule(
                cell_tdb_uri="human_scref_cell_metadata",
                counts_tdb_uri="human_scref_counts_matrix",
                gene_tdb_uri="human_scref_gene_metadata",
                gene_order="gene_order.tsv",
                label_id_column="cellTypeOntologyID",
                batch_size=1000,
                num_workers=1,
            )
        """
        super().__init__()
        self.cell_tdb_uri = cell_tdb_uri
        self.gene_tdb_uri = gene_tdb_uri
        self.counts_tdb_uri = counts_tdb_uri
        self.val_samples = val_samples
        self.label_column = label_column
        self.study_column = study_column
        self.sample_column = sample_column
        self.tissue_column = tissue_column
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.min_sample_size = min_sample_size
        self.min_cell_type_size = min_cell_type_size
        self.num_workers = num_workers
        self.agg_cell_type_gex = agg_cell_type_gex
        self.sample_sampling = sample_sampling
        self.cell_sampling = cell_sampling
        self.pad_value_expression = pad_value_expression
        self.mask_prob = mask_prob
        self.mask_tissue_prob = mask_tissue_prob
        self.cell_type_sqrt_sampling = cell_type_sqrt_sampling
        self.max_seq_len = max_seq_len
        self.lognorm = lognorm
        self.target_sum = target_sum
        self.random_seed = random_seed
        random.seed(random_seed)

        self.min_n_expressed_cells_per_gene = int(perc_min_expressed_cells_per_gene * sample_size)

        self.cell_tdb = tiledb.open(self.cell_tdb_uri, "r")
        self.gene_tdb = tiledb.open(self.gene_tdb_uri, "r")

        self.counts_shape = (
            self.cell_tdb.nonempty_domain()[0][1] + 1,
            self.gene_tdb.nonempty_domain()[0][1] + 1,
        )

        # open cell metadata df
        self.data_df = self.cell_tdb.query().df[:]

        assert (
            self.label_column in self.data_df.columns
        ), f"label_column: {self.label_column} not present in obs columns, make sure it is present!"
        assert (
            self.study_column in self.data_df.columns
        ), f"study_column: {self.study_column} not present in obs columns, make sure it is present!"

        if cell_type_mapping is not None:
            self.data_df[self.label_column] = self.data_df[self.label_column].map(cell_type_mapping)

        # filter out cell types with less than min_cell_type_size cells
        if self.min_cell_type_size is not None:
            self.cell_type_proportions = self.data_df[self.label_column].value_counts(normalize=True)
            cell_types = [ct for ct, p in self.cell_type_proportions.items() if p >= self.min_cell_type_size]
            self.data_df = self.data_df[self.data_df[self.label_column].isin(cell_types)]

        # filter out sample without enough cells
        if self.min_sample_size is not None:
            self.study_sample_df = self.data_df.groupby([self.study_column, self.sample_column]).size()
            self.study_sample_df = (
                self.study_sample_df[self.study_sample_df >= self.min_sample_size].copy().reset_index()
            )
            self.data_df = self.data_df[
                self.data_df[self.study_column].isin(self.study_sample_df[self.study_column])
                & self.data_df[self.sample_column].isin(self.study_sample_df[self.sample_column])
            ]

        self.cell_type_sampled = dict(self.data_df[self.label_column].value_counts().items())
        if self.cell_type_sqrt_sampling:
            self.cell_type_sampled = {
                ct: int(np.sqrt(c)) for ct, c in self.data_df[self.label_column].value_counts().items()
            }

        s = sum(self.cell_type_sampled.values())
        self.cell_type_proportions = {ct: c / s for ct, c in self.cell_type_sampled.items()}
        self.cell_type_input_order = get_cell_type_order(self.cell_type_proportions, self.sample_size)

        if restrict_to_conditions is not None:
            for key, value in restrict_to_conditions.items():
                self.data_df = self.data_df[self.data_df[key] == value]

        # concat study and sample in the case there are duplicate sample names
        self.sampleID = "study::::sample"
        self.data_df[self.sampleID] = self.data_df[self.study_column] + "::::" + self.data_df[self.sample_column]

        self.tissue2idx = {"[PAD]": 0, "[MASK]": 1}
        if self.tissue_column in self.data_df.columns:
            self.tissue2idx.update({t: idx + 2 for idx, t in enumerate(self.data_df[self.tissue_column].unique())})

        self.val_df = None
        if self.val_samples is not None:
            # split out validation samples
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

        gene_df = self.gene_tdb.query().df[:]
        self.gene2idx = {"[PAD]": 0, "[MASK]": 1}
        self.gene2idx.update({gene: idx + 2 for idx, gene in enumerate(gene_df["cellarr_gene_index"].tolist())})
        self.n_genes = len(self.gene2idx)  # used when creating training model

        gp = self.data_df.groupby(self.sampleID)
        self.train_int2sample = dict(enumerate(gp.groups.keys()))
        self.train_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
        self.train_dataset = CellarrDataset(
            data_df=self.data_df,
            sample2cells=self.train_sample2cells,
            int2sample=self.train_int2sample,
            tissue2idx=self.tissue2idx,
            sample_size=self.sample_size,
            label_column=self.label_column,
            study_column=self.study_column,
            sample_column=self.sample_column,
            tissue_column=self.tissue_column,
            agg_cell_type_gex=self.agg_cell_type_gex,
            sample_sampling=sample_sampling,
            cell_sampling=cell_sampling,
            cell_type_input_order=self.cell_type_input_order,
        )

        self.val_dataset = None
        if len(self.val_df) > 0:
            gp = self.val_df.groupby(self.sampleID)
            self.val_int2sample = dict(enumerate(gp.groups.keys()))
            self.val_sample2cells = {x: gp.groups[x].tolist() for x in gp.groups.keys()}
            self.val_dataset = CellarrDataset(
                data_df=self.val_df,
                sample2cells=self.val_sample2cells,
                int2sample=self.val_int2sample,
                tissue2idx=self.tissue2idx,
                sample_size=self.sample_size,
                label_column=self.label_column,
                study_column=self.study_column,
                sample_column=self.sample_column,
                tissue_column=self.tissue_column,
                sample_sampling=sample_sampling,
                cell_sampling=cell_sampling,
                cell_type_input_order=self.cell_type_input_order,
                agg_cell_type_gex=self.agg_cell_type_gex,
            )

        self.cell_tdb.close()
        self.gene_tdb.close()

        self.counts_tdb_uri = counts_tdb_uri
        self.counts_tdb = tiledb.open(self.counts_tdb_uri, "r", config=config)

    def collate(self, batch: list[BagOfCellsDatasetOutput]) -> BagOfCellsBatch:
        """Collate tensors.

        Parameters
        ----------
        batch:
            Batch to collate.

        Returns
        -------
        tuple
            A Tuple[torch.Tensor, torch.Tensor, list, list] containing information
            on the collated tensors.
        """
        # call the counts tdb once per batch
        cell_indices = list(itertools.chain.from_iterable([x.cell_indices for x in batch]))
        results = self.counts_tdb.multi_index[cell_indices, :]
        counts = coo_matrix(
            (results["data"], (results["cell_index"], results["gene_index"])),
            shape=self.counts_shape,
        ).tocsr()
        counts = counts[cell_indices, :].astype(np.float32)

        # do log1-p normalization
        if self.lognorm:
            counts_per_cell = counts.sum(axis=1)
            counts_per_cell = np.ravel(counts_per_cell)
            counts_per_cell = counts_per_cell / self.target_sum
            counts = counts / counts_per_cell[:, None]
            counts = counts.log1p()
        counts = counts.toarray()

        # zero out the items which were padded
        cell_pad_mask = np.array(list(itertools.chain.from_iterable([x.cell_pad_mask for x in batch])))
        counts = torch.tensor(np.where(cell_pad_mask[:, None], 0.0, counts), dtype=torch.float32)

        samples = []
        for i, item in zip(range(0, len(cell_indices), self.sample_size), batch):
            X = counts[i : i + self.sample_size, :]
            sample = self.process_sample(X, item)
            samples.append(sample)

        gex = pad_sequence([x.expression_matrix for x in samples], batch_first=True, padding_value=0.0)

        tissue_idx = [item.tissue_idx for item in samples]
        if None not in tissue_idx:
            tissue_idx = torch.tensor(tissue_idx, dtype=torch.long)
            tissue_idx, tissue_labels = self.mask_tissue(tissue_idx)
        else:
            tissue_idx = None
            tissue_labels = None

        # pad sequences
        is_masked_gene_id = pad_sequence([x.is_masked_gene_id for x in samples], batch_first=True, padding_value=0.0)
        is_masked_gene_expression = pad_sequence(
            [x.is_masked_gene_expression for x in samples], batch_first=True, padding_value=0.0
        )

        # 0 = pad_token_idx
        gene_ids = pad_sequence([x.gene_ids for x in samples], batch_first=True, padding_value=0)
        labels = pad_sequence([x.labels for x in samples], batch_first=True, padding_value=-100)
        # fetch number of genes which were not filtered out in each sample
        n_expressed_genes = torch.Tensor([x.n_expressed_genes for x in samples])

        padding_mask = [torch.zeros_like(x.is_masked_gene_id) for x in samples]
        padding_mask = pad_sequence(padding_mask, padding_value=1.0, batch_first=True).bool()

        return BagOfCellsBatch(
            expression_matrix=gex,
            sample_label=[x.sample_label for x in samples],
            study_label=[x.study_label for x in samples],
            gene_ids=gene_ids,
            labels=labels,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            padding_mask=padding_mask,
            tissue_idx=tissue_idx,
            tissue_labels=tissue_labels,
            n_expressed_genes=n_expressed_genes,
            disease=[x.disease for x in samples],
        )

    def process_sample(self, X: torch.Tensor, item: BagOfCellsDatasetOutput) -> BagOfCellsSample:
        """Process a sample."""
        # filter genes which are present in less than % of cells
        expressed_gene_indices = ((X > 0).sum(0) > self.min_n_expressed_cells_per_gene).nonzero().squeeze(-1)
        X = X[:, expressed_gene_indices]
        n_expressed_genes = len(expressed_gene_indices)

        if X.shape[1] > self.max_seq_len:
            selected_genes = torch.randperm(X.shape[1])[: self.max_seq_len]
            X = X[:, selected_genes]
            expressed_gene_indices = expressed_gene_indices[selected_genes]

        if self.agg_cell_type_gex:
            output = [X[start:end].sum(0) for _, (start, end) in self.cell_type_input_order.items()]
            X = torch.stack(output)

        expressed_gene_indices += 2  # shift by 2 to account for [MASK] and [PAD] tokens

        # do the masking
        expressed_gene_indices_w_mask, is_masked_gene_id, is_masked_gene_expression = self.mask(expressed_gene_indices)

        return BagOfCellsSample(
            expression_matrix=X.T,
            gene_ids=expressed_gene_indices_w_mask,
            labels=expressed_gene_indices,
            is_masked_gene_id=is_masked_gene_id,
            is_masked_gene_expression=is_masked_gene_expression,
            tissue_idx=item.tissue_idx,
            cell_type_indices=item.cell_type_indices,
            study_label=item.study_label,
            sample_label=item.sample_label,
            n_expressed_genes=n_expressed_genes,
            disease=item.disease,
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
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            collate_fn=self.collate,
            # multiprocessing_context="spawn",
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
            # persistent_workers=True,
            # multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        """Load the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object containing the test dataset.
        """
        return self.val_dataloader()
