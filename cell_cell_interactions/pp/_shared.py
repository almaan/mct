"""Shared preprocessing helpers used by all three datamodules.

These functions were previously duplicated across
``adata_rnaseq_datamodule.py``, ``cellarr_datamodule.py``, and
``spatial_datamodule.py``. The bodies are byte-identical to the originals;
this module is a straight extraction, not a rewrite. Numerical parity with
the pre-extraction state is enforced by ``tests/test_inference_parity.py``.

The gene-ID token offset ``expressed_gene_indices += 2`` at
``cellarr_datamodule.py:554`` and the ``tissue2idx = {"[PAD]": 0,
"[MASK]": 1}`` construction convention live at the call sites, NOT here, so
that anyone reading the datamodule can still see the checkpoint-coupled bits
inline.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from cell_cell_interactions.pp.data_types import BagOfCellsBatch, BagOfCellsSample

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


def mask_gene_indices(
    gene_indices: torch.Tensor,
    mask_prob: float,
    mask_token: int,
    n_genes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the BERT-style masking strategy to a sequence of gene token IDs.

    Byte-identical to the ``mask`` method previously defined on each dataset.
    Returns ``(gene_indices_w_mask, is_masked, is_masked_gene_expression)``.
    """
    # Step 1: Calculate the number of tokens to mask
    gene_indices_w_mask = gene_indices.clone()
    seq_len = gene_indices_w_mask.size(0)
    num_masked_genes = int(seq_len * mask_prob)

    # Step 2: Randomly select indices for masking
    masked_indices = torch.randperm(seq_len)[:num_masked_genes]

    # Step 3: Create a mask of the same size as the sequence
    is_masked = torch.zeros_like(gene_indices_w_mask).type_as(gene_indices_w_mask)
    is_masked[masked_indices] = 1

    # Step 4: Randomly assign the masking strategy
    probs = torch.rand(num_masked_genes)

    # 70% Replace with mask token
    mask_gene_prob = GENE_MASKING_STRATEGY["mask_gene_id"]
    gene_indices_w_mask[masked_indices[probs < mask_gene_prob]] = mask_token

    # 10% replace with random gene
    random_gene_prob = GENE_MASKING_STRATEGY["random"] + mask_gene_prob
    random_indices = masked_indices[(probs >= mask_gene_prob) & (probs < random_gene_prob)]
    gene_indices_w_mask[random_indices] = torch.randint(2, n_genes, (len(random_indices),))

    # 10% mask gene expression
    mask_gene_expression_prob = GENE_MASKING_STRATEGY["mask_gene_expression"] + random_gene_prob
    mask_gene_expression = masked_indices[(probs >= random_gene_prob) & (probs < mask_gene_expression_prob)]
    is_masked_gene_expression = torch.zeros_like(gene_indices_w_mask).type_as(gene_indices_w_mask)
    is_masked_gene_expression[mask_gene_expression] = 1

    # 10% we do not mask so we skip it
    return gene_indices_w_mask, is_masked, is_masked_gene_expression


def mask_tissue_indices(
    tissue_indices: torch.Tensor,
    mask_tissue_prob: float,
    mask_token: int,
    n_tissues: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the BERT-style masking strategy to a batch of tissue tokens.

    Byte-identical to the ``mask_tissue`` method previously defined on
    ``AdataRnaseqDataModule`` and ``CellarrDataModule``.
    Returns ``(tissue_indices, tissue_labels)``.
    """
    # Step 1: Calculate the number of tokens to mask
    tissue_labels = torch.ones_like(tissue_indices) * -100  # -100 is the ignore label
    seq_len = tissue_indices.size(0)
    num_masked_tissues = int(seq_len * mask_tissue_prob)
    if num_masked_tissues == 0:
        return tissue_indices, tissue_labels

    # Step 2: Randomly select indices for masking
    masked_indices = torch.randperm(seq_len)[:num_masked_tissues]
    # assign the tissue labels
    tissue_labels[masked_indices] = tissue_indices[masked_indices]

    # Step 3: Randomly assign the masking strategy
    probs = torch.rand(num_masked_tissues)

    # 80% Replace with mask token
    mask_tissue_prob_thresh = TISSUE_MASKING_STRATEGY["mask"]
    tissue_indices[masked_indices[probs < mask_tissue_prob_thresh]] = mask_token

    # 10% replace with random tissue
    random_gene_prob = TISSUE_MASKING_STRATEGY["random"] + mask_tissue_prob_thresh
    random_indices = masked_indices[(probs >= mask_tissue_prob_thresh) & (probs < random_gene_prob)]
    tissue_indices[random_indices] = torch.randint(2, n_tissues, (len(random_indices),))

    # 10% we do not mask so we skip it
    return tissue_indices, tissue_labels


def stratified_pack_cells(
    sample_data: Any,
    cell_type_input_order: dict[str, tuple[int, int]],
    cell_pad_idx: int,
) -> tuple[list[int], list[int]]:
    """Stratified sampling of cell types into fixed-width slots per cell type.

    Byte-identical to the ``order_cells_in_input`` method previously
    duplicated on ``AdataRnaseqDataset`` and ``CellarrDataset``.

    ``sample_data`` may be a dict, a ``pandas.Series``, or anything else
    that responds to ``.get(cell_type, [])``.
    """
    cell_indices: list[int] = []
    cell_pad_mask: list[int] = []
    for ct, (start, end) in cell_type_input_order.items():
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
            cell_indices += ct_data + [cell_pad_idx] * n_to_fill
            cell_pad_mask += [0] * len(ct_data) + [1] * n_to_fill
        else:
            cell_indices += [cell_pad_idx] * n_ct_cells
            cell_pad_mask += [1] * n_ct_cells

    return cell_indices, cell_pad_mask


def collate_bag_of_cells(
    samples: list[BagOfCellsSample],
    include_tissue: bool = False,
    include_sample_metadata: bool = False,
    include_cell_indices: bool = False,
) -> BagOfCellsBatch:
    """Pad-collate a list of ``BagOfCellsSample`` into a ``BagOfCellsBatch``.

    Byte-identical to the ``collate`` bodies previously duplicated across the
    three datamodules. Feature-flags let each datamodule opt in to the tissue
    fields (adata/cellarr) or spatial-only ``cell_indices`` field. The
    fields turned on match today's behavior exactly:

    - ``include_tissue`` — CellarrDataModule's ``collate`` builds
      ``tissue_idx`` and ``tissue_labels`` from ``samples[i].tissue_idx`` via
      ``mask_tissue``. Callers must set these on the resulting batch AFTER
      this function returns; this helper only handles the common padding
      block.
    - ``include_sample_metadata`` — adata/cellarr collates attach
      ``sample_label``, ``study_label``, ``disease`` list attributes.
    - ``include_cell_indices`` — spatial collate attaches ``cell_indices``
      and ``cell_idx`` fields.

    This helper is intentionally conservative in what it puts into the
    returned ``BagOfCellsBatch``; callers that need tissue-masking output
    still call ``mask_tissue_indices`` themselves.
    """
    gex = pad_sequence([x.expression_matrix for x in samples], batch_first=True, padding_value=0.0)

    is_masked_gene_id = pad_sequence(
        [x.is_masked_gene_id for x in samples], batch_first=True, padding_value=0.0
    )
    is_masked_gene_expression = pad_sequence(
        [x.is_masked_gene_expression for x in samples], batch_first=True, padding_value=0.0
    )

    # 0 = pad_token_idx
    gene_ids = pad_sequence([x.gene_ids for x in samples], batch_first=True, padding_value=0)
    labels = pad_sequence([x.labels for x in samples], batch_first=True, padding_value=-100)

    n_expressed_genes = torch.Tensor([x.n_expressed_genes for x in samples])

    padding_mask = [torch.zeros_like(x.is_masked_gene_id) for x in samples]
    padding_mask = pad_sequence(padding_mask, padding_value=1.0, batch_first=True).bool()

    batch = BagOfCellsBatch(
        expression_matrix=gex,
        gene_ids=gene_ids,
        labels=labels,
        is_masked_gene_id=is_masked_gene_id,
        is_masked_gene_expression=is_masked_gene_expression,
        padding_mask=padding_mask,
        n_expressed_genes=n_expressed_genes,
    )

    if include_sample_metadata:
        batch.sample_label = [x.sample_label for x in samples]
        batch.study_label = [x.study_label for x in samples]
        batch.disease = [x.disease for x in samples]

    if include_cell_indices:
        batch.cell_indices = pad_sequence(
            [x.cell_indices for x in samples], batch_first=True, padding_value=-1
        )
        batch.cell_idx = torch.tensor([x.cell_idx for x in samples])

    return batch
