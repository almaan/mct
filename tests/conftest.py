"""Shared test fixtures.

The parity tests defined in ``test_inference_parity.py`` and
``test_checkpoint_load.py`` are the gate that lets us refactor MCT without
silently changing model outputs. They rely on the fixtures here to build a
small, fully deterministic setup.

- ``tiny_config`` builds a ``MulticellularGraphConfig`` with values small
  enough that a forward pass runs in milliseconds on CPU.
- ``tiny_batch`` builds a matching ``BagOfCellsBatch`` on a pinned seed.
- ``released_checkpoint_path`` points at the local checkpoint if present;
  tests that need it are skipped when it is missing so CI still runs.

None of the values here are value-coupled to released checkpoints. This is a
random-init model on synthetic data; its only job is to detect numerical
drift introduced by refactors.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.pp.data_types import BagOfCellsBatch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_SEED = 20240101


def _seed_all(seed: int = DEFAULT_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def seed() -> int:
    return DEFAULT_SEED


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def tiny_config() -> MulticellularGraphConfig:
    """A tiny model config used across parity tests.

    Values are intentionally small; do not reuse for training or benchmarking.
    """
    return MulticellularGraphConfig(
        gene_emb_dim=16,
        hidden_dim=32,
        n_genes=64,
        max_seq_len=32,
        dropout=0.0,
        transformer_dropout=0.0,
        n_heads=2,
        n_layers=2,
        n_cells_bag=8,
        n_input=8,
        batch_size=2,
        padding_token_idx=0,
        mask_token_idx=1,
        n_tissues=0,
        alpha_reconstruction_loss_weight=0.0,
        gamma_tissue_loss_weight=0.0,
    )


@pytest.fixture(scope="session")
def tiny_batch(tiny_config: MulticellularGraphConfig) -> BagOfCellsBatch:
    """A deterministic tiny batch matching ``tiny_config``."""
    _seed_all()

    batch_size = tiny_config.batch_size
    seq_len = 16
    n_input = tiny_config.n_input
    n_genes = tiny_config.n_genes

    expression_matrix = torch.rand(batch_size, seq_len, n_input)
    # Gene IDs must avoid pad/mask token IDs to keep results meaningful.
    gene_ids = torch.randint(low=2, high=n_genes, size=(batch_size, seq_len))
    labels = gene_ids.clone()

    is_masked_gene_id = torch.zeros(batch_size, seq_len, dtype=torch.long)
    is_masked_gene_id[:, :3] = 1

    is_masked_gene_expression = torch.zeros(batch_size, seq_len, dtype=torch.float)

    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -2:] = True  # last two positions are padding

    return BagOfCellsBatch(
        expression_matrix=expression_matrix,
        gene_ids=gene_ids,
        labels=labels,
        is_masked_gene_id=is_masked_gene_id,
        is_masked_gene_expression=is_masked_gene_expression,
        padding_mask=padding_mask,
    )


@pytest.fixture(scope="session")
def released_checkpoint_path() -> Path:
    """Path to the released checkpoint if available locally.

    Set ``MCT_TEST_CHECKPOINT`` to override. Otherwise falls back to
    ``checkpoints/multicellular_transformer.ckpt`` under the repo root.
    """
    env = os.environ.get("MCT_TEST_CHECKPOINT")
    if env:
        return Path(env)
    return REPO_ROOT / "checkpoints" / "multicellular_transformer.ckpt"
