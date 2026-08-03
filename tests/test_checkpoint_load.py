"""Checkpoint-load contract tests.

The released checkpoint plus its paired index JSONs must load into the
current code without user-side state_dict surgery. These tests are skipped
when the checkpoint is not available locally so CI without secrets still
runs; set ``MCT_TEST_CHECKPOINT`` to point at an alternate path.

When a refactor forces a state_dict rename, add a remap in
``load_from_checkpoint`` / ``on_load_checkpoint`` and add the new key names
to the invariants below rather than deleting the assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel


def _load_raw(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def test_checkpoint_file_present(released_checkpoint_path: Path) -> None:
    """Sanity: the checkpoint file exists and is a torch archive."""
    if not released_checkpoint_path.exists():
        pytest.skip(
            f"Checkpoint not found at {released_checkpoint_path}. "
            "Set MCT_TEST_CHECKPOINT or place the file to enable these tests."
        )
    ckpt = _load_raw(released_checkpoint_path)
    assert isinstance(ckpt, dict), "Checkpoint is not a dict-like torch archive."


def test_hyperparameters_are_readable(released_checkpoint_path: Path) -> None:
    """The saved ``hyper_parameters['config']`` must deserialize as MulticellularGraphConfig."""
    if not released_checkpoint_path.exists():
        pytest.skip("Checkpoint not available.")
    ckpt = _load_raw(released_checkpoint_path)
    assert "hyper_parameters" in ckpt, "Missing 'hyper_parameters' block in checkpoint."
    hp = ckpt["hyper_parameters"]
    assert "config" in hp, "Missing 'config' inside 'hyper_parameters'."
    config = hp["config"]
    assert isinstance(config, MulticellularGraphConfig), (
        f"Expected MulticellularGraphConfig in checkpoint, got {type(config).__name__}. "
        "If the config dataclass was renamed or moved, add a loader shim."
    )


def test_state_dict_has_expected_top_level_modules(released_checkpoint_path: Path) -> None:
    """Guard against silent state_dict key renames.

    If you intentionally rename a top-level module, update this list AND
    ship a remap in the loader so old checkpoints still work.
    """
    if not released_checkpoint_path.exists():
        pytest.skip("Checkpoint not available.")
    ckpt = _load_raw(released_checkpoint_path)
    assert "state_dict" in ckpt, "Missing 'state_dict' in checkpoint."

    keys = list(ckpt["state_dict"].keys())
    prefixes = {k.split(".")[0] for k in keys}
    # 'model' is the PL module attribute; this reflects current layout, not a promise
    # about what internal modules will always be named. If you change this, ship a shim.
    assert "model" in prefixes, (
        f"Expected top-level 'model' prefix in state_dict. Got prefixes: {sorted(prefixes)}. "
        "A rename here requires a loader shim in MulticellularTransformerPLModel."
    )


def test_load_from_checkpoint_roundtrip(released_checkpoint_path: Path) -> None:
    """End-to-end: the Lightning module must load without manual surgery."""
    if not released_checkpoint_path.exists():
        pytest.skip("Checkpoint not available.")
    model = MulticellularTransformerPLModel.load_from_checkpoint(
        str(released_checkpoint_path),
        map_location="cpu",
    )
    assert isinstance(model, MulticellularTransformerPLModel)
    assert isinstance(model.config, MulticellularGraphConfig)
    # Basic shape sanity: gene embedding table matches vocab.
    gene_id_emb = model.model.encoder.gene_id_emb
    assert gene_id_emb.num_embeddings == model.config.n_genes, (
        f"gene_id_emb vocab ({gene_id_emb.num_embeddings}) does not match "
        f"config.n_genes ({model.config.n_genes}). Checkpoint contract broken."
    )
