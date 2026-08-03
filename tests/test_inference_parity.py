"""Numerical parity tests for the model forward pass.

Purpose: guard against silent numerical drift introduced by refactors.

The test builds a small random-init ``MulticellularTransformerModule``,
runs a deterministic forward pass, and compares outputs against a snapshot
file. If the snapshot is missing, the test writes it and passes with a
warning; subsequent runs enforce bit-parity on CPU.

When output is expected to change (documented kernel change, intentional
model edit), delete ``tests/fixtures/inference_parity.pt`` and re-run to
regenerate. Explain the change in ``CHANGELOG.md`` under a Migration
heading.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.modeling import MulticellularTransformerModule
from cell_cell_interactions.pp.data_types import BagOfCellsBatch

SNAPSHOT_NAME = "inference_parity.pt"


def _build_model(config: MulticellularGraphConfig, seed: int) -> MulticellularTransformerModule:
    torch.manual_seed(seed)
    model = MulticellularTransformerModule(config)
    model.eval()
    return model


def _forward(model: MulticellularTransformerModule, batch: BagOfCellsBatch) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(
            expression_matrix=batch.expression_matrix,
            gene_ids=batch.gene_ids,
            is_masked_gene_expression=batch.is_masked_gene_expression,
            padding_mask=batch.padding_mask,
            tissue=None,
        )
    return {
        "last_hidden_state": output.last_hidden_state.detach().clone(),
        "gene_id_logits": output.gene_id_logits.detach().clone(),
        "gex_reconstruction": output.gex_reconstruction.detach().clone(),
    }


def test_forward_is_deterministic_across_two_runs(
    tiny_config: MulticellularGraphConfig,
    tiny_batch: BagOfCellsBatch,
    seed: int,
) -> None:
    """Two builds with the same seed must produce bit-identical output."""
    model_a = _build_model(tiny_config, seed)
    model_b = _build_model(tiny_config, seed)

    out_a = _forward(model_a, tiny_batch)
    out_b = _forward(model_b, tiny_batch)

    for key in out_a:
        torch.testing.assert_close(out_a[key], out_b[key], atol=0.0, rtol=0.0)


def test_forward_matches_snapshot(
    tiny_config: MulticellularGraphConfig,
    tiny_batch: BagOfCellsBatch,
    fixtures_dir: Path,
    seed: int,
) -> None:
    """Forward output must match the committed snapshot.

    If snapshot is missing, write it and warn. This makes first-time setup
    self-bootstrapping while still failing loudly on drift afterwards.
    """
    model = _build_model(tiny_config, seed)
    out = _forward(model, tiny_batch)

    snapshot_path = fixtures_dir / SNAPSHOT_NAME
    if not snapshot_path.exists():
        torch.save(out, snapshot_path)
        warnings.warn(
            f"Wrote new parity snapshot at {snapshot_path}. Commit it to gate future refactors.",
            stacklevel=2,
        )
        return

    snapshot = torch.load(snapshot_path, map_location="cpu")
    assert set(snapshot.keys()) == set(out.keys()), (
        "Snapshot keys drifted. Model output surface changed. "
        f"Snapshot: {sorted(snapshot.keys())}, current: {sorted(out.keys())}."
    )
    for key in out:
        torch.testing.assert_close(
            out[key],
            snapshot[key],
            atol=0.0,
            rtol=0.0,
            msg=lambda m, k=key: (
                f"Parity broke for '{k}'. Either a refactor changed numerical output "
                f"(fix it), or the change is intentional (delete "
                f"tests/fixtures/{SNAPSHOT_NAME}, regenerate, and note in CHANGELOG).\n{m}"
            ),
        )


@pytest.mark.parametrize("with_tissue", [False, True])
def test_output_shapes(
    tiny_config: MulticellularGraphConfig,
    tiny_batch: BagOfCellsBatch,
    seed: int,
    with_tissue: bool,
) -> None:
    """Output tensor shapes must match the config-implied contract."""
    if with_tissue:
        config = MulticellularGraphConfig(**{**tiny_config.to_dict(), "n_tissues": 3})
        tissue = torch.tensor([0, 1], dtype=torch.long)
    else:
        config = tiny_config
        tissue = None

    model = _build_model(config, seed)
    with torch.no_grad():
        output = model(
            expression_matrix=tiny_batch.expression_matrix,
            gene_ids=tiny_batch.gene_ids,
            is_masked_gene_expression=tiny_batch.is_masked_gene_expression,
            padding_mask=tiny_batch.padding_mask,
            tissue=tissue,
        )

    batch_size, seq_len, _ = tiny_batch.expression_matrix.shape
    assert output.last_hidden_state.shape == (batch_size, seq_len, config.gene_emb_dim)
    assert output.gene_id_logits.shape == (batch_size, seq_len, config.n_genes)
    assert output.gex_reconstruction.shape == (batch_size, seq_len, config.n_input)
    if with_tissue:
        assert output.tissue_logits.shape == (batch_size, config.n_tissues)
    else:
        assert output.tissue_logits is None
