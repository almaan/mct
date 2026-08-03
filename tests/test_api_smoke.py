"""Smoke tests for the public API layer.

These exercise import surface and pure-Python behavior. Anything that needs a
released checkpoint delegates to the fixture in ``conftest.py`` and skips
gracefully when the checkpoint is not available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cell_cell_interactions.api import (
    AdataInferencePreset,
    CellarrInferencePreset,
    MCTIndex,
    MCTModel,
    SpatialInferencePreset,
    get_context_aware_gene_embeddings,
    load_model,
)
from cell_cell_interactions.modeling.data_types import (
    MulticellularGraphConfig,
    MulticellularTransformerOutput,
)
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.data_types import BagOfCellsBatch


def test_top_level_reexports_are_importable() -> None:
    import cell_cell_interactions as cci

    for name in (
        "MCTModel",
        "MCTIndex",
        "load_model",
        "AdataInferencePreset",
        "CellarrInferencePreset",
        "SpatialInferencePreset",
        "embed_adata",
        "embed_dataloader",
        "get_context_aware_gene_embeddings",
        "GeneEmbedding",
        "calculate_condition_specificity",
        "plot_gene_set",
        "plot_similarity_across_conditions_heatmap",
    ):
        assert hasattr(cci, name), f"missing top-level re-export: {name}"


def test_mct_index_from_dir(tmp_path: Path) -> None:
    gene2idx = {"GENEA": 2, "GENEB": 3, "[PAD]": 0, "[MASK]": 1}
    (tmp_path / "gene2idx.json").write_text(json.dumps(gene2idx))
    (tmp_path / "cell_type_input_order.json").write_text(json.dumps({"T": [0, 4]}))
    (tmp_path / "tissue2idx.json").write_text(json.dumps({"lung": 2}))

    index = MCTIndex.from_dir(tmp_path)
    assert index.gene2idx == gene2idx
    assert index.idx2gene == {2: "GENEA", 3: "GENEB", 0: "[PAD]", 1: "[MASK]"}
    assert index.cell_type_input_order == {"T": (0, 4)}
    assert index.tissue2idx == {"lung": 2}


def test_mct_index_from_dir_missing_files_is_ok(tmp_path: Path) -> None:
    (tmp_path / "gene2idx.json").write_text(json.dumps({"GENEA": 2}))
    index = MCTIndex.from_dir(tmp_path)
    assert index.gene2idx == {"GENEA": 2}
    assert index.cell_type_input_order is None
    assert index.tissue2idx is None


def test_mct_index_from_dir_raises_when_dir_missing() -> None:
    with pytest.raises(FileNotFoundError):
        MCTIndex.from_dir("/nonexistent/path/aaaaa")


def test_gene_embeddings_shape_matches_vocab(
    tiny_config: MulticellularGraphConfig,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(tiny_config)
    pl_model.eval()

    gene2idx = {f"G{i}": i for i in range(2, tiny_config.n_genes)}
    gene2idx["[PAD]"] = 0
    gene2idx["[MASK]"] = 1
    index = MCTIndex(gene2idx=gene2idx, idx2gene={v: k for k, v in gene2idx.items()})
    model = MCTModel(pl_model, index)

    all_emb = model.gene_embeddings()
    assert len(all_emb) == tiny_config.n_genes
    assert all(isinstance(v, np.ndarray) for v in all_emb.values())
    assert all(v.shape == (tiny_config.gene_emb_dim,) for v in all_emb.values())

    subset = model.gene_embeddings(["G3", "G5", "not_a_gene"])
    assert set(subset.keys()) == {"G3", "G5"}


def test_gene_embeddings_raises_without_index(
    tiny_config: MulticellularGraphConfig,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(tiny_config)
    model = MCTModel(pl_model, MCTIndex())
    with pytest.raises(RuntimeError, match="gene2idx"):
        model.gene_embeddings()


def test_embed_batch_matches_direct_forward(
    tiny_config: MulticellularGraphConfig,
    tiny_batch: BagOfCellsBatch,
    seed: int,
) -> None:
    """MCTModel.embed_batch must be byte-identical to module(...) call."""
    torch.manual_seed(seed)
    pl_model_a = MulticellularTransformerPLModel(tiny_config)
    pl_model_a.eval()
    torch.manual_seed(seed)
    pl_model_b = MulticellularTransformerPLModel(tiny_config)
    pl_model_b.eval()

    model = MCTModel(pl_model_a, MCTIndex())
    out_via_api = model.embed_batch(tiny_batch)

    with torch.no_grad():
        out_direct = pl_model_b.model(
            expression_matrix=tiny_batch.expression_matrix,
            gene_ids=tiny_batch.gene_ids,
            is_masked_gene_expression=tiny_batch.is_masked_gene_expression,
            padding_mask=tiny_batch.padding_mask,
            tissue=None,
        )

    torch.testing.assert_close(
        out_via_api.last_hidden_state, out_direct.last_hidden_state, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        out_via_api.gene_id_logits, out_direct.gene_id_logits, atol=0.0, rtol=0.0
    )


def test_get_context_aware_gene_embeddings(
    tiny_config: MulticellularGraphConfig,
    tiny_batch: BagOfCellsBatch,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(tiny_config)
    pl_model.eval()
    idx2gene = {i: f"G{i}" for i in range(tiny_config.n_genes)}
    idx2gene[0] = "[PAD]"
    idx2gene[1] = "[MASK]"

    model = MCTModel(pl_model, MCTIndex(idx2gene=idx2gene))
    out = model.embed_batch(tiny_batch)
    per_cell = get_context_aware_gene_embeddings(model, out, tiny_batch)

    assert len(per_cell) == tiny_batch.gene_ids.shape[0]
    for cell_map in per_cell:
        assert all(not sym.startswith("[") for sym in cell_map)
        for vec in cell_map.values():
            assert vec.shape == (tiny_config.gene_emb_dim,)


def test_load_model_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "no_such.ckpt")


def test_presets_construct_without_erroring() -> None:
    """We can't actually build the datamodules without real data; just verify
    the presets themselves accept their documented kwargs."""
    p1 = AdataInferencePreset(adata_file_path="x.h5ad", gene2idx={"A": 2})
    assert p1.batch_size == 1
    p2 = CellarrInferencePreset(
        cell_tdb_uri="/a", gene_tdb_uri="/b", counts_tdb_uri="/c"
    )
    assert p2.lognorm is True
    p3 = SpatialInferencePreset(adata_file_path="x.h5ad", gene2idx={"A": 2})
    assert p3.max_seq_len == 2500
