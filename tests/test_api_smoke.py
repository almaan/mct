"""Smoke tests for the scanpy-style public API layer."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import cell_cell_interactions as cci
from cell_cell_interactions.api.model import (
    MCTIndex,
    MCTModel,
    _align_to_cell_type_order,
    _anndata_to_pseudobulk,
    _find_checkpoint,
    _normalize_log1p,
    load,
)
from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel


# ---------------------------------------------------------------------- #
# Top-level namespace
# ---------------------------------------------------------------------- #


def test_top_level_namespaces_exist() -> None:
    assert callable(cci.load)
    for name in ("rank_similar_genes", "pairwise_cosine_similarity", "gene_neighbors", "regress_out"):
        assert hasattr(cci.tl, name), f"cci.tl.{name} missing"
    for name in ("similarity_ranking", "similarity_heatmap", "plot_gene_set"):
        assert hasattr(cci.pl, name), f"cci.pl.{name} missing"
    for name in ("filter_to_model_vocabulary", "harmonize_cell_types", "log_normalize"):
        assert hasattr(cci.pp, name), f"cci.pp.{name} missing"


# ---------------------------------------------------------------------- #
# MCTIndex
# ---------------------------------------------------------------------- #


def test_mct_index_from_dir_loads_present_files(tmp_path: Path) -> None:
    gene2idx = {"[PAD]": 0, "[MASK]": 1, "GENEA": 2, "GENEB": 3}
    (tmp_path / "gene2idx.json").write_text(json.dumps(gene2idx))
    (tmp_path / "cell_type_input_order.json").write_text(json.dumps({"T": [0, 4]}))
    (tmp_path / "tissue2idx.json").write_text(json.dumps({"lung": 2}))

    index = MCTIndex.from_dir(tmp_path)
    assert index.gene2idx == gene2idx
    assert index.idx2gene[2] == "GENEA"
    assert index.cell_type_input_order == {"T": (0, 4)}
    assert index.tissue2idx == {"lung": 2}


def test_mct_index_from_dir_tolerates_missing(tmp_path: Path) -> None:
    (tmp_path / "gene2idx.json").write_text(json.dumps({"GENEA": 2}))
    index = MCTIndex.from_dir(tmp_path)
    assert index.gene2idx == {"GENEA": 2}
    assert index.cell_type_input_order is None


def test_mct_index_from_dir_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MCTIndex.from_dir(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------- #
# Checkpoint auto-discovery
# ---------------------------------------------------------------------- #


def _make_dummy_ckpt(path: Path) -> None:
    path.write_bytes(b"not-a-real-ckpt")


def test_find_checkpoint_prefers_canonical_names(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    _make_dummy_ckpt(tmp_path / "checkpoints" / "model.ckpt")
    _make_dummy_ckpt(tmp_path / "checkpoints" / "extra.ckpt")

    found = _find_checkpoint(tmp_path)
    assert found.name == "model.ckpt"


def test_find_checkpoint_falls_back_to_any_glob(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    _make_dummy_ckpt(tmp_path / "checkpoints" / "arbitrary.ckpt")

    found = _find_checkpoint(tmp_path)
    assert found.name == "arbitrary.ckpt"


def test_find_checkpoint_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find"):
        _find_checkpoint(tmp_path)


def test_load_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/model/dir/aaa")


def test_load_requires_something() -> None:
    with pytest.raises(ValueError, match="model_dir"):
        load(None)


# ---------------------------------------------------------------------- #
# gene_embeddings — no data required
# ---------------------------------------------------------------------- #


def test_gene_embeddings_returns_dataframe(
    tiny_config: MulticellularGraphConfig,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(tiny_config)
    pl_model.eval()

    gene2idx = {f"G{i}": i for i in range(2, tiny_config.n_genes)}
    gene2idx["[PAD]"] = 0
    gene2idx["[MASK]"] = 1
    idx2gene = {v: k for k, v in gene2idx.items()}
    model = MCTModel(pl_model, MCTIndex(gene2idx=gene2idx, idx2gene=idx2gene))

    all_emb = model.gene_embeddings()
    assert isinstance(all_emb, pd.DataFrame)
    assert all_emb.shape == (tiny_config.n_genes - 2, tiny_config.gene_emb_dim)
    assert all_emb.columns.tolist() == [f"dim_{j}" for j in range(tiny_config.gene_emb_dim)]
    assert "[PAD]" not in all_emb.index

    subset = model.gene_embeddings(["G3", "G5", "not_a_gene"])
    assert subset.index.tolist() == ["G3", "G5"]

    with_special = model.gene_embeddings(drop_special_tokens=False)
    assert "[PAD]" in with_special.index and "[MASK]" in with_special.index


# ---------------------------------------------------------------------- #
# AnnData -> pseudobulk plumbing
# ---------------------------------------------------------------------- #


def test_pseudobulk_pipeline_shapes() -> None:
    genes = ["EGFR", "TP53", "MYC", "GAPDH"]
    obs = pd.DataFrame(
        {
            "cell_type": ["T", "B", "T", "B", "T"],
            "sample": ["s1"] * 5,
        }
    )
    X = np.arange(len(obs) * len(genes)).reshape(len(obs), len(genes)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    gene2idx = {"[PAD]": 0, "[MASK]": 1, "EGFR": 2, "TP53": 3, "MYC": 4, "GAPDH": 5}
    ct_order = {"T": (0, 3), "B": (3, 4)}

    pseudobulk, gene_ids = _anndata_to_pseudobulk(adata, "cell_type", gene2idx)
    assert pseudobulk.shape == (4, 2)  # genes x cell_types (only T, B present)
    assert set(pseudobulk.columns) == {"T", "B"}
    assert list(gene_ids) == [gene2idx[g] for g in pseudobulk.index]

    normalized = _normalize_log1p(pseudobulk, target_sum=1e4)
    assert (normalized.values >= 0).all()

    aligned = _align_to_cell_type_order(normalized, ct_order)
    assert aligned.shape == (4, 2)
    # 'T' slot is index 0 in ct_order, 'B' is index 1
    np.testing.assert_allclose(aligned[:, 0], normalized["T"].values)
    np.testing.assert_allclose(aligned[:, 1], normalized["B"].values)


def test_normalize_log1p_zero_input_returns_zeros() -> None:
    df = pd.DataFrame(np.zeros((3, 2)))
    out = _normalize_log1p(df, target_sum=1e4)
    assert (out.values == 0).all()


# ---------------------------------------------------------------------- #
# End-to-end: MCTModel.embed on a tiny synthetic AnnData
# ---------------------------------------------------------------------- #


def _tiny_model_and_index(config: MulticellularGraphConfig, seed: int) -> MCTModel:
    """Build a tiny MCTModel with cell_type_input_order sized to n_input.

    When agg_cell_type_gex mode is intended (the mode ``embed`` targets),
    the input tensor width equals the number of cell types, so
    len(ct_order) must equal config.n_input for the ``gene_expression_encoder``
    Linear(n_input, hidden_dim) layer to accept the tensor.
    """
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(config)
    pl_model.eval()
    gene2idx = {f"G{i}": i for i in range(2, config.n_genes)}
    gene2idx["[PAD]"] = 0
    gene2idx["[MASK]"] = 1
    ct_names = [f"CT{i}" for i in range(config.n_input)]
    ct_order = {name: (i, i + 1) for i, name in enumerate(ct_names)}
    return MCTModel(
        pl_model,
        MCTIndex(
            gene2idx=gene2idx,
            idx2gene={v: k for k, v in gene2idx.items()},
            cell_type_input_order=ct_order,
        ),
    )


def test_embed_returns_per_sample_dataframes(
    tiny_config: MulticellularGraphConfig,
    seed: int,
) -> None:
    model = _tiny_model_and_index(tiny_config, seed)

    genes = [f"G{i}" for i in (2, 3, 4, 5, 6)]
    ct_labels = list(model.cell_type_input_order.keys())
    # Cycle through the model's cell types across cells so pseudobulk has content.
    obs = pd.DataFrame(
        {
            "cell_type": [ct_labels[i % len(ct_labels)] for i in range(2 * len(ct_labels))],
            "sample": ["s1"] * len(ct_labels) + ["s2"] * len(ct_labels),
        }
    )
    X = np.abs(np.random.default_rng(0).normal(size=(len(obs), len(genes)))).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    out = model.embed(adata, cell_type_key="cell_type", sample_key="sample")
    assert set(out) == {"s1", "s2"}
    for sample, df in out.items():
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == tiny_config.gene_emb_dim
        assert df.shape[0] == len(genes)
        assert set(df.index) == set(genes)


def test_forward_batch_matches_direct_module_call(
    tiny_config: MulticellularGraphConfig,
    tiny_batch,
    seed: int,
) -> None:
    """forward_batch must be byte-identical to calling module(...) directly."""
    torch.manual_seed(seed)
    pl_a = MulticellularTransformerPLModel(tiny_config)
    pl_a.eval()
    torch.manual_seed(seed)
    pl_b = MulticellularTransformerPLModel(tiny_config)
    pl_b.eval()

    mct = MCTModel(pl_a, MCTIndex())
    out_api = mct.forward_batch(tiny_batch)
    with torch.no_grad():
        out_direct = pl_b.model(
            expression_matrix=tiny_batch.expression_matrix,
            gene_ids=tiny_batch.gene_ids,
            is_masked_gene_expression=tiny_batch.is_masked_gene_expression,
            padding_mask=tiny_batch.padding_mask,
            tissue=None,
        )
    torch.testing.assert_close(
        out_api.last_hidden_state, out_direct.last_hidden_state, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        out_api.gene_id_logits, out_direct.gene_id_logits, atol=0.0, rtol=0.0
    )


def test_embed_dataloader_returns_dataframes_per_batch(
    tiny_config: MulticellularGraphConfig,
    tiny_batch,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    pl_model = MulticellularTransformerPLModel(tiny_config)
    pl_model.eval()
    idx2gene = {i: f"G{i}" for i in range(tiny_config.n_genes)}
    idx2gene[0] = "[PAD]"
    idx2gene[1] = "[MASK]"
    mct = MCTModel(pl_model, MCTIndex(idx2gene=idx2gene))

    # Fake a dataloader by yielding the same batch twice.
    def fake_loader():
        yield tiny_batch
        yield tiny_batch

    results = mct.embed_dataloader(fake_loader(), max_batches=2)
    assert len(results) == 2
    for entry in results:
        assert entry["gene_ids"].shape == tuple(tiny_batch.gene_ids.shape)
        embs = entry["embeddings"]
        assert len(embs) == tiny_batch.gene_ids.shape[0]
        for df in embs:
            assert df.shape[1] == tiny_config.gene_emb_dim
            # No special tokens leak through
            assert all(not sym.startswith("[") for sym in df.index)


def test_embed_raises_on_missing_column(
    tiny_config: MulticellularGraphConfig, seed: int
) -> None:
    model = _tiny_model_and_index(tiny_config, seed)
    obs = pd.DataFrame({"cell_type": ["T"], "wrong_sample": ["s1"]})
    adata = ad.AnnData(
        X=np.ones((1, 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G2", "G3", "G4"]),
    )
    with pytest.raises(KeyError, match="missing required column"):
        model.embed(adata, cell_type_key="cell_type", sample_key="sample")


# ---------------------------------------------------------------------- #
# cci.tl helpers on the same fake embedding matrix
# ---------------------------------------------------------------------- #


def test_pairwise_cosine_similarity_basic() -> None:
    emb = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=["A", "B", "C"], columns=["dim_0", "dim_1"]
    )
    out = cci.tl.pairwise_cosine_similarity(emb, pairs=[("A", "C"), ("A", "B"), ("A", "MISSING")])
    assert list(out["pair"]) == ["A_C", "A_B"]
    np.testing.assert_allclose(out["score"].values, [np.sqrt(2) / 2, 0.0], atol=1e-6)


def test_pairwise_cosine_similarity_lists_form() -> None:
    emb = pd.DataFrame(np.eye(3), index=["A", "B", "C"])
    out = cci.tl.pairwise_cosine_similarity(emb, pairs_a=["A", "B"], pairs_b=["B", "A"])
    assert out.shape[0] == 2


def test_pairwise_cosine_similarity_empty_result() -> None:
    emb = pd.DataFrame(np.eye(2), index=["A", "B"])
    out = cci.tl.pairwise_cosine_similarity(emb, pairs=[("X", "Y")])
    assert out.empty


def test_rank_similar_genes_by_symbol_and_vector() -> None:
    emb = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], index=["A", "B", "C"], columns=["dim_0", "dim_1"]
    )
    by_sym = cci.tl.rank_similar_genes(emb, "A", top_n=2)
    assert by_sym["gene"].tolist() == ["A", "B"]

    by_vec = cci.tl.rank_similar_genes(emb, np.array([1.0, 0.0]), top_n=1)
    assert by_vec["gene"].iloc[0] == "A"


def test_gene_neighbors_shape() -> None:
    emb = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 4)), index=[f"g{i}" for i in range(10)])
    out = cci.tl.gene_neighbors(emb, n_neighbors=3)
    assert set(out.columns) == {"gene", "neighbor", "rank", "distance"}
    assert out.shape == (10 * 3, 4)
    assert out["rank"].min() == 1 and out["rank"].max() == 3


# ---------------------------------------------------------------------- #
# cci.pp helpers
# ---------------------------------------------------------------------- #


def test_filter_to_model_vocabulary_subsets_var_names() -> None:
    adata = ad.AnnData(
        X=np.zeros((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"c": [0, 1]}),
        var=pd.DataFrame(index=["EGFR", "MADE_UP", "TP53"]),
    )
    filtered = cci.pp.filter_to_model_vocabulary(adata, {"EGFR": 2, "TP53": 3})
    assert filtered.var_names.tolist() == ["EGFR", "TP53"]


def test_log_normalize_dataframe() -> None:
    df = pd.DataFrame({"c1": [1.0, 2.0, 0.0], "c2": [5.0, 0.0, 3.0]}, index=["A", "B", "C"])
    out = cci.pp.log_normalize(df, target_sum=1e4)
    total = df.values.sum()
    expected = np.log1p(df.values / total * 1e4)
    np.testing.assert_allclose(out.values, expected)


def test_harmonize_cell_types_writes_new_key() -> None:
    adata = ad.AnnData(
        X=np.zeros((3, 1), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["T_naive", "B_mem", "unknown"]}),
        var=pd.DataFrame(index=["G"]),
    )
    out = cci.pp.harmonize_cell_types(
        adata,
        cell_type_key="cell_type",
        mapping={"T_naive": "T", "B_mem": "B"},
        new_key="cell_type_h",
    )
    assert out.obs["cell_type_h"].tolist() == ["T", "B", "unknown"]
    assert out.obs["cell_type"].tolist() == ["T_naive", "B_mem", "unknown"]
