# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   **Public API layer, scanpy-style.** `cell_cell_interactions.api` exposes
    one entry point and three submodules:

    ```python
    import cell_cell_interactions as cci
    mct = cci.load("/models/mct_v1")             # auto-discovers ckpt + index JSONs
    emb = mct.gene_embeddings()                  # (n_genes x dims) DataFrame
    per_sample = mct.embed(adata,
                           cell_type_key="cell_type",
                           sample_key="donor_id")
    ranking = cci.tl.rank_similar_genes(emb, "EGFR")
    cci.pl.similarity_ranking(ranking)
    ```

-   `cci.load(model_dir=None, *, checkpoint_path=None, index_dir=None, device=None)` —
    auto-discovers a checkpoint in `checkpoints/` (or `ckpts/`, or the root),
    preferring `model.ckpt` / `multicellular_transformer.ckpt` / `best.ckpt` /
    `last.ckpt`, then any `*.ckpt`. Index JSONs are found alongside the ckpt
    or in the model dir.
-   `MCTModel.gene_embeddings(genes=None)` — DataFrame indexed by gene symbol.
    Replaces the raw `state_dict['model.gene_id_classifier.decoder.weight']`
    slice used in the paper tutorials.
-   `MCTModel.embed(adata, cell_type_key, sample_key, tissue_key=None,
    target_sum=1e4)` — AnnData → per-sample `{sample_id: DataFrame}`. Handles
    pseudobulk, cell-type ordering, log-normalization, and forward pass.
-   `cci.tl.rank_similar_genes`, `cci.tl.pairwise_cosine_similarity`,
    `cci.tl.gene_neighbors`, `cci.tl.regress_out` — DataFrame in, DataFrame
    out. Legacy tools (`GeneEmbedding`, `calculate_condition_specificity`,
    `get_most_common_neighbors`, `jaccard_similarity_across_items`)
    re-exported here as well.
-   `cci.pl.similarity_ranking`, `cci.pl.similarity_heatmap`, plus legacy
    `plot_gene_set` and `plot_similarity_across_conditions_heatmap`. Every
    function returns an `Axes` and never calls `plt.show()`.
-   `cci.pp.filter_to_model_vocabulary`, `cci.pp.harmonize_cell_types`,
    `cci.pp.log_normalize` — preprocessing utilities that operate on
    AnnData / DataFrame, do not touch the model.
-   `cell_cell_interactions.utils.sample_embeddings.api`
    (`collect_sample_embeddings`, `collect_sample_neighbors`) and
    `cell_cell_interactions.utils.depmap.api` (`run_gene_ko_effect`,
    `DEPMAP_DISEASE_MAP`) — library-form wrappers around the existing
    `run_*.py` scripts. Scripts remain as CLI shims.
-   `tests/test_inference_parity.py` and `tests/test_checkpoint_load.py`:
    fixture-based numerical parity + checkpoint-load contract tests.
    Snapshot at `tests/fixtures/inference_parity.pt`.
-   `tests/test_api_smoke.py`: import, checkpoint auto-discovery,
    gene-embeddings, AnnData-embed-end-to-end, and every `cci.tl` / `cci.pp`
    helper.
-   `docs/api.md`: full API reference with runnable examples for every
    function.
-   `tutorials/api_quickstart.ipynb`: illustrative walkthrough.
-   Basic tool, preprocessing and plotting functions.

### Changed

-   `cell_cell_interactions.__init__` now re-exports the public scanpy-style
    API surface. Users can `import cell_cell_interactions as cci`.
-   `cell_cell_interactions.pp._shared`: `mask` (gene masking),
    `mask_tissue`, and `order_cells_in_input` (stratified packing) were
    duplicated across the three datamodules. Consolidated into shared
    functions; the three `DataModule` classes now delegate. Numerical output
    verified byte-identical against the pre-extraction state via
    `tests/test_inference_parity.py`.
-   `cell_cell_interactions.tl.grn.GeneEmbedding`:
    -   Initialize `cluster_definitions = None` in `__init__`;
        `cluster_definitions_as_df` now raises a clear `RuntimeError` if the
        clustering step wasn't run.
    -   Removed the shadowed three-argument `generate_weighted_vector`
        definition; the two-argument version (used by the tutorials) is the
        only survivor.
    -   `get_adata` writes its intermediate matrix to
        `tempfile.NamedTemporaryFile(...)` instead of `./.tmp.txt`.
-   Removed duplicate `min_n_expressed_cells_per_gene` / `astype(str)` compute
    lines in `AdataRnaseqDataModule.__init__` and
    `AdataRnaseqDataModuleForInference.__init__` (pure dead code).
-   Fixed stale `CellarrDataModule` docstring example that referenced
    `MetricLearningZarrDataModule` and non-existent `gene_order` /
    `label_id_column` kwargs.

### Reproducibility contract

Loading a released checkpoint plus its paired index JSONs and running any
tutorial must produce the same numbers as before. Public CLI names and old
import paths still work. The parity test in
`tests/test_inference_parity.py` gates every refactor.

### Followups (not yet done)

-   Rewrite the six existing tutorial notebooks to use the new API. Requires
    a checkpoint and end-to-end re-execution to verify numeric parity.
-   Consolidate the four remaining `collate` bodies once a way to keep the
    per-datamodule tissue/spatial fields clean is settled.
-   Relocate `GeneEmbedding` plotting methods to `pl/` behind deprecation
    shims (currently mixed into the analysis class).
-   Public API surface for the three `train_*` entrypoints.
