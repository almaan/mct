# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   `cell_cell_interactions.api` — public API layer.
    -   `load_model(...)` + `MCTModel` wrapper: replaces raw
        `MulticellularTransformerPLModel.load_from_checkpoint(...)` +
        `model.model(...)` + manual `.to(device)` glue.
    -   `MCTModel.gene_embeddings(genes=None)`: replaces the raw
        `state_dict["model.gene_id_classifier.decoder.weight"]` access shown in
        the paper tutorials.
    -   `MCTModel.embed_batch(batch)`: byte-identical wrapper around
        `model.model(...)`.
    -   `AdataInferencePreset`, `CellarrInferencePreset`,
        `SpatialInferencePreset`: reduce the 30-kwarg datamodule ctors down to a
        handful of user-facing fields, plus `extra={}` for advanced knobs.
    -   `embed_adata(...)`, `embed_dataloader(...)`,
        `get_context_aware_gene_embeddings(...)`: replaces the per-notebook
        demux helpers.
    -   `MCTIndex.from_dir(...)`: loads any subset of `gene2idx.json`,
        `cell_type_input_order.json`, `tissue2idx.json`, `cell_type_map.json`.
-   `cell_cell_interactions.utils.sample_embeddings.api`
    (`collect_sample_embeddings`, `collect_sample_neighbors`) and
    `cell_cell_interactions.utils.depmap.api` (`run_gene_ko_effect`) as
    library-form wrappers around the existing `run_*.py` scripts.
-   `tests/test_inference_parity.py` and `tests/test_checkpoint_load.py`:
    fixture-based numerical parity + checkpoint-load contract tests. Snapshot
    at `tests/fixtures/inference_parity.pt`.
-   `tests/test_api_smoke.py`: import, index-loading, and
    `embed_batch`-vs-direct-forward byte-parity checks for the new API.
-   `docs/api.md`: public API reference.
-   `tutorials/api_quickstart.ipynb`: illustrative walkthrough of the new API.
-   Basic tool, preprocessing and plotting functions.

### Changed

-   `cell_cell_interactions.pp._shared`: `mask` (gene masking),
    `mask_tissue`, and `order_cells_in_input` (stratified packing) were
    duplicated across the three datamodules. Consolidated into shared
    functions; the three `DataModule` classes now delegate. Numerical output
    verified byte-identical against the pre-extraction state via
    `tests/test_inference_parity.py`.
-   `cell_cell_interactions.__init__` now re-exports the public API surface.
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
import paths (`from cell_cell_interactions.pp.cellarr_datamodule import
CellarrDataModule`, `GENE_MASKING_STRATEGY`, `TISSUE_MASKING_STRATEGY`, etc.)
still work. The parity test in `tests/test_inference_parity.py` gates every
refactor.

### Followups (not yet done)

-   Rewrite the six existing tutorial notebooks to use the new API. Requires a
    checkpoint and end-to-end re-execution to verify numeric output is
    unchanged.
-   Consolidate the four remaining `collate` bodies once a way to keep the
    per-datamodule tissue/spatial fields clean is settled.
-   Relocate `GeneEmbedding` plotting methods (`plot_similarities`,
    `plot_metagene`, `plot_metagenes_scores`) from `tl/` to `pl/` behind a
    deprecation shim.
-   Public API surface for the three `train_*` entrypoints.
