# API reference

`cell_cell_interactions.api` is the stable public surface for the Multicellular Transformer (MCT).
Everything below is importable from the top level:

```python
from cell_cell_interactions.api import load_model, CellarrInferencePreset, embed_adata
# or, equivalently:
from cell_cell_interactions import load_model, CellarrInferencePreset, embed_adata
```

Lower-level modules (`cell_cell_interactions.modeling`, `cell_cell_interactions.pp`, `cell_cell_interactions.tl`, `cell_cell_interactions.pl`)
remain importable but their signatures are subject to internal refactoring. Prefer the API
surface below whenever it covers your use case.

---

## Model loading

### `load_model(checkpoint_path, index_dir=None, map_location="cpu", eval_mode=True) -> MCTModel`

Load a Lightning `.ckpt` produced by MCT training. `index_dir` should point at a directory
containing any subset of `gene2idx.json` (or `gene_symbol2idx.json` / `ensemble2idx.json`),
`cell_type_input_order.json`, `tissue2idx.json`, `cell_type_map.json`. If `index_dir` is not
given, the loader searches the checkpoint's directory and the repository's `tutorials/data/`
before returning an empty index.

### `class MCTModel`

Wraps `MulticellularTransformerPLModel` with a small user-facing surface:

- `MCTModel.config` — the `MulticellularGraphConfig` recorded with the checkpoint.
- `MCTModel.device` — active torch device.
- `MCTModel.module` / `MCTModel.lightning_module` — escape hatches to the underlying modules.
- `MCTModel.to(device)` — move to a device, returns self.
- `MCTModel.eval()` — set eval mode, returns self.
- `MCTModel.gene_embeddings(genes=None) -> dict[str, np.ndarray]` — zero-shot gene embeddings
  from the gene-ID classifier head. Replaces the raw
  `state_dict["model.gene_id_classifier.decoder.weight"]` slice from the paper tutorials.
- `MCTModel.embed_batch(batch) -> MulticellularTransformerOutput` — thin wrapper around
  `model.model(...)` that handles moving every tensor in a `BagOfCellsBatch` to the model
  device. Byte-identical to calling the underlying module directly.

### `class MCTIndex`

Container for the JSON metadata files paired with a checkpoint. Fields (all optional):

- `gene2idx: dict[str, int]`
- `idx2gene: dict[int, str]`
- `cell_type_input_order: dict[str, tuple[int, int]]`
- `tissue2idx: dict[str, int]`
- `cell_type_map: dict[str, str]`

Construct from a directory with `MCTIndex.from_dir(path)`.

---

## Data presets

Each preset is a dataclass; `.build()` returns the underlying Lightning datamodule.
Extra keyword arguments accepted by the datamodule but not surfaced here can be passed via
the `extra=` field.

### `AdataInferencePreset(adata_file_path, gene2idx, ...)`

Wraps `AdataRnaseqDataModule` with defaults appropriate for inference (`val_set_proportion=1.0`,
`mask_prob=0.0`, `num_workers=0`).

### `CellarrInferencePreset(cell_tdb_uri, gene_tdb_uri, counts_tdb_uri, ...)`

Wraps `CellarrDataModule` with the same inference defaults plus `lognorm=True`,
`target_sum=1e4`.

### `SpatialInferencePreset(adata_file_path, gene2idx, ...)`

Wraps `SpatialDataModule`.

---

## Inference

### `embed_adata(model, preset, return_batches=False, loader="val", max_batches=None) -> list[dict]`

Build a preset's datamodule and iterate the chosen loader end-to-end. Returns a list of dicts,
one per batch, each containing `last_hidden_state`, `gene_ids`, and `sample_label`.

### `embed_dataloader(model, dataloader, return_batches=False, max_batches=None) -> list[dict]`

Same as `embed_adata` but takes an already-built dataloader.

### `get_context_aware_gene_embeddings(model, output, batch, idx2gene=None, skip_gene_ids=(0, 1)) -> list[dict[str, np.ndarray]]`

Per-cell `{gene_symbol: vector}` map. Replaces the per-notebook demux helper.

---

## Analysis

Re-exported from `cell_cell_interactions.tl`:

- `GeneEmbedding(embeddings)` — cosine-similarity gene analysis and metagene clustering.
- `calculate_condition_specificity(df, condition, min_n_conditions, min_n_samples, k_neighbors_within_condition, k_neighbors_across_conditions)` — per-sample gene-neighbor Jaccard aggregation.
- `get_most_common_neighbors(item, k=50)`, `jaccard_similarity_across_items(item, k=50)` — building blocks.

## Plotting

Re-exported from `cell_cell_interactions.pl`:

- `plot_gene_set(gene_set, embed, thresh=0.4, seed=10, edge_font_size=15)` — cosine-similarity network for a gene set.
- `plot_similarity_across_conditions_heatmap(gene_df, k, min_items, condition, labels_fontsize)` — Jaccard heatmap.

---

## Sample-level and perturbation helpers

Not re-exported from the top level; import from the utils submodules.

### `cell_cell_interactions.utils.sample_embeddings.api`

- `collect_sample_embeddings(model, dataloader, output_dir, max_steps=None, save_every_n_batches=5000, idx2tissue=None)`
- `collect_sample_neighbors(model, dataloader, output_dir, n_neighbors=50, max_steps=None, save_every_n_batches=1000, idx2tissue=None)`
- `calculate_neighbors(embeddings, n_neighbors=20, metric="cosine")`

### `cell_cell_interactions.utils.depmap.api`

- `run_gene_ko_effect(model, dataloader, tumor_cell_type, genes_to_knock_out, name2idx, output_dir, max_steps=None, save_every_n_batches=10, collect_all_gex=False)`
- `DEPMAP_DISEASE_MAP` — canonical disease-code lookup.

The CLI scripts `run_collect_sample_embeddings.py`, `run_collect_sample_neighbors.py`,
`run_gene_ko_effect_on_tumor_cells_adata.py`, and `run_gene_ko_effect_on_tumor_cells_cellarr.py`
continue to work; they are thin argparser shims over the same computation.

---

## Reproducibility contract

Nothing above changes the behavior of a released checkpoint. Every refactor step is guarded
by parity tests in `tests/test_inference_parity.py` and `tests/test_checkpoint_load.py`. If a
checkpoint that used to load starts producing different numbers, that's a bug; open an issue.

Quickstart in `tutorials/api_quickstart.ipynb`.
