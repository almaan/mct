# Inputs and API contracts

## Checkpoint and identifiers

Keep `.ckpt`, `gene2idx.json` (or `gene_symbol2idx.json` / `ensemble2idx.json`), `cell_type_input_order.json`, and applicable `tissue2idx.json` / `cell_type_map.json` together. Token numbering and cell-slot order are part of the trained model. Do not regenerate them from a new dataset or swap mappings merely because shapes match. The loader prefers the first recognized mapping filename: explicitly verify whether its keys are symbols or Ensembl IDs. Separate symbol and Ensembl maps may share token IDs only when they are paired artifacts.

Check `mct.config`, `mct.gene2idx`, `mct.index`, and `mct.cell_type_input_order`. Resolve gene aliases and duplicate identifiers before inference; adding arbitrary suffixes to duplicate symbols creates names outside the vocabulary. Record how duplicates were aggregated. `gene_embeddings()` drops special tokens by default but can retain unnamed `__idx_N__` rows when the mapping is incomplete; exclude them from biological analyses.

## AnnData convenience path

Supply nonnegative raw counts in `.X`, genes in `.var_names`, and nonmissing sample/cell-type columns in `.obs`. The convenience method always sums by cell type and applies global pseudobulk `log1p(count / total * target_sum)` with default `target_sum=1e4`. Passing already log-normalized input normalizes it again. This is not per-cell Scanpy normalization and is not automatically equivalent to the paper datamodules.

**Use `mct.embed` only for a compatible pseudobulk checkpoint:** verify `mct.config.agg_cell_type_gex`, `mct.config.n_input == len(mct.cell_type_input_order)`, and the exact trained label/order convention. For nonaggregated models, cell-type entries denote ranges of individual-cell slots; one column per type cannot fill that input. Use a datamodule instead. Shape equality alone is insufficient to establish preprocessing parity.

```python
# adata contains raw counts; metadata columns here are example user choices.
adata = cci.pp.filter_to_model_vocabulary(adata, mct.gene2idx, copy=True)
if mct.index.cell_type_map:
    adata = cci.pp.harmonize_cell_types(
        adata, cell_type_key="cell_type", mapping=mct.index.cell_type_map,
        new_key="mct_cell_type",
    )
    cell_type_key = "mct_cell_type"
else:
    cell_type_key = "cell_type"

# Inspect/filter unmapped labels explicitly before this call.
mct.eval()
per_sample = mct.embed(
    adata, cell_type_key=cell_type_key, sample_key="sample_id",
)
# dict[str, DataFrame], each matrix has genes as rows and embedding dimensions as columns.
```

Unrecognized types are omitted from aligned inputs, but their counts can still affect global normalization: inspect them before calling `embed`, then explicitly filter or remap them and record the decision. Validate that each sample has supported cells and useful expressed genes. This method retains vocabulary genes even when their sample expression is zero; establish an expression eligibility rule before neighborhood comparisons rather than interpreting these as observed gene support.

If using `tissue_key`, require one recognized tissue per sample and the paired tissue vocabulary. The implementation otherwise chooses the first tissue or silently omits conditioning. Preserve original sample IDs separately: returned dictionary keys are strings. Avoid merging multiple biopsies/conditions under one donor ID; use a unique sample ID and retain donor as separate metadata. Large `.X` matrices are densified within samples; use dataloaders for large corpora.

## Datamodule path and paper reproduction

Choose `AdataRnaseqDataModule`, `CellarrDataModule`, or `SpatialDataModule` from `cell_cell_interactions.pp.*_datamodule`. Read the constructor and dataset preprocessing before building it; the tutorials show the strategy but some contain obsolete arguments (`cat_covariates`) and private paths.

Match checkpoint gene maps, tissue maps, fixed cell-type slot layout, `sample_size`, `agg_cell_type_gex`, normalization, expressed-gene filtering, and sequence limit. Check the dataset's actual maps/order against checkpoint artifacts: some constructors derive them from the input data. For ordinary inference disable gene and tissue masking where supported and use evaluation mode. Datamodules can sample bags with replacement; one loop iteration is not necessarily one unique sample, nor an independent replicate. Retain bag IDs and seed and ensure intended sample coverage.

Run one batch first:

```python
batch = next(iter(dataloader))  # configured datamodule dataloader
assert batch.expression_matrix.shape[-1] == mct.config.n_input
out = mct.forward_batch(batch)
# out.last_hidden_state: (batch, gene_positions, embedding_dimension)
# out.gene_id_logits: (batch, gene_positions, vocabulary)
# out.gex_reconstruction: (batch, gene_positions, input_slots)
results = mct.embed_dataloader(dataloader, max_batches=1)
# Each entry: sample_label, gene_ids, embeddings (one DataFrame per batch item).
```

Resolve gene positions using `batch.gene_ids`, never a presumed alphabetical order. For raw outputs exclude padded positions (`batch.padding_mask`) and special/unmapped IDs. `embed_dataloader` excludes IDs below 2 and unmapped IDs; check padding conventions as well. Stream batches and save outputs rather than accumulating a full atlas in its returned list.

AnnData datamodules read a file and expect their documented expression preparation; CellArr can perform its own normalization. Spatial inputs also require `adata.obsp['spatial_connectivities']`, appropriate feature identifiers, and neighborhood metadata; inspect `SpatialDataModule` before constructing them. Do not apply the AnnData convenience preprocessing recipe to every input path.

## Analysis API behavior that matters

- `cci.tl.pairwise_cosine_similarity(E, pairs=[(a,b), ...])` returns `a,b,pair,score` and silently drops unavailable pairs. Join labels by both endpoints, not row position or ambiguous underscore-concatenated names.
- `cci.tl.rank_similar_genes(E, query, top_n=..., subset=...)` accepts a name or vector. Name queries include themselves; request an extra result and explicitly remove self.
- `cci.tl.gene_neighbors(E, n_neighbors=k, metric='cosine')` returns `gene,neighbor,rank,distance`; use `k < len(E)` and cosine similarity = `1-distance`. It skips the first nearest result assuming self is first; tied/identical vectors can violate that. For ties, explicitly filter self and recompute enough neighbors to retain k nonself rows.
- `cci.tl.regress_out` optionally adjusts scores for categorical study effects while preserving selected covariates. It requires `statsmodels`, which is not a core declared dependency. Fit per pair or justify pooling; inspect study/condition confounding before adjustment.
- `cci.tl.GeneEmbedding` consumes `{gene: vector}`, not a DataFrame. Its graph construction uses Scanpy defaults, not explicitly cosine distance.
- `cci.pl.similarity_ranking` and `similarity_heatmap` return Axes. Legacy plotting re-exports may have different return/show behavior; inspect them before saving.

## Known helper limitations in this checkout

`utils/sample_embeddings/api.py` and `utils/depmap/api.py` call `model.embed_batch`, while `MCTModel` exposes `forward_batch`. Do not present these helpers as ready-to-run against this version. Sample neighbors can be computed directly with `gene_neighbors` on contextual matrices; sample vectors can be pooled directly. For KO, implement a matched loop with `forward_batch` as described in the advanced reference. Recheck the installed version before applying this guidance if the API changes.

Collection helpers also make assumptions about batch size, padding, output-directory contents, and availability of a datamodule reference. Verify these before reusing them. Documentation of an entry point is not evidence that a paper-scale run has been validated.
