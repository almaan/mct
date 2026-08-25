# API reference

The `cell_cell_interactions` package uses a scanpy-style layout. One loader,
three submodules:

```python
import cell_cell_interactions as cci

mct = cci.load("/models/mct_v1")          # <- single call, auto-discovers files
emb  = mct.gene_embeddings()              # <- DataFrame (genes x dims)
rank = cci.tl.rank_similar_genes(emb, "EGFR")
ax   = cci.pl.similarity_ranking(rank)
adata = cci.pp.filter_to_model_vocabulary(adata, mct.gene2idx)
sample_embs = mct.embed(adata, cell_type_key="cell_type", sample_key="donor")
```

Table of contents:

- [`cci.load` and `MCTModel`](#loading-a-model)
- [`cci.tl` — analysis tools](#ccitl)
- [`cci.pl` — plotting](#ccipl)
- [`cci.pp` — preprocessing](#ccipp)
- [Reproducibility contract](#reproducibility-contract)

---

## Loading a model

### `cci.load(model_dir=None, *, checkpoint_path=None, index_dir=None, device=None, map_location="cpu", eval_mode=True) -> MCTModel`

Two calling styles:

**Directory mode (recommended).** Point at a directory that follows the
canonical layout used by the model registry:

```
/models/mct_v1/
├── checkpoints/
│   └── model.ckpt           # or multicellular_transformer.ckpt / best.ckpt / last.ckpt / *.ckpt
├── gene2idx.json            # or gene_symbol2idx.json / ensemble2idx.json
├── cell_type_input_order.json
├── tissue2idx.json          # optional
└── cell_type_map.json       # optional
```

```python
mct = cci.load("/models/mct_v1", device="cuda")
```

**Explicit mode.** Give the two paths yourself:

```python
mct = cci.load(
    checkpoint_path="/scratch/best.ckpt",
    index_dir="/scratch/index/",
)
```

### `class MCTModel`

Loaded model. Every attribute is a small facade; the underlying Lightning
module is accessible via `mct.lightning_module` and the raw torch module via
`mct.module` for advanced use.

| Attribute | Description |
|---|---|
| `mct.config` | `MulticellularGraphConfig` recorded with the checkpoint. |
| `mct.device` | Current torch device. |
| `mct.gene2idx` / `mct.idx2gene` | Token id maps. |
| `mct.cell_type_input_order` | Ordered cell-type slot layout. |
| `mct.index` | Full `MCTIndex` dataclass. |

Methods:

#### `mct.gene_embeddings(genes=None, drop_special_tokens=True) -> pd.DataFrame`

Zero-shot gene embeddings from the model's gene-ID classifier head. Rows are
gene symbols, columns are `dim_0 .. dim_{D-1}`.

```python
>>> mct.gene_embeddings().shape
(11743, 64)
>>> mct.gene_embeddings(["EGFR", "TP53", "MYC"])
             dim_0     dim_1  ...   dim_63
EGFR       -0.041     0.132   ...    0.087
TP53        0.019    -0.045   ...   -0.012
MYC         0.098     0.021   ...    0.056
```

#### `mct.embed(adata, cell_type_key="cell_type", sample_key="sample", tissue_key=None, target_sum=1e4) -> dict[str, pd.DataFrame]`

**Contextual gene embeddings per sample.** This is the paper's context-aware
representation: the surrounding cellular ensemble shapes each gene's vector,
so the same gene gets a different embedding in different samples.

The AnnData is grouped by `sample_key` and, within each sample, pseudobulked
by `cell_type_key` using the cell-type order the model was trained under.
Counts are log-normalized to `target_sum` (matches training-time preprocessing —
changing it WILL shift outputs).

```python
>>> per_sample = mct.embed(adata,
...                        cell_type_key="cell_type",
...                        sample_key="donor_id")
>>> list(per_sample)[:3]
['donor_A', 'donor_B', 'donor_C']
>>> per_sample["donor_A"].loc["EGFR"]
dim_0    -0.123
dim_1     0.045
...
```

Returned DataFrames have one row per gene present in that sample and are
indexed by gene symbol.

#### `mct.embed_dataloader(dataloader, max_batches=None) -> list[dict]`

Same contextual embeddings as `mct.embed`, but takes a dataloader from a
`CellarrDataModule` / `AdataRnaseqDataModule` / `SpatialDataModule`. Use this
for paper-reproduction runs where the datamodule owns gene filtering,
normalization, and cell-type stratification.

Each entry has:
- `sample_label` — list of sample identifiers (one per batch item).
- `gene_ids` — token id array `(B, seq_len)`.
- `embeddings` — list of DataFrames (one per batch item), indexed by gene symbol.

```python
>>> from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule
>>> dm = CellarrDataModule(
...     cell_tdb_uri="/data/cell_metadata",
...     gene_tdb_uri="/data/gene_annotation",
...     counts_tdb_uri="/data/counts",
...     cell_type_mapping=mct.index.cell_type_map,
... )
>>> results = mct.embed_dataloader(dm.val_dataloader(), max_batches=10)
>>> results[0]["embeddings"][0].loc["EGFR"].shape
(64,)
```

#### `mct.forward_batch(batch) -> MulticellularTransformerOutput`

Raw one-shot forward pass on a `BagOfCellsBatch`. Byte-identical to calling
`mct.module(...)` with the same tensors; the wrapper just handles moving
inputs to `mct.device`.

```python
>>> batch = next(iter(dm.val_dataloader()))
>>> out = mct.forward_batch(batch)
>>> out.last_hidden_state.shape         # (B, n_expressed_genes, gene_emb_dim)
torch.Size([1, 4823, 64])
>>> out.gene_id_logits.shape            # masked-gene-modeling logits
torch.Size([1, 4823, 12000])
```

### `class MCTIndex`

Container for the JSON metadata paired with a checkpoint. Fields (all
optional): `gene2idx`, `idx2gene`, `cell_type_input_order`, `tissue2idx`,
`cell_type_map`. Load with `MCTIndex.from_dir(path)`.

---

## `cci.tl`

Tools that turn embeddings into tables. All accept a DataFrame and return one
(scanpy-style).

### `cci.tl.pairwise_cosine_similarity(emb, pairs=None, pairs_a=None, pairs_b=None) -> pd.DataFrame`

Score cosine similarity between named pairs. Two calling styles:
`pairs=[(a1, b1), ...]` or `pairs_a=[a1, ...], pairs_b=[b1, ...]`. Missing
symbols on either side are silently dropped.

```python
>>> emb = mct.gene_embeddings()
>>> cci.tl.pairwise_cosine_similarity(
...     emb,
...     pairs=[("EGFR", "EGF"), ("KIT", "KITLG"), ("PDGFRA", "PDGFA")],
... )
        a       b            pair    score
0    EGFR     EGF        EGFR_EGF   0.634
1     KIT   KITLG       KIT_KITLG   0.712
2  PDGFRA   PDGFA   PDGFRA_PDGFA   0.681
```

### `cci.tl.rank_similar_genes(emb, query, top_n=20, subset=None) -> pd.DataFrame`

Rank genes by cosine similarity to a query gene symbol or an arbitrary
vector.

```python
>>> cci.tl.rank_similar_genes(mct.gene_embeddings(), "EGFR", top_n=5)
     gene  similarity
0    EGFR    1.0000
1   ERBB2    0.8123
2   ERBB3    0.7756
3    EREG    0.7412
4  HBEGF    0.7328
```

### `cci.tl.gene_neighbors(emb, n_neighbors=20, metric="cosine") -> pd.DataFrame`

Long-form k-NN table over all genes. Useful for feeding into downstream
graph analysis.

```python
>>> nbrs = cci.tl.gene_neighbors(mct.gene_embeddings(), n_neighbors=10)
>>> nbrs[nbrs["gene"] == "EGFR"].head(3)
     gene neighbor  rank  distance
0    EGFR    ERBB2     1     0.188
1    EGFR    ERBB3     2     0.224
2    EGFR     EREG     3     0.259
```

### `cci.tl.regress_out(scores, regress_out, keep, score_col="score", adjusted_col="score_adj", fisher_transform=True) -> pd.DataFrame`

Regress out categorical batch/study effects from a score column. Requires
`statsmodels`.

```python
>>> scores = cci.tl.pairwise_cosine_similarity(emb, pairs=my_rl_pairs)
>>> scores = scores.merge(meta, on="pair")                # study, tissue columns
>>> adjusted = cci.tl.regress_out(scores, regress_out="study", keep=["tissue"])
>>> adjusted[["pair", "score", "score_adj"]].head()
```

### Legacy re-exports

Also importable from `cci.tl`: `GeneEmbedding`, `calculate_condition_specificity`,
`get_most_common_neighbors`, `jaccard_similarity_across_items`. These are the
same functions used in the paper tutorials.

---

## `cci.pl`

Every function returns the `matplotlib.axes.Axes` it drew on and never calls
`plt.show()`.

### `cci.pl.similarity_ranking(ranking, top_n=20, ax=None, ...) -> Axes`

Bar plot of the top-N most-similar genes to a query.

```python
>>> ranking = cci.tl.rank_similar_genes(emb, "EGFR", top_n=20)
>>> ax = cci.pl.similarity_ranking(ranking, title="EGFR neighbors")
>>> ax.figure.savefig("egfr_neighbors.png", dpi=200, bbox_inches="tight")
```

### `cci.pl.similarity_heatmap(emb, genes=None, cmap="vlag", ...) -> Axes`

Cosine-similarity heatmap over a small gene set.

```python
>>> ax = cci.pl.similarity_heatmap(emb, genes=["EGFR", "ERBB2", "ERBB3", "ERBB4"])
```

### Legacy re-exports

`cci.pl.plot_gene_set`, `cci.pl.plot_similarity_across_conditions_heatmap`.

---

## `cci.pp`

Preprocessing on AnnData / DataFrame objects. Nothing here calls the model.

### `cci.pp.filter_to_model_vocabulary(adata, gene2idx, copy=False) -> AnnData`

Subset an AnnData to genes present in the model's vocabulary.

```python
>>> adata_ok = cci.pp.filter_to_model_vocabulary(adata, mct.gene2idx)
>>> adata_ok.n_vars, adata.n_vars
(11743, 22432)
```

### `cci.pp.harmonize_cell_types(adata, cell_type_key="cell_type", mapping=None, drop_unmapped=False, new_key=None) -> AnnData`

Apply a `{source: harmonized}` cell-type remap.

```python
>>> adata = cci.pp.harmonize_cell_types(
...     adata,
...     cell_type_key="cell_type",
...     mapping=mct.index.cell_type_map,
...     new_key="cell_type_h",
... )
```

### `cci.pp.log_normalize(df_or_adata, target_sum=1e4, copy=False)`

Global `log1p(x / total * target_sum)` — the same normalization
`MCTModel.embed` applies to pseudobulk under the hood. Useful for previewing
what the model sees.

```python
>>> import pandas as pd
>>> df = pd.DataFrame({"c1": [1, 2, 0], "c2": [5, 0, 3]}, index=["A", "B", "C"])
>>> cci.pp.log_normalize(df).round(3)
        c1     c2
A    0.693  2.301
B    1.099  0.000
C    0.000  1.386
```

---

## Sample-level and perturbation helpers

Not in the top-level namespace; import from utils submodules.

### `cell_cell_interactions.utils.sample_embeddings.api`

- `collect_sample_embeddings(model, dataloader, output_dir, max_steps=None, save_every_n_batches=5000, idx2tissue=None)`
- `collect_sample_neighbors(model, dataloader, output_dir, n_neighbors=50, max_steps=None, save_every_n_batches=1000, idx2tissue=None)`
- `calculate_neighbors(embeddings, n_neighbors=20, metric="cosine")`

### `cell_cell_interactions.utils.depmap.api`

- `run_gene_ko_effect(model, dataloader, tumor_cell_type, genes_to_knock_out, name2idx, output_dir, max_steps=None, save_every_n_batches=10, collect_all_gex=False)`
- `DEPMAP_DISEASE_MAP`

CLI shims (`python -m cell_cell_interactions.utils.sample_embeddings.run_collect_sample_embeddings`, etc.) continue to work.

---

## Reproducibility contract

Nothing above changes a released checkpoint's behavior. Every refactor is
guarded by `tests/test_inference_parity.py` and `tests/test_checkpoint_load.py`.
If loading a checkpoint that used to work starts producing different numbers,
that's a bug; open an issue.

Quickstart notebook: `tutorials/api_quickstart.ipynb`.
