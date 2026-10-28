# Analysis strategies

Contents: 1. Gene/signature neighbors · 2. Programs and networks · 3. Pair benchmarks · 4. Condition specificity · 5. Differential interactions · 6. Sample structure · 7. Paper coverage.

Use `E` for a gene-by-dimension matrix and `E_s` for one sample's contextual matrix. Use one checkpoint/representation consistently within a comparison. Embedding coordinates from independently trained models are not aligned; compare their within-model rankings/metrics rather than subtracting their vectors.

## 1. Related genes and signature expansion

Rank cosine similarity between a query vector and candidate gene vectors with `cci.tl.rank_similar_genes`. For a receptor query, restrict candidates to ligands; for a transcription factor, use eligible targets. Remove self by identifier. Return the score and candidate universe with the ranking.

To expand a gene signature, intersect its genes with E, inspect coverage, and average their vectors (optionally L2-normalize each first to give genes equal directional weight). Pass the resulting centroid as `query`; exclude seed genes from discoveries. Weighted centroids can encode signed hypotheses but can cancel to near zero: inspect norm and document weights. This is an extension of embedding geometry, not a separate trained MCT predictor. Evaluate held-out members or external pathways rather than treating nearest neighbors as validation of themselves.

For contextual comparisons, rank within each sample first, then compare recurring partners and uncertainty across independent samples. Averaging vectors before ranking answers a different question and can hide heterogeneous programs.

## 2. Gene programs, enrichment, and candidate networks

Choose a gene universe (all covered genes, a biologically justified HVG subset, or HVGs plus receptors/ligands). Report its effect on interpretation and use the same universe for comparisons. Build a gene-neighbor graph from embeddings, cluster with Leiden, and inspect stability across resolutions/seeds. Cluster labels are arbitrary and are not aligned across samples; compare membership overlap or pathway annotations.

The tutorial-compatible adapter is:

```python
ge = cci.tl.GeneEmbedding({g: row.to_numpy() for g, row in E.iterrows()})
gdata = ge.get_adata(resolution=20)  # example, not a universal optimum
programs = {k: v for k, v in ge.get_metagenes(gdata).items() if len(v) >= 4}
```

`get_adata` runs Scanpy neighbors on raw E, Leiden, and UMAP. For an explicitly cosine-based graph, construct AnnData with **genes as observations**, call Scanpy neighbors with `use_rep='X', metric='cosine'`, then Leiden; record this change from the helper's defaults. Do not cluster the 2D UMAP coordinates. The paper explores high resolutions and excludes clusters with fewer than four genes; this is a reproduction setting, not a universal biological threshold.

Enrich programs against GO/Reactome using covered, eligible genes as the background. Specify organism, library/version, correction method, and the family of tests across programs and pathways. A network plot is a visualization of similarity edges; use held-out annotations and independent samples to assess biological coherence. For CCC enrichment, the paper selects communication-related pathways using keywords and manual review: preserve the curated list if reproducing that result.

Map programs back to cell types using expression dot plots or expression-based module scores in the original AnnData. This helps distinguish a shared multicellular program from a single cell-type marker set. Expression module scores and embedding similarity are different quantities.

For candidate GRNs, restrict one endpoint to TFs and rank TF–target cosine scores; validate against regulons. PPI benchmarking uses the same geometry with undirected pairs. A function in `tl/grn.py` or a TF annotation does not make the underlying symmetric similarity a directional or signed regulatory estimate.

## 3. Receptor–ligand and other pair benchmarks

Load a curated positive set and a declared candidate universe. Intersect both endpoints with available embeddings, record missing pairs, then score with `cci.tl.pairwise_cosine_similarity`. The optional mapping `(cosine + 1)/2` preserves ranking and AUROC but is not probability calibration.

For fair method comparisons:

1. Use the same gene/pair universe, labels, and sampled negatives for all models. Exclude known positives from negatives and deduplicate pairs; canonicalize unordered PPI pairs but retain receptor/ligand roles.
2. Include hard negatives that share one endpoint with a positive. Record the negative ratio and random seeds. The paper's LIANA method describes 1:10 positives:negatives; the CellChat notebook uses a different sampling recipe. Select the intended benchmark explicitly.
3. Report AUROC and precision–recall/average precision, positive prevalence, repeated-seed variation, and per-receptor metrics when both classes exist. Random unlabeled pairs can contain unknown positives.
4. Compare with expression correlation over **sample-level aggregated expression**, not a mismatched per-cell baseline. For supervised extensions, split before fitting and consider gene/receptor-held-out splits to reduce endpoint leakage.

Resources: `tutorials/data/receptor_ligand_with_negatives.csv`, `receptor_ligand_liana_consensus.csv`, `cellchatdb.csv`, `ppi_pairs_liana.csv`, and `tf_regulon_pairs_liana.csv`. Inspect columns before joining; their schemas differ. The paper methods and CellChat tutorial differ in resource and filtering choices, so do not silently combine their reported metrics.

The paper's single-gene analysis excludes multimeric complexes. If extending to complexes, require constituent coverage and declare an aggregation rule (such as mean constituent vectors), rather than interpreting a complex string as one gene. Partial-complex means change the candidate and must be flagged. CellChatDB pair classification does not itself run CellChat or infer sender/receiver cell types.

## 4. Condition-specific genes: stable within, rewired between

For each sample, compute a ranked cosine-neighbor list for each eligible gene. The paper collects 50 neighbors, compares the first 20 within conditions, and uses the most frequent 40 to summarize each condition. Keep the eligibility universe comparable across samples or quantify how missing genes alter neighborhoods.

Build a table with **one row per gene and independent sample**: `gene_name`, `sample`, condition column, and `neighbors` (ordered list/array of consistently named genes). From the long API output:

```python
neighbors = cci.tl.gene_neighbors(E_s, n_neighbors=min(50, len(E_s) - 1))
# Require at least two eligible genes and verify nonself neighbors first.
rows = (neighbors.sort_values(['gene', 'rank'])
        .groupby('gene')['neighbor'].agg(list).rename('neighbors').reset_index()
        .rename(columns={'gene': 'gene_name'}))
rows['sample'] = sample_id
# Join sample-level condition metadata by sample, with many-to-one validation.
```

For each gene, compute:

- Within-condition consistency: mean Jaccard `|A∩B| / |A∪B|` between the top-k neighbor sets of distinct samples in each condition.
- Across-condition consistency: Jaccard between each condition's frequent-neighbor summary.
- Specificity: mean within-condition overlap divided by mean across-condition overlap. High scores indicate reproducible rewiring, not necessarily differential expression.

Call `cci.tl.calculate_condition_specificity(df, condition='disease', ...)` for compatibility with the supplied tables. Pre-filter to enough independent samples per gene/condition and at least two distinct retained conditions yourself: `min_n_conditions` currently counts rows, not distinct condition labels. The implementation includes zero diagonals in the Jaccard matrix mean and discards zero-overlap cases. Consequently it differs from the paper's off-diagonal equation and depends on group size. Preserve and disclose this when reproducing the repository; for a new implementation use distinct sample pairs and explicitly handle zero denominators, labeling it as a methodological change.

Inspect both overlap components and sample counts, not just their ratio. Balance sample counts or assess sensitivity; repeated bags of the same sample are not independent evidence. Use donor-aware resampling/permutation for uncertainty. Display neighbor-overlap heatmaps and partner lists for selected genes, then compare with expression. The shipped tissue/lupus/COVID tables are examples, not reusable scores for a new cohort.

## 5. Differential contextual interactions

Score a fixed set of pairs separately in every sample, then join condition/study/donor metadata by sample. Preserve `sample,a,b,score,condition` and availability counts; do not pool cells or count stochastic bags as biological replicates. Require adequate support in every group (the paper requires at least three samples per group for its SLE pair analysis).

Estimate condition means and their difference, with donor-aware uncertainty and multiple-testing correction when testing many pairs. The paper also uses `(mean_disease - mean_control) / mean_control`; report it only when the control denominator is meaningful, since cosine can be zero or negative. Do not silently add an epsilon and interpret the resulting large ratios as biology.

Optionally adjust study effects with `cci.tl.regress_out` per pair, preserving condition/tissue; inspect the design for confounding and retain raw results. A batch correction cannot identify a disease effect if study and disease are inseparable. Compare pair-score shifts against expression correlation or abundance changes: this directly tests the claim that embedding neighborhoods reveal changes missed by expression alone.

## 6. Whole-sample structure

Mean-pool valid contextual gene vectors per bag; omit padding and special tokens. For `per_sample` DataFrames, a sample matrix is:

```python
sample_vectors = pd.DataFrame({s: e.mean(axis=0) for s, e in per_sample.items()}).T
```

Here `pd` is pandas; require nonempty, filtered matrices. When multiple bags represent a sample, aggregate them explicitly or retain repeated-measure IDs. Unequal gene coverage and cell composition can drive sample distances; inspect these with batch/study metadata and test a common-gene sensitivity analysis.

Construct a neighbor graph in sample-vector space, use UMAP for display and Leiden for clustering, and evaluate ARI/NMI against independent tissue or disease labels. Paper settings include 15–30 neighbors and resolution 1.0; scale neighbors to sample count and do not tune and report on the same labels without disclosure. Compare to PCA of sample-level expression with the same samples and split. A separated UMAP alone is not evidence of biological generalization.

## 7. Paper coverage and implementation sources

The source manuscript is `rsc/BRAID_MLTD_GGI_v2.pdf`, *Learning gene co-regulation across cells with Multicellular Transformer*. Figure/section references refer to that version. “Strategy” below means a described analysis, not a verified end-to-end reproduction.

| Paper analysis | Skill location | Repository starting point / status |
|---|---|---|
| Fig. 2a, S3–S5: RL, PPI, TF–target benchmarks | §3 | `zero_shot_receptor_ligand_analysis.ipynb`, `benchmarking_ccc_rl_cellchat.ipynb`, pair CSVs |
| Fig. 2b, S6: cell-type attribution | Advanced §2 | Requires custom differentiable analysis; no dedicated tutorial |
| Fig. 2c/f, S8–S9: programs and enrichment | §2 | `zero_shot_gene_sets_analysis.ipynb`, `inference_and_context_aware_gene_sets.ipynb` |
| Fig. 2d/e, S7: protein integration and candidate complexes | Advanced §4 | External protein/structure resources; no complete packaged workflow |
| Fig. 3a–e, S10–S13: condition specificity | §4 | `condition_specific_genes_analysis.ipynb`, example parquet tables |
| Fig. 3f: disease-associated RL shifts | §5 | Compose contextual inference and pair-scoring API |
| Fig. 3g, S14: sample clustering | §6 | Mean pooling; sample collection helper needs compatibility check |
| Fig. 4, S15: perturbation effects | Advanced §1 | `utils/depmap/`; custom loop needed for current wrapper and gene-logit analysis |
| Fig. 5, S16: spatial validation and cell-type matrix | Advanced §3 | `pp/spatial_datamodule.py`, `train_spatial`; downstream analysis custom |
| Model training/context-size ablations | Training reference | Training entry points; repeated matched runs required for ablations |

All notebook names in this table are under `tutorials/`. Baseline model weights, full cohorts, protein features, and some figure-specific code are external to these tutorials. For exact reproduction, reconcile manuscript settings with the particular notebook/script and record differences.
