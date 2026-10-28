# Advanced analyses

Contents: 1. Perturbations · 2. Attribution · 3. Spatial analysis · 4. Protein integration.

These strategies cover manuscript analyses beyond the complete public tutorials. Implement and validate a small case before scaling. They require additional data or custom analysis code; there are no `cci.tl` one-call methods for these workflows.

## 1. In-silico gene knockout

Choose a checkpoint appropriate to the tissue and readout. The paper uses a lung-cancer-fine-tuned model; population readouts require a trained expression reconstruction head and cell-type-pseudobulk layout. A head present in the checkpoint is not evidence that its loss was enabled or that it predicts viability.

Build an unmasked baseline batch and run `mct.forward_batch`. For each target present in a sample, delete its token position consistently from `expression_matrix`, `gene_ids`, `is_masked_gene_expression`, and `padding_mask`, preserve tissue conditioning, and run the matched perturbed batch. Use batch size one initially: each sample can have a different target position. Preserve the input batch (e.g. `dataclasses.replace` with new tensors), metadata, gene order, model state, and all other preprocessing. Do not resample the baseline and perturbation independently.

Deleting a gene token removes that gene's expression across all input cells/slots. It does not represent a cell-type-specific knockout. Merely setting its expression to zero while retaining its identity token is not necessarily equivalent: the convenience input path retains zero-expression genes. Document the intervention explicitly.

Choose one of two readouts:

- **Gene–gene dependency:** compare `gene_id_logits` for matched remaining positions, aligned by token ID after deletion. The paper describes a maximum log-ratio over predicted identity vectors. Raw logits can be negative, so literal log ratios of logits are undefined; verify the original scoring implementation for exact reproduction. For a new analysis, a defensible explicit alternative is to compute log-softmax probabilities, take baseline-minus-KO log-probability differences divided by `log(2)`, then summarize the vocabulary axis. Declare this as a chosen probability-based score, not verified paper parity. Compare known interacting targets with matched controls, aggregate across independent samples, and report the aggregation/direction.
- **Population readout:** for a pseudobulk model, reconstruct `(genes, cell_type_slots)`, aggregate over the specified gene set, and compare cell-type totals. The paper uses `(KO - baseline) / baseline`; negative values mean lower reconstructed expression. For a relative allocation measure, apply softmax across slots per gene before summing, as the DepMap helper does. Absolute totals and relative allocations answer different questions. State whether the baseline includes the deleted gene or only shared genes: including it adds its direct removed contribution. Handle zero denominators explicitly.

Compare tumor and non-tumor changes to identify selectivity rather than only generic housekeeping/ribosomal dependencies. Benchmark with a declared DepMap release, matched gene IDs and cancer subtype/cell lines, ranking overlap, and correlation. Preserve sign conventions: reduced reconstructed expression and measured CRISPR dependency are not automatically on the same scale. Relative reconstructed abundance is a model proxy, not a measured survival rate.

`cell_cell_interactions/utils/depmap/api.py` documents the workflow but currently calls missing `MCTModel.embed_batch`; it also requires a dataloader-attached datamodule and assumes batch size one in several operations. Use a direct `forward_batch` loop for this checkout, checking slot indices against paired metadata. The gene-logit analysis needs custom scoring even if the population wrapper is repaired.

## 2. Cell-type attribution of an interaction score

Define a scalar score that depends on the input: for example cosine similarity between two **contextual** output gene vectors for a fixed pair. A cosine between static classifier weights does not depend on a sample and cannot explain which input cells drove that sample's score.

Use integrated gradients along a declared expression baseline (such as all zeros, with gene IDs/slots/tissue held fixed). Use `mct.module` directly in evaluation mode with gradients enabled; `mct.forward_batch`, `mct.embed`, and detached DataFrames cannot supply this computation because inference disables gradients or detaches tensors. Keep the score definition, baseline, path steps, and normalization explicit.

Attribute to the floating-point expression matrix, not integer token IDs. Aggregate absolute attributions across genes and cells within the trained cell-type slot ranges; for pseudobulk, each column is already a type. Aggregate across biological samples with uncertainty. The paper's figure applies softmax to aggregated absolute attributions; these are relative importance weights and lose sign. Check convergence/completeness and sensitivity to the baseline. Attribution measures model dependence, not experimentally established signaling direction. A library such as Captum is an optional external dependency, not a bundled MCT API.

## 3. Spatial colocalization and cell-type interaction matrices

### Compare gene similarity with independent spatial evidence

Use a spatial dataset with expression, coordinates, tissue/section labels, and overlapping genes. Define the spatial weight matrix within each section before scoring. The paper uses a 50 µm neighborhood for Xenium colocalization and six neighboring grid spots for Visium. Distinguish these validation neighborhoods from the 350 µm bags used for spatial fine-tuning.

For a matched gene pair X,Y, compute bivariate Moran's I from centered expression and the declared spatial weights. The manuscript form is:

`I_XY = N * sum_ij(w_ij * (X_i-mean(X)) * (Y_j-mean(Y))) / (sum_i((X_i-mean(X))^2) * sum_ij(w_ij))`.

Record weight normalization and expression scaling; the statistic is directional and variance-dependent, whereas embedding cosine is symmetric. Preserve the manuscript convention for reproduction or explicitly symmetrize/standardize for a new analysis. Do not substitute univariate Moran's I. A spatial library can build the graph, but verify that the function actually computes the required bivariate statistic. Exclude constant genes and empty graphs.

Compare contextual pair cosines from matched tissues with spatial scores on identical eligible pairs. The paper samples 200 genes, evaluates pairwise colocalization, and examines correlation above colocalization percentile thresholds; record whether thresholds use absolute scores as in the figure caption. Report pair counts and spatial/donor-aware uncertainty at each threshold. Use expression-correlation baselines and independent sections; a shrinking tail with higher correlation alone does not establish improved generalization.

### Spatial fine-tuning

Use `SpatialDataModule` and `train_spatial` with spatial neighbor bags and compatible annotations. The paper uses 350 µm neighborhoods, 512-cell bags, and Leiden-derived labels when cell types are absent. This changes the expression encoder's input context; retain pretrained transferable weights and learn compatible input/reconstruction layers. Keep evaluation regions/sections separate to prevent overlapping neighborhoods leaking between training and validation. Compare base and spatially fine-tuned models on the same genes and held-out spatial data.

### Aggregate marker-pair evidence to cell-type pairs

Supply declared marker sets per cell type/state (the paper uses supplementary Table S2). For each marker pair, obtain similarity `s_xy` and each gene's mean similarity to the background universe, `mean_x` and `mean_y`. The paper normalizes as `2*s_xy / (mean_x + mean_y)`, averages these normalized values across marker pairs for a cell-type pair, then takes log2. Use the same aggregation for a spatial-score comparator.

Check denominator size and positivity before division/log2; signed or near-zero similarities can make this normalization undefined or unstable. Report excluded cases or choose a clearly labeled alternative for a new analysis. Handle overlapping markers/self-pairs consistently and test sensitivity to marker selection. This is a marker-derived association matrix, not direct observation of physical contact or a sender–receiver probability.

## 4. Protein information and structural follow-up

For receptor–ligand prediction, map genes to explicit UniProt proteins/isoforms and retain sequence provenance. Obtain protein model embeddings, pool residue vectors per protein using a stated convention, and calculate a protein cosine score. The paper combines MCT and ESMC scores as two features in XGBoost with five-fold cross-validation. The public CellChat notebook also compares ESM2; those are distinct models/protocols.

Split before fitting or selecting features and report held-out metrics. Use common pairs and unchanged negatives for MCT-only, protein-only, and combined baselines. Consider receptor/gene-disjoint folds for novel-partner claims. A supervised combined classifier is not a zero-shot MCT score even if its input embeddings were pretrained.

For novel complex candidates, rank contextual pair similarity across sufficiently supported samples (the paper requires at least three), filter with secretion/transmembrane annotations, then prioritize structural follow-up. The manuscript screens its top 70 pairs with AlphaFold3. Structure prediction, ipTM, secretion, and embedding similarity are complementary evidence; none alone validates binding. The repository does not bundle a structure-prediction service or all protein artifacts. Generate a candidate table with provenance before any separately authorized external submission.
