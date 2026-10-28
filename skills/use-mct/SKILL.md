---
name: use-mct
description: Use the Multicellular Transformer (MCT, cell_cell_interactions) Python package to load checkpoints, extract static or contextual gene embeddings, discover gene programs and condition-specific neighborhoods, score receptor-ligand and regulatory associations, analyze samples and spatial patterns, and design in-silico perturbation analyses. Includes brief training and fine-tuning guidance. Use for MCT analysis and inference requests or adapting its paper workflows to new data.
---

# Use Multicellular Transformer

Use the public Python API to turn a biological question into a representation, a score, and a validation strategy. Adapt the analysis to the user's samples; use notebooks as methodological examples rather than scripts to copy verbatim.

## Locate the package and choose a representation

Locate the MCT checkout containing `pyproject.toml`, `cell_cell_interactions/`, and `tutorials/`. Paths below referring to package code or tutorials are relative to that checkout, not to the skill installation directory. If only the installed package is available, inspect its source/signatures; request a checkout only when the task requires repository resources. Do not assume private notebook paths exist.

Read [input and API guidance](references/inputs-and-api.md) before contextual inference. Read only the relevant sections of [analysis strategies](references/analyses.md) for downstream work, or [advanced analyses](references/advanced.md) for perturbations, attribution, spatial analysis, and protein integration.

MCT treats **genes as tokens** and encodes each gene's expression across a bag of cells. Cells are input features, not rows of the gene embedding matrix. Distinguish:

| Representation | Obtain it with | Appropriate questions |
|---|---|---|
| Static pretrained gene vectors, one row per gene | `mct.gene_embeddings()` | General gene associations, zero-shot receptor–ligand ranking, reusable gene programs |
| Contextual gene vectors, one row per gene per sample/bag | `mct.embed(...)` for compatible pseudobulk models; otherwise `mct.embed_dataloader(...)` | Condition-specific neighborhoods, differential pair scores, context-specific programs |
| Sample vector | Mean of valid contextual gene vectors within a bag | Sample clustering, tissue/disease structure |
| Gene identity logits / expression reconstruction | `mct.forward_batch(batch)` | Perturbation readouts; reconstruction requires an appropriately trained head |

Static vectors are decoder-classifier weights; contextual vectors are final transformer hidden states. Do not substitute one for the other or describe cosine similarity as a probability, binding measurement, directed regulation, or causal effect. The API returns gene-by-dimension matrices, **not** an independently indexed vector for every cell–gene pair. Cell-type interpretation requires expression/marker annotations, attribution, or reconstruction outputs.

## Start with the public API

Install from the checkout using the README's environment recipe (`pip install -e ".[tutorials]"`; Python 3.10/3.11 is a practical choice for the pinned dependencies). Load a checkpoint with its own paired indices:

```python
import cell_cell_interactions as cci

mct = cci.load(
    checkpoint_path="checkpoints/multicellular_transformer.ckpt",
    index_dir="tutorials/data",  # use only with the matching tutorial checkpoint
    device="cpu",              # choose cuda when available and needed
)
E = mct.gene_embeddings()
# Exclude unnamed vocabulary rows before biological interpretation.
E = E.loc[E.index.isin(mct.gene2idx)]
ranking = cci.tl.rank_similar_genes(E, "EGFR", top_n=21)
ranking = ranking.loc[ranking.gene != "EGFR"].head(20)
ax = cci.pl.similarity_ranking(ranking)
```

`cci.load(model_dir)` also discovers a checkpoint and JSONs; use explicit paths when multiple checkpoints or gene-ID mappings are present. Do not assume every checkpoint uses gene symbols. Check requested-gene coverage before ranking. `docs/api.md` and `tutorials/api_quickstart.ipynb` describe the facade; current implementation in `cell_cell_interactions/api/` resolves discrepancies.

## Turn embeddings into an analysis

1. Define the biological unit (sample/donor, condition, tissue, or spatial neighborhood), candidate genes/pairs, and intended comparison before computing embeddings.
2. Choose static or contextual vectors using the table above. Preserve sample IDs, gene IDs, and checkpoint provenance with every result.
3. Check finite, nonzero vectors, unique gene identifiers, gene/pair coverage, input layout, and adequate independent sample support. Missing genes are missing observations, not zero scores.
4. Apply the relevant strategy below. Keep inference and downstream statistics separate so embeddings can be reused.
5. Validate against appropriate reference interactions, expression baselines, independent samples, or spatial evidence. Report effect sizes and uncertainty with the gene universe, filtering, and comparison design.

| Goal | Strategy/reference |
|---|---|
| Find genes related to a query or gene signature | Cosine ranking; signature centroid; analyses §1 |
| Discover programs and candidate regulatory networks | Neighbor graph → Leiden → enrichment; analyses §2 |
| Rank receptor–ligand, PPI, or TF–target pairs | Candidate-pair cosine scores → matched benchmark; analyses §3 |
| Find rewiring without requiring differential expression | Per-sample neighbors → within/across-condition Jaccard; analyses §4 |
| Compare interactions across disease/tissue | Per-sample pair scores → replicate-aware contrast; analyses §5 |
| Compare whole samples | Pool contextual genes → neighbor graph/clustering; analyses §6 |
| Estimate knockout effects | Matched baseline and token-deletion forwards; advanced §1 |
| Attribute a pair score to cell types | Differentiable contextual score and integrated gradients; advanced §2 |
| Compare with spatial colocalization / cell-type interactions | Matched gene pairs, spatial weights, marker-set aggregation; advanced §3 |
| Combine MCT with protein evidence | Independent protein similarity features and held-out evaluation; advanced §4 |

## Train or fine-tune briefly

Use [training guidance](references/training.md) when requested. Choose `train_adata_rnaseq`, `train_cellarr`, or `train_spatial`; inspect that entry point's `--help`, set paths and metadata keys explicitly, and preserve its generated indices with the checkpoint. Distinguish training from scratch, initialization from pretrained weights, and resuming a Lightning run. Match architecture and preprocessing rather than copying tutorial defaults. Ordinary downstream analysis needs inference, not retraining.

## Deliver usable results

Save machine-readable rankings, pair scores, neighbor lists, gene sets, or sample embeddings alongside figures. Include checkpoint/index identities, package revision, preprocessing, sample/gene exclusions, random seeds, graph settings, and statistical design. Distinguish reproduced results from proposed extensions and unrun code. For paper reproduction, consult the coverage map in the analysis reference: some advanced analyses are described in the paper but have no complete public notebook or working high-level wrapper.
