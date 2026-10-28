# Multicellular Transformer

Official code for **Learning gene co-regulation across cells with Multicellular Transformer**.

Multicellular Transformer (MCT) learns context-aware gene representations from bags of cells.
Genes are tokens; each cell contributes its expressed subset; a sample is a bag of cells over
a shared vocabulary. Pretraining is BERT-style masked gene prediction with optional
expression reconstruction. Output: **one context-aware embedding per gene per cell per
sample**, i.e. the same gene gets different vectors depending on what surrounds it.

The package ships as a small Python API for using the pretrained model on your own AnnData,
plus training entry points and downstream analysis tools (gene sets, receptor–ligand, GRN,
condition-specific genes, sample embeddings, in-silico DepMap perturbation).

## Quickstart

```python
import cell_cell_interactions as cci
import scanpy as sc

mct = cci.load("/models/mct_v1", device="cuda")            # single call, auto-discovers files
emb = mct.gene_embeddings()                                 # zero-shot: DataFrame (genes x dims)

ranking = cci.tl.rank_similar_genes(emb, "EGFR", top_n=20)  # analysis: DataFrame
cci.pl.similarity_ranking(ranking)                          # plotting: returns Axes

adata = sc.read_h5ad("your.h5ad")
adata = cci.pp.filter_to_model_vocabulary(adata, mct.gene2idx)
sample_embs = mct.embed(adata,                              # context-aware: {sample: DataFrame}
                        cell_type_key="cell_type",
                        sample_key="donor_id")
sample_embs["donor_42"].loc["EGFR"]
```

Full walkthrough: [`tutorials/api_quickstart.ipynb`](tutorials/api_quickstart.ipynb).
Full reference (every function with a runnable example): [`docs/api.md`](docs/api.md).

## Agent skill

[`skills/use-mct/SKILL.md`](skills/use-mct/SKILL.md) provides an
[Agent Skills](https://agentskills.io/specification)-format guide for Claude Code,
Codex, and other compatible coding agents. It focuses on choosing and analyzing
gene embeddings, with paper-to-workflow mappings, advanced analysis strategies,
and brief training guidance. It also documents current API limitations.

Copy the entire `skills/use-mct/` folder into your client's skill directory to
install it, or ask an agent to read its `SKILL.md` directly. Keep the `references/`
folder alongside the entry point; repository paths in the guide refer to your
MCT checkout. Installing the skill does not install MCT or download model weights.

## Public API surface

Scanpy-style layout — one loader, three submodules:

| Symbol | Purpose |
|---|---|
| `cci.load(model_dir, device=..., ...)` | Auto-discovers `.ckpt` + `gene2idx.json` / `cell_type_input_order.json` / `tissue2idx.json` under `model_dir`. Returns an `MCTModel`. |
| `mct.gene_embeddings(genes=None)` | Zero-shot gene embeddings as a DataFrame indexed by symbol. No context. |
| `mct.embed(adata, cell_type_key, sample_key, tissue_key=None)` | **Contextual** per-sample embeddings from an AnnData. Handles pseudobulk + normalization + forward pass. |
| `mct.embed_dataloader(dataloader)` | Contextual embeddings via a `CellarrDataModule` / `AdataRnaseqDataModule` dataloader. Paper-reproduction path. |
| `mct.forward_batch(batch)` | Raw forward pass on a `BagOfCellsBatch`. Advanced use. |
| `cci.tl.rank_similar_genes`, `pairwise_cosine_similarity`, `gene_neighbors`, `regress_out` | DataFrame in, DataFrame out. |
| `cci.pl.similarity_ranking`, `similarity_heatmap`, `plot_gene_set`, `plot_similarity_across_conditions_heatmap` | Every function returns an `Axes` and never calls `plt.show()`. |
| `cci.pp.filter_to_model_vocabulary`, `harmonize_cell_types`, `log_normalize` | AnnData / DataFrame preprocessing. |
| `cell_cell_interactions.utils.sample_embeddings.api` | `collect_sample_embeddings`, `collect_sample_neighbors`. |
| `cell_cell_interactions.utils.depmap.api` | `run_gene_ko_effect`, `DEPMAP_DISEASE_MAP`. |

Advanced users can still reach in via `cci.MCTModel.module` / `.lightning_module` and the
lower-level `cell_cell_interactions.{modeling,pp,tl,pl}` modules; the API layer is a facade,
not a fork.

## Repository Layout

- `cell_cell_interactions/api/`: public API — `load`, `MCTModel`, `tl`, `pl`, `pp`.
- `cell_cell_interactions/modeling/`: PyTorch Lightning model, configuration dataclasses, losses, trainers, training entry points.
- `cell_cell_interactions/pp/`: preprocessing and data modules for AnnData, CellArr/TileDB, and spatial data.
- `cell_cell_interactions/tl/`: analysis utilities for condition-specific genes and gene embedding/GRN workflows.
- `cell_cell_interactions/pl/`: plotting utilities used by the analysis notebooks.
- `cell_cell_interactions/utils/`: scripts and library wrappers for sample embeddings and DepMap in-silico perturbation workflows.
- `tutorials/`: reproducible notebooks and small tutorial data.
- `docs/`: API reference.

## Installation

Python 3.10 or newer. A fresh conda or mamba environment is recommended because the project
depends on PyTorch, Scanpy, Lightning, and CellArr/TileDB.

```bash
conda create -n multicellular-transformer python=3.10
conda activate multicellular-transformer
pip install -e ".[tutorials,test]"
```

For development checks and pre-commit hooks:

```bash
pip install -e ".[dev,test,tutorials]"
pre-commit install
```

Weights & Biases logging is optional. Install the tracking extra only if you pass
`--wandb-api-key` to training scripts:

```bash
pip install -e ".[tracking]"
```

## Quick Verification

After installation, run lightweight checks from the repository root:

```bash
python -m compileall -q cell_cell_interactions
python -m pytest
```

The test suite includes numerical parity tests that gate every internal refactor; if a
released checkpoint stops producing the same numbers, `pytest` fails.

If you only want to inspect notebooks without training,
`python -m compileall -q cell_cell_interactions` is the fastest sanity check.

## Download Checkpoint

The pretrained checkpoint used by the tutorials is hosted on Google Drive:

<https://drive.google.com/file/d/1HVqCSiizGpFhVKqxnv6hEEOP_JWrWkSX/view>

After installing the tutorial dependencies, download it from the repository root with:

```bash
mkdir -p checkpoints
gdown --fuzzy "https://drive.google.com/file/d/1HVqCSiizGpFhVKqxnv6hEEOP_JWrWkSX/view" -O checkpoints/multicellular_transformer.ckpt
```

Then load with:

```python
import cell_cell_interactions as cci
mct = cci.load(checkpoint_path="checkpoints/multicellular_transformer.ckpt",
               index_dir="tutorials/data")
```

Or, for the recommended layout (`model_dir` with `checkpoints/` + index JSONs alongside):

```python
mct = cci.load("/models/mct_v1")
```

If `gdown` is not available, install it with `pip install gdown`, or download the file
manually from the Google Drive link and save it to the same path.

## Tutorials

Notebooks are designed to be run from the repository root so relative paths resolve
correctly. Start Jupyter with:

```bash
jupyter lab
```

New users should start with:

- `tutorials/api_quickstart.ipynb` — one-stop tour of the public API.

Paper-tracking tutorials (reproduce the figures / benchmarks):

- `tutorials/inference_using_cellarr.ipynb`: run inference with a pretrained Multicellular Transformer model on CellArr inputs.
- `tutorials/inference_and_context_aware_gene_sets.ipynb`: extract context-aware gene representations and analyze gene sets.
- `tutorials/condition_specific_genes_analysis.ipynb`: score condition-specific genes from sample-level nearest-neighbor outputs.
- `tutorials/zero_shot_gene_sets_analysis.ipynb`: zero-shot gene set analysis.
- `tutorials/zero_shot_receptor_ligand_analysis.ipynb`: zero-shot receptor-ligand and gene-gene analysis.
- `tutorials/benchmarking_ccc_rl_cellchat.ipynb`: receptor-ligand benchmarking against CellChat-style resources.

Small input resources used by tutorials live in `tutorials/data/`, including gene index
files, cell type maps, receptor-ligand pairs, PPI pairs, TF regulon pairs, and example
condition-specific gene tables. Larger model checkpoints and full paper-scale datasets are
distributed separately.

## Training Entry Points

Training scripts use `typed-argument-parser`, so command-line flags are generated from
`MulticellularTransformerArgparser` in `cell_cell_interactions/modeling/argparser.py`.

```bash
python -m cell_cell_interactions.modeling.train_adata_rnaseq --help
python -m cell_cell_interactions.modeling.train_cellarr --help
python -m cell_cell_interactions.modeling.train_spatial --help
```

Common outputs:

- `checkpoints/`: Lightning checkpoints.
- `logs/`: local logger output.
- `gene2idx.json`, `tissue2idx.json`, `cell_type_input_order.json`: metadata needed to reproduce inference with the trained checkpoint. These are part of the checkpoint contract; keep them paired.

## Reproducibility Notes

- Set `--random-seed` in scripts and record the generated metadata JSON files with each checkpoint.
- Keep the gene index mapping paired with the checkpoint that produced it; token IDs are part of the model contract.
- Run notebooks from the repository root or update paths explicitly in the first notebook cells.
- Do not commit notebook checkpoint folders (`.ipynb_checkpoints/`) or platform artifacts such as `.DS_Store`.

## Pretrained Model Zoo

The following checkpoint variants are planned for the release package. The recommended
default is the 12k gene-context, single-cell model without gene reconstruction loss.

- 12k gene context, single-cell, without gene reconstruction loss (recommended).
- 12k gene context, single-cell, with gene reconstruction loss.
- 12k gene context, pseudobulk, with gene reconstruction loss.
- 12k gene context, finetuned on lung cancer atlas, with gene reconstruction loss.

Each checkpoint ships with its matching gene index and, when applicable, cell type and
tissue mappings. Point `cci.load(model_dir=...)` at the unpacked directory.

## Citation

Citation details will be added after publication.
