# Multicellular Transformer

Official code for **Learning gene co-regulation across cells with Multicellular Transformer**.

Multicellular Transformer learns context-aware gene representations from bags of cells. The package includes data modules for AnnData, CellArr/TileDB, and spatial-neighborhood inputs; model training code; downstream utilities for gene sets, receptor-ligand analyses, gene regulatory networks, condition-specific genes, and sample embeddings; and Jupyter tutorials with small example resources.

## Repository Layout

- `cell_cell_interactions/modeling/`: PyTorch Lightning model, configuration dataclasses, losses, trainers, and training entry points.
- `cell_cell_interactions/pp/`: preprocessing and data modules for AnnData, CellArr/TileDB, and spatial data.
- `cell_cell_interactions/tl/`: analysis utilities for condition-specific genes and gene embedding/GRN workflows.
- `cell_cell_interactions/pl/`: plotting utilities used by the analysis notebooks.
- `cell_cell_interactions/utils/`: scripts for sample embeddings and DepMap in silico perturbation workflows.
- `tutorials/`: reproducible notebooks and small tutorial data.
- `tutorials/old/`: archived benchmark notebooks retained for reference.

## Installation

Use Python 3.10 or newer. A fresh conda or mamba environment is recommended because the project depends on PyTorch, Scanpy, Lightning, and CellArr/TileDB.

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

Weights & Biases logging is optional. Install the tracking extra only if you pass `--wandb-api-key` to training scripts:

```bash
pip install -e ".[tracking]"
```

## Quick Verification

After installation, run lightweight checks from the repository root:

```bash
python -m compileall -q cell_cell_interactions
python -m pytest
```

If you only need to inspect notebooks and not train models, `python -m compileall -q cell_cell_interactions` is the fastest sanity check.

## Download Checkpoint

The pretrained checkpoint used by the tutorials is hosted on Google Drive:

<https://drive.google.com/file/d/1HVqCSiizGpFhVKqxnv6hEEOP_JWrWkSX/view>

After installing the tutorial dependencies, download it from the repository root with:

```bash
mkdir -p checkpoints
gdown --fuzzy "https://drive.google.com/file/d/1HVqCSiizGpFhVKqxnv6hEEOP_JWrWkSX/view" -O checkpoints/multicellular_transformer.ckpt
```

Use this local path in notebooks and scripts whenever a checkpoint path is requested:

```text
checkpoints/multicellular_transformer.ckpt
```

If `gdown` is not available, install it with `pip install gdown`, or download the file manually from the Google Drive link and save it to the same path.

## Tutorials

The notebooks are designed to be run from the repository root so relative paths resolve correctly. Start Jupyter with:

```bash
jupyter lab
```

Current tutorials:

- `tutorials/inference_using_cellarr.ipynb`: run inference with a pretrained Multicellular Transformer model on CellArr inputs.
- `tutorials/inference_and_context_aware_gene_sets.ipynb`: extract context-aware gene representations and analyze gene sets.
- `tutorials/condition_specific_genes_analysis.ipynb`: score condition-specific genes from sample-level nearest-neighbor outputs.
- `tutorials/zero_shot_gene_sets_analysis.ipynb`: zero-shot gene set analysis.
- `tutorials/zero_shot_receptor_ligand_analysis.ipynb`: zero-shot receptor-ligand and gene-gene analysis.
- `tutorials/benchmarking_ccc_rl_cellchat.ipynb`: receptor-ligand benchmarking against CellChat-style resources.

Small input resources used by tutorials live in `tutorials/data/`, including gene index files, cell type maps, receptor-ligand pairs, PPI pairs, TF regulon pairs, and example condition-specific gene tables. Larger model checkpoints and full paper-scale datasets are expected to be distributed separately from the source repository; place them at the paths configured in the relevant notebook or script.

## Training Entry Points

Training scripts use `typed-argument-parser`, so command-line flags are generated from `MulticellularTransformerArgparser` in `cell_cell_interactions/modeling/argparser.py`.

```bash
python -m cell_cell_interactions.modeling.train_adata_rnaseq --help
python -m cell_cell_interactions.modeling.train_cellarr --help
python -m cell_cell_interactions.modeling.train_spatial --help
```

Common outputs include:

- `checkpoints/`: Lightning checkpoints.
- `logs/`: local logger output.
- `gene2idx.json`, `tissue2idx.json`, and `cell_type_input_order.json`: metadata needed to reproduce inference with the trained checkpoint.

## Reproducibility Notes

- Set `--random-seed` in scripts and record the generated metadata JSON files with each checkpoint.
- Keep the gene index mapping paired with the checkpoint that produced it; token IDs are part of the model contract.
- Run notebooks from the repository root or update paths explicitly in the first notebook cells.
- Do not commit notebook checkpoint folders (`.ipynb_checkpoints/`) or platform artifacts such as `.DS_Store`.

## Pretrained Model Zoo

The following checkpoint variants are planned for the release package. The recommended default is the 12k gene-context, single-cell model without gene reconstruction loss.

- 12k gene context, single-cell, without gene reconstruction loss (recommended).
- 12k gene context, single-cell, with gene reconstruction loss.
- 12k gene context, pseudobulk, with gene reconstruction loss.
- 12k gene context, finetuned on lung cancer atlas, with gene reconstruction loss.

Each checkpoint should be distributed with its matching gene index and, when applicable, cell type and tissue mappings.

## Citation

Citation details will be added after publication.
