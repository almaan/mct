# Brief training and fine-tuning guide

## Choose the data path

From the repository root, inspect the appropriate CLI:

```bash
python -m cell_cell_interactions.modeling.train_adata_rnaseq --help
python -m cell_cell_interactions.modeling.train_cellarr --help
python -m cell_cell_interactions.modeling.train_spatial --help
```

Use AnnData for a cohort `.h5ad`, CellArr/TileDB for large corpora, and spatial inputs for neighborhood bags. Flags are generated from `modeling/argparser.py` with underscores converted to dashes. Defaults include developer paths and exploratory hyperparameters: set input/output paths, gene index, sample/cell-type keys, bag size, masking, architecture, learning rate, and run length explicitly. Read the selected entry point because not all parser fields apply to every path.

For example, the AnnData invocation has this shape (paths are placeholders):

```bash
python -m cell_cell_interactions.modeling.train_adata_rnaseq \
  --adata-file-path /path/cohort_preprocessed.h5ad \
  --gene2idx-file-path /path/gene2idx.json \
  --sample-key sample --cell-type-key cell_type \
  --output-dir /path/runs --experiment-name cohort_mct \
  --n-cells-bag 512 --min-n-cells-per-sample 512 \
  --batch-size 1 --random-seed 32
```

This is an invocation template, not the paper's complete hyperparameter recipe. Set compatible architecture, training precision/device, normalization, and optimization flags before launching. The AnnData entry point currently hardcodes a `tissue` metadata column; inspect required disease/feature columns in its datamodule as well.

## Distinguish initialization from resumption

- **From scratch:** build vocabulary and input layout for the training corpus, set a valid gene embedding dimension/head count and sequence limit, and retain generated maps. Learn masked gene identity prediction, with expression reconstruction and tissue losses only when configured.
- **Fine-tuning:** supply `--pretrained-ckpt-path` with matching gene IDs and transferable architecture. The AnnData entry point removes tissue, expression encoder, and reconstruction decoder weights before a non-strict load, allowing new cell-slot layouts. Inspect actual missing/unexpected keys rather than assuming all weights transfer or are frozen. A frozen pretrained block is not implied by “fine-tuning.”
- **Resuming:** use `--ckpt-path` for Lightning training-state continuation with compatible model/data configuration. This is different from initializing a new model from selected pretrained weights.

Split at sample/donor level, with study-held-out evaluation where needed; spatial neighborhoods require region/section separation. Record the actual split and seed. Paper reproduction also requires matching corpus, cell sampling, filtering, masking, normalization, loss weights, architecture, and optimization schedule; changing context size requires compatible input dimensions and a matched training run, not an inference-time toggle.

## Validate and preserve the result

Check one batch for finite inputs, valid IDs, expected masking and input dimensions; then run a short training/validation check before scaling. Verify masking is enabled for masked training objectives and disabled for ordinary downstream inference. Confirm that the reconstruction loss was trained before interpreting reconstructed expression for KO analysis.

Save resolved configuration, split IDs, seeds, checkpoint, gene map, cell-type slot layout, tissue map, and label harmonization map together. Some entry points save only a subset of these automatically; explicitly preserve the rest from the datamodule. Reload with `cci.load` and perform one representative inference to verify the artifact bundle. Optional W&B logging uses the `tracking` extra; local training does not require it.
