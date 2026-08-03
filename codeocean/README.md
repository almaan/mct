# CodeOcean Capsule Notes

This directory contains the files needed to run the reviewer-facing CodeOcean capsule for the Multicellular Transformer repository.

## Capsule Scope

The capsule is configured for peer-review inspection and lightweight reproduction, not full paper-scale retraining. The CodeOcean run:

- checks that the Python source compiles;
- runs the repository test suite;
- reads the bundled tutorial resources;
- loads the pretrained Multicellular Transformer checkpoint;
- generates receptor-ligand benchmark summaries, top-gene tables, condition-specific gene summaries, and figures under `/results`.

Full-scale CellArr corpora, large `.h5ad` datasets, and third-party baseline model files used by exploratory notebooks are intentionally not part of the default run unless they are separately cleared, uploaded, and documented.

## CodeOcean Layout

Upload the source bundle to `/code` and the data bundle to `/data`.

Expected data layout:

```text
/data/
  checkpoints/
    multicellular_transformer.ckpt
  tutorials/
    data/
      gene_symbol2idx.json
      receptor_ligand_with_negatives.csv
      receptors_and_ligands.csv
      tumor_microenvironment_hvg_genes.csv
      immune_across_organs_condition_specific_genes.parquet
      lupus_disease_condition_specific_genes.parquet
      covid_stephenson_disease_condition_specific_genes.parquet
      immune_across_organs_neighbors_sample.parquet
      ...
```

The small tutorial resources currently tracked in `tutorials/data/` should be uploaded under `/data/tutorials/data`. The pretrained checkpoint should be uploaded under `/data/checkpoints/multicellular_transformer.ckpt`; it is not downloaded during runtime.

## Environment Setup

Use a Python 3.10-compatible CodeOcean base environment. Install dependencies during the CodeOcean environment build, not inside `run.sh`, using either the package manager UI or a postInstall command such as:

```bash
pip install -r /code/requirements-codeocean.txt
```

The run script sets `PYTHONPATH=/code`, so the local package source is imported directly from the uploaded capsule.

## Running

Set `/code/run.sh` as the CodeOcean file to run. It writes outputs to:

- `/results/logs`: run configuration, test log, metrics JSON, and full run summary;
- `/results/tables`: CSV tables for scored receptor-ligand pairs, top genes, data inventory, and condition-specific summaries;
- `/results/figures`: static PNG figures generated with a headless Matplotlib backend.

For local dry-runs without the checkpoint, use:

```bash
CODEOCEAN_DATA_DIR="$PWD" \
CODEOCEAN_RESULTS_DIR="/tmp/mct-codeocean-results" \
MCT_STRICT_INPUTS=0 \
bash run.sh
```

In CodeOcean, leave `MCT_STRICT_INPUTS` unset so the run fails clearly if the checkpoint is missing.

## Peer-Review Submission

Before pressing "Submit for Publication", confirm that the capsule metadata lists the target journal and that CodeOcean has the corresponding editor information if the journal workflow does not provide it automatically. The peer-review copy is private and frozen; update it by withdrawing, editing, resubmitting, and notifying CodeOcean support.

