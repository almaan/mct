import os

import numpy as np
import pandas as pd
import scanpy as sc
from cellarr import MatrixOptions, build_cellarrdataset
from tap import Tap

ADATA_FILEPATHS = [
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/sc-eqtl/sc_eqtl_preprocessed_all_genes_v2.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/aida/aida_preprocessed_all_genes_v2.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/hbca/hbca_preprocessed_all_genes_v2.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/hlca/hlca_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/lupus/lupus_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/tabula-sapiens/tabula_sapiens_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/immune-across-organs/immune_across_organs_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/covid-stephenson/covid_stephenson_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/glioblastoma/glioblastoma_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/hgca/hgca_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/heart-litvinukova/heart_litvinukova_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/covid-ren/covid_ren_preprocessed_all_genes.h5ad",
    "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/msk-spectrum/msk_spectrum_preprocessed_all_genes.h5ad",
]

OBS_COLS = [
    "sample",
    "tissue",
    "assay",
    "cell_type",
    "cell_type_ontology_term_id",
    "disease",
    "sex",
    "cell_type_remapped_v2",
    "study",
]


def run(
    output_dir: str,
):
    """Run the script to build the CellArrDataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # collect cell metadata
    cell_metadata = []
    for anndata_fp in ADATA_FILEPATHS:
        adata = sc.read_h5ad(anndata_fp, "r")
        adata.obs["study"] = anndata_fp
        cell_metadata.append(adata.obs[OBS_COLS].copy())

    cell_metadata = pd.concat(cell_metadata)
    cell_metadata.to_parquet(os.path.join(output_dir, "cell_metadata.parquet"))
    print(f"Cell metadata details, nr of rows {len(cell_metadata)}, columns {cell_metadata.columns}")
    print("Finished reading anndata files, building the CellArr dataset...")
    # Build CellArrDataset
    dataset = build_cellarrdataset(
        output_path=output_dir.rstrip("/"),
        files=ADATA_FILEPATHS,
        matrix_options=MatrixOptions(dtype=np.float32),
        num_threads=os.cpu_count(),
        cell_metadata=cell_metadata,
    )
    print(dataset)


class BuildCellArrDatasetArgParser(Tap):
    """Arguments for the preprocess_dataset script."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    output_dir: str = (
        "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/hbca-aida-eqtl-cellarr-final-v3/"
    )


if __name__ == "__main__":
    args = BuildCellArrDatasetArgParser().parse_args()
    run(args.output_dir)
