import pandas as pd
import scanpy as sc
from tap import Tap


def preprocess_cross_tissue_immune(
    adata_file_path: str,
    output_file_path: str,
    n_highly_variable_genes: int,
    cell_type_key: str,
    min_n_cells_per_cell_type: int,
    sample_cols: list[str] | None,  # list of columns to use for sample identification
    min_n_cells_per_sample: int,
    min_n_cells_per_gene: int,  # for filtering
    min_n_counts_per_cell: int,  # for filtering
    ccc_genes: list[str] = None,  # list of genes involved in cell-cell communication to include
):
    """Preprocess the cross-tissue immune dataset Domínguez Conde et al. (2022) Science"""
    adata = sc.read_h5ad(adata_file_path)
    # remove it as we don't need it
    del adata.raw

    # do the filtering
    sc.pp.filter_cells(adata, min_genes=min_n_counts_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_n_cells_per_gene)

    # remove rare cell types
    cell_type_counts = adata.obs[cell_type_key].value_counts()
    cell_types_include = cell_type_counts[cell_type_counts >= min_n_cells_per_cell_type].index
    adata = adata[adata.obs[cell_type_key].isin(cell_types_include), :]

    # create sample id col
    if "sample_id" not in adata.obs.columns:
        adata.obs["sample_id"] = adata.obs[sample_cols].apply(lambda x: "_".join(x), axis=1)

    # remove samples with too few cells
    sample_counts = adata.obs["sample_id"].value_counts()
    samples_include = sample_counts[sample_counts >= min_n_cells_per_sample].index
    adata = adata[adata.obs["sample_id"].isin(samples_include), :]

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_highly_variable_genes,
        subset=False,
        flavor="seurat_v3",
        batch_key="sample_id",
    )

    if ccc_genes is not None:
        adata.var["ccc_gene"] = adata.var.gene_symbols.isin(ccc_genes)
        adata.var["select_gene"] = adata.var.highly_variable | adata.var.ccc_gene
        adata = adata[:, adata.var["select_gene"]]
    else:
        adata = adata[:, adata.var.highly_variable]

    # save the data
    adata.write_h5ad(output_file_path)


class PreprocessDatasetArgParser(Tap):
    """Arguments for the preprocess_dataset script."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    adata_file_path: str = "/Users/wiatrakm/Downloads/breast_atlas.h5ad"  # path to the input AnnData file
    output_file_path: str = (
        "/Users/wiatrakm/Downloads/breast_atlas_preprocessed_2k_hvg.h5ad"  # path to the output AnnData file
    )
    n_highly_variable_genes: int = 2000  # number of highly variable genes to select
    cell_type_key: str = "cell_type"  #  "Majority_voting_CellTypist"  # key for the cell type labels
    min_n_cells_per_cell_type: int = 2000  # minimum number of cells per cell type
    sample_cols: list[str] | None = (
        None  # ["donor_id", "tissue", "assay"]  # list of columns to use for sample identification
    )
    min_n_cells_per_sample: int = 1024  # minimum number of cells per sample
    min_n_cells_per_gene: int = 100  # minimum number of cells per gene
    min_n_counts_per_cell: int = 5  # minimum number of counts per cell
    ccc_genes_file_path: str = None  # (
    #     "/Users/wiatrakm/Downloads/liana_consensus_lig_rec.csv"  # path to the file with CCC genes
    # )


def main(args: PreprocessDatasetArgParser):
    """Main function for the preprocess_dataset script."""
    ccc_genes = None
    if args.ccc_genes_file_path is not None:
        ccc_df = pd.read_csv(args.ccc_genes_file_path)
        ccc_genes = list(set(ccc_df.ligand.tolist() + ccc_df.receptor.tolist()))

    preprocess_cross_tissue_immune(
        adata_file_path=args.adata_file_path,
        output_file_path=args.output_file_path,
        n_highly_variable_genes=args.n_highly_variable_genes,
        cell_type_key=args.cell_type_key,
        min_n_cells_per_cell_type=args.min_n_cells_per_cell_type,
        sample_cols=args.sample_cols,
        min_n_cells_per_sample=args.min_n_cells_per_sample,
        min_n_cells_per_gene=args.min_n_cells_per_gene,
        min_n_counts_per_cell=args.min_n_counts_per_cell,
        ccc_genes=ccc_genes,
    )


if __name__ == "__main__":
    args = PreprocessDatasetArgParser().parse_args()
    main(args)
