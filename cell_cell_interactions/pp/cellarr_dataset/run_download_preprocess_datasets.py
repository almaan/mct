"""Download and preprocess the Multicellular Transformer pretraining corpus.

The corpus-level preprocessing follows Appendix A.2/A.4.1 of the MCT paper:

* use raw molecule/read counts, never the normalized visualization matrix;
* remove genes with fewer than five total counts;
* remove cells expressing fewer than 100 genes;
* harmonize CELLxGENE cell-type labels with the released training mapping;
* retain the 54 harmonized types in the released checkpoint input order; and
* remove samples containing fewer than 1,024 retained cells.

Counts remain raw in ``layers["counts"]`` in the processed H5AD files.
``CellarrDataModule`` performs the model's normalize-total (10,000) and log1p
transformations while loading.
Downloaded CELLxGENE source files are held only in a temporary directory and
removed one by one after their processed H5AD has been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    """A CELLxGENE dataset in the MCT pretraining corpus."""

    dataset_id: str
    title: str
    url: str
    publication_url: str
    sample_column_candidates: tuple[str, ...]


# The URLs below are the CELLxGENE H5AD exports selected for this rebuild.
DATASETS = {
    "aida": DatasetSpec(
        dataset_id="aida",
        title="AIDA Phase 1 Data Freeze v1",
        url="https://datasets.cellxgene.cziscience.com/c51e913d-832a-44a1-8f43-d20c51a1799f.h5ad",
        publication_url="https://doi.org/10.1016/j.cell.2025.02.017",
        sample_column_candidates=("sample_id", "sample", "sampleID"),
    ),
    "covid_ren": DatasetSpec(
        dataset_id="covid_ren",
        title="COVID-19 immune features revealed by a large-scale single-cell transcriptome atlas",
        url="https://datasets.cellxgene.cziscience.com/481a2da9-2e61-424a-87e4-171e940006a4.h5ad",
        publication_url="https://doi.org/10.1016/j.cell.2021.01.053",
        sample_column_candidates=("sampleID", "sample_id", "sample"),
    ),
    "covid_stephenson": DatasetSpec(
        dataset_id="covid_stephenson",
        title="Single-cell multi-omics analysis of the immune response in COVID-19",
        url="https://datasets.cellxgene.cziscience.com/fe2e847c-1602-4f1b-86a4-112e4dc7a8e3.h5ad",
        publication_url="https://doi.org/10.1038/s41591-021-01329-2",
        sample_column_candidates=("sample_id", "sample", "sampleID"),
    ),
    "glioblastoma": DatasetSpec(
        dataset_id="glioblastoma",
        title="Harmonized single-cell landscape, intercellular crosstalk and tumor architecture of glioblastoma",
        url="https://datasets.cellxgene.cziscience.com/230c8701-7291-4dd5-bf36-688490c681ee.h5ad",
        publication_url="https://doi.org/10.1101/2022.08.27.505439",
        sample_column_candidates=("sample_id", "sample", "sampleID"),
    ),
    "hbca": DatasetSpec(
        dataset_id="hbca",
        title="Integrated Human Breast Cell Atlas",
        url="https://datasets.cellxgene.cziscience.com/28d9d02e-e78b-4f9d-a22b-f9d3b28a0792.h5ad",
        publication_url="https://doi.org/10.1038/s41588-024-01688-9",
        sample_column_candidates=("sample_id", "sample", "sampleID"),
    ),
    "heart": DatasetSpec(
        dataset_id="heart",
        title="Cells of the adult human heart",
        url="https://datasets.cellxgene.cziscience.com/49312f8c-a75a-4672-9c8a-9078acd35bdb.h5ad",
        publication_url="https://doi.org/10.1038/s41586-020-2797-4",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
    "gut": DatasetSpec(
        dataset_id="gut",
        title="Cells of the human intestinal tract mapped across space and time",
        url="https://datasets.cellxgene.cziscience.com/c54fbb37-2a1f-451a-815b-3b1fc034b62e.h5ad",
        publication_url="https://doi.org/10.1038/s41586-021-03852-1",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
    "hlca": DatasetSpec(
        dataset_id="hlca",
        title="The integrated Human Lung Cell Atlas",
        url="https://datasets.cellxgene.cziscience.com/5f863718-02a6-46cb-83e7-52be988b915b.h5ad",
        publication_url="https://doi.org/10.1038/s41591-023-02327-2",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
    "developing_immune": DatasetSpec(
        dataset_id="developing_immune",
        title="Mapping the developing human immune system across organs",
        url="https://datasets.cellxgene.cziscience.com/df38503b-8dba-4800-957d-6613747924bb.h5ad",
        publication_url="https://doi.org/10.1126/science.abo0510",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
    "lupus": DatasetSpec(
        dataset_id="lupus",
        title="Single-cell RNA-seq reveals the cell-type-specific molecular and genetic associations to lupus",
        url="https://datasets.cellxgene.cziscience.com/c55dc602-d168-4d15-acc1-5de4f2f5d551.h5ad",
        publication_url="https://doi.org/10.1126/science.abf1970",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
    "msk_spectrum": DatasetSpec(
        dataset_id="msk_spectrum",
        title="MSK SPECTRUM - Ovarian cancer mutational processes drive site-specific immune evasion",
        url="https://datasets.cellxgene.cziscience.com/f8f7f2b0-b8a5-4087-8048-8d3f5b6a49dd.h5ad",
        publication_url="https://doi.org/10.1038/s41586-022-05496-1",
        sample_column_candidates=("sample_id", "sample", "sampleID"),
    ),
    "sc_eqtl": DatasetSpec(
        dataset_id="sc_eqtl",
        title="Single-cell eQTL mapping identifies cell type specific genetic control of autoimmune disease",
        url="https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad",
        publication_url="https://doi.org/10.1126/science.abf3041",
        sample_column_candidates=("sample", "sample_id", "sampleID", "donor_id"),
    ),
    "tabula_sapiens": DatasetSpec(
        dataset_id="tabula_sapiens",
        title="Tabula Sapiens - All Cells",
        url="https://datasets.cellxgene.cziscience.com/1c88f927-bcbb-4bb1-9881-281842945a2d.h5ad",
        publication_url="https://doi.org/10.1126/science.abl4896",
        sample_column_candidates=("sample", "sample_id", "sampleID"),
    ),
}

# Preserve the file order in the original CellArr builder. Besides making the
# union deterministic, this is the best available reconstruction of the tissue
# category encounter order because a released tissue2idx artifact is absent.
DEFAULT_DATASET_ORDER = (
    "sc_eqtl",
    "aida",
    "hbca",
    "hlca",
    "lupus",
    "tabula_sapiens",
    "developing_immune",
    "covid_stephenson",
    "glioblastoma",
    "gut",
    "heart",
    "covid_ren",
    "msk_spectrum",
)

MIN_GENE_COUNTS = 5
MIN_GENES_PER_CELL = 100
MIN_CELLS_PER_SAMPLE = 1024
DEFAULT_WRITE_CHUNK_SIZE = 2048
REQUIRED_CELLXGENE_OBS_COLUMNS = (
    "assay",
    "cell_type",
    "cell_type_ontology_term_id",
    "disease",
    "donor_id",
    "sex",
    "tissue",
)
DONOR_COLUMN_CANDIDATES = ("donor_id", "donor")
ASSAY_COLUMN_CANDIDATES = ("assay", "assay_ontology_term_id")

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CELL_TYPE_MAP = PACKAGE_ROOT / "tutorials" / "data" / "cell_type_map.json"
DEFAULT_CELL_TYPE_INPUT_ORDER = PACKAGE_ROOT / "tutorials" / "data" / "cell_type_input_order.json"
DEFAULT_GENE2IDX = PACKAGE_ROOT / "tutorials" / "data" / "ensemble2idx.json"


def format_duration(seconds: float) -> str:
    """Format a duration for progress messages."""
    return str(timedelta(seconds=round(seconds)))


def report_stage(dataset_id: str, stage: str, started_at: float) -> None:
    """Print the elapsed time for a completed preprocessing stage."""
    print(
        f"[{dataset_id}] {stage} completed in {format_duration(time.perf_counter() - started_at)}",
        flush=True,
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """Load a JSON object."""
    with path.open() as handle:
        return json.load(handle)


def matrix_values(matrix) -> np.ndarray:
    """Return stored values without densifying a sparse matrix."""
    if sparse.issparse(matrix):
        return matrix.data
    return np.asarray(matrix).ravel()


def materialize_matrix(matrix):
    """Load one backed matrix into memory without loading other H5AD matrices."""
    if sparse.issparse(matrix):
        return matrix.copy()
    if hasattr(matrix, "to_memory"):
        return matrix.to_memory()
    return np.asarray(matrix)


def assert_raw_counts(matrix, source: str) -> None:
    """Fail if a matrix is not a finite, nonnegative, integer-like count matrix."""
    values = matrix_values(matrix)
    block_size = 10_000_000
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        assert np.isfinite(block).all(), f"{source} contains non-finite values"
        assert (block >= 0).all(), f"{source} contains negative values"
        assert np.allclose(block, np.rint(block)), (
            f"{source} is not integer-like. It is probably normalized; raw counts cannot be "
            "recovered exactly from this matrix. Supply an H5AD containing raw.X or a counts layer."
        )


def select_raw_counts(adata: ad.AnnData) -> tuple[ad.AnnData, str]:
    """Select the canonical raw-count matrix and retain matching obs/var metadata."""
    if adata.raw is not None:
        counts = materialize_matrix(adata.raw.X)
        assert_raw_counts(counts, "raw.X")
        return ad.AnnData(X=counts, obs=adata.obs.copy(), var=adata.raw.var.copy()), "raw.X"

    for layer_name in ("counts", "raw_counts"):
        if layer_name in adata.layers:
            counts = materialize_matrix(adata.layers[layer_name])
            assert_raw_counts(counts, f"layers[{layer_name!r}]")
            count_adata = ad.AnnData(
                X=counts,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            return count_adata, f"layers[{layer_name!r}]"

    counts = materialize_matrix(adata.X)
    assert_raw_counts(counts, "X")
    return ad.AnnData(X=counts, obs=adata.obs.copy(), var=adata.var.copy()), "X"


def calculate_expressed_genes_per_cell_chunked(
    matrix,
    retained_gene_indices: np.ndarray,
    dataset_id: str,
    chunk_size: int,
) -> np.ndarray:
    """Count expressed retained genes using only bounded row-slice copies."""
    expressed_genes = np.zeros(matrix.shape[0], dtype=np.int32)
    with tqdm(total=matrix.shape[0], desc=f"{dataset_id} cell QC", unit="cells", dynamic_ncols=True) as progress:
        for start in range(0, matrix.shape[0], chunk_size):
            end = min(start + chunk_size, matrix.shape[0])
            chunk = matrix[start:end, retained_gene_indices]
            expressed_genes[start:end] = np.asarray((chunk > 0).sum(axis=1)).ravel()
            progress.update(end - start)
    return expressed_genes


def calculate_expressed_genes_per_cell_without_full_copy(
    matrix: sparse.csr_matrix,
    retained_genes: np.ndarray,
) -> np.ndarray:
    """Count retained expressed genes from CSR structure without row chunks."""
    matrix.eliminate_zeros()
    expressed_genes = np.diff(matrix.indptr).astype(np.int32, copy=False)
    removed_gene_indices = np.flatnonzero(~retained_genes)
    if len(removed_gene_indices) == 0:
        return expressed_genes

    # Low-count genes contribute at most four stored values each, and nonhuman
    # features are normally a very small fraction of CELLxGENE feature sets.
    removed_genes = matrix[:, removed_gene_indices]
    expressed_genes -= np.diff(removed_genes.indptr).astype(np.int32, copy=False)
    return expressed_genes


def write_filtered_h5ad(
    matrix,
    retained_cell_indices: np.ndarray,
    retained_gene_indices: np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    output_path: Path,
    dataset_id: str,
    chunk_size: int,
) -> None:
    """Stream a filtered CSR matrix into ``layers['counts']`` without a second copy."""
    metadata = ad.AnnData(X=None, obs=obs, var=var, uns=uns)
    metadata.write_h5ad(output_path, compression="gzip")

    with h5py.File(output_path, "a") as handle:
        matrix_group = handle["layers"].create_group("counts")
        matrix_group.attrs["encoding-type"] = "csr_matrix"
        matrix_group.attrs["encoding-version"] = "0.1.0"
        matrix_group.attrs["shape"] = np.asarray((len(obs), len(var)), dtype=np.int64)
        data = matrix_group.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            chunks=(1_000_000,),
            dtype=np.float32,
            compression="gzip",
        )
        indices = matrix_group.create_dataset(
            "indices",
            shape=(0,),
            maxshape=(None,),
            chunks=(1_000_000,),
            dtype=np.int32,
            compression="gzip",
        )
        indptr = matrix_group.create_dataset(
            "indptr",
            shape=(len(obs) + 1,),
            dtype=np.int64,
            compression="gzip",
        )

        output_row = 0
        output_nnz = 0
        with tqdm(total=len(obs), desc=f"{dataset_id} H5AD write", unit="cells", dynamic_ncols=True) as progress:
            for start in range(0, len(retained_cell_indices), chunk_size):
                cell_indices = retained_cell_indices[start : start + chunk_size]
                chunk = matrix[cell_indices, :]
                if not sparse.issparse(chunk):
                    chunk = sparse.csr_matrix(chunk)
                chunk = chunk.tocsr()[:, retained_gene_indices].astype(np.float32, copy=False)

                next_nnz = output_nnz + chunk.nnz
                data.resize((next_nnz,))
                indices.resize((next_nnz,))
                data[output_nnz:next_nnz] = chunk.data
                indices[output_nnz:next_nnz] = chunk.indices
                next_row = output_row + chunk.shape[0]
                indptr[output_row : next_row + 1] = chunk.indptr.astype(np.int64) + output_nnz

                output_row = next_row
                output_nnz = next_nnz
                progress.update(chunk.shape[0])

        assert output_row == len(obs), f"{dataset_id}: wrote {output_row:,} of {len(obs):,} cells"


def resolve_sample_column(obs: pd.DataFrame, spec: DatasetSpec) -> str:
    """Resolve a sample identifier, falling back to donor plus assay."""
    for column in spec.sample_column_candidates:
        if column in obs.columns:
            assert obs[column].notna().all(), f"{spec.dataset_id}: {column!r} contains missing sample identifiers"
            values = obs[column].astype(str)
            assert (~values.isin({"", "nan", "<NA>"})).all(), (
                f"{spec.dataset_id}: {column!r} contains empty sample identifiers"
            )
            return column

    donor_column = next((column for column in DONOR_COLUMN_CANDIDATES if column in obs.columns), None)
    assert donor_column is not None, (
        f"{spec.dataset_id}: none of the expected sample columns {spec.sample_column_candidates} or donor "
        f"columns {DONOR_COLUMN_CANDIDATES} are present. Available obs columns: {obs.columns.tolist()}"
    )
    assert obs[donor_column].notna().all(), f"{spec.dataset_id}: {donor_column!r} contains missing identifiers"
    donor_values = obs[donor_column].astype(str)
    assert (~donor_values.isin({"", "nan", "<NA>"})).all(), (
        f"{spec.dataset_id}: {donor_column!r} contains empty identifiers"
    )

    assay_column = next((column for column in ASSAY_COLUMN_CANDIDATES if column in obs.columns), None)
    if assay_column is None:
        obs["sample"] = donor_values
        components = donor_column
    else:
        assert obs[assay_column].notna().all(), f"{spec.dataset_id}: {assay_column!r} contains missing values"
        assay_values = obs[assay_column].astype(str)
        assert (~assay_values.isin({"", "nan", "<NA>"})).all(), (
            f"{spec.dataset_id}: {assay_column!r} contains empty values"
        )
        obs["sample"] = donor_values + "__" + assay_values
        components = f"{donor_column} + {assay_column}"

    print(
        f"[{spec.dataset_id}] No author sample column found; generated 'sample' from {components}",
        flush=True,
    )
    return "sample"


def assert_cellxgene_schema(adata: ad.AnnData, spec: DatasetSpec) -> None:
    """Validate the CELLxGENE fields used by the unified corpus."""
    missing = sorted(set(REQUIRED_CELLXGENE_OBS_COLUMNS) - set(adata.obs.columns))
    assert not missing, f"{spec.dataset_id}: missing required CELLxGENE obs columns: {missing}"
    assert not adata.obs[list(REQUIRED_CELLXGENE_OBS_COLUMNS)].isna().any().any(), (
        f"{spec.dataset_id}: required CELLxGENE metadata contains nulls"
    )
    assert adata.obs_names.is_unique, f"{spec.dataset_id}: obs names are not unique"
    assert adata.var_names.is_unique, f"{spec.dataset_id}: gene identifiers are not unique"
    assert any(str(gene).startswith("ENSG") for gene in adata.var_names), (
        f"{spec.dataset_id}: no human Ensembl gene identifiers were found"
    )
    if "organism_ontology_term_id" in adata.obs:
        organisms = set(adata.obs["organism_ontology_term_id"].astype(str))
        assert organisms == {"NCBITaxon:9606"}, f"{spec.dataset_id}: non-human organism terms: {organisms}"


def harmonize_cell_types(
    adata: ad.AnnData,
    spec: DatasetSpec,
    cell_type_map: dict[str, str],
    checkpoint_cell_types: set[str],
) -> tuple[int, list[str], list[dict]]:
    """Map source labels and remove labels not represented by the checkpoint."""
    assert adata.obs["cell_type"].notna().all(), f"{spec.dataset_id}: cell_type contains missing labels"
    source_labels = adata.obs["cell_type"].astype(str)
    unmapped = sorted(set(source_labels) - set(cell_type_map))
    source_counts = source_labels.value_counts()
    unmapped_stats = [
        {
            "cell_type": cell_type,
            "absolute_count": int(source_counts[cell_type]),
            "relative_fraction": float(source_counts[cell_type] / len(source_labels)),
            "relative_percent": float(100 * source_counts[cell_type] / len(source_labels)),
        }
        for cell_type in unmapped
    ]
    if unmapped_stats:
        print(
            f"[{spec.dataset_id}] WARNING: {len(unmapped_stats)} cell types are absent from the released training "
            f"mapping and will be removed ({len(source_labels):,} cells before harmonization):",
            flush=True,
        )
        for stats in sorted(unmapped_stats, key=lambda item: (-item["absolute_count"], item["cell_type"])):
            print(
                f"[{spec.dataset_id}]   {stats['cell_type']}: {stats['absolute_count']:,} cells "
                f"({stats['relative_percent']:.4f}%)",
                flush=True,
            )

    adata.obs["cell_type_original"] = source_labels
    mapped_labels = source_labels.map(cell_type_map)
    non_checkpoint_types = sorted(set(mapped_labels.dropna()) - checkpoint_cell_types)
    keep = mapped_labels.isin(checkpoint_cell_types).to_numpy()
    n_removed = int((~keep).sum())
    assert keep.any(), (
        f"{spec.dataset_id}: all {len(source_labels):,} cells would be removed because none map to a cell type "
        "represented by the checkpoint"
    )
    adata.obs["cell_type"] = mapped_labels
    adata._inplace_subset_obs(keep)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return n_removed, non_checkpoint_types, unmapped_stats


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    spec: DatasetSpec,
    cell_type_map: dict[str, str],
    checkpoint_cell_types: set[str],
    chunk_size: int | None = None,
) -> dict:
    """Preprocess one dataset while retaining only one full expression matrix."""
    assert chunk_size is None or chunk_size > 0, "chunk_size must be positive"
    preprocessing_started = time.perf_counter()
    stage_started = time.perf_counter()
    print(f"[{spec.dataset_id}] Loading and validating raw counts", flush=True)
    source_adata = sc.read_h5ad(input_path, backed="r")
    assert_cellxgene_schema(source_adata, spec)
    input_cells, input_genes = source_adata.shape
    source_layers = list(source_adata.layers.keys())
    had_raw = source_adata.raw is not None

    adata, count_source = select_raw_counts(source_adata)
    source_adata.file.close()
    del source_adata
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    report_stage(spec.dataset_id, "Raw-count loading and validation", stage_started)

    stage_started = time.perf_counter()
    print(f"[{spec.dataset_id}] Calculating gene and cell filters without copying the full matrix", flush=True)
    # CELLxGENE may retain viral or ERCC features in otherwise human datasets.
    # The released checkpoint vocabulary contains human ENSG features only.
    human_genes = np.fromiter(
        (str(gene).startswith("ENSG") for gene in adata.var_names),
        dtype=bool,
        count=adata.n_vars,
    )
    non_human_features_removed = int((~human_genes).sum())
    assert human_genes.any(), f"{spec.dataset_id}: no human genes remain"

    gene_totals = np.asarray(adata.X.sum(axis=0, dtype=np.float64)).ravel()
    retained_genes = human_genes & (gene_totals >= MIN_GENE_COUNTS)
    retained_gene_indices = np.flatnonzero(retained_genes)
    assert len(retained_gene_indices) > 0, f"{spec.dataset_id}: no genes remain after raw-count filtering"
    filtered_var = adata.var.iloc[retained_gene_indices].copy()
    filtered_var["n_counts"] = gene_totals[retained_gene_indices]
    cells_after_gene_filter = adata.n_obs
    genes_after_count_filter = len(retained_gene_indices)

    if chunk_size is None:
        print(f"[{spec.dataset_id}] Calculating cell filter directly from CSR structure", flush=True)
        expressed_genes = calculate_expressed_genes_per_cell_without_full_copy(adata.X, retained_genes)
        cell_filter_strategy = "CSR structure; no full matrix copy and no row chunks"
    else:
        expressed_genes = calculate_expressed_genes_per_cell_chunked(
            adata.X,
            retained_gene_indices,
            spec.dataset_id,
            chunk_size,
        )
        cell_filter_strategy = f"row chunks of at most {chunk_size} cells"
    retained_cells = expressed_genes >= MIN_GENES_PER_CELL
    retained_cell_indices = np.flatnonzero(retained_cells)
    cells_after_cell_filter = len(retained_cell_indices)
    assert cells_after_cell_filter > 0, f"{spec.dataset_id}: no cells remain after expressed-gene filtering"

    filtered_obs = adata.obs.iloc[retained_cell_indices].copy()
    filtered_obs["n_genes"] = expressed_genes[retained_cell_indices]
    filtered_obs["_source_row_index"] = retained_cell_indices
    metadata_adata = ad.AnnData(X=None, obs=filtered_obs, var=filtered_var)
    report_stage(spec.dataset_id, "Gene and cell filtering", stage_started)

    stage_started = time.perf_counter()
    print(f"[{spec.dataset_id}] Harmonizing cell types and filtering samples", flush=True)
    sample_column = resolve_sample_column(metadata_adata.obs, spec)
    removed_non_checkpoint_cells, non_checkpoint_types, unmapped_cell_types = harmonize_cell_types(
        metadata_adata,
        spec,
        cell_type_map,
        checkpoint_cell_types,
    )

    metadata_adata.obs["sample"] = metadata_adata.obs[sample_column].astype(str)
    sample_sizes = metadata_adata.obs["sample"].value_counts()
    retained_samples = sample_sizes[sample_sizes >= MIN_CELLS_PER_SAMPLE].index
    keep_samples = metadata_adata.obs["sample"].isin(retained_samples).to_numpy()
    removed_small_sample_cells = int((~keep_samples).sum())
    metadata_adata._inplace_subset_obs(keep_samples)
    assert metadata_adata.n_obs > 0, f"{spec.dataset_id}: no cells remain after preprocessing"
    report_stage(spec.dataset_id, "Cell-type harmonization and sample filtering", stage_started)

    output_cell_indices = metadata_adata.obs.pop("_source_row_index").to_numpy(dtype=np.int64)
    # Prefixing prevents collisions when CellArr concatenates studies whose raw
    # cell barcodes are only unique within a sequencing library.
    metadata_adata.obs["cell_id_original"] = metadata_adata.obs_names.astype(str)
    metadata_adata.obs_names = pd.Index(
        [f"{spec.dataset_id}:{cell_id}" for cell_id in metadata_adata.obs_names.astype(str)],
        name="cell_id",
    )
    assert metadata_adata.obs_names.is_unique

    metadata_adata.obs["study"] = spec.dataset_id
    metadata_adata.obs["cell_type_remapped_v2"] = metadata_adata.obs["cell_type"].astype(str)
    for column in metadata_adata.obs.select_dtypes(include="category").columns:
        metadata_adata.obs[column] = metadata_adata.obs[column].astype(str)

    preprocessing_metadata = {
        "mct_preprocessing": {
            "count_source": count_source,
            "gene_filter": f"total raw counts >= {MIN_GENE_COUNTS}",
            "cell_filter": f"expressed genes >= {MIN_GENES_PER_CELL}",
            "sample_filter": f"retained cells >= {MIN_CELLS_PER_SAMPLE}",
            "highly_variable_gene_selection": "none",
            "normalization": "none; CellarrDataModule applies normalize_total(target_sum=10000) and log1p",
            "matrix_location": "layers/counts",
            "cell_type_mapping": "released MCT cell_type_map.json composed with cell_type_input_order.json",
            "memory_strategy": "one full raw CSR matrix; filtered output is streamed to disk",
            "cell_filter_strategy": cell_filter_strategy,
        }
    }
    stage_started = time.perf_counter()
    print(f"[{spec.dataset_id}] Writing processed H5AD: {output_path}", flush=True)
    write_filtered_h5ad(
        adata.X,
        output_cell_indices,
        retained_gene_indices,
        metadata_adata.obs,
        metadata_adata.var,
        preprocessing_metadata,
        output_path,
        spec.dataset_id,
        chunk_size or DEFAULT_WRITE_CHUNK_SIZE,
    )
    report_stage(spec.dataset_id, "Processed H5AD write", stage_started)
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    print(f"[{spec.dataset_id}] Total preprocessing time: {format_duration(preprocessing_seconds)}")

    return {
        "dataset_id": spec.dataset_id,
        "title": spec.title,
        "url": spec.url,
        "publication_url": spec.publication_url,
        "processed_file": output_path.name,
        "count_source": count_source,
        "processed_matrix_location": "layers/counts",
        "source_had_raw": had_raw,
        "source_layers": source_layers,
        "non_human_features_removed": non_human_features_removed,
        "sample_source_column": sample_column,
        "input_cells": input_cells,
        "input_genes": input_genes,
        "cells_after_gene_filter": cells_after_gene_filter,
        "genes_after_count_filter": genes_after_count_filter,
        "cells_after_cell_filter": cells_after_cell_filter,
        "cells_removed_for_non_checkpoint_cell_type": removed_non_checkpoint_cells,
        "unmapped_source_cell_types": unmapped_cell_types,
        "non_checkpoint_harmonized_cell_types": non_checkpoint_types,
        "cells_removed_for_small_sample": removed_small_sample_cells,
        "output_cells": metadata_adata.n_obs,
        "output_genes": metadata_adata.n_vars,
        "output_samples": int(metadata_adata.obs["sample"].nunique()),
        "output_cell_types": sorted(metadata_adata.obs["cell_type"].unique().tolist()),
        "chunk_size": chunk_size,
        "preprocessing_seconds": preprocessing_seconds,
    }


def download_dataset(spec: DatasetSpec, output_path: Path) -> bool:
    """Download one CELLxGENE H5AD, or reuse an existing raw file."""
    if output_path.is_file():
        print(f"[{spec.dataset_id}] Reusing existing raw H5AD: {output_path}", flush=True)
        return False
    assert not output_path.exists(), f"Raw H5AD path exists but is not a file: {output_path}"
    print(f"Downloading {spec.dataset_id}: {spec.url}", flush=True)
    with tqdm(
        desc=f"{spec.dataset_id} download",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=0.5,
        dynamic_ncols=True,
    ) as progress:

        def update_progress(block_number: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.total = total_size
            downloaded = block_number * block_size
            if total_size > 0:
                downloaded = min(downloaded, total_size)
            progress.update(max(0, downloaded - progress.n))

        urllib.request.urlretrieve(spec.url, output_path, reporthook=update_progress)
        progress.total = output_path.stat().st_size
        progress.n = progress.total
        progress.refresh()
    return True


def download_preprocess_and_remove_source(
    spec: DatasetSpec,
    raw_directory: Path,
    processed_path: Path,
    cell_type_map: dict[str, str],
    checkpoint_cell_types: set[str],
    chunk_size: int | None = None,
) -> dict:
    """Download or reuse, preprocess, and then delete one source H5AD."""
    dataset_started = time.perf_counter()
    raw_path = raw_directory / f"{spec.dataset_id}.source.h5ad"
    download_started = time.perf_counter()
    raw_was_downloaded = download_dataset(spec, raw_path)
    download_seconds = time.perf_counter() - download_started
    download_size_bytes = raw_path.stat().st_size
    raw_status = "Downloaded" if raw_was_downloaded else "Reused"
    print(
        f"[{spec.dataset_id}] {raw_status} {download_size_bytes / 1024**3:.2f} GiB raw H5AD "
        f"in {format_duration(download_seconds)}"
    )
    print(f"Preprocessing {spec.dataset_id}: {raw_path}")
    report = preprocess_dataset(
        raw_path,
        processed_path,
        spec,
        cell_type_map,
        checkpoint_cell_types,
        chunk_size,
    )
    raw_path.unlink()
    print(f"Removed temporary source H5AD: {raw_path}")
    report["download_size_bytes"] = download_size_bytes
    report["download_seconds"] = download_seconds
    report["total_seconds"] = time.perf_counter() - dataset_started
    report["raw_input_status"] = "downloaded" if raw_was_downloaded else "reused"
    report["run_status"] = "processed"
    print(
        f"[{spec.dataset_id}] Dataset complete in {format_duration(report['total_seconds'])} "
        f"(download {format_duration(download_seconds)}, "
        f"preprocessing {format_duration(report['preprocessing_seconds'])})"
    )
    return report


def reuse_processed_dataset(
    spec: DatasetSpec,
    processed_path: Path,
    previous_report: dict | None,
) -> dict:
    """Return manifest metadata for an existing processed H5AD without opening it."""
    report = {} if previous_report is None else dict(previous_report)
    report.update(
        {
            "dataset_id": spec.dataset_id,
            "title": spec.title,
            "url": spec.url,
            "publication_url": spec.publication_url,
            "processed_file": processed_path.name,
            "run_status": "reused_processed",
        }
    )
    print(f"[{spec.dataset_id}] Processed H5AD already exists; skipping: {processed_path}", flush=True)
    return report


def process_or_reuse_dataset(
    spec: DatasetSpec,
    processed_directory: Path,
    raw_directory: Path | None,
    previous_report: dict | None,
    cell_type_map: dict[str, str],
    checkpoint_cell_types: set[str],
    chunk_size: int | None = None,
) -> dict:
    """Reuse a processed dataset, otherwise resume from raw or download it."""
    processed_path = processed_directory / f"{spec.dataset_id}.preprocessed.h5ad"
    if processed_path.is_file():
        return reuse_processed_dataset(spec, processed_path, previous_report)
    assert not processed_path.exists(), f"Processed H5AD path exists but is not a file: {processed_path}"
    assert raw_directory is not None, "A raw download directory is required for an unprocessed dataset"
    return download_preprocess_and_remove_source(
        spec,
        raw_directory,
        processed_path,
        cell_type_map,
        checkpoint_cell_types,
        chunk_size,
    )


def validate_checkpoint_artifacts(
    cell_type_map: dict[str, str],
    cell_type_input_order: dict[str, list[int]],
    gene2idx: dict[str, int],
) -> None:
    """Validate the released mappings against the authoritative checkpoint shape."""
    assert len(cell_type_map) == 309, f"Expected 309 released cell-type mappings, found {len(cell_type_map)}"
    assert len(cell_type_input_order) == 54, (
        f"Expected 54 checkpoint cell types, found {len(cell_type_input_order)}"
    )
    ranges = list(cell_type_input_order.values())
    assert ranges[0][0] == 0 and ranges[-1][1] == 1024
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False)), (
        "Checkpoint cell-type input ranges must be contiguous"
    )
    assert gene2idx.get("[PAD]") == 0 and gene2idx.get("[MASK]") == 1
    assert len(gene2idx) == 58_056, f"Expected 58,056 checkpoint gene tokens, found {len(gene2idx)}"
    assert set(gene2idx.values()) == set(range(len(gene2idx))), "Checkpoint gene indices must be contiguous"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--temporary-dir",
        type=Path,
        default=None,
        help=(
            "Directory used directly for raw downloads. Existing raw H5ADs are reused and successful inputs are "
            "removed after processing. Defaults to a newly created system temporary directory."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DEFAULT_DATASET_ORDER),
        help="Dataset IDs to process; defaults to all 13 in the original CellArr build order.",
    )
    parser.add_argument("--cell-type-map", type=Path, default=DEFAULT_CELL_TYPE_MAP)
    parser.add_argument("--cell-type-input-order", type=Path, default=DEFAULT_CELL_TYPE_INPUT_ORDER)
    parser.add_argument("--gene2idx", type=Path, default=DEFAULT_GENE2IDX)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Opt into row-chunked cell QC and set its cells per chunk. By default, cell QC uses CSR structure "
            "directly without row chunks or a full matrix copy."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Download selected datasets, preprocess them, and write a build manifest."""
    run_started = time.perf_counter()
    args = parse_args()
    assert args.chunk_size is None or args.chunk_size > 0, "--chunk-size must be positive"
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.processed_dir / "pretraining_manifest.json"
    previous_manifest = load_json(manifest_path) if manifest_path.is_file() else None
    if previous_manifest is not None:
        assert previous_manifest.get("format_version") == 1, "Unsupported existing preprocessing manifest version"
        previous_reports = {report["dataset_id"]: report for report in previous_manifest["datasets"]}
        print(f"Updating existing preprocessing manifest: {manifest_path}", flush=True)
    else:
        assert not manifest_path.exists(), f"Manifest path exists but is not a file: {manifest_path}"
        previous_reports = {}

    cell_type_map = load_json(args.cell_type_map)
    cell_type_input_order = load_json(args.cell_type_input_order)
    gene2idx = load_json(args.gene2idx)
    validate_checkpoint_artifacts(cell_type_map, cell_type_input_order, gene2idx)

    missing_processed_dataset_ids = [
        dataset_id
        for dataset_id in args.datasets
        if not (args.processed_dir / f"{dataset_id}.preprocessed.h5ad").is_file()
    ]
    raw_directory = None
    remove_raw_directory_after_run = False
    if missing_processed_dataset_ids:
        if args.temporary_dir is None:
            raw_directory = Path(tempfile.mkdtemp(prefix="mct-pretraining-"))
            remove_raw_directory_after_run = True
        else:
            args.temporary_dir.mkdir(parents=True, exist_ok=True)
            raw_directory = args.temporary_dir
        print(f"Using raw download directory: {raw_directory}", flush=True)

    dataset_reports = []
    for dataset_number, dataset_id in enumerate(args.datasets, start=1):
        spec = DATASETS[dataset_id]
        print(f"\nDataset {dataset_number}/{len(args.datasets)}: {dataset_id} — {spec.title}", flush=True)
        dataset_reports.append(
            process_or_reuse_dataset(
                spec,
                args.processed_dir,
                raw_directory,
                previous_reports.get(dataset_id),
                cell_type_map,
                set(cell_type_input_order),
                args.chunk_size,
            )
        )

    if remove_raw_directory_after_run:
        raw_directory.rmdir()
        print(f"Removed empty temporary raw directory: {raw_directory}", flush=True)

    artifacts = {
        "cell_type_map": args.cell_type_map.name,
        "cell_type_input_order": args.cell_type_input_order.name,
        "gene2idx": args.gene2idx.name,
    }
    for source in (args.cell_type_map, args.cell_type_input_order, args.gene2idx):
        destination = args.processed_dir / source.name
        if destination.exists():
            assert destination.is_file(), f"Model-contract artifact path is not a file: {destination}"
            assert file_sha256(source) == file_sha256(destination), (
                f"Existing model-contract artifact does not match {source}: {destination}"
            )
            print(f"Reusing model-contract artifact: {destination}", flush=True)
        else:
            shutil.copyfile(source, destination)

    processed_this_run = [report for report in dataset_reports if report["run_status"] == "processed"]
    reused_this_run = [report for report in dataset_reports if report["run_status"] == "reused_processed"]

    manifest = {
        "format_version": 1,
        "preprocessing": {
            "min_gene_counts": MIN_GENE_COUNTS,
            "min_genes_per_cell": MIN_GENES_PER_CELL,
            "min_cells_per_sample": MIN_CELLS_PER_SAMPLE,
            "gene_set": "union across processed datasets; no highly-variable-gene selection",
            "normalization": "raw counts; normalization is deferred to CellarrDataModule",
            "source_h5ads": "existing raw files are reused; raw files are removed after successful preprocessing",
            "matrix_memory_strategy": "one full raw CSR matrix; no full filtered matrix copy",
            "chunk_size": args.chunk_size,
            "total_download_seconds": sum(report["download_seconds"] for report in processed_this_run),
            "total_preprocessing_seconds": sum(report["preprocessing_seconds"] for report in processed_this_run),
            "total_run_seconds": time.perf_counter() - run_started,
            "checkpoint_cell_types": len(cell_type_input_order),
            "checkpoint_gene_tokens": len(gene2idx),
        },
        "artifact_files": artifacts,
        "artifact_sha256": {
            name: file_sha256(args.processed_dir / filename) for name, filename in artifacts.items()
        },
        "dataset_specs": [asdict(DATASETS[dataset_id]) for dataset_id in args.datasets],
        "datasets": dataset_reports,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote preprocessing manifest: {manifest_path}", flush=True)
    print(
        f"Finished {len(dataset_reports)} datasets in "
        f"{format_duration(manifest['preprocessing']['total_run_seconds'])}: "
        f"processed {len(processed_this_run)}, reused {len(reused_this_run)}, "
        f"download {format_duration(manifest['preprocessing']['total_download_seconds'])}, "
        f"preprocessing {format_duration(manifest['preprocessing']['total_preprocessing_seconds'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
