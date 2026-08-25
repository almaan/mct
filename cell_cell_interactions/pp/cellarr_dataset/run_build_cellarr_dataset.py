"""Build the unified MCT pretraining CellArr dataset from processed H5ADs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# CellArr 0.6.1 uses spawn workers and creates fresh TileDB Config objects for
# matrix writes, consolidation, and the returned CellArrDataset. Environment
# settings are inherited by every spawned process and apply to every new Config.
TILEDB_CONCURRENCY = 1
os.environ["TILEDB_SM_COMPUTE_CONCURRENCY_LEVEL"] = str(TILEDB_CONCURRENCY)
os.environ["TILEDB_SM_IO_CONCURRENCY_LEVEL"] = str(TILEDB_CONCURRENCY)


OBS_COLUMNS = (
    "cell_id_original",
    "sample",
    "donor_id",
    "tissue",
    "assay",
    "cell_type_original",
    "cell_type",
    "cell_type_ontology_term_id",
    "disease",
    "sex",
    "cell_type_remapped_v2",
    "study",
)


def initialize_tiledb() -> object:
    """Initialize and verify a single-threaded process-global TileDB context."""
    import tiledb

    config = tiledb.Config()
    assert int(config["sm.compute_concurrency_level"]) == TILEDB_CONCURRENCY
    assert int(config["sm.io_concurrency_level"]) == TILEDB_CONCURRENCY
    context = tiledb.default_ctx(config)
    context_config = context.config()
    assert int(context_config["sm.compute_concurrency_level"]) == TILEDB_CONCURRENCY
    assert int(context_config["sm.io_concurrency_level"]) == TILEDB_CONCURRENCY
    print("Configured TileDB compute and I/O concurrency to 1 thread each", flush=True)
    return context


def load_json(path: Path) -> dict:
    """Load a JSON object."""
    with path.open() as handle:
        return json.load(handle)


def resolve_manifest_files(manifest_path: Path, manifest: dict) -> list[Path]:
    """Resolve and validate processed H5AD paths in manifest order."""
    files = [manifest_path.parent / dataset["processed_file"] for dataset in manifest["datasets"]]
    missing = [str(path) for path in files if not path.is_file()]
    assert not missing, f"Processed H5AD files listed in the manifest are missing: {missing}"
    assert files, "The preprocessing manifest contains no datasets"
    return files


def ensure_counts_layers(files: list[Path]) -> None:
    """Expose legacy X-only processed matrices as CellArr's required counts layer."""
    for path in files:
        with h5py.File(path, "r+") as handle:
            assert "layers" in handle and isinstance(handle["layers"], h5py.Group), (
                f"{path}: missing H5AD layers group"
            )
            layers = handle["layers"]
            if "counts" in layers:
                continue

            assert "X" in handle, f"{path}: neither layers['counts'] nor X contains the processed count matrix"
            # This creates a second HDF5 name for the same object, not a data
            # copy. Removing the old name then leaves exactly one stored matrix.
            layers["counts"] = handle["X"]
            del handle["X"]
            print(f"Moved legacy X matrix to layers['counts'] without copying data: {path}", flush=True)


def collect_cell_metadata(files: list[Path], checkpoint_cell_types: set[str]) -> pd.DataFrame:
    """Collect validated CellArr row metadata in the H5AD concatenation order."""
    import scanpy as sc

    frames = []
    seen_studies = set()
    for path in files:
        adata = sc.read_h5ad(path, backed="r")
        missing = sorted(set(OBS_COLUMNS) - set(adata.obs.columns))
        assert not missing, f"{path}: missing required obs columns: {missing}"
        assert adata.obs_names.is_unique, f"{path}: cell identifiers are not unique"
        assert not adata.obs[list(OBS_COLUMNS)].isna().any().any(), f"{path}: required metadata contains nulls"

        studies = set(adata.obs["study"].astype(str))
        assert len(studies) == 1, f"{path}: expected one study identifier, found {studies}"
        study = next(iter(studies))
        assert study not in seen_studies, f"Duplicate study identifier in manifest: {study}"
        seen_studies.add(study)

        invalid_cell_types = sorted(set(adata.obs["cell_type"].astype(str)) - checkpoint_cell_types)
        assert not invalid_cell_types, f"{path}: cell types outside checkpoint input order: {invalid_cell_types}"

        frame = adata.obs[list(OBS_COLUMNS)].copy()
        frame.insert(0, "cell_id", adata.obs_names.astype(str))
        frames.append(frame)
        adata.file.close()

    metadata = pd.concat(frames, ignore_index=True)
    assert metadata["cell_id"].is_unique, "Cell identifiers are not unique across processed datasets"
    return metadata


def copy_model_contract(manifest_path: Path, manifest: dict, output_dir: Path) -> None:
    """Copy checkpoint mappings and preprocessing provenance next to CellArr."""
    artifact_files = manifest["artifact_files"]
    destinations = {
        "cell_type_map": "cell_type_map.json",
        "cell_type_input_order": "cell_type_input_order.json",
        "gene2idx": "gene2idx.json",
    }
    for artifact_name, destination_name in destinations.items():
        source = manifest_path.parent / artifact_files[artifact_name]
        assert source.is_file(), f"Missing model-contract artifact: {source}"
        shutil.copyfile(source, output_dir / destination_name)
    shutil.copyfile(manifest_path, output_dir / "pretraining_manifest.json")


def prepare_output_dir(output_dir: Path, manifest_path: Path) -> None:
    """Create an empty output directory, replacing an existing safe target."""
    resolved_output = output_dir.resolve()
    resolved_manifest = manifest_path.resolve()
    protected_paths = {
        Path("/").resolve(),
        Path.cwd().resolve(),
        Path.home().resolve(),
        resolved_manifest.parent,
    }
    assert resolved_output not in protected_paths, f"Refusing to replace protected directory: {resolved_output}"
    assert resolved_output not in resolved_manifest.parents, (
        f"Refusing to replace {resolved_output} because it contains the preprocessing manifest"
    )

    if output_dir.exists():
        assert output_dir.is_dir(), f"CellArr output path exists but is not a directory: {output_dir}"
        assert not output_dir.is_symlink(), f"Refusing to replace symlinked output directory: {output_dir}"
        print(f"Replacing existing CellArr output directory: {output_dir}", flush=True)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)


def build_cellarr_dataset(manifest_path: Path, output_dir: Path, num_threads: int) -> None:
    """Build and validate the CellArr dataset."""
    assert manifest_path.is_file(), f"Preprocessing manifest does not exist: {manifest_path}"
    assert num_threads > 0, "--num-threads must be positive"

    # CellArr imports TileDB. Establish the process-global TileDB context first so
    # TileDB does not size its own thread pools from every CPU visible on the host.
    tiledb_context = initialize_tiledb()
    from cellarr import MatrixOptions, build_cellarrdataset

    manifest = load_json(manifest_path)
    assert manifest.get("format_version") == 1, "Unsupported preprocessing manifest version"

    cell_type_order_path = manifest_path.parent / manifest["artifact_files"]["cell_type_input_order"]
    checkpoint_cell_types = set(load_json(cell_type_order_path))
    files = resolve_manifest_files(manifest_path, manifest)
    ensure_counts_layers(files)
    cell_metadata = collect_cell_metadata(files, checkpoint_cell_types)

    prepare_output_dir(output_dir, manifest_path)
    cell_metadata.to_parquet(output_dir / "cell_metadata.parquet", index=False)
    print(f"Building CellArr from {len(files)} datasets and {len(cell_metadata):,} cells")
    dataset = build_cellarrdataset(
        output_path=str(output_dir),
        files=[str(path) for path in files],
        matrix_options=MatrixOptions(dtype=np.float32),
        num_threads=num_threads,
        cell_metadata=cell_metadata,
        optimize_tiledb=True,
    )
    del tiledb_context
    print(dataset)

    for component in ("cell_metadata", "gene_annotation", "sample_metadata", "counts"):
        assert (output_dir / component).exists(), f"CellArr did not create expected component: {component}"
    copy_model_contract(manifest_path, manifest, output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="CellArr metadata-scanning worker processes; TileDB always uses one thread (default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cellarr_dataset(args.manifest, args.output_dir, args.num_threads)
