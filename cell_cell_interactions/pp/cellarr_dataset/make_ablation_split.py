"""Emit a fixed sample-level ablation split for CellArr pretraining.

Reads the built CellArr's ``cell_metadata`` TileDB, applies the same
``min_cell_type_size`` + ``min_sample_size`` filters as
``CellarrDataModule``, then partitions samples deterministically into:

    - test  (10 percent of samples)
    - val   (10 percent of samples)
    - train (80 percent of samples), further split into nested subsets
      1.00 supset 0.50 supset 0.25 supset 0.10 for data-scaling ablation.

Output JSON schema (v1):

    {
      "format_version": 1,
      "random_seed": <int>,
      "sample_column": "sample",
      "test_samples":  [...],
      "val_samples":   [...],
      "train_samples": {
         "1.00": [...],
         "0.50": [...],
         "0.25": [...],
         "0.10": [...]
      }
    }

The output is consumed by ``CellarrDataModule`` via ``--split-file`` +
``--train-fraction`` in the training scripts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TRAIN_FRACTIONS = (1.00, 0.50, 0.25, 0.10, 0.05)


def load_sample_ids(
    cell_metadata_uri: str,
    sample_column: str,
    label_column: str,
    min_sample_size: int,
    min_cell_type_size: float,
    cell_type_mapping_file: Path | None,
) -> list[str]:
    """Return sorted valid sample IDs after applying datamodule filters."""
    import tiledb

    config = tiledb.Config()
    config["sm.compute_concurrency_level"] = 1
    config["sm.io_concurrency_level"] = 1

    with tiledb.open(cell_metadata_uri, "r", config=config) as handle:
        data_df = handle.query(attrs=[sample_column, label_column, "study"]).df[:]

    if cell_type_mapping_file is not None:
        cell_type_mapping = json.loads(cell_type_mapping_file.read_text())
        data_df[label_column] = data_df[label_column].map(cell_type_mapping)

    if min_cell_type_size is not None and min_cell_type_size > 0:
        proportions = data_df[label_column].value_counts(normalize=True)
        keep_types = proportions[proportions >= min_cell_type_size].index
        data_df = data_df[data_df[label_column].isin(keep_types)]

    if min_sample_size is not None and min_sample_size > 0:
        sizes = data_df.groupby([("study"), sample_column]).size()
        keep = sizes[sizes >= min_sample_size].reset_index()
        data_df = data_df[
            data_df["study"].isin(keep["study"]) & data_df[sample_column].isin(keep[sample_column])
        ]

    samples = sorted(data_df[sample_column].astype(str).unique().tolist())
    assert samples, "No samples survive filters; check min_sample_size / min_cell_type_size"
    return samples


def partition(samples: list[str], seed: int) -> dict:
    """Deterministic test / val / nested train partition."""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(0.10 * n))
    n_val = max(1, int(0.10 * n))

    test_samples = sorted(shuffled[:n_test])
    val_samples = sorted(shuffled[n_test : n_test + n_val])
    train_pool = shuffled[n_test + n_val :]  # keep shuffled order for nested subsets

    train_samples: dict[str, list[str]] = {}
    n_pool = len(train_pool)
    for frac in TRAIN_FRACTIONS:
        k = max(1, int(round(frac * n_pool)))
        subset = sorted(train_pool[:k])
        train_samples[f"{frac:.2f}"] = subset

    # sanity: nested containment
    prev: set[str] | None = None
    for frac in sorted(TRAIN_FRACTIONS):
        this = set(train_samples[f"{frac:.2f}"])
        if prev is not None:
            assert prev.issubset(this), f"Nested subset broken between {frac} and previous"
        prev = this

    assert set(test_samples).isdisjoint(val_samples)
    assert set(test_samples).isdisjoint(train_samples["1.00"])
    assert set(val_samples).isdisjoint(train_samples["1.00"])

    return {
        "test_samples": test_samples,
        "val_samples": val_samples,
        "train_samples": train_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cellarr-dir", type=Path, required=True, help="Root dir of built CellArr dataset")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path")
    parser.add_argument("--sample-column", type=str, default="sample")
    parser.add_argument("--label-column", type=str, default="cell_type")
    parser.add_argument("--min-sample-size", type=int, default=256)
    parser.add_argument("--min-cell-type-size", type=float, default=0.01)
    parser.add_argument("--cell-type-mapping-file", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_metadata_uri = str(args.cellarr_dir / "cell_metadata")
    assert (args.cellarr_dir / "cell_metadata").exists(), f"Missing cell_metadata under {args.cellarr_dir}"

    samples = load_sample_ids(
        cell_metadata_uri=cell_metadata_uri,
        sample_column=args.sample_column,
        label_column=args.label_column,
        min_sample_size=args.min_sample_size,
        min_cell_type_size=args.min_cell_type_size,
        cell_type_mapping_file=args.cell_type_mapping_file,
    )
    print(f"Total valid samples after filters: {len(samples)}")

    parts = partition(samples, args.random_seed)
    print(f"Test: {len(parts['test_samples'])}  Val: {len(parts['val_samples'])}")
    for frac, ids in parts["train_samples"].items():
        print(f"Train {frac}: {len(ids)}")

    output = {
        "format_version": 1,
        "random_seed": args.random_seed,
        "sample_column": args.sample_column,
        **parts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote split file: {args.output}")


if __name__ == "__main__":
    main()
