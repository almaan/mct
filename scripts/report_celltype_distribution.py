"""Cell-type distribution report over a built CellArr dataset.

For every sample (unique study::::sample) counts the number of DISTINCT
cell types present and produces:
  1. Histogram: distribution of "cell types per sample"
  2. Cumulative curve: count of samples with n_types >= k, for k in
     [1, max_types], showing saturation.

Uses the same filtering as ``CellarrDataModule``
(``min_cell_type_size`` on population-wide cell-type frequency and
``min_sample_size`` on per-sample cell count) so the reported numbers
match what training actually sees. Both filters can be disabled.

Emits:
  - per_sample_celltype_counts.csv    one row per sample, n_cell_types
  - celltype_summary.json             overall stats (n_samples,
                                      n_cell_types, mean/median/quantiles)
  - report.html                       self-contained (base64 figures + tables)
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiledb


# --------------------------------------------------------------------- #
# Data loading + filtering (parallel to CellarrDataModule.__init__)
# --------------------------------------------------------------------- #


def load_cell_metadata(
    cell_tdb_uri: str,
    label_column: str,
    sample_column: str,
    study_column: str,
) -> pd.DataFrame:
    config = tiledb.Config()
    config["sm.compute_concurrency_level"] = 1
    config["sm.io_concurrency_level"] = 1
    with tiledb.open(cell_tdb_uri, "r", config=config) as handle:
        df = handle.query(attrs=[label_column, sample_column, study_column]).df[:]
    df[label_column] = df[label_column].astype(str)
    df[sample_column] = df[sample_column].astype(str)
    df[study_column] = df[study_column].astype(str)
    return df


def apply_filters(
    df: pd.DataFrame,
    label_column: str,
    sample_column: str,
    study_column: str,
    min_cell_type_size: float | None,
    min_sample_size: int | None,
    cell_type_mapping: dict | None,
) -> pd.DataFrame:
    if cell_type_mapping:
        df = df.copy()
        df[label_column] = df[label_column].map(cell_type_mapping).fillna(df[label_column])

    if min_cell_type_size is not None and min_cell_type_size > 0:
        props = df[label_column].value_counts(normalize=True)
        keep = props[props >= min_cell_type_size].index
        df = df[df[label_column].isin(keep)]

    if min_sample_size is not None and min_sample_size > 0:
        sizes = df.groupby([study_column, sample_column]).size()
        keep = sizes[sizes >= min_sample_size].reset_index()
        df = df.merge(keep[[study_column, sample_column]], on=[study_column, sample_column], how="inner")
    return df


def per_sample_celltype_counts(
    df: pd.DataFrame,
    label_column: str,
    sample_column: str,
    study_column: str,
) -> pd.DataFrame:
    # sample identity = study::::sample to avoid collapsing duplicate sample names
    # across studies, matching CellarrDataModule.__init__ convention.
    df = df.copy()
    df["sample_uid"] = df[study_column] + "::::" + df[sample_column]

    grouped = df.groupby("sample_uid")
    n_types = grouped[label_column].nunique().rename("n_cell_types")
    n_cells = grouped.size().rename("n_cells")

    # Shannon entropy (nats) of cell-type frequencies within each sample.
    # Also normalised entropy (0-1) = H / log(n_types), 1 = perfectly uniform.
    def _entropy(labels: pd.Series) -> float:
        vc = labels.value_counts(normalize=True).values
        return float(-(vc * np.log(vc)).sum())

    def _max_type_fraction(labels: pd.Series) -> float:
        return float(labels.value_counts(normalize=True).iloc[0])

    entropy = grouped[label_column].apply(_entropy).rename("entropy")
    max_type_fraction = grouped[label_column].apply(_max_type_fraction).rename("max_type_fraction")

    # H_max = log(n_types)                              (perfectly uniform bound)
    # H_min = min entropy given (n_cells, n_types) when one dominant type takes
    #         (n_cells - (n_types - 1)) cells and every other type has one cell.
    #         = -p_dom * log(p_dom) + ((n_types - 1)/n_cells) * log(n_cells)
    # entropy_position = (H - H_min) / (H_max - H_min): 0 = at min (peaky),
    #                                                   1 = at max (uniform).
    n_types_arr = n_types.astype(float)
    n_cells_arr = n_cells.astype(float)
    log_n_types = np.log(n_types_arr.replace(1, np.nan))
    p_dom = (n_cells_arr - (n_types_arr - 1)) / n_cells_arr
    h_min = -p_dom * np.log(p_dom) + ((n_types_arr - 1) / n_cells_arr) * np.log(n_cells_arr)
    h_min = h_min.rename("entropy_min_bound")
    h_max = log_n_types.rename("entropy_max_bound")
    norm_entropy = (entropy / log_n_types).fillna(0.0).rename("entropy_normalised")
    denom = (h_max - h_min).replace(0, np.nan)
    entropy_position = ((entropy - h_min) / denom).fillna(0.0).clip(0.0, 1.0).rename("entropy_position")
    peakiness = (1.0 - entropy_position).rename("peakiness")

    out = pd.concat(
        [n_types, n_cells, entropy, norm_entropy, max_type_fraction,
         h_min, h_max, entropy_position, peakiness],
        axis=1,
    ).reset_index()
    sample_lookup = df[["sample_uid", study_column, sample_column]].drop_duplicates("sample_uid")
    out = out.merge(sample_lookup, on="sample_uid", how="left")
    cols = ["sample_uid", study_column, sample_column, "n_cells", "n_cell_types",
            "entropy", "entropy_min_bound", "entropy_max_bound",
            "entropy_normalised", "entropy_position", "peakiness", "max_type_fraction"]
    return out[cols].sort_values("n_cell_types", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------- #


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_histogram(counts: np.ndarray, bin_width: int = 1):
    import matplotlib.pyplot as plt

    max_k = int(counts.max())
    bins = np.arange(0.5, max_k + 1.5, bin_width)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    ax.hist(counts, bins=bins, color="#3b7dd8", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Number of distinct cell types in sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of per-sample cell-type diversity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axvline(np.median(counts), color="black", ls="--", lw=1, label=f"median = {int(np.median(counts))}")
    ax.axvline(np.mean(counts), color="crimson", ls=":", lw=1, label=f"mean = {np.mean(counts):.1f}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_entropy(entropy: np.ndarray, norm_entropy: np.ndarray):
    """Two-panel histogram: raw Shannon entropy (nats) + normalised entropy."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    ax1.hist(entropy, bins=40, color="#3b7dd8", edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Shannon entropy of cell-type distribution (nats)")
    ax1.set_ylabel("Number of samples")
    ax1.set_title("Per-sample cell-type entropy (raw)")
    ax1.axvline(float(np.median(entropy)), color="black", ls="--", lw=1,
                label=f"median = {np.median(entropy):.2f}")
    ax1.axvline(float(np.mean(entropy)), color="crimson", ls=":", lw=1,
                label=f"mean = {np.mean(entropy):.2f}")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.hist(norm_entropy, bins=40, color="#8f4d2a", edgecolor="black", alpha=0.85, range=(0, 1))
    ax2.set_xlabel("Normalised entropy  H / log(n_types)")
    ax2.set_ylabel("Number of samples")
    ax2.set_title("Per-sample cell-type entropy (normalised)")
    ax2.set_xlim(0, 1.02)
    ax2.axvline(float(np.median(norm_entropy)), color="black", ls="--", lw=1,
                label=f"median = {np.median(norm_entropy):.2f}")
    ax2.axvline(float(np.mean(norm_entropy)), color="crimson", ls=":", lw=1,
                label=f"mean = {np.mean(norm_entropy):.2f}")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_entropy_vs_ntypes(
    n_types: np.ndarray,
    entropy: np.ndarray,
    norm_entropy: np.ndarray,
    entropy_min: np.ndarray,
):
    """Scatter with both maximal (uniform) and minimal (peaky) entropy bounds."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    ax1.scatter(n_types, entropy, s=8, alpha=0.35, color="#2a4d8f", label="samples")
    max_k = int(n_types.max())
    ks = np.arange(1, max_k + 1)
    ax1.plot(ks, np.log(ks), color="crimson", lw=1.5, ls="--", label="max = log(n_types)  (uniform)")
    # H_min for each sample (already computed per-sample, one dot each) — sort
    # by n_types to draw a monotone lower envelope.
    order = np.argsort(n_types)
    ax1.plot(n_types[order], entropy_min[order], color="#e6a700", lw=1.5, ls="--",
             label="min (one dominant type)")
    ax1.set_xlabel("Number of distinct cell types")
    ax1.set_ylabel("Shannon entropy (nats)")
    ax1.set_title("Entropy vs n_cell_types with H_min / H_max bounds")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(n_types, norm_entropy, s=8, alpha=0.35, color="#8f4d2a")
    ax2.axhline(1.0, color="crimson", ls="--", lw=1.5, label="uniform (H = log n)")
    ax2.set_xlabel("Number of distinct cell types")
    ax2.set_ylabel("Normalised entropy  H / log(n_types)")
    ax2.set_title("Normalised entropy vs n_cell_types")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_peakiness(
    entropy_position: np.ndarray, max_type_fraction: np.ndarray
):
    """Two-panel: entropy_position (0 = peaky, 1 = uniform) + max-type share."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)

    ax1.hist(entropy_position, bins=40, range=(0, 1),
             color="#4a8f4a", edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Entropy position  (H − H_min) / (H_max − H_min)")
    ax1.set_ylabel("Number of samples")
    ax1.set_title("Where samples sit between min and max entropy\n(0 = fully peaky, 1 = uniform)")
    ax1.set_xlim(0, 1.02)
    ax1.axvline(float(np.median(entropy_position)), color="black", ls="--", lw=1,
                label=f"median = {np.median(entropy_position):.2f}")
    ax1.axvline(float(np.mean(entropy_position)), color="crimson", ls=":", lw=1,
                label=f"mean = {np.mean(entropy_position):.2f}")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.hist(max_type_fraction, bins=40, range=(0, 1),
             color="#a24a8f", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Max cell-type fraction  p_max")
    ax2.set_ylabel("Number of samples")
    ax2.set_title("Dominant cell-type share per sample\n(1 = one type covers all cells)")
    ax2.set_xlim(0, 1.02)
    ax2.axvline(float(np.median(max_type_fraction)), color="black", ls="--", lw=1,
                label=f"median = {np.median(max_type_fraction):.2f}")
    ax2.axvline(float(np.mean(max_type_fraction)), color="crimson", ls=":", lw=1,
                label=f"mean = {np.mean(max_type_fraction):.2f}")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative(counts: np.ndarray):
    import matplotlib.pyplot as plt

    max_k = int(counts.max())
    # k range covers 1..max_k+1 so the < k CDF actually reaches all samples
    # at k = max_k + 1.
    ks = np.arange(1, max_k + 2)
    n_lt_k = np.array([(counts < k).sum() for k in ks])
    frac_lt_k = n_lt_k / len(counts)
    n_eq_k = np.array([(counts == k).sum() for k in ks])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)
    ax1.plot(ks, n_lt_k, lw=2, color="#2a4d8f", marker="o", ms=3)
    ax1.set_xlabel("k (upper bound on distinct cell types)")
    ax1.set_ylabel("Number of samples with n_types < k")
    ax1.set_title("Cumulative distribution (absolute)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, frac_lt_k, lw=2, color="#8f4d2a", marker="o", ms=3)
    ax2.set_xlabel("k (upper bound on distinct cell types)")
    ax2.set_ylabel("Fraction of samples with n_types < k")
    ax2.set_title("Cumulative distribution (fraction)")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)
    for level in (0.5, 0.9):
        ax2.axhline(level, color="gray", ls=":", lw=0.8)

    fig.tight_layout()
    return fig, ks, n_lt_k, frac_lt_k, n_eq_k


# --------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------- #


def build_html_report(
    output_path: Path,
    metadata: dict,
    summary: dict,
    counts_df: pd.DataFrame,
    saturation_df: pd.DataFrame,
    fig_hist_b64: str,
    fig_cum_b64: str,
    fig_entropy_b64: str,
    fig_entropy_scatter_b64: str,
    fig_peakiness_b64: str,
) -> None:
    css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
        table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        table.tbl th { background: #f4f4f4; }
        .meta { background: #f8f8f8; padding: 8px 12px; border-radius: 4px;
                font-family: monospace; font-size: 12px; }
        .meta pre { margin: 0; white-space: pre-wrap; }
    """

    def img(b64, alt):
        return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'

    def df_html(df):
        return df.to_html(index=False, float_format="%.4f", classes="tbl")

    top = counts_df.head(20)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Cell-type distribution report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Cell-type distribution report</h1>",
        "<div class='meta'><pre>",
        json.dumps(metadata, indent=2, default=str),
        "</pre></div>",
        "<h2>Overall summary</h2>",
        "<div class='meta'><pre>",
        json.dumps(summary, indent=2, default=str),
        "</pre></div>",
        "<h2>Distribution — cell types per sample</h2>",
        img(fig_hist_b64, "Histogram cell types per sample"),
        "<h2>Per-sample cell-type entropy</h2>",
        "<p>Shannon entropy H = &minus;&Sigma; p<sub>i</sub> log p<sub>i</sub> "
        "over the cell-type frequency distribution within each sample. "
        "Normalised = H / log(n_types) &isin; [0, 1]; 1 = perfectly uniform, "
        "0 = single type.</p>",
        img(fig_entropy_b64, "Entropy histogram"),
        "<h2>Entropy vs number of cell types</h2>",
        img(fig_entropy_scatter_b64, "Entropy vs n_cell_types"),
        "<h2>Distance to minimal entropy (peakiness)</h2>",
        "<p>Left: <code>entropy_position = (H &minus; H_min) / (H_max &minus; H_min)</code>, "
        "where H_min corresponds to one dominant cell type covering all remaining cells "
        "beyond one-per-type singletons. 0 = at minimum entropy (fully peaky), "
        "1 = at maximum entropy (uniform).<br>"
        "Right: dominant cell-type share <code>p_max</code> per sample.</p>",
        img(fig_peakiness_b64, "Peakiness histograms"),
        "<h2>Cumulative distribution curve</h2>",
        img(fig_cum_b64, "Cumulative n_types < k"),
        "<h2>Saturation table (n_samples with n_types &lt; k and exactly = k)</h2>",
        df_html(saturation_df),
        "<h2>Top 20 most diverse samples</h2>",
        df_html(top),
        "</body></html>",
    ]
    output_path.write_text("\n".join(parts))


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cellarr-dir", type=Path,
        default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr"),
    )
    p.add_argument("--output-dir", type=Path,
                   default=Path("/cv/scratch/u/andera29/mct_pretraining/celltype_report"))
    p.add_argument("--label-column", default="cell_type")
    p.add_argument("--sample-column", default="sample")
    p.add_argument("--study-column", default="study")
    p.add_argument("--min-sample-size", type=int, default=0,
                   help="Filter samples with fewer cells than this. 0 = no filter.")
    p.add_argument("--min-cell-type-size", type=float, default=0.0,
                   help="Filter cell types with lower population fraction. 0 = no filter.")
    p.add_argument("--cell-type-mapping-file", type=Path, default=None,
                   help="Optional JSON dict mapping raw cell types to harmonised labels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parent.parent
    metadata = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    cell_tdb_uri = str(args.cellarr_dir / "cell_metadata")
    print(f"[load] reading {cell_tdb_uri}", flush=True)
    df = load_cell_metadata(cell_tdb_uri, args.label_column, args.sample_column, args.study_column)
    metadata["raw_n_rows"] = int(len(df))
    metadata["raw_n_samples"] = int(df.groupby([args.study_column, args.sample_column]).ngroups)
    metadata["raw_n_cell_types"] = int(df[args.label_column].nunique())

    mapping = None
    if args.cell_type_mapping_file is not None and args.cell_type_mapping_file.is_file():
        mapping = json.loads(args.cell_type_mapping_file.read_text())

    print(
        f"[filter] min_sample_size={args.min_sample_size} "
        f"min_cell_type_size={args.min_cell_type_size} mapping={mapping is not None}",
        flush=True,
    )
    df = apply_filters(
        df,
        label_column=args.label_column,
        sample_column=args.sample_column,
        study_column=args.study_column,
        min_cell_type_size=args.min_cell_type_size if args.min_cell_type_size > 0 else None,
        min_sample_size=args.min_sample_size if args.min_sample_size > 0 else None,
        cell_type_mapping=mapping,
    )
    metadata["filtered_n_rows"] = int(len(df))
    metadata["filtered_n_samples"] = int(df.groupby([args.study_column, args.sample_column]).ngroups)
    metadata["filtered_n_cell_types"] = int(df[args.label_column].nunique())

    counts_df = per_sample_celltype_counts(df, args.label_column, args.sample_column, args.study_column)
    counts_df.to_csv(args.output_dir / "per_sample_celltype_counts.csv", index=False)

    counts = counts_df["n_cell_types"].values.astype(int)
    quantiles = np.percentile(counts, [10, 25, 50, 75, 90, 95, 99])

    summary = {
        "n_samples": int(len(counts)),
        "n_cell_types_total": metadata["filtered_n_cell_types"],
        "min_types_per_sample": int(counts.min()),
        "max_types_per_sample": int(counts.max()),
        "mean_types_per_sample": float(counts.mean()),
        "median_types_per_sample": float(np.median(counts)),
        "quantiles_p10_p25_p50_p75_p90_p95_p99": [float(q) for q in quantiles],
        "n_samples_with_1_type": int((counts == 1).sum()),
        "n_samples_ge_5_types": int((counts >= 5).sum()),
        "n_samples_ge_10_types": int((counts >= 10).sum()),
        "n_samples_ge_20_types": int((counts >= 20).sum()),
    }
    (args.output_dir / "celltype_summary.json").write_text(json.dumps(summary, indent=2))

    entropy = counts_df["entropy"].values
    norm_entropy = counts_df["entropy_normalised"].values
    entropy_min = counts_df["entropy_min_bound"].values
    entropy_position = counts_df["entropy_position"].values
    max_type_fraction = counts_df["max_type_fraction"].values

    summary["entropy_mean"] = float(entropy.mean())
    summary["entropy_median"] = float(np.median(entropy))
    summary["entropy_max_observed"] = float(entropy.max())
    summary["max_possible_entropy_log_ntotal"] = float(np.log(summary["n_cell_types_total"]))
    summary["normalised_entropy_mean"] = float(norm_entropy.mean())
    summary["normalised_entropy_median"] = float(np.median(norm_entropy))
    summary["entropy_position_mean"] = float(entropy_position.mean())
    summary["entropy_position_median"] = float(np.median(entropy_position))
    summary["max_type_fraction_mean"] = float(max_type_fraction.mean())
    summary["max_type_fraction_median"] = float(np.median(max_type_fraction))
    summary["n_samples_p_max_ge_0_5"] = int((max_type_fraction >= 0.5).sum())
    summary["n_samples_p_max_ge_0_8"] = int((max_type_fraction >= 0.8).sum())

    (args.output_dir / "celltype_summary.json").write_text(json.dumps(summary, indent=2))

    print("[plot] histogram + entropy + peakiness + cumulative", flush=True)
    fig_hist = plot_histogram(counts)
    fig_entropy = plot_entropy(entropy, norm_entropy)
    fig_entropy_scatter = plot_entropy_vs_ntypes(counts, entropy, norm_entropy, entropy_min)
    fig_peakiness = plot_peakiness(entropy_position, max_type_fraction)
    fig_cum, ks, n_lt_k, frac_lt_k, n_eq_k = plot_cumulative(counts)

    saturation_df = pd.DataFrame({
        "k": ks,
        "n_samples_lt_k": n_lt_k,
        "frac_samples_lt_k": frac_lt_k,
        "n_samples_eq_k": n_eq_k,
    })
    saturation_df.to_csv(args.output_dir / "saturation.csv", index=False)

    build_html_report(
        args.output_dir / "report.html",
        metadata=metadata,
        summary=summary,
        counts_df=counts_df,
        saturation_df=saturation_df,
        fig_hist_b64=_fig_to_base64(fig_hist),
        fig_cum_b64=_fig_to_base64(fig_cum),
        fig_entropy_b64=_fig_to_base64(fig_entropy),
        fig_entropy_scatter_b64=_fig_to_base64(fig_entropy_scatter),
        fig_peakiness_b64=_fig_to_base64(fig_peakiness),
    )
    print(f"[done] outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
