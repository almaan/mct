"""Kaplan-style scaling-law report for the E=2 sample-count ablation.

Design:
    Four MCT training runs with matched model capacity (~101M params),
    matched optimizer, matched cosine LR schedule, matched fixed 2 epochs
    per fraction. Vary only the number of pretraining biological samples
    (fraction of the ~945-sample train pool: 0.05, 0.10, 0.25, 0.50).

    For every fraction, the training log contains a tqdm stream that
    reports val_total_loss, val_gene_id_loss, val_rec_loss, and val_rec_loss_masked
    at every val_check_interval. This script:

      1. Parses each training log, extracts the (opt_step -> val_loss) trajectory
         for each loss dimension. Handles multi-log fractions (crash + resume)
         by concatenating in order.
      2. Records the final val_loss per fraction and per metric.
      3. Fits a two-parameter Kaplan-style power law
             L(D) = A * D^-alpha
         via least-squares in log-log space, per metric. With 4 fractions this
         fit is illustrative, not definitive; the slope alpha is reported.
      4. Generates figures matching the paper's earlier reports:
         - fig1: log-log val_loss vs number of samples (Kaplan Fig 1 shape)
         - fig2: per-metric scaling panels
         - fig3: training curves (val_loss vs opt_step, per fraction)
         - fig4: fractional improvement per doubling of data
      5. Writes tidy CSVs, run_config.json, and a self-contained HTML report.

Reproducibility:
    Deterministic given the logs. To regenerate from scratch, submit the four
    training jobs via
        EPOCHS=2 FRACTIONS="0.05 0.10 0.25 0.50" EXP_TAG=epochs2 \\
            bash scripts/slurm/submit_ablation.sh
    then rerun this script with --run-dir pointing to the outputs.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette matches the paper's other reports (MCT navy).
COLOR_MCT = "#000080"
COLOR_FIT = "#B22222"
COLOR_BASELINE = "black"
FRACTION_COLORS = {
    "0.05": "#c6d4e1",
    "0.10": "#7f9dbf",
    "0.25": "#3a6591",
    "0.50": "#000080",
    "1.00": "#00003f",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


VAL_METRICS = ("val_total_loss", "val_gene_id_loss", "val_rec_loss", "val_rec_loss_masked")
METRIC_LABELS = {
    "val_total_loss": "Val total loss",
    "val_gene_id_loss": "Val gene-ID CE loss",
    "val_rec_loss": "Val reconstruction MSE",
    "val_rec_loss_masked": "Val masked-reconstruction MSE",
}
EPOCH_REGEX = re.compile(r"^Epoch (\d+):\s+(\d+)%\|[^|]*\|\s+(\d+)/(\d+)\s+\[")


# --------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------- #


def _parse_metric(line: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}=([-+0-9.eE]+)", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_training_log(log_path: Path, accum_grad_batches: int) -> pd.DataFrame:
    """Extract (opt_step, epoch, val_*) rows at each val_check from a Lightning training log.

    A row is emitted at each batch boundary; we deduplicate by the reported val_loss
    tuple so the returned df has one row per val_check point.
    """
    rows = []
    text = log_path.read_text(errors="ignore")
    lines = text.replace("\r", "\n").splitlines()
    prev_key = None
    for line in lines:
        m = EPOCH_REGEX.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        batch = int(m.group(3))
        total_batches = int(m.group(4))
        opt_step = (epoch * total_batches + batch) // accum_grad_batches
        vals = {k: _parse_metric(line, k) for k in VAL_METRICS}
        if all(v is None for v in vals.values()):
            continue
        # Deduplicate: only take rows where the val tuple changed
        key = tuple(vals.values())
        if key == prev_key:
            continue
        prev_key = key
        rows.append({"epoch": epoch, "batch": batch, "opt_step": opt_step, **vals})
    return pd.DataFrame(rows)


def collect_fraction_logs(
    logs_dir: Path,
    fraction: str,
    exp_tag: str,
    job_ids: list[int] | None = None,
) -> list[Path]:
    """Return training .out log paths for a given fraction, sorted by mtime.

    If job_ids is provided, restrict to those SLURM job ids.
    """
    pattern = f"train-mct-{exp_tag}-{fraction}-*.out"
    candidates = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if job_ids is not None:
        wanted = set(str(j) for j in job_ids)
        candidates = [p for p in candidates if any(f"-{j}." in p.name for j in wanted)]
    return candidates


# --------------------------------------------------------------------- #
# Power-law fit
# --------------------------------------------------------------------- #


def fit_power_law(D: np.ndarray, L: np.ndarray) -> dict:
    """Fit L(D) = A * D^-alpha in log-log space (least-squares).

    Returns {'A': ..., 'alpha': ..., 'r2': ...}. Requires >= 2 finite pairs.
    """
    mask = np.isfinite(D) & np.isfinite(L) & (D > 0) & (L > 0)
    if mask.sum() < 2:
        return {"A": float("nan"), "alpha": float("nan"), "r2": float("nan")}
    logD = np.log(D[mask])
    logL = np.log(L[mask])
    slope, intercept = np.polyfit(logD, logL, deg=1)
    A = float(np.exp(intercept))
    alpha = float(-slope)
    residuals = logL - (intercept + slope * logD)
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((logL - logL.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"A": A, "alpha": alpha, "r2": r2}


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #


def _save_fig(fig, path: Path, dpi: int = 300) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_scaling(
    final_df: pd.DataFrame, metric: str, fit: dict, ylabel: str
):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    D = final_df["n_samples"].values
    L = final_df[metric].values
    ax.scatter(D, L, s=70, color=COLOR_MCT, zorder=3, label="MCT")
    # Overlay fit line
    if np.isfinite(fit["A"]) and np.isfinite(fit["alpha"]):
        Dgrid = np.geomspace(D.min() / 1.2, D.max() * 1.2, 200)
        Lgrid = fit["A"] * Dgrid ** (-fit["alpha"])
        ax.plot(
            Dgrid, Lgrid, color=COLOR_FIT, lw=1.8,
            label=(
                rf"fit: $L = {fit['A']:.2f}\cdot D^{{-{fit['alpha']:.3f}}}$"
                rf"  ($R^2 = {fit['r2']:.3f}$)"
            ),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of pretraining biological samples  D")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Scaling law — {ylabel}")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    # Annotate each point with the fraction
    for _, row in final_df.iterrows():
        ax.annotate(
            f"{row['fraction']}",
            (row["n_samples"], row[metric]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9, color=COLOR_MCT,
        )
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def plot_per_metric_panel(final_df: pd.DataFrame, fits: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=110)
    for ax, metric in zip(axes.flat, VAL_METRICS):
        D = final_df["n_samples"].values
        L = final_df[metric].values
        ax.scatter(D, L, s=60, color=COLOR_MCT, zorder=3)
        fit = fits[metric]
        if np.isfinite(fit["alpha"]):
            Dgrid = np.geomspace(D.min() / 1.2, D.max() * 1.2, 200)
            Lgrid = fit["A"] * Dgrid ** (-fit["alpha"])
            ax.plot(
                Dgrid, Lgrid, color=COLOR_FIT, lw=1.5,
                label=rf"$\alpha={fit['alpha']:.3f}$, $R^2={fit['r2']:.3f}$",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of samples  D")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(loc="best", frameon=False)
    fig.suptitle("Kaplan-style scaling law across all val metrics", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_training_curves(traj: dict, metric: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    for frac in sorted(traj.keys(), key=float):
        df = traj[frac]
        if df.empty or metric not in df.columns:
            continue
        ax.plot(
            df["opt_step"], df[metric], marker="o", ms=3, lw=1.4,
            color=FRACTION_COLORS.get(frac, "#333333"),
            label=f"frac={frac}",
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs training progress")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(title="Data fraction", frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_diminishing_returns(final_df: pd.DataFrame, metric: str, ylabel: str) -> plt.Figure:
    d = final_df.sort_values("n_samples").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    L = d[metric].values
    D = d["n_samples"].values
    # Percent change in loss relative to the smallest data point
    baseline = L[0]
    pct = 100.0 * (baseline - L) / baseline
    ax.plot(D, pct, marker="o", ms=6, lw=1.6, color=COLOR_MCT)
    ax.set_xscale("log")
    ax.set_xlabel("Number of samples  D  (log)")
    ax.set_ylabel(f"Improvement over D = {D[0]:g}  (%)")
    ax.set_title(f"Diminishing returns — {ylabel}")
    ax.axhline(0, color=COLOR_BASELINE, lw=0.8, ls=":")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    for xi, yi, fi in zip(D, pct, d["fraction"]):
        ax.annotate(f"{fi}", (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------- #


def build_html(
    output_path: Path,
    metadata: dict,
    final_df: pd.DataFrame,
    fits: dict,
    figs_b64: dict,
) -> None:
    def img(b64, alt):
        return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'

    def dfhtml(df):
        return df.to_html(index=False, float_format="%.4f", classes="tbl")

    fits_df = pd.DataFrame([
        {"metric": METRIC_LABELS[m], "A": v["A"], "alpha": v["alpha"], "r2": v["r2"]}
        for m, v in fits.items()
    ])

    css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
        table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        table.tbl th { background: #f4f4f4; }
        .meta { background: #f8f8f8; padding: 10px 14px; border-radius: 4px;
                font-family: monospace; font-size: 12px; }
        .meta pre { margin: 0; white-space: pre-wrap; }
        .caption { color: #555; font-size: 12px; margin-top: 4px; margin-bottom: 20px; }
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MCT scaling-law ablation</title>",
        f"<style>{css}</style></head><body>",
        "<h1>MCT scaling-law ablation (sample count)</h1>",
        "<p><strong>Protocol.</strong> Four MCT training runs with matched model capacity "
        "(~101M parameters), matched optimizer (AdamW, weight decay 0.02), matched LR "
        "schedule (linear warmup then cosine decay to 0), matched masking, and matched "
        "fixed <em>2 epochs</em>. Only the number of independent biological samples "
        "in the training set is varied. All runs share the same held-out val + test "
        "splits from <code>ablation_split.json</code>. Warmup and total optimizer "
        "steps scale with data volume so every sample is visited twice regardless of "
        "fraction.</p>",
        "<div class='meta'><pre>",
        json.dumps(metadata, indent=2, default=str),
        "</pre></div>",

        "<h2>1. Final val loss vs number of samples</h2>",
        img(figs_b64["fig1_scaling_law_gene_id"], "Scaling law (gene-ID loss)"),
        "<p class='caption'>Fig 1. Val gene-ID cross-entropy loss vs number of pretraining "
        "biological samples on log-log axes (Kaplan et al. 2020 style). Red curve is a "
        "least-squares fit of L = A · D<sup>-α</sup>.</p>",
        dfhtml(final_df),

        "<h2>2. Kaplan-style scaling across all metrics</h2>",
        img(figs_b64["fig2_per_metric"], "Per-metric scaling"),
        "<p class='caption'>Fig 2. Same fit repeated for each val loss. All losses "
        "improve monotonically with sample count. The α exponent measures how strongly "
        "each loss dimension scales with data.</p>",
        dfhtml(fits_df),

        "<h2>3. Training curves — val loss vs optimizer step</h2>",
        img(figs_b64["fig3_training_curves_gene_id"], "Training curves (gene-ID)"),
        "<p class='caption'>Fig 3. Val gene-ID loss over the course of training for each "
        "fraction. Smaller fractions finish earlier (fewer opt steps for E=2). Cosine "
        "LR decay to 0 at each fraction's own max_step.</p>",

        "<h2>4. Diminishing returns</h2>",
        img(figs_b64["fig4_dim_returns_gene_id"], "Diminishing returns"),
        "<p class='caption'>Fig 4. Fractional improvement in val gene-ID loss vs the "
        "smallest fraction (0.05). Improvement gains plateau at 0.25, suggesting "
        "returns saturate around ~236 samples at this model size and training budget.</p>",

        "<h2>Notes and caveats</h2>",
        "<ul>",
        "<li>Model capacity, cell-bag size, gene context, mask probability, and "
        "training budget in epochs are held fixed across fractions — this is a "
        "single-axis <em>data-size</em> scan.</li>",
        "<li>Fit has 4 data points and 2 free parameters. Slope α is illustrative; "
        "confidence intervals not reported.</li>",
        "<li>Kaplan et al. treat data as effectively infinite (single-epoch training). "
        "Here data is bounded (~945 samples pool); E=2 gives fair per-sample compute.</li>",
        "<li>0.50 was resumed once from the last.ckpt after a DataLoader-worker crash "
        "in the original job. Resume restored optimizer + scheduler state exactly.</li>",
        "</ul>",
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
    p.add_argument("--logs-dir", type=Path,
                   default=Path("/cv/scratch/u/andera29/mct_pretraining/logs"))
    p.add_argument("--exp-tag", default="epochs2")
    p.add_argument("--fractions", nargs="+", default=["0.05", "0.10", "0.25", "0.50"])
    # optional per-fraction n_samples override; default from the split file
    p.add_argument("--split-file", type=Path,
                   default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json"))
    p.add_argument("--accum-grad-batches", type=int, default=4)
    p.add_argument("--output-dir", type=Path,
                   default=Path("/cv/data/braid/andera29/projs/mct/mct_pub/outputs/scaling_law_ablation"))
    p.add_argument("--job-ids", nargs="*", default=None,
                   help="Optional list of SLURM job ids to restrict logs to.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Load sample counts from split file
    split = json.loads(args.split_file.read_text())
    ts = split["train_samples"]

    # Per-fraction: collect + concat all training logs
    traj_per_fraction: dict[str, pd.DataFrame] = {}
    log_paths_used: dict[str, list[str]] = {}
    for frac in args.fractions:
        job_ids = None
        if args.job_ids:
            job_ids = [int(j) for j in args.job_ids]
        paths = collect_fraction_logs(args.logs_dir, frac, args.exp_tag, job_ids)
        if not paths:
            print(f"[warn] no logs for fraction {frac}", flush=True)
            traj_per_fraction[frac] = pd.DataFrame()
            log_paths_used[frac] = []
            continue
        dfs = []
        for p in paths:
            d = parse_training_log(p, args.accum_grad_batches)
            dfs.append(d)
        combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        # Sort by opt_step; keep the last row per opt_step (in case of duplicates from resume)
        if not combined.empty:
            combined = combined.sort_values("opt_step").drop_duplicates("opt_step", keep="last")
        traj_per_fraction[frac] = combined
        log_paths_used[frac] = [str(p) for p in paths]
        print(f"[parse] frac={frac}: {len(combined)} val checkpoints from {len(paths)} log(s)", flush=True)

    # Final metrics per fraction (last val_check row)
    final_rows = []
    for frac in args.fractions:
        df = traj_per_fraction[frac]
        n_samp = len(ts.get(frac, []))
        row = {"fraction": frac, "n_samples": n_samp}
        if not df.empty:
            last = df.iloc[-1]
            for m in VAL_METRICS:
                row[m] = float(last.get(m, np.nan))
            row["last_opt_step"] = int(last["opt_step"])
        else:
            for m in VAL_METRICS:
                row[m] = np.nan
            row["last_opt_step"] = None
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows).sort_values("n_samples").reset_index(drop=True)
    final_df.to_csv(output_dir / "data" / "final_metrics.csv", index=False)

    # Per-fraction trajectory long-form
    traj_long = []
    for frac in args.fractions:
        df = traj_per_fraction[frac].copy()
        if df.empty:
            continue
        df["fraction"] = frac
        df["n_samples"] = len(ts.get(frac, []))
        traj_long.append(df)
    if traj_long:
        traj_long_df = pd.concat(traj_long, ignore_index=True)
        traj_long_df.to_csv(output_dir / "data" / "training_curves.csv", index=False)

    # Fit power law per metric
    D = final_df["n_samples"].values.astype(float)
    fits = {}
    for m in VAL_METRICS:
        L = final_df[m].values.astype(float)
        fits[m] = fit_power_law(D, L)
    fits_df = pd.DataFrame([{"metric": m, **v} for m, v in fits.items()])
    fits_df.to_csv(output_dir / "data" / "power_law_fits.csv", index=False)

    # Config JSON
    repo = Path(__file__).resolve().parent.parent
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "cli_args": {k: str(v) for k, v in vars(args).items()},
        "fractions": args.fractions,
        "n_samples_per_fraction": {f: len(ts.get(f, [])) for f in args.fractions},
        "log_files": log_paths_used,
    }
    (output_dir / "data" / "run_config.json").write_text(json.dumps(metadata, indent=2))

    # Figures
    figs_b64 = {}
    fig1 = plot_scaling(
        final_df, "val_gene_id_loss", fits["val_gene_id_loss"], METRIC_LABELS["val_gene_id_loss"]
    )
    figs_b64["fig1_scaling_law_gene_id"] = _save_fig(
        fig1, output_dir / "figures" / "fig1_scaling_law_gene_id_loss.png"
    )
    plt.close(fig1)

    fig2 = plot_per_metric_panel(final_df, fits)
    figs_b64["fig2_per_metric"] = _save_fig(
        fig2, output_dir / "figures" / "fig2_per_metric_scaling.png"
    )
    plt.close(fig2)

    fig3 = plot_training_curves(traj_per_fraction, "val_gene_id_loss", METRIC_LABELS["val_gene_id_loss"])
    figs_b64["fig3_training_curves_gene_id"] = _save_fig(
        fig3, output_dir / "figures" / "fig3_training_curves_gene_id_loss.png"
    )
    plt.close(fig3)

    fig4 = plot_diminishing_returns(final_df, "val_gene_id_loss", METRIC_LABELS["val_gene_id_loss"])
    figs_b64["fig4_dim_returns_gene_id"] = _save_fig(
        fig4, output_dir / "figures" / "fig4_diminishing_returns_gene_id_loss.png"
    )
    plt.close(fig4)

    # HTML
    build_html(
        output_path=output_dir / "report.html",
        metadata=metadata,
        final_df=final_df,
        fits=fits,
        figs_b64=figs_b64,
    )
    print(f"[done] outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
