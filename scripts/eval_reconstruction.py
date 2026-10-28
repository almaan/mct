"""Gene-expression reconstruction analysis for the ablation ckpts.

For each ckpt discovered under <run-dir>/ablation_paperhp_frac<X>/, grabs
a small number of test-set batches, runs the model forward, and compares
predicted vs actual expression (the same tensor that
``compute_loss.rec_loss`` uses under the hood).

Reports:
  - Per-batch MSE
  - Per-gene Pearson correlation across the cells in each batch
  - Aggregate scatter of predicted vs actual expression across sampled cells

Outputs:
  - reconstruction_per_batch.csv       one row per (frac, batch_idx)
  - reconstruction_per_gene.csv        one row per (frac, batch_idx, gene_idx)
  - reconstruction_report.html         self-contained (scatter + summary)
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel  # noqa: E402
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule  # noqa: E402


BEST_CKPT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)-val_total_loss=([\d.]+)\.ckpt$")


@dataclass
class CkptRef:
    fraction: str
    ckpt_kind: str
    path: Path
    ckpt_step: int


def discover_ckpts(run_dir: Path, fractions: list[str], ckpt_kind: str) -> list[CkptRef]:
    refs: list[CkptRef] = []
    for frac in fractions:
        ckpt_dir = run_dir / f"ablation_paperhp_frac{frac}" / "checkpoints"
        if not ckpt_dir.exists():
            print(f"[WARN] no ckpt dir for frac={frac}", flush=True)
            continue
        if ckpt_kind == "last":
            p = ckpt_dir / "last.ckpt"
            if not p.exists():
                continue
            step = int(torch.load(str(p), map_location="cpu", weights_only=False).get("global_step", -1))
            refs.append(CkptRef(frac, "last", p, step))
        else:
            candidates = []
            for p in ckpt_dir.glob("epoch=*-step=*-val_total_loss=*.ckpt"):
                m = BEST_CKPT_PATTERN.search(p.name)
                if not m:
                    continue
                _, step, val = m.groups()
                candidates.append((float(val), int(step), p))
            if not candidates:
                continue
            candidates.sort()
            val, step, path = candidates[0]
            refs.append(CkptRef(frac, "best", path, step))
    return refs


def build_datamodule(config, input_dir: Path, split_file: Path, cell_type_map_file: Path,
                     fraction: float, seed: int) -> CellarrDataModule:
    cell_type_mapping = None
    if cell_type_map_file.is_file():
        cell_type_mapping = json.loads(cell_type_map_file.read_text())
    return CellarrDataModule(
        cell_tdb_uri=str(input_dir / "cell_metadata"),
        gene_tdb_uri=str(input_dir / "gene_annotation"),
        counts_tdb_uri=str(input_dir / "counts"),
        val_samples=None,
        label_column="cell_type",
        study_column="study",
        sample_column="sample",
        tissue_column="tissue",
        batch_size=config.batch_size,
        sample_size=config.n_cells_bag,
        num_workers=2,
        min_sample_size=256,
        min_cell_type_size=0.01,
        agg_cell_type_gex=config.agg_cell_type_gex,
        sample_sampling="weighted",
        cell_sampling="stratified",
        pad_value_expression=0.0,
        mask_prob=config.mask_prob_gene_id,
        mask_tissue_prob=0.25,
        perc_min_expressed_cells_per_gene=0.01,
        max_seq_len=config.max_seq_len,
        lognorm=True,
        target_sum=1e4,
        random_seed=seed,
        cell_type_mapping=cell_type_mapping,
        split_file=str(split_file),
        train_fraction=fraction,
        prefetch_factor=2,
        persistent_workers=False,
    )


def per_gene_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-gene Pearson correlation across cells within a batch.

    pred, true: (n_cells, n_genes)
    Returns:    (n_genes,) with NaN where variance is 0.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p * t).sum(axis=0) / denom
    r[~np.isfinite(r)] = np.nan
    return r


def eval_reconstruction(
    ref: CkptRef, input_dir: Path, split_file: Path, cell_type_map_file: Path,
    n_batches: int, device: str, seed: int,
) -> tuple[list[dict], list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    print(f"[recon] frac={ref.fraction} loading ckpt", flush=True)
    ckpt = torch.load(str(ref.path), map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]

    dm = build_datamodule(config, input_dir, split_file, cell_type_map_file,
                          fraction=float(ref.fraction), seed=seed)
    assert dm.test_dataset is not None

    pl_model = MulticellularTransformerPLModel(config)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
    pl_model.load_state_dict(sd, strict=False)
    pl_model.eval().to(device)
    if device == "cuda":
        pl_model.to(torch.bfloat16)

    loader = dm.test_dataloader()
    per_batch_rows: list[dict] = []
    per_gene_rows: list[dict] = []
    scatter_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            # Move to device + dtype
            for f in ("expression_matrix", "gene_ids", "labels",
                      "is_masked_gene_id", "is_masked_gene_expression",
                      "padding_mask", "tissue_idx", "tissue_labels", "n_expressed_genes"):
                v = getattr(batch, f, None)
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point() and device == "cuda":
                        v = v.to(device=device, dtype=torch.bfloat16)
                    else:
                        v = v.to(device=device)
                    setattr(batch, f, v)

            out = pl_model.model(
                expression_matrix=batch.expression_matrix,
                gene_ids=batch.gene_ids,
                is_masked_gene_expression=batch.is_masked_gene_expression,
                padding_mask=batch.padding_mask,
                tissue=batch.tissue_idx,
            )
            # Shapes: gex_reconstruction and expression_matrix are
            # (B, seq_len, n_cells_bag). Iterate per bag.
            true = batch.expression_matrix.detach().float().cpu().numpy()
            pred = out.gex_reconstruction.detach().float().cpu().numpy()
            gene_ids = batch.gene_ids.detach().cpu().numpy()

            for bag_idx in range(true.shape[0]):
                # Trim to actual seq_len (padding mask masks out pad tokens)
                pad = batch.padding_mask[bag_idx].detach().cpu().numpy().astype(bool)
                valid = ~pad
                true_bag = true[bag_idx][valid]          # (seq, n_cells_bag)
                pred_bag = pred[bag_idx][valid]
                gids = gene_ids[bag_idx][valid]

                mse = float(np.mean((pred_bag - true_bag) ** 2))
                # Pearson per gene across cells
                pearson_by_gene = per_gene_pearson(pred_bag.T, true_bag.T)  # cells as rows
                pearson_valid = pearson_by_gene[np.isfinite(pearson_by_gene)]
                mean_r = float(np.nanmean(pearson_by_gene)) if pearson_valid.size else float("nan")
                median_r = float(np.nanmedian(pearson_by_gene)) if pearson_valid.size else float("nan")
                frac_pos = float((pearson_valid > 0).mean()) if pearson_valid.size else float("nan")

                per_batch_rows.append({
                    "fraction": ref.fraction,
                    "ckpt_kind": ref.ckpt_kind,
                    "ckpt_step": ref.ckpt_step,
                    "batch_idx": i,
                    "bag_idx_in_batch": bag_idx,
                    "n_genes_in_bag": int(true_bag.shape[0]),
                    "n_cells_in_bag": int(true_bag.shape[1]),
                    "mse": mse,
                    "pearson_mean_across_genes": mean_r,
                    "pearson_median_across_genes": median_r,
                    "frac_genes_with_positive_r": frac_pos,
                })
                # Store one gene-level row per bag (sample of first 500 genes to keep CSV manageable)
                keep = min(500, len(gids))
                for gj in range(keep):
                    per_gene_rows.append({
                        "fraction": ref.fraction,
                        "ckpt_kind": ref.ckpt_kind,
                        "ckpt_step": ref.ckpt_step,
                        "batch_idx": i,
                        "bag_idx_in_batch": bag_idx,
                        "gene_id": int(gids[gj]),
                        "pearson_r": float(pearson_by_gene[gj]),
                        "gene_mean_true": float(true_bag[gj].mean()),
                        "gene_mean_pred": float(pred_bag[gj].mean()),
                    })
                # First bag of first batch: keep flattened arrays for scatter
                if i == 0 and bag_idx == 0:
                    scatter_pairs.append((true_bag.reshape(-1), pred_bag.reshape(-1)))

    return per_batch_rows, per_gene_rows, scatter_pairs


# --------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------- #


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fraction_colors(fractions):
    import matplotlib.cm as cm
    fs = sorted({float(f) for f in fractions})
    n = len(fs)
    cmap = cm.get_cmap("viridis")
    frac2color = {f: cmap(0.15 + 0.7 * (i / max(n - 1, 1))) for i, f in enumerate(fs)}
    return {f: frac2color[float(f)] for f in fractions}


def plot_summary(per_batch_df: pd.DataFrame, colors: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    metrics = [
        ("mse", "MSE (predicted vs actual)"),
        ("pearson_median_across_genes", "Median per-gene Pearson r"),
        ("frac_genes_with_positive_r", "Fraction of genes with r > 0"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        agg = per_batch_df.groupby("fraction")[col].agg(["mean", "std"]).reset_index().sort_values("fraction")
        xs = np.arange(len(agg))
        ys = agg["mean"].values
        errs = agg["std"].values
        cs = [colors[f] for f in agg["fraction"]]
        ax.bar(xs, ys, yerr=errs, color=cs, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{float(f):.2f}" for f in agg["fraction"]])
        ax.set_xlabel("Train fraction")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Reconstruction quality across fractions (mean over sampled batches)", y=1.02)
    fig.tight_layout()
    return fig


def plot_scatter(scatter_data: dict, colors: dict):
    """One scatter per fraction: predicted vs actual over the first bag."""
    import matplotlib.pyplot as plt
    fs = sorted(scatter_data.keys(), key=float)
    fig, axes = plt.subplots(1, len(fs), figsize=(4.5 * len(fs), 4.5), dpi=110, squeeze=False)
    for ax, frac in zip(axes[0], fs):
        pairs = scatter_data[frac]
        if not pairs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        true_vals, pred_vals = pairs[0]
        # Subsample to keep the PNG small
        rng = np.random.default_rng(0)
        idx = rng.choice(len(true_vals), size=min(20000, len(true_vals)), replace=False)
        ax.scatter(true_vals[idx], pred_vals[idx], s=3, alpha=0.15, color=colors[frac])
        lim = max(true_vals.max(), pred_vals.max())
        ax.plot([0, lim], [0, lim], color="black", ls="--", lw=1)
        ax.set_xlabel("True log-normalised expression")
        ax.set_ylabel("Predicted")
        ax.set_title(f"frac={frac}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Predicted vs actual expression (first test bag per fraction)", y=1.02)
    fig.tight_layout()
    return fig


def plot_pearson_hist(per_gene_df: pd.DataFrame, colors: dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    for frac, sub in per_gene_df.groupby("fraction"):
        vals = sub["pearson_r"].dropna().values
        ax.hist(vals, bins=50, range=(-0.5, 1.0), alpha=0.45,
                color=colors[frac], label=f"frac={frac} (n={len(vals)})")
    ax.set_xlabel("Per-gene Pearson r (across cells within bag)")
    ax.set_ylabel("Gene count")
    ax.set_title("Distribution of per-gene reconstruction correlation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------- #


def build_html(output_path: Path, metadata: dict, per_batch_df: pd.DataFrame,
               per_gene_df: pd.DataFrame, fig_summary_b64: str,
               fig_scatter_b64: str, fig_hist_b64: str):
    css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1300px; margin: 24px auto; padding: 0 16px; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
        table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        table.tbl th { background: #f4f4f4; }
        .meta { background: #f8f8f8; padding: 8px 12px; border-radius: 4px;
                font-family: monospace; font-size: 12px; }
        .meta pre { margin: 0; white-space: pre-wrap; }
    """
    def img(b64, alt): return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'
    def dfhtml(d): return d.to_html(index=False, float_format="%.4f", classes="tbl")

    summary = per_batch_df.groupby("fraction").agg(
        n_bags=("mse", "count"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        pearson_median=("pearson_median_across_genes", "mean"),
        frac_pos_r=("frac_genes_with_positive_r", "mean"),
    ).reset_index().sort_values("fraction")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MCT ablation — reconstruction report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>MCT ablation — gene-expression reconstruction</h1>",
        "<div class='meta'><pre>", json.dumps(metadata, indent=2, default=str), "</pre></div>",
        "<h2>1. Summary across fractions</h2>",
        img(fig_summary_b64, "Reconstruction summary"),
        dfhtml(summary),
        "<h2>2. Per-gene Pearson r distribution</h2>",
        img(fig_hist_b64, "Per-gene Pearson histogram"),
        "<h2>3. Predicted vs actual scatter (first test bag)</h2>",
        img(fig_scatter_b64, "True vs predicted scatter"),
        "<h2>4. Per-bag detail</h2>",
        dfhtml(per_batch_df.sort_values(["fraction", "batch_idx", "bag_idx_in_batch"])),
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
    p.add_argument("--run-dir", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/runs"))
    p.add_argument("--output-dir", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/ablation_recon"))
    p.add_argument("--input-dir", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr"))
    p.add_argument("--split-file", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json"))
    p.add_argument("--cell-type-map", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/cell_type_map.json"))
    p.add_argument("--fractions", nargs="+", default=["0.10", "0.25", "0.50", "1.00"])
    p.add_argument("--ckpt-kind", choices=("best", "last"), default="last")
    p.add_argument("--n-batches", type=int, default=4,
                   help="Number of test batches to run through the model per checkpoint.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable, falling back to cpu", flush=True)
        args.device = "cpu"

    metadata = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(REPO),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    refs = discover_ckpts(args.run_dir, args.fractions, args.ckpt_kind)
    print(f"Discovered {len(refs)} ckpts:", flush=True)
    for r in refs:
        print(f"  frac={r.fraction} kind={r.ckpt_kind} step={r.ckpt_step}", flush=True)
    if not refs:
        raise SystemExit("no checkpoints found")

    per_batch_rows: list[dict] = []
    per_gene_rows: list[dict] = []
    scatter_data: dict = {}
    for r in refs:
        try:
            pb, pg, sc = eval_reconstruction(
                r, args.input_dir, args.split_file, args.cell_type_map,
                n_batches=args.n_batches, device=args.device, seed=args.seed,
            )
            per_batch_rows.extend(pb)
            per_gene_rows.extend(pg)
            scatter_data[r.fraction] = sc
        except Exception as e:
            print(f"[ERR] frac={r.fraction}: {e}", flush=True)
            scatter_data[r.fraction] = []

    per_batch_df = pd.DataFrame(per_batch_rows)
    per_gene_df = pd.DataFrame(per_gene_rows)
    per_batch_df.to_csv(args.output_dir / "reconstruction_per_batch.csv", index=False)
    per_gene_df.to_csv(args.output_dir / "reconstruction_per_gene.csv", index=False)

    colors = _fraction_colors([r.fraction for r in refs])
    fig_summary = plot_summary(per_batch_df, colors)
    fig_scatter = plot_scatter(scatter_data, colors)
    fig_hist = plot_pearson_hist(per_gene_df, colors)

    build_html(
        args.output_dir / "reconstruction_report.html",
        metadata=metadata,
        per_batch_df=per_batch_df,
        per_gene_df=per_gene_df,
        fig_summary_b64=_fig_to_base64(fig_summary),
        fig_scatter_b64=_fig_to_base64(fig_scatter),
        fig_hist_b64=_fig_to_base64(fig_hist),
    )
    print(f"[done] outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
