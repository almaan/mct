"""Ablation-study evaluation script.

Loads each ablation training run's checkpoint and evaluates it on:
  1. Test loss (fixed held-out test set from the ablation split JSON) —
     total, gene_id, tissue, rec, rec_masked.
  2. Zero-shot ligand-receptor prediction on CellChatDB using cosine
     similarity of gene embeddings extracted from the classifier head
     (identical methodology to
     tutorials/benchmarking_ccc_rl_cellchat.ipynb).

Emits:
  - test_metrics.csv                 one row per (fraction, ckpt_kind)
  - lr_benchmark.csv                 one row per (fraction, ckpt_kind, seed, annotation)
  - lr_benchmark_summary.csv         mean +- std across seeds
  - report.html                      self-contained (base64 figures + inline tables)
  - run_metadata.json                CLI args + versions
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel  # noqa: E402
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule  # noqa: E402


# --------------------------------------------------------------------- #
# Notebook helpers copied verbatim from benchmarking_ccc_rl_cellchat.ipynb
# (see cells d1fdb748, a8b03ff9, 2a28b984, 4e8579c4). Kept literal so the
# ablation numbers are directly comparable to the paper's reported numbers.
# --------------------------------------------------------------------- #


def sample_negatives_w_complexes(
    positives_df: pd.DataFrame,
    potential_genes: set,
    num_negative_per_positive: int = 2,
    seed: int = 1,
) -> pd.DataFrame:
    """Sample negatives by swapping one gene in a partner complex for a random gene."""
    np.random.seed(seed)
    positives_df = positives_df.copy()
    positives_df["seed"] = seed
    negative_samples = []

    for _, row in tqdm(positives_df.iterrows(), total=len(positives_df), leave=False):
        partner_a = row["partner_a"]  # list of genes
        partner_b = row["partner_b"]  # list of genes

        n_negatives_a = min(len(partner_a), num_negative_per_positive // 2)
        if n_negatives_a > 0 and len(partner_a) > 0:
            genes_to_replace_in_a = np.random.choice(
                len(partner_a), size=min(n_negatives_a, len(partner_a)), replace=False
            )
            for idx_to_replace in genes_to_replace_in_a:
                excluded_genes = set(partner_b) | {partner_a[idx_to_replace]}
                available_genes = list(potential_genes - excluded_genes)
                if available_genes:
                    replacement_gene = np.random.choice(available_genes)
                    new_partner_a = partner_a.copy()
                    new_partner_a[idx_to_replace] = replacement_gene
                    negative_samples.append({
                        "partner_a": new_partner_a,
                        "partner_b": partner_b,
                        "annotation": row["annotation"],
                        "label": 0,
                        "seed": seed,
                    })

        n_negatives_b = min(len(partner_b), num_negative_per_positive // 2)
        if n_negatives_b > 0 and len(partner_b) > 0:
            genes_to_replace_in_b = np.random.choice(
                len(partner_b), size=min(n_negatives_b, len(partner_b)), replace=False
            )
            for idx_to_replace in genes_to_replace_in_b:
                excluded_genes = set(partner_a) | {partner_b[idx_to_replace]}
                available_genes = list(potential_genes - excluded_genes)
                if available_genes:
                    replacement_gene = np.random.choice(available_genes)
                    new_partner_b = partner_b.copy()
                    new_partner_b[idx_to_replace] = replacement_gene
                    negative_samples.append({
                        "partner_a": partner_a,
                        "partner_b": new_partner_b,
                        "annotation": row["annotation"],
                        "label": 0,
                        "seed": seed,
                    })

    negative_df = pd.DataFrame(negative_samples)
    if len(negative_df) > 0:
        negative_df["gene_pair"] = negative_df.apply(
            lambda row: "_".join(sorted(["_".join(row["partner_a"]), "_".join(row["partner_b"])])),
            axis=1,
        )
    else:
        negative_df["gene_pair"] = []

    positives_df["gene_pair"] = positives_df.apply(
        lambda row: "_".join(sorted(["_".join(row["partner_a"]), "_".join(row["partner_b"])])),
        axis=1,
    )

    df = pd.concat([positives_df, negative_df], ignore_index=True)
    df = df.drop_duplicates(subset=["gene_pair", "seed"], keep="first")
    return df


def compute_cosine_similarity_complexes(
    row: dict,
    gene_symbol2emb: dict,
) -> float | None:
    """Mean-pool constituent gene embeddings per partner, then cosine sim."""
    try:
        lig_genes = [gene_symbol2emb[item] for item in row["partner_a"] if item in gene_symbol2emb]
        rec_genes = [gene_symbol2emb[item] for item in row["partner_b"] if item in gene_symbol2emb]
        if len(lig_genes) == 0 or len(rec_genes) == 0:
            return None
        lig_emb = torch.stack(lig_genes).mean(dim=0)
        rec_emb = torch.stack(rec_genes).mean(dim=0)
        cs = cosine_similarity(lig_emb, rec_emb, dim=0).item()
        return (cs + 1) / 2  # normalise to [0, 1]
    except Exception:
        return None


def mean_roc_across_seeds(
    df: pd.DataFrame, score_col: str, seeds: list, fpr_grid: np.ndarray
):
    aucs = []
    tprs = []
    for seed in seeds:
        df_seed = df[df["seed"] == seed].dropna(subset=[score_col])
        if df_seed["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df_seed["label"], df_seed[score_col])
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    if not tprs:
        return None, None, float("nan"), float("nan")
    tprs = np.asarray(tprs)
    return tprs.mean(axis=0), tprs.std(axis=0), float(np.mean(aucs)), float(np.std(aucs))


def mean_pr_across_seeds(
    df: pd.DataFrame, score_col: str, seeds: list, recall_grid: np.ndarray
):
    aps = []
    precisions = []
    for seed in seeds:
        df_seed = df[df["seed"] == seed].dropna(subset=[score_col])
        if df_seed["label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(df_seed["label"], df_seed[score_col])
        aps.append(average_precision_score(df_seed["label"], df_seed[score_col]))
        precision = precision[::-1]
        recall = recall[::-1]
        precisions.append(np.interp(recall_grid, recall, precision))
    if not precisions:
        return None, None, float("nan"), float("nan")
    precisions = np.asarray(precisions)
    return precisions.mean(axis=0), precisions.std(axis=0), float(np.mean(aps)), float(np.std(aps))


# --------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------- #

BEST_CKPT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)-val_total_loss=([\d.]+)\.ckpt$")


@dataclass
class CkptRef:
    fraction: str
    ckpt_kind: str  # "best" | "last"
    path: Path
    ckpt_step: int
    val_total_loss: float | None


def discover_ckpts(run_dir: Path, fractions: list[str], ckpt_kind: str) -> list[CkptRef]:
    refs: list[CkptRef] = []
    for frac in fractions:
        ckpt_dir = run_dir / f"ablation_paperhp_frac{frac}" / "checkpoints"
        if not ckpt_dir.exists():
            print(f"[WARN] no ckpt dir for frac={frac}: {ckpt_dir}", flush=True)
            continue

        if ckpt_kind == "last":
            last = ckpt_dir / "last.ckpt"
            if not last.exists():
                print(f"[WARN] no last.ckpt for frac={frac}", flush=True)
                continue
            step = _read_global_step(last)
            refs.append(CkptRef(frac, "last", last, step, None))
            continue

        # "best" = lowest val_total_loss embedded in the filename
        candidates = []
        for p in ckpt_dir.glob("epoch=*-step=*-val_total_loss=*.ckpt"):
            m = BEST_CKPT_PATTERN.search(p.name)
            if not m:
                continue
            _, step, val = m.groups()
            candidates.append((float(val), int(step), p))
        if not candidates:
            print(f"[WARN] no best-val ckpt for frac={frac}", flush=True)
            continue
        candidates.sort()  # ascending val_loss
        val, step, path = candidates[0]
        refs.append(CkptRef(frac, "best", path, step, val))
    return refs


def _read_global_step(ckpt_path: Path) -> int:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return int(ckpt.get("global_step", -1))


# --------------------------------------------------------------------- #
# Test-loss eval
# --------------------------------------------------------------------- #


def eval_test_loss(
    ref: CkptRef,
    input_dir: Path,
    split_file: Path,
    cell_type_map_file: Path,
    device: str,
    limit_test_batches: int | None,
    seed: int,
) -> dict:
    """Load ckpt, restore datamodule + model, iterate test loader, aggregate losses."""
    ckpt = torch.load(str(ref.path), map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["config"]

    cell_type_mapping = None
    if cell_type_map_file.is_file():
        cell_type_mapping = json.loads(cell_type_map_file.read_text())

    print(f"[test-loss] frac={ref.fraction} building datamodule", flush=True)
    dm = CellarrDataModule(
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
        num_workers=4,
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
        train_fraction=float(ref.fraction),
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert dm.test_dataset is not None, "Split file did not produce a test set"

    print(f"[test-loss] frac={ref.fraction} loading model", flush=True)
    pl_model = MulticellularTransformerPLModel(config)
    # Strip compile-related prefixes if the ckpt was saved from torch.compile'd
    # module (state_dict keys may start with "_orig_mod." on newer torch).
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = pl_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[test-loss]   missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    if unexpected:
        print(f"[test-loss]   unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}", flush=True)
    pl_model.eval()
    pl_model.to(device)
    if device == "cuda":
        pl_model.to(torch.bfloat16)

    loader = dm.test_dataloader()
    losses = {
        "test_total_loss": [],
        "test_gene_id_loss": [],
        "test_tissue_loss": [],
        "test_rec_loss": [],
        "test_rec_loss_masked": [],
    }

    n_seen = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"test frac={ref.fraction}", leave=False)):
            if limit_test_batches is not None and i >= limit_test_batches:
                break
            for f in (
                "expression_matrix", "gene_ids", "labels", "is_masked_gene_id",
                "is_masked_gene_expression", "padding_mask", "tissue_idx",
                "tissue_labels", "n_expressed_genes",
            ):
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
            batch_losses = pl_model.compute_loss(batch, out)
            for k in losses:
                v = batch_losses[k.replace("test_", "")].detach().float().cpu().item()
                if np.isfinite(v):
                    losses[k].append(v)
            n_seen += 1

    return {
        "fraction": ref.fraction,
        "ckpt_kind": ref.ckpt_kind,
        "ckpt_step": ref.ckpt_step,
        "n_test_batches": n_seen,
        **{k: float(np.mean(v)) if v else float("nan") for k, v in losses.items()},
    }


# --------------------------------------------------------------------- #
# L-R zero-shot benchmark
# --------------------------------------------------------------------- #


def load_cellchatdb(cellchat_csv: Path) -> pd.DataFrame:
    """Preprocess CellChatDB per notebook: keep single-partner rows, split by ' - ' and '+'."""
    df = pd.read_csv(str(cellchat_csv))
    df = df[["interaction_name_2", "annotation"]].copy()
    df["is_multi_unit_complex"] = df["interaction_name_2"].apply(lambda x: "+" in x)
    df = df[~df["is_multi_unit_complex"]].copy()

    df["partner_a"] = df["interaction_name_2"].apply(
        lambda x: x.split(" - ")[0].replace("(", "").replace(")", "")
    )
    df["partner_b"] = df["interaction_name_2"].apply(
        lambda x: x.split(" - ")[-1].replace("(", "").replace(")", "")
    )
    df["partner_a"] = df["partner_a"].apply(lambda x: x.split("+"))
    df["partner_b"] = df["partner_b"].apply(lambda x: x.split("+"))
    df["label"] = 1
    return df


def extract_gene_embeddings(ckpt_path: Path, gene_symbol2idx: dict) -> dict:
    """Extract classifier-decoder weight and index into a symbol -> tensor dict."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    key_candidates = [
        "model.gene_id_classifier.decoder.weight",
        "_orig_mod.model.gene_id_classifier.decoder.weight",
    ]
    weight = None
    for k in key_candidates:
        if k in sd:
            weight = sd[k]
            break
    assert weight is not None, f"gene_id_classifier.decoder.weight missing in ckpt {ckpt_path}"
    return {sym: weight[idx] for sym, idx in gene_symbol2idx.items() if idx < weight.shape[0]}


def eval_lr_for_ckpt(
    ref: CkptRef,
    cellchat_df: pd.DataFrame,
    gene_symbol2idx: dict,
    seeds: list[int],
    num_negative: int,
    fpr_grid: np.ndarray,
    recall_grid: np.ndarray,
) -> list[dict]:
    """Run L-R benchmark for one ckpt, return long-form rows (per seed, per annotation)."""
    print(f"[lr] frac={ref.fraction} extracting embeddings", flush=True)
    gene_symbol2emb = extract_gene_embeddings(ref.path, gene_symbol2idx)
    potential_genes = set(gene_symbol2emb)

    # Build negatives + scores per seed
    per_seed_dfs = []
    for seed in seeds:
        print(f"[lr] frac={ref.fraction} seed={seed} sampling negatives", flush=True)
        with_negs = sample_negatives_w_complexes(
            cellchat_df[["partner_a", "partner_b", "annotation", "label"]].copy(),
            potential_genes,
            num_negative_per_positive=num_negative,
            seed=seed,
        )
        with_negs["mct_cs"] = with_negs.apply(
            lambda row: compute_cosine_similarity_complexes(row, gene_symbol2emb), axis=1
        )
        per_seed_dfs.append(with_negs)

    combined = pd.concat(per_seed_dfs, ignore_index=True)

    rows = []
    # Overall
    _, _, auroc_mean, auroc_std = mean_roc_across_seeds(combined, "mct_cs", seeds, fpr_grid)
    _, _, auprc_mean, auprc_std = mean_pr_across_seeds(combined, "mct_cs", seeds, recall_grid)
    for seed in seeds:
        seed_df = combined[combined["seed"] == seed].dropna(subset=["mct_cs"])
        if seed_df["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(seed_df["label"], seed_df["mct_cs"])
        prec, rec, _ = precision_recall_curve(seed_df["label"], seed_df["mct_cs"])
        rows.append({
            "fraction": ref.fraction,
            "ckpt_kind": ref.ckpt_kind,
            "ckpt_step": ref.ckpt_step,
            "seed": seed,
            "annotation": "all",
            "auroc": float(auc(fpr, tpr)),
            "auprc": float(average_precision_score(seed_df["label"], seed_df["mct_cs"])),
            "n_pos": int((seed_df["label"] == 1).sum()),
            "n_neg": int((seed_df["label"] == 0).sum()),
        })

    # Per annotation
    for annotation in sorted(combined["annotation"].dropna().unique()):
        sub = combined[combined["annotation"] == annotation]
        for seed in seeds:
            seed_df = sub[sub["seed"] == seed].dropna(subset=["mct_cs"])
            if seed_df["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(seed_df["label"], seed_df["mct_cs"])
            rows.append({
                "fraction": ref.fraction,
                "ckpt_kind": ref.ckpt_kind,
                "ckpt_step": ref.ckpt_step,
                "seed": seed,
                "annotation": annotation,
                "auroc": float(auc(fpr, tpr)),
                "auprc": float(average_precision_score(seed_df["label"], seed_df["mct_cs"])),
                "n_pos": int((seed_df["label"] == 1).sum()),
                "n_neg": int((seed_df["label"] == 0).sum()),
            })
    return rows


def summarise_lr(long_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        long_df.groupby(["fraction", "ckpt_kind", "ckpt_step", "annotation"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            n_seeds=("seed", "nunique"),
            n_pos=("n_pos", "mean"),
            n_neg=("n_neg", "mean"),
        )
        .reset_index()
    )
    for col in ("auroc_std", "auprc_std"):
        agg[col] = agg[col].fillna(0.0)
    return agg


# --------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------- #


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fraction_colors(fractions: list[str]) -> dict:
    # Monotonic viridis: smaller frac -> lighter
    import matplotlib.cm as cm
    fs = sorted({float(f) for f in fractions})
    n = len(fs)
    cmap = cm.get_cmap("viridis")
    frac2color = {f: cmap(0.15 + 0.7 * (i / max(n - 1, 1))) for i, f in enumerate(fs)}
    return {f: frac2color[float(f)] for f in fractions}


def _plot_test_losses(test_df: pd.DataFrame, colors: dict):
    import matplotlib.pyplot as plt

    metrics = [
        ("test_total_loss", "Total loss"),
        ("test_gene_id_loss", "Gene-ID loss"),
        ("test_rec_loss_masked", "Reconstruction loss (masked)"),
        ("test_tissue_loss", "Tissue loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=110)
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metrics):
        if col not in test_df.columns or test_df[col].isna().all():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            ax.set_title(title)
            continue
        d = test_df.sort_values("fraction")
        xs = [float(f) for f in d["fraction"]]
        ys = d[col].values
        cols = [colors[f] for f in d["fraction"]]
        ax.bar(range(len(xs)), ys, color=cols)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.2f}" for x in xs])
        ax.set_xlabel("Train fraction")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Test-set losses by training-data fraction", y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


def _plot_lr_overall(summary_df: pd.DataFrame, colors: dict):
    import matplotlib.pyplot as plt

    overall = summary_df[summary_df["annotation"] == "all"].sort_values("fraction")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)
    for ax, (metric_mean, metric_std, ylabel) in zip(
        (ax1, ax2),
        (
            ("auroc_mean", "auroc_std", "AUROC"),
            ("auprc_mean", "auprc_std", "AUPRC"),
        ),
    ):
        xs = [float(f) for f in overall["fraction"]]
        ys = overall[metric_mean].values
        errs = overall[metric_std].values
        cols = [colors[f] for f in overall["fraction"]]
        ax.bar(range(len(xs)), ys, yerr=errs, color=cols, capsize=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.2f}" for x in xs])
        ax.set_xlabel("Train fraction")
        ax.set_ylabel(f"{ylabel} (mean ± std across seeds)")
        ax.set_title(f"L-R zero-shot — {ylabel}")
        ax.axhline(0.5 if ylabel == "AUROC" else float("nan"), color="k", ls="--", lw=1, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_lr_per_annotation(summary_df: pd.DataFrame, colors: dict):
    import matplotlib.pyplot as plt

    annots = sorted(a for a in summary_df["annotation"].unique() if a != "all")
    if not annots:
        return None
    fig, axes = plt.subplots(len(annots), 2, figsize=(11, 3.8 * len(annots)), dpi=110, squeeze=False)
    for row, annotation in enumerate(annots):
        sub = summary_df[summary_df["annotation"] == annotation].sort_values("fraction")
        for col, (mean_col, std_col, ylabel) in enumerate((
            ("auroc_mean", "auroc_std", "AUROC"),
            ("auprc_mean", "auprc_std", "AUPRC"),
        )):
            ax = axes[row, col]
            xs = [float(f) for f in sub["fraction"]]
            ys = sub[mean_col].values
            errs = sub[std_col].values
            cols = [colors[f] for f in sub["fraction"]]
            ax.bar(range(len(xs)), ys, yerr=errs, color=cols, capsize=4)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([f"{x:.2f}" for x in xs])
            ax.set_xlabel("Train fraction")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{annotation} — {ylabel}")
            if ylabel == "AUROC":
                ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)
            ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def build_html_report(
    output_path: Path,
    metadata: dict,
    test_df: pd.DataFrame,
    lr_summary_df: pd.DataFrame,
    fractions: list[str],
) -> None:
    colors = _fraction_colors(fractions)
    fig_test = _plot_test_losses(test_df, colors) if not test_df.empty else None
    fig_lr = _plot_lr_overall(lr_summary_df, colors)
    fig_lr_ann = _plot_lr_per_annotation(lr_summary_df, colors)

    def img_tag(fig, alt):
        if fig is None:
            return f"<p><em>{alt}: no data</em></p>"
        b64 = _fig_to_base64(fig)
        return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'

    def df_html(df):
        return df.to_html(index=False, float_format="%.4f", classes="tbl")

    css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1200px; margin: 24px auto; padding: 0 16px; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
        table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
        table.tbl th { background: #f4f4f4; }
        .meta { background: #f8f8f8; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; }
        .meta pre { margin: 0; white-space: pre-wrap; }
    """

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MCT ablation report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>MCT training-data ablation — evaluation report</h1>",
        "<div class='meta'><pre>",
        json.dumps(metadata, indent=2, default=str),
        "</pre></div>",
    ]

    parts.append("<h2>1. Test-set losses</h2>")
    parts.append(img_tag(fig_test, "Test losses by fraction"))
    if not test_df.empty:
        parts.append(df_html(test_df))
    else:
        parts.append("<p><em>Test loss evaluation skipped or produced no rows.</em></p>")

    parts.append("<h2>2. L-R zero-shot benchmark — overall</h2>")
    parts.append(img_tag(fig_lr, "L-R overall AUROC/AUPRC"))
    overall = lr_summary_df[lr_summary_df["annotation"] == "all"].sort_values("fraction")
    parts.append(df_html(overall))

    parts.append("<h2>3. L-R zero-shot benchmark — per annotation</h2>")
    parts.append(img_tag(fig_lr_ann, "L-R per annotation"))
    per_annot = lr_summary_df[lr_summary_df["annotation"] != "all"].sort_values(["annotation", "fraction"])
    parts.append(df_html(per_annot))

    parts.append("</body></html>")
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
    p.add_argument("--output-dir", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/ablation_eval"))
    p.add_argument("--input-dir", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr"))
    p.add_argument("--split-file", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json"))
    p.add_argument("--cellchat-csv", type=Path, default=REPO / "tutorials/data/cellchatdb.csv")
    p.add_argument("--gene-symbol2idx", type=Path, default=REPO / "tutorials/data/gene_symbol2idx.json")
    p.add_argument("--cell-type-map", type=Path, default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/cell_type_map.json"))
    p.add_argument("--fractions", nargs="+", default=["0.10", "0.25", "0.50", "1.00"])
    p.add_argument("--ckpt-kind", choices=("best", "last"), default="best")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-negative", type=int, default=2)
    p.add_argument("--limit-test-batches", type=int, default=None)
    p.add_argument("--skip-test-loss", action="store_true")
    p.add_argument("--skip-lr", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu", flush=True)
        args.device = "cpu"

    metadata = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(REPO),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    refs = discover_ckpts(args.run_dir, args.fractions, args.ckpt_kind)
    print(f"Discovered {len(refs)} ckpts:", flush=True)
    for r in refs:
        print(f"  frac={r.fraction} kind={r.ckpt_kind} step={r.ckpt_step} val={r.val_total_loss} path={r.path}", flush=True)
    if not refs:
        raise SystemExit("No checkpoints discovered.")

    # ---------- test-loss ----------
    test_rows: list[dict] = []
    if not args.skip_test_loss:
        for r in refs:
            try:
                row = eval_test_loss(
                    r,
                    input_dir=args.input_dir,
                    split_file=args.split_file,
                    cell_type_map_file=args.cell_type_map,
                    device=args.device,
                    limit_test_batches=args.limit_test_batches,
                    seed=args.seed,
                )
                test_rows.append(row)
            except Exception as e:
                print(f"[ERR] test-loss frac={r.fraction} failed: {e}", flush=True)
    test_df = pd.DataFrame(test_rows)
    if not test_df.empty:
        test_df.to_csv(args.output_dir / "test_metrics.csv", index=False)

    # ---------- L-R benchmark ----------
    lr_rows: list[dict] = []
    if not args.skip_lr:
        gene_symbol2idx = json.loads(args.gene_symbol2idx.read_text())
        cellchat_df = load_cellchatdb(args.cellchat_csv)
        fpr_grid = np.linspace(0.0, 1.0, 200)
        recall_grid = np.linspace(0.0, 1.0, 200)
        for r in refs:
            try:
                lr_rows.extend(eval_lr_for_ckpt(
                    r, cellchat_df, gene_symbol2idx,
                    seeds=args.seeds, num_negative=args.num_negative,
                    fpr_grid=fpr_grid, recall_grid=recall_grid,
                ))
            except Exception as e:
                print(f"[ERR] L-R frac={r.fraction} failed: {e}", flush=True)

    lr_df = pd.DataFrame(lr_rows)
    lr_summary = pd.DataFrame()
    if not lr_df.empty:
        lr_df.to_csv(args.output_dir / "lr_benchmark.csv", index=False)
        lr_summary = summarise_lr(lr_df)
        lr_summary.to_csv(args.output_dir / "lr_benchmark_summary.csv", index=False)

    # ---------- HTML ----------
    build_html_report(
        args.output_dir / "report.html",
        metadata=metadata,
        test_df=test_df,
        lr_summary_df=lr_summary if not lr_summary.empty else pd.DataFrame(
            columns=["fraction", "ckpt_kind", "ckpt_step", "annotation",
                     "auroc_mean", "auroc_std", "auprc_mean", "auprc_std", "n_seeds"]
        ),
        fractions=args.fractions,
    )
    print(f"[done] outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
