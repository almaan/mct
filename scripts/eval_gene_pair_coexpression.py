"""Gene-pair co-expression baseline report.

Scores each (gene1, gene2) pair from a positives+negatives CSV by two baselines
and reports AUROC / AUPRC in the same style as
``rsc/benchmarking_ccc_rl_cellchat.ipynb``:

  1. **cell** — Pearson correlation of log-normalized expression across
     individual cells drawn from the cellarr training pool (the same
     TileDB store + ablation split used for the scaling analysis).
  2. **sample** — Pearson correlation of per-sample pseudobulks
     (mean of log-normalized expression across a sample's cells) across
     samples in that same training pool.

Both scores are shifted to [0, 1] as (r + 1) / 2 so higher = more
co-expressed. AUROC / AUPRC are computed per seed exactly like the
notebook, then mean ± std across seeds.

Usage
-----
    python scripts/eval_gene_pair_coexpression.py \
        --pairs-csv /cv/data/braid/andera29/projs/mct/mct_pub/rsc/rl_df_w_negs_with_gene_pairs.csv \
        --cellarr-dir /cv/scratch/u/andera29/mct_pretraining/cellarr \
        --split-file /cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json \
        --train-fraction 1.00 \
        --cells-per-sample 128 \
        --output-dir /cv/data/braid/andera29/projs/mct/mct_pub/outputs/gene_pair_coexpression
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiledb
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


TILEDB_CONFIG = tiledb.Config()
for _k in (
    "sm.compute_concurrency_level",
    "sm.io_concurrency_level",
    "sm.num_async_threads",
    "sm.num_reader_threads",
    "sm.num_tbb_threads",
    "sm.num_writer_threads",
    "vfs.num_threads",
):
    TILEDB_CONFIG[_k] = 1


COLOR_CELL = "#000080"
COLOR_SAMPLE = "#B22222"

ALL_STRATUM = "All pairs"


# --------------------------------------------------------------------- #
# Gene symbol -> counts column index
# --------------------------------------------------------------------- #


def build_symbol2col(cellarr_dir: Path, gene_info_csv: Path) -> dict[str, int]:
    """Map HGNC symbol -> column index in the cellarr counts matrix.

    The counts TileDB is indexed by (cell, gene) integer positions.
    ``gene_annotation`` stores one row per column with ENSG id;
    ``gene_order_info.csv`` provides the ENSG -> symbol mapping.
    """
    with tiledb.open(str(cellarr_dir / "gene_annotation"), "r") as g:
        gene_df = g.query().df[:]
    ensg2col = {ensg: i for i, ensg in enumerate(gene_df["cellarr_gene_index"].tolist())}

    info = pd.read_csv(gene_info_csv)
    symbol2ensg = dict(zip(info["feature_name"].astype(str), info["feature_id"].astype(str)))
    return {sym: ensg2col[ensg] for sym, ensg in symbol2ensg.items() if ensg in ensg2col}


# --------------------------------------------------------------------- #
# CellChatDB annotation
# --------------------------------------------------------------------- #


def annotate_pairs(pairs: pd.DataFrame, cellchatdb_csv: Path) -> pd.DataFrame:
    """Attach CellChatDB `annotation` to every row of `pairs`.

    - Positives: exact join on the sorted (gene1, gene2) pair key.
    - Negatives: no direct CellChatDB entry; each negative in the notebook
      was derived from a positive by keeping one partner intact. Since we
      lost the source-positive back-pointer when the CSV was written, we
      approximate: look up all CellChatDB positives that contain gene1 OR
      gene2 as a partner, and take the plurality annotation. Unresolved
      negatives are labeled "Unknown" and excluded from stratified plots.
    """
    cc = pd.read_csv(cellchatdb_csv)
    cc = cc[~cc["interaction_name_2"].str.contains(r"\+")].copy()
    cc["g1"] = cc["interaction_name_2"].apply(
        lambda x: x.split(" - ")[0].replace("(", "").replace(")", "")
    )
    cc["g2"] = cc["interaction_name_2"].apply(
        lambda x: x.split(" - ")[-1].replace("(", "").replace(")", "")
    )
    cc["sorted_pair"] = cc.apply(
        lambda r: "_".join(sorted([r["g1"], r["g2"]])), axis=1
    )
    pair2ann = dict(zip(cc["sorted_pair"], cc["annotation"]))

    # Per-gene tally of annotations across CellChatDB positives
    from collections import Counter, defaultdict
    gene2ann_counter: dict[str, Counter] = defaultdict(Counter)
    for _, r in cc.iterrows():
        gene2ann_counter[r["g1"]][r["annotation"]] += 1
        gene2ann_counter[r["g2"]][r["annotation"]] += 1

    def gene_pair_key(row):
        return "_".join(sorted([str(row["gene1"]), str(row["gene2"])]))

    def infer_neg_annotation(row):
        c = Counter()
        for g in (row["gene1"], row["gene2"]):
            c.update(gene2ann_counter.get(g, {}))
        if not c:
            return "Unknown"
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return "Unknown"
        return top[0][0]

    pairs = pairs.copy()
    pairs["_key"] = pairs.apply(gene_pair_key, axis=1)
    pos_ann = pairs["_key"].map(pair2ann)
    neg_ann = pairs.apply(infer_neg_annotation, axis=1)
    pairs["annotation"] = np.where(pairs["label"] == 1, pos_ann, neg_ann)
    pairs.drop(columns=["_key"], inplace=True)
    pairs["annotation"] = pairs["annotation"].fillna("Unknown")
    return pairs


# --------------------------------------------------------------------- #
# Fetch counts + log-normalize
# --------------------------------------------------------------------- #


def stream_lognorm_target(
    counts_tdb: tiledb.Array,
    cell_indices: np.ndarray,
    target_cols: np.ndarray,
    n_genes: int,
    chunk: int = 20000,
    target_sum: float = 1e4,
) -> np.ndarray:
    """Stream cells in chunks from TileDB, log-normalize, keep only target columns.

    Never materializes the full (all-cells x all-genes) sparse matrix. Library
    size is computed from each cell's full gene row before the target-column
    slice is taken, so the lognorm matches the datamodule's behavior.
    """
    n_cells = len(cell_indices)
    n_targets = len(target_cols)
    X = np.zeros((n_cells, n_targets), dtype=np.float32)
    for start in range(0, n_cells, chunk):
        idx = np.asarray(cell_indices[start : start + chunk], dtype=np.int64)
        uniq = np.unique(idx)
        res = counts_tdb.multi_index[uniq.tolist(), :]
        row_pos = np.searchsorted(uniq, res["cell_index"])
        m = sparse.coo_matrix(
            (res["data"], (row_pos, res["gene_index"])),
            shape=(len(uniq), n_genes),
        ).tocsr()
        # library size from full row, before column slicing
        lib = np.asarray(m.sum(axis=1)).ravel()
        lib = np.where(lib > 0, lib, 1.0)
        scale = (target_sum / lib).astype(np.float32)
        m = sparse.diags(scale) @ m
        m = m.tocsr()
        m.data = np.log1p(m.data).astype(np.float32)
        gather = np.searchsorted(uniq, idx)
        X[start : start + len(idx), :] = m[gather, :][:, target_cols].toarray()
    return X


# --------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------- #


def pearson_corr_columns(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of columns of X (n, G)."""
    X = X.astype(np.float32, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(Xc, axis=0, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    Xn = Xc / norm
    return Xn.T @ Xn  # (G, G)


def group_mean(X: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-group mean of rows of X and the unique group order."""
    uniq, inv = np.unique(group_ids, return_inverse=True)
    n_groups = len(uniq)
    sums = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, inv, X)
    np.add.at(counts, inv, 1)
    counts = np.maximum(counts, 1)
    return (sums / counts[:, None]).astype(np.float32), uniq


# --------------------------------------------------------------------- #
# Sampling training cells
# --------------------------------------------------------------------- #


def choose_training_cells(
    cell_metadata_uri: Path,
    train_samples: list[str],
    cells_per_sample: int,
    sample_column: str,
    seed: int,
) -> pd.DataFrame:
    """Return a DataFrame (cell_index, sample) of subsampled train cells."""
    with tiledb.open(str(cell_metadata_uri), "r") as c:
        meta = c.query(attrs=[sample_column]).df[:]
    meta = meta.reset_index(drop=False)  # exposes the row index as 'cell_id' domain
    # The cell TileDB's row index IS the counts row index (both share (0..N-1)).
    # After query().df[:] pandas keeps the numeric index; use that.
    cell_idx_col = "index" if "index" in meta.columns else meta.columns[0]
    if cell_idx_col != "index":
        meta = meta.rename(columns={cell_idx_col: "index"})
    train_set = set(train_samples)
    meta = meta[meta[sample_column].isin(train_set)]
    rng = np.random.default_rng(seed)
    parts = []
    for samp, grp in meta.groupby(sample_column, sort=False):
        idxs = grp["index"].to_numpy()
        if len(idxs) > cells_per_sample:
            idxs = rng.choice(idxs, size=cells_per_sample, replace=False)
        parts.append(pd.DataFrame({"cell_index": idxs, "sample": samp}))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #


def score_pairs(pairs_df: pd.DataFrame, corr: np.ndarray, sym2local: dict[str, int]) -> np.ndarray:
    """Return per-row score = (Pearson r + 1) / 2, NaN if a gene is absent."""
    scores = np.full(len(pairs_df), np.nan, dtype=np.float64)
    g1 = pairs_df["gene1"].to_numpy()
    g2 = pairs_df["gene2"].to_numpy()
    for i, (a, b) in enumerate(zip(g1, g2)):
        ia = sym2local.get(a)
        ib = sym2local.get(b)
        if ia is None or ib is None:
            continue
        r = corr[ia, ib]
        if not np.isfinite(r):
            continue
        scores[i] = (float(r) + 1.0) / 2.0
    return scores


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #


def _save_fig(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def mean_roc_across_seeds(df: pd.DataFrame, score_col: str, seeds, fpr_grid: np.ndarray):
    aucs, tprs = [], []
    for s in seeds:
        d = df[df["seed"] == s].dropna(subset=[score_col])
        if d["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(d["label"], d[score_col])
        aucs.append(roc_auc_score(d["label"], d[score_col]))
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    tprs = np.asarray(tprs)
    return tprs.mean(axis=0), tprs.std(axis=0), float(np.mean(aucs)), float(np.std(aucs))


def mean_pr_across_seeds(df: pd.DataFrame, score_col: str, seeds, recall_grid: np.ndarray):
    aps, precisions = [], []
    for s in seeds:
        d = df[df["seed"] == s].dropna(subset=[score_col])
        if d["label"].nunique() < 2:
            continue
        p, r, _ = precision_recall_curve(d["label"], d[score_col])
        aps.append(average_precision_score(d["label"], d[score_col]))
        precisions.append(np.interp(recall_grid, r[::-1], p[::-1]))
    precisions = np.asarray(precisions)
    return precisions.mean(axis=0), precisions.std(axis=0), float(np.mean(aps)), float(np.std(aps))


def plot_roc(df: pd.DataFrame, methods: dict, seeds, title_suffix: str = "") -> plt.Figure:
    fpr_grid = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    for name, (col, color) in methods.items():
        mtpr, stpr, mauc, sauc = mean_roc_across_seeds(df, col, seeds, fpr_grid)
        ax.plot(fpr_grid, mtpr, lw=2, color=color, label=f"{name}: {mauc:.3f} ± {sauc:.3f}")
        ax.fill_between(fpr_grid, np.maximum(mtpr - stpr, 0), np.minimum(mtpr + stpr, 1),
                        color=color, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random: 0.5")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    subtitle = f"{title_suffix} — n_pos={n_pos}, n_neg={n_neg}" if title_suffix else ""
    ax.set_title(f"ROC — gene-pair co-expression baselines\n{subtitle}")
    ax.legend(loc="lower right", frameon=False, title="AUROC (mean ± std across seeds)")
    fig.tight_layout()
    return fig


def plot_pr(df: pd.DataFrame, methods: dict, seeds, title_suffix: str = "") -> plt.Figure:
    recall_grid = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    for name, (col, color) in methods.items():
        mp, sp, map_, sap = mean_pr_across_seeds(df, col, seeds, recall_grid)
        ax.plot(recall_grid, mp, lw=2, color=color, label=f"{name}: {map_:.4f} ± {sap:.4f}")
        ax.fill_between(recall_grid, np.maximum(mp - sp, 0), np.minimum(mp + sp, 1),
                        color=color, alpha=0.15)
    pos_rate = df["label"].mean()
    ax.hlines(pos_rate, xmin=0, xmax=1, colors="k", linestyles="--",
              lw=1.5, label=f"Random: {pos_rate:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    subtitle = f"{title_suffix} — n_pos={n_pos}, n_neg={n_neg}" if title_suffix else ""
    ax.set_title(f"PR — gene-pair co-expression baselines\n{subtitle}")
    ax.legend(loc="upper right", frameon=False, title="AUPRC (mean ± std across seeds)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------- #


def build_html(
    output_path: Path,
    metadata: dict,
    metrics_df: pd.DataFrame,
    coverage: dict,
    figs_b64: dict,
    strata: list[str],
) -> None:
    def img(b64, alt):
        return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'

    def slug(stratum: str) -> str:
        return "all" if stratum == ALL_STRATUM else (
            stratum.lower().replace(" ", "_").replace("-", "_")
        )

    css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
        h2 { margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
        table.tbl { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
        table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; }
        table.tbl th { background: #f4f4f4; }
        .meta { background: #f8f8f8; padding: 10px 14px; border-radius: 4px;
                font-family: monospace; font-size: 12px; }
        .meta pre { margin: 0; white-space: pre-wrap; }
        .caption { color: #555; font-size: 12px; margin: 4px 0 20px 0; }
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Gene-pair co-expression baselines</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Gene-pair co-expression baselines</h1>",
        "<p><strong>Protocol.</strong> Positive and negative gene pairs are read from "
        "<code>rl_df_w_negs_with_gene_pairs.csv</code> — the same construction as in "
        "<code>rsc/benchmarking_ccc_rl_cellchat.ipynb</code>. Each pair is scored by two "
        "unsupervised co-expression baselines computed on the cellarr training pool used "
        "for the scaling ablation:</p>",
        "<ul>",
        "<li><b>Cell co-expression</b> — Pearson correlation of log-normalized expression "
        "across individual cells subsampled from every training sample.</li>",
        "<li><b>Sample co-expression</b> — Pearson correlation of per-sample pseudobulks "
        "(mean of log-normalized expression across each sample's cells) across samples.</li>",
        "</ul>",
        "<p>Both scores are shifted to [0, 1] via <code>(r + 1) / 2</code>. AUROC and AUPRC "
        "are computed per seed (same seeds as the CSV), then averaged.</p>",
        "<div class='meta'><pre>",
        json.dumps({**metadata, "coverage": coverage}, indent=2, default=str),
        "</pre></div>",
        "<h2>Per-seed metrics (stratified by CellChatDB annotation)</h2>",
        "<p>Positive pairs' annotations come from a direct join on CellChatDB. "
        "Negative pairs inherit an annotation by plurality vote over the "
        "CellChatDB positives their two genes participate in; ties and unmatched "
        "negatives are labeled <code>Unknown</code> and excluded from per-annotation "
        "plots (they are still included in <em>All pairs</em>).</p>",
        metrics_df.to_html(index=False, float_format="%.4f", classes="tbl"),
    ]
    for stratum in strata:
        sl = slug(stratum)
        parts += [
            f"<h2>ROC + PR — {stratum}</h2>",
            img(figs_b64[f"roc__{sl}"], f"ROC {stratum}"),
            "<p class='caption'>Mean ROC across seeds; band = ±1 std.</p>",
            img(figs_b64[f"pr__{sl}"], f"PR {stratum}"),
            "<p class='caption'>Mean PR across seeds; band = ±1 std. Dashed line = "
            "positive-class prevalence.</p>",
        ]
    parts += [
        "<h2>Notes</h2>",
        "<ul>",
        f"<li>Cells were drawn from the training pool defined by "
        f"<code>{metadata['split_file']}</code> at fraction "
        f"<code>{metadata['train_fraction']}</code> "
        f"({coverage['n_train_samples']} samples).</li>",
        f"<li>Up to <code>{metadata['cells_per_sample']}</code> cells were sampled per "
        f"training sample, giving a working set of "
        f"<code>{coverage['n_cells_used']}</code> cells.</li>",
        f"<li>Gene-symbol coverage: <code>{coverage['n_genes_matched']}</code> of "
        f"<code>{coverage['n_genes_requested']}</code> unique symbols in the CSV mapped "
        f"to the cellarr gene table; <code>{coverage['n_pairs_scored']}</code> of "
        f"<code>{coverage['n_pairs_total']}</code> rows scored (rest dropped as NaN).</li>",
        "<li>Log-normalization: <code>log1p(1e4 · x / library_size)</code>, with library "
        "size computed over all cellarr genes.</li>",
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
    p.add_argument(
        "--pairs-csv", type=Path,
        default=Path("/cv/data/braid/andera29/projs/mct/mct_pub/rsc/rl_df_w_negs_with_gene_pairs.csv"),
    )
    p.add_argument(
        "--cellarr-dir", type=Path,
        default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr"),
    )
    p.add_argument(
        "--split-file", type=Path,
        default=Path("/cv/scratch/u/andera29/mct_pretraining/cellarr/ablation_split.json"),
    )
    p.add_argument("--train-fraction", default="1.00")
    p.add_argument(
        "--gene-info-csv", type=Path,
        default=Path("/cv/data/braid/andera29/projs/mct/mct_pub/tutorials/data/gene_order_info.csv"),
    )
    p.add_argument(
        "--cellchatdb-csv", type=Path,
        default=Path("/cv/data/braid/andera29/projs/mct/mct_pub/tutorials/data/cellchatdb.csv"),
    )
    p.add_argument("--sample-column", default="sample")
    p.add_argument("--cells-per-sample", type=int, default=128,
                   help="Max cells drawn per training sample (for both baselines).")
    p.add_argument("--fetch-chunk", type=int, default=20000)
    p.add_argument("--sampling-seed", type=int, default=0)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("/cv/data/braid/andera29/projs/mct/mct_pub/outputs/gene_pair_coexpression"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[load] pairs csv: {args.pairs_csv}", flush=True)
    pairs = pd.read_csv(args.pairs_csv)
    for col in ("label", "seed", "gene1", "gene2"):
        assert col in pairs.columns, f"missing column {col!r} in pairs csv"
    unique_symbols = sorted(set(pairs["gene1"]).union(pairs["gene2"]))
    print(f"       {len(pairs)} rows, {pairs['seed'].nunique()} seeds, "
          f"{len(unique_symbols)} unique symbols", flush=True)

    print(f"[annot] joining annotations from {args.cellchatdb_csv}", flush=True)
    pairs = annotate_pairs(pairs, args.cellchatdb_csv)
    print("       annotation counts:", pairs["annotation"].value_counts().to_dict(),
          flush=True)

    print("[map ] resolving gene symbols against cellarr gene table", flush=True)
    symbol2col = build_symbol2col(args.cellarr_dir, args.gene_info_csv)
    target_symbols = [s for s in unique_symbols if s in symbol2col]
    n_matched = len(target_symbols)
    print(f"       matched {n_matched}/{len(unique_symbols)} symbols", flush=True)
    target_cols = np.asarray([symbol2col[s] for s in target_symbols], dtype=np.int64)
    sym2local = {s: i for i, s in enumerate(target_symbols)}

    print(f"[split] loading ablation split at fraction {args.train_fraction}", flush=True)
    split = json.loads(args.split_file.read_text())
    frac_key = f"{float(args.train_fraction):.2f}"
    train_samples = split["train_samples"][frac_key]
    print(f"       {len(train_samples)} train samples", flush=True)

    print(f"[cells] choosing up to {args.cells_per_sample} cells per sample", flush=True)
    cell_choice = choose_training_cells(
        cell_metadata_uri=args.cellarr_dir / "cell_metadata",
        train_samples=train_samples,
        cells_per_sample=args.cells_per_sample,
        sample_column=args.sample_column,
        seed=args.sampling_seed,
    )
    print(f"       {len(cell_choice)} cells from {cell_choice['sample'].nunique()} samples",
          flush=True)

    print("[fetch] streaming counts + lognorm from cellarr TileDB", flush=True)
    with tiledb.open(str(args.cellarr_dir / "gene_annotation"), "r") as g:
        n_genes = g.nonempty_domain()[0][1] + 1
    counts_tdb = tiledb.open(str(args.cellarr_dir / "counts"), "r", config=TILEDB_CONFIG)
    try:
        cell_indices = cell_choice["cell_index"].to_numpy()
        X_cell = stream_lognorm_target(
            counts_tdb, cell_indices, target_cols, n_genes,
            chunk=args.fetch_chunk, target_sum=1e4,
        )
    finally:
        counts_tdb.close()
    print(f"       X_cell shape={X_cell.shape}", flush=True)

    print("[corr ] cell-level Pearson correlation over target genes", flush=True)
    cell_corr = pearson_corr_columns(X_cell)

    print("[bulk ] per-sample pseudobulk (mean of lognorm) + correlation", flush=True)
    sample_ids = cell_choice["sample"].to_numpy()
    X_sample, sample_order = group_mean(X_cell, sample_ids)
    print(f"       X_sample shape={X_sample.shape}", flush=True)
    sample_corr = pearson_corr_columns(X_sample)

    print("[score] scoring pairs", flush=True)
    pairs = pairs.copy()
    pairs["cell_cs"] = score_pairs(pairs, cell_corr, sym2local)
    pairs["sample_cs"] = score_pairs(pairs, sample_corr, sym2local)
    pairs.to_csv(output_dir / "data" / "scored_pairs.csv", index=False)

    seeds = sorted(pairs["seed"].unique())
    methods = {
        "Cell co-expression": ("cell_cs", COLOR_CELL),
        "Sample co-expression (pseudobulk)": ("sample_cs", COLOR_SAMPLE),
    }

    strata = [ALL_STRATUM] + sorted(
        a for a in pairs["annotation"].unique() if a != "Unknown"
    )

    def subset_for(stratum: str) -> pd.DataFrame:
        return pairs if stratum == ALL_STRATUM else pairs[pairs["annotation"] == stratum]

    print("[metric] per-seed AUROC / AUPRC (stratified)", flush=True)
    rows = []
    for stratum in strata:
        sub = subset_for(stratum)
        for name, (col, _) in methods.items():
            for s in seeds:
                d = sub[sub["seed"] == s].dropna(subset=[col])
                if d["label"].nunique() < 2:
                    continue
                rows.append({
                    "stratum": stratum,
                    "method": name,
                    "seed": int(s),
                    "n_scored": int(len(d)),
                    "n_pos": int((d["label"] == 1).sum()),
                    "n_neg": int((d["label"] == 0).sum()),
                    "auroc": float(roc_auc_score(d["label"], d[col])),
                    "auprc": float(average_precision_score(d["label"], d[col])),
                })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "data" / "per_seed_metrics.csv", index=False)

    summary = (
        metrics_df.groupby(["stratum", "method"])[["auroc", "auprc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "stratum", "method", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
    ]
    summary.to_csv(output_dir / "data" / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    print("[plot ] ROC + PR curves (stratified)", flush=True)
    figs_b64: dict[str, str] = {}
    for stratum in strata:
        sub = subset_for(stratum)
        slug = (
            "all" if stratum == ALL_STRATUM
            else stratum.lower().replace(" ", "_").replace("-", "_")
        )
        fig_roc = plot_roc(sub, methods, seeds, title_suffix=stratum)
        figs_b64[f"roc__{slug}"] = _save_fig(
            fig_roc, output_dir / "figures" / f"roc_{slug}.png"
        )
        plt.close(fig_roc)
        fig_pr = plot_pr(sub, methods, seeds, title_suffix=stratum)
        figs_b64[f"pr__{slug}"] = _save_fig(
            fig_pr, output_dir / "figures" / f"pr_{slug}.png"
        )
        plt.close(fig_pr)

    coverage = {
        "n_train_samples": len(train_samples),
        "n_samples_with_cells": int(cell_choice["sample"].nunique()),
        "n_cells_used": int(len(cell_choice)),
        "n_genes_requested": int(len(unique_symbols)),
        "n_genes_matched": int(n_matched),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_scored": int(
            pairs[["cell_cs", "sample_cs"]].notna().all(axis=1).sum()
        ),
    }
    repo = Path(__file__).resolve().parent.parent
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(repo),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "pairs_csv": str(args.pairs_csv),
        "cellarr_dir": str(args.cellarr_dir),
        "split_file": str(args.split_file),
        "train_fraction": args.train_fraction,
        "cells_per_sample": int(args.cells_per_sample),
        "sampling_seed": int(args.sampling_seed),
    }
    (output_dir / "data" / "run_config.json").write_text(json.dumps(metadata, indent=2))

    build_html(
        output_path=output_dir / "report.html",
        metadata=metadata,
        metrics_df=metrics_df,
        coverage=coverage,
        figs_b64=figs_b64,
        strata=strata,
    )
    print(f"[done] outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
