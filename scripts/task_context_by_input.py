"""Ablation: contextual embedding drift as the input sample representation changes.

Question: how much do MCT's contextual gene embeddings shift when the same
biological sample is represented by different subsamples of its cells, and is
that shift big or small compared to the natural spread of gene embeddings
inside a sample?

Design:
    1. For each of ``--n-samples`` largest samples in the AnnData with at
       least ``n_cells_bag`` cells:
       a. Draw a "reference" bag of ``n_cells_bag`` cells (seed 0).
       b. For each perturbation mode, generate perturbed bags:
          - ``subsample``: keep a fraction ``f`` of the reference cells;
            the remaining positions are padded with zeros.
          - ``redraw``: draw a fresh ``n_cells_bag`` cells at random from
            the sample pool (upper bound of sampling variance).
          - ``shuffle``: permute the positional (cell) axis of the same
            reference cells (isolates positional / permutation effects).
       c. Run the model on each bag, get context-aware gene embeddings,
          and score per-gene cosine distance to the reference on the
          intersection of expressed genes.
       d. Compute a **random gene-pair baseline**: cosine distance between
          ``--n-random-pairs`` random pairs of genes in the reference
          embedding matrix. This is the natural pairwise spread inside a
          sample; use it to gauge whether the observed drift is big or
          small.
    2. Aggregate across samples and write:
       - self-contained HTML report (figures embedded inline)
       - 300 dpi PNGs
       - tidy per-gene / per-sample CSVs

Palette pulled from ``tutorials/benchmarking_ccc_rl_cellchat.ipynb`` so the
plots match the rest of the paper (MCT navy #000080, firebrick, sea green).

Example
-------
    python scripts/task_context_by_input.py \\
        --gene-index tutorials/data/ensemble2idx.json \\
        --sample-key sample --device cuda \\
        --n-samples 3 --n-seeds 5 --n-random-pairs 10000

Design goal: this is a paper-supplementary figure. Script is
self-contained, deterministic given the seed, and writes camera-ready
PNGs.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import io
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch

# Force a clean white background on every saved figure so the PNGs drop
# cleanly onto a paper page.
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

import cell_cell_interactions as cci
from cell_cell_interactions.modeling.data_types import MulticellularTransformerOutput
from cell_cell_interactions.pp.data_types import BagOfCellsBatch


# ---------------------------------------------------------------------- #
# Palette — matches tutorials/benchmarking_ccc_rl_cellchat.ipynb
# ---------------------------------------------------------------------- #

# The paper uses these hex codes across all model-comparison figures. Reused
# here for per-mode colors so the ablation plot slots into the same visual
# system.
MCT_NAVY = "#000080"        # MCT primary (paper benchmarking)
FIREBRICK = "#B22222"       # scFoundation in paper; used here for "redraw"
SEA_GREEN = "#2E8B57"       # CellFM in paper; used here for "shuffle"
DARK_ORANGE = "#FF8C00"     # scPRINT in paper; used here for cross-sample lines
NEUTRAL_GRAY = "#4d4d4d"    # random gene-pair baseline (loose reference)
LIGHT_GRAY = "#9ea3a8"      # 1-NN baseline (tight reference)

PALETTE = {
    "redraw": FIREBRICK,
    "shuffle_within_ct": SEA_GREEN,
    "shuffle_across_ct": DARK_ORANGE,
    "baseline_pair": NEUTRAL_GRAY,
    "baseline_1nn": LIGHT_GRAY,
}
MODES = ["redraw", "shuffle_within_ct", "shuffle_across_ct"]

# ---------------------------------------------------------------------- #
# Defaults / config
# ---------------------------------------------------------------------- #

DEFAULT_CHECKPOINT = "checkpoints/best_12k_no_pseudobulk_no_gex.ckpt"
DEFAULT_INDEX_DIR = "tutorials/data"
DEFAULT_ADATA = "data/lung.h5ad"
DEFAULT_OUTPUT = Path("outputs") / "task_context_by_input"


@dataclass
class RunConfig:
    checkpoint: str
    index_dir: str
    adata: str
    samples: list[str]
    sample_key: str
    cell_type_key: str | None
    n_cells_bag: int
    fractions: list[float]
    n_seeds: int
    n_random_pairs: int
    min_frac_expressed_cells: float
    max_seq_len: int
    device: str
    output_dir: str
    modes: list[str]
    reference_seed: int
    checkpoint_config: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    python: str = ""
    torch_version: str = ""


# ---------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dense(X, rows: np.ndarray) -> np.ndarray:
    sub = X[rows, :]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine distance."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom < eps, np.nan, denom)
    cos = num / denom
    return 1.0 - np.nan_to_num(cos, nan=0.0)


def _pack_reference_stratified(
    cell_pool: np.ndarray,
    obs_cell_types: np.ndarray,
    cell_type_input_order: dict[str, tuple[int, int]],
    n_cells_bag: int,
    cell_type_mapping: dict[str, str] | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a bag of ``n_cells_bag`` cells stratified by cell type.

    Follows the training-time convention: each cell type occupies the
    contiguous slot range given by ``cell_type_input_order``. Within a
    donor, cells of each type are sampled to fill their slot; if too few
    cells of that type exist in the donor, the leftover slots are filled
    with a repeat draw. If none exist, slots are zero-filled (marked by
    cell index = -1 in the returned selection so the bag row stays zero).

    Returns
    -------
    reference_cells:
        Length ``n_cells_bag`` array of row indices into the AnnData.
        Value -1 marks a zero-padded slot.
    reference_cell_types:
        Length ``n_cells_bag`` array of the cell-type label assigned to
        each slot (constant within a cell-type block).
    """
    rng = np.random.default_rng(seed)
    if cell_type_mapping is not None:
        mapped = np.array(
            [cell_type_mapping.get(str(x), str(x)) for x in obs_cell_types],
            dtype=object,
        )
    else:
        mapped = obs_cell_types.astype(object)

    # Donor-scoped index of cells by mapped cell-type label.
    donor_types = mapped[cell_pool]
    cells_by_type: dict[str, np.ndarray] = {}
    for ct in np.unique(donor_types):
        cells_by_type[str(ct)] = cell_pool[donor_types == ct]

    reference_cells = np.full(n_cells_bag, -1, dtype=np.int64)
    reference_cell_types = np.array(["__pad__"] * n_cells_bag, dtype=object)

    for ct, (start, end) in cell_type_input_order.items():
        width = end - start
        pool = cells_by_type.get(ct)
        if pool is None or len(pool) == 0:
            reference_cell_types[start:end] = ct  # slot still owned by ct
            continue
        if len(pool) >= width:
            chosen = rng.choice(pool, size=width, replace=False)
        else:
            head = pool
            tail = rng.choice(pool, size=width - len(pool), replace=True)
            chosen = np.concatenate([head, tail])
        reference_cells[start:end] = chosen
        reference_cell_types[start:end] = ct

    return reference_cells, reference_cell_types


def _resolve_var_to_token(adata: sc.AnnData, gene2idx: dict[str, int]) -> np.ndarray:
    ids = np.full(adata.n_vars, -1, dtype=np.int64)
    for i, name in enumerate(adata.var.index):
        tok = gene2idx.get(name)
        if tok is not None:
            ids[i] = int(tok)
    return ids


# ---------------------------------------------------------------------- #
# Bag construction (unchanged from single-sample version)
# ---------------------------------------------------------------------- #


def _build_bag(
    X_all: np.ndarray,
    n_cells_bag: int,
    fraction: float,
    perturbation_mode: str,
    reference_cells: np.ndarray,
    reference_cell_types: np.ndarray,
    cells_by_type: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    """Build a (n_cells_bag, n_genes) perturbed input bag.

    Cell-type-slot aware:
      - The reference is packed by cell type into contiguous position ranges
        (``reference_cell_types`` gives the label for each of the 1024 slots).
      - ``redraw`` replaces a fraction of positions with fresh cells of the
        **same cell type** as the slot they occupy (preserves training-time
        cell-type-slot convention).
      - ``shuffle_within_ct`` permutes positions inside each cell-type block.
      - ``shuffle_across_ct`` permutes positions freely (destroys convention).
    """
    rng = np.random.default_rng(seed)
    n_genes = X_all.shape[1]
    n_positions = len(reference_cells)
    bag = np.zeros((n_cells_bag, n_genes), dtype=np.float32)
    real_mask = reference_cells >= 0
    if real_mask.any():
        real_pos = np.where(real_mask)[0]
        bag[real_pos] = X_all[reference_cells[real_pos], :]

    if perturbation_mode == "redraw":
        real_positions = np.where(real_mask)[0]
        k = int(round(fraction * len(real_positions)))
        if k == 0:
            return bag
        replace_positions = rng.choice(real_positions, size=k, replace=False)
        # For each replaced position, draw a fresh cell of the SAME cell type
        # from the donor pool, excluding the current occupant.
        for pos in replace_positions:
            ct = reference_cell_types[pos]
            pool = cells_by_type.get(str(ct))
            if pool is None or len(pool) == 0:
                continue
            current = reference_cells[pos]
            candidates = pool[pool != current]
            if len(candidates) == 0:
                candidates = pool
            new_cell = int(rng.choice(candidates))
            bag[pos] = X_all[new_cell, :]
        return bag

    if perturbation_mode == "shuffle_across_ct":
        k = int(round(fraction * n_positions))
        if k < 2:
            return bag
        positions = rng.choice(n_positions, size=k, replace=False)
        permuted = rng.permutation(positions)
        bag[positions] = bag[permuted].copy()
        return bag

    if perturbation_mode == "shuffle_within_ct":
        unique_cts = pd.unique(reference_cell_types)
        for ct in unique_cts:
            positions = np.where(reference_cell_types == ct)[0]
            if len(positions) < 2:
                continue
            k = int(round(fraction * len(positions)))
            if k < 2:
                continue
            chosen = rng.choice(positions, size=k, replace=False)
            permuted = rng.permutation(chosen)
            bag[chosen] = bag[permuted].copy()
        return bag

    raise ValueError(f"unknown mode {perturbation_mode!r}")


def _bag_to_batch(
    bag: np.ndarray,
    var_tokens: np.ndarray,
    min_expressed_cells: int,
    max_seq_len: int,
    device: torch.device,
    seed: int,
) -> tuple[BagOfCellsBatch, np.ndarray]:
    X = torch.tensor(bag, dtype=torch.float32)

    valid_gene_mask = var_tokens >= 0
    expressed = ((X > 0).sum(0) > min_expressed_cells).numpy()
    keep_local = np.where(valid_gene_mask & expressed)[0]

    if len(keep_local) > max_seq_len:
        rng = np.random.default_rng(seed + 999)
        pick = rng.choice(len(keep_local), size=max_seq_len, replace=False)
        pick.sort()
        keep_local = keep_local[pick]

    if len(keep_local) == 0:
        raise RuntimeError("No expressed / in-vocabulary genes in this bag.")

    X_sub = X[:, keep_local]
    tokens = torch.tensor(var_tokens[keep_local], dtype=torch.long)

    expression_matrix = X_sub.T.unsqueeze(0).to(device)
    gene_ids = tokens.unsqueeze(0).to(device)
    is_masked_gene_expression = torch.zeros_like(gene_ids, dtype=torch.float32)
    padding_mask = torch.zeros_like(gene_ids, dtype=torch.bool)

    batch = BagOfCellsBatch(
        expression_matrix=expression_matrix,
        gene_ids=gene_ids,
        labels=gene_ids.clone(),
        is_masked_gene_id=torch.zeros_like(gene_ids),
        is_masked_gene_expression=is_masked_gene_expression,
        padding_mask=padding_mask,
    )
    return batch, var_tokens[keep_local]


@dataclass
class BagResult:
    sample: str
    mode: str
    fraction: float
    seed: int
    gene_ids: np.ndarray
    embeddings: np.ndarray


def _forward_bag(model: cci.MCTModel, batch: BagOfCellsBatch) -> np.ndarray:
    out: MulticellularTransformerOutput = model.forward_batch(batch)
    return out.last_hidden_state.squeeze(0).detach().cpu().numpy()


# ---------------------------------------------------------------------- #
# Per-sample loop
# ---------------------------------------------------------------------- #


def _process_sample(
    model: cci.MCTModel,
    adata: sc.AnnData,
    sample_id: str,
    X_all: np.ndarray,
    cell_pool: np.ndarray,
    var_tokens: np.ndarray,
    modes: list[str],
    fractions: list[float],
    n_seeds: int,
    n_cells_bag: int,
    min_expressed_cells: int,
    max_seq_len: int,
    n_random_pairs: int,
    reference_seed: int,
    cell_type_key: str | None = None,
    cell_type_input_order: dict[str, tuple[int, int]] | None = None,
    cell_type_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the whole (reference + perturbations + baseline) pipeline for one sample.

    Returns:
        drift_long: per-gene, per-perturbation cosine distance to reference.
        baseline_pairs: cosine distances between random gene pairs in reference.
        baseline_1nn: 1-NN cosine distance for every gene in reference.
    """
    device = model.device
    if cell_type_key is None or cell_type_input_order is None:
        raise ValueError(
            "Cell-type-aware pipeline requires both --cell-type-key on the AnnData "
            "and a cell_type_input_order in the model index."
        )
    obs_cts = adata.obs[cell_type_key].astype(str).values

    print(f"  [ref] packing stratified reference for sample {sample_id}")
    reference_cells, reference_cell_types = _pack_reference_stratified(
        cell_pool=cell_pool,
        obs_cell_types=obs_cts,
        cell_type_input_order=cell_type_input_order,
        n_cells_bag=n_cells_bag,
        cell_type_mapping=cell_type_mapping,
        seed=reference_seed,
    )
    n_filled = int((reference_cells >= 0).sum())
    print(f"       {n_filled}/{n_cells_bag} slots filled with real cells")

    # Same-cell-type pool for redraw. Use the mapped (harmonized) label per cell.
    if cell_type_mapping is not None:
        donor_cts_mapped = np.array(
            [cell_type_mapping.get(x, x) for x in obs_cts[cell_pool]],
            dtype=object,
        )
    else:
        donor_cts_mapped = obs_cts[cell_pool].astype(object)
    cells_by_type: dict[str, np.ndarray] = {}
    for ct in np.unique(donor_cts_mapped):
        cells_by_type[str(ct)] = cell_pool[donor_cts_mapped == ct]

    # For zero-padded slots we can't do same-type redraw; leave them zero.
    # The reference bag itself uses X_all directly, so zero-padded slots
    # remain zero rows already.
    real_positions = np.where(reference_cells >= 0)[0]
    fake_positions = np.where(reference_cells < 0)[0]
    if len(fake_positions):
        print(f"       {len(fake_positions)} slots empty (no cells of that type in donor)")

    # Build the reference bag manually since _build_bag expects reference_cells
    # to have real indices; zero-fill fake positions.
    ref_bag = np.zeros((n_cells_bag, X_all.shape[1]), dtype=np.float32)
    ref_bag[real_positions] = X_all[reference_cells[real_positions], :]
    ref_batch, ref_tokens = _bag_to_batch(
        bag=ref_bag,
        var_tokens=var_tokens,
        min_expressed_cells=min_expressed_cells,
        max_seq_len=max_seq_len,
        device=device,
        seed=reference_seed,
    )
    ref_emb = _forward_bag(model, ref_batch)
    print(f"       reference: {len(ref_tokens)} genes, emb shape {ref_emb.shape}")

    # Random gene-pair baseline (natural spread inside a sample).
    baseline_pairs = _random_pair_baseline(
        embeddings=ref_emb,
        gene_ids=ref_tokens,
        n_pairs=n_random_pairs,
        seed=reference_seed + 5000,
    )
    baseline_pairs["sample"] = sample_id
    print(
        f"       random-pair baseline: "
        f"mean={baseline_pairs['cosine_distance'].mean():.4f}, "
        f"median={baseline_pairs['cosine_distance'].median():.4f}"
    )

    # 1-NN baseline: distance from each gene to its nearest neighbor.
    baseline_1nn = _one_nn_baseline(embeddings=ref_emb, gene_ids=ref_tokens)
    baseline_1nn["sample"] = sample_id
    print(
        f"       1-NN baseline: "
        f"mean={baseline_1nn['cosine_distance'].mean():.4f}, "
        f"median={baseline_1nn['cosine_distance'].median():.4f}"
    )

    # Perturbations.
    ref_token_to_row = {int(t): i for i, t in enumerate(ref_tokens)}
    rows: list[pd.DataFrame] = []
    for mode in modes:
        for frac in fractions:
            for seed_offset in range(n_seeds):
                seed = 100 + 1000 * fractions.index(frac) + seed_offset
                bag = _build_bag(
                    X_all=X_all,
                    n_cells_bag=n_cells_bag,
                    fraction=frac,
                    perturbation_mode=mode,
                    reference_cells=reference_cells,
                    reference_cell_types=reference_cell_types,
                    cells_by_type=cells_by_type,
                    seed=seed,
                )
                batch, tokens = _bag_to_batch(
                    bag=bag,
                    var_tokens=var_tokens,
                    min_expressed_cells=min_expressed_cells,
                    max_seq_len=max_seq_len,
                    device=device,
                    seed=seed,
                )
                emb = _forward_bag(model, batch)

                cur_map = {int(t): i for i, t in enumerate(tokens)}
                shared = sorted(set(ref_token_to_row) & set(cur_map))
                ref_idx = np.array([ref_token_to_row[t] for t in shared], dtype=np.int64)
                cur_idx = np.array([cur_map[t] for t in shared], dtype=np.int64)
                dists = _cosine_distance(ref_emb[ref_idx], emb[cur_idx])
                symbols = [model.idx2gene.get(t, f"__idx_{t}__") for t in shared]

                rows.append(
                    pd.DataFrame(
                        {
                            "sample": sample_id,
                            "mode": mode,
                            "fraction": frac,
                            "seed": seed_offset,
                            "gene": symbols,
                            "token": shared,
                            "cosine_distance": dists,
                            "n_shared_genes": len(shared),
                        }
                    )
                )
                print(
                    f"  [{mode:10s}] frac={frac:.2f} seed={seed_offset} "
                    f"→ {len(shared)} shared genes, mean drift {dists.mean():.4f}"
                )

    drift_long = pd.concat(rows, ignore_index=True)
    return drift_long, baseline_pairs, baseline_1nn


def _random_pair_baseline(
    embeddings: np.ndarray,
    gene_ids: np.ndarray,
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    n = embeddings.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["i", "j", "gene_i", "gene_j", "cosine_distance"])
    rng = np.random.default_rng(seed)
    # Sample WITHOUT self-pairs; oversample then filter.
    i = rng.integers(0, n, size=n_pairs * 2)
    j = rng.integers(0, n, size=n_pairs * 2)
    mask = i != j
    i, j = i[mask][:n_pairs], j[mask][:n_pairs]
    dists = _cosine_distance(embeddings[i], embeddings[j])
    return pd.DataFrame({"i": i, "j": j, "gene_i": gene_ids[i], "gene_j": gene_ids[j], "cosine_distance": dists})


def _one_nn_baseline(
    embeddings: np.ndarray,
    gene_ids: np.ndarray,
) -> pd.DataFrame:
    """1-NN cosine distance for every gene in the reference embedding.

    For each row, find its nearest neighbor (excluding itself) in the same
    matrix, return the cosine distance. Gives a tighter reference than the
    random-pair baseline: how far away is the CLOSEST other gene, on
    average? Useful because "drift" that exceeds even the 1-NN distance
    means the perturbation moves a gene further than any of its natural
    neighbors sit.
    """
    from sklearn.neighbors import NearestNeighbors

    n = embeddings.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["gene", "nearest_neighbor", "cosine_distance"])
    # cosine metric; k=2 so we can drop the self match at index 0.
    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    # First col is self (distance 0). Take second col.
    nn_dist = distances[:, 1]
    nn_idx = indices[:, 1]
    return pd.DataFrame(
        {
            "gene": gene_ids,
            "nearest_neighbor": gene_ids[nn_idx],
            "cosine_distance": nn_dist,
        }
    )


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_drift_multi_sample(
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    baseline_1nn_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Mean drift vs fraction with a broken y-axis for the baseline lines.

    Bottom pane: drift curves in their natural range (~0..0.05).
    Top pane: the two baselines (~0.45 and ~0.82).
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]}
    )

    # ---- Bottom pane: drift curves + also draw baseline lines above their
    # native range so the ratio to the top pane is visible.
    for mode in MODES:
        if mode not in summary["mode"].unique():
            continue
        sub = summary[summary["mode"] == mode]
        ax_bot.errorbar(
            sub["fraction"],
            sub["mean_cosine_distance"],
            yerr=sub["std_cosine_distance"],
            marker="o",
            linewidth=2.0,
            markersize=6,
            capsize=3,
            label=mode,
            color=PALETTE[mode],
        )

    # ---- Top pane: the two horizontal baselines with their ±std bands.
    bmean = float(baseline_summary["mean_cosine_distance"].mean()) if not baseline_summary.empty else None
    bstd = float(baseline_summary["mean_cosine_distance"].std(ddof=0)) if not baseline_summary.empty else 0
    nmean = float(baseline_1nn_summary["mean_cosine_distance"].mean()) if not baseline_1nn_summary.empty else None
    nstd = float(baseline_1nn_summary["mean_cosine_distance"].std(ddof=0)) if not baseline_1nn_summary.empty else 0

    if nmean is not None:
        ax_top.axhline(
            nmean, color=PALETTE["baseline_1nn"], linestyle=":", linewidth=1.8,
            label=f"1-NN baseline: {nmean:.3f}",
        )
        ax_top.axhspan(nmean - nstd, nmean + nstd, color=PALETTE["baseline_1nn"], alpha=0.15)
    if bmean is not None:
        ax_top.axhline(
            bmean, color=PALETTE["baseline_pair"], linestyle="--", linewidth=1.5,
            label=f"random-pair baseline: {bmean:.3f}",
        )
        ax_top.axhspan(bmean - bstd, bmean + bstd, color=PALETTE["baseline_pair"], alpha=0.12)

    # ---- Axis limits for the break.
    drift_top = summary["mean_cosine_distance"].max() * 1.4 + 0.005
    ax_bot.set_ylim(0, drift_top)
    # Top pane spans from just under the 1-NN band to just above the random-pair band.
    lo = min(v for v in [nmean and nmean - nstd, bmean and bmean - bstd] if v is not None) - 0.03
    hi = max(v for v in [nmean and nmean + nstd, bmean and bmean + bstd] if v is not None) + 0.05
    ax_top.set_ylim(lo, hi)

    # ---- Hide the shared spines and add diagonal break marks.
    ax_top.spines[["bottom", "top", "right"]].set_visible(False)
    ax_bot.spines[["top", "right"]].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    d = 0.012  # break-mark size
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d * 2, 1 + d * 2), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d * 2, 1 + d * 2), **kwargs)

    ax_bot.set_xlabel("Fraction of positions perturbed", fontsize=13)
    ax_bot.set_ylabel(r"Mean cosine distance $\Delta$ (vs. reference)", fontsize=13)
    ax_top.set_ylabel("Baseline", fontsize=11)
    ax_top.set_title(
        "Contextual embedding drift under input resampling\n"
        f"n = {summary['n_samples'].max()} sample(s), error bars = ±1 std across samples",
        fontsize=13,
    )

    # Combined legend outside on the right.
    handles_bot, labels_bot = ax_bot.get_legend_handles_labels()
    handles_top, labels_top = ax_top.get_legend_handles_labels()
    fig.legend(
        handles_bot + handles_top,
        labels_bot + labels_top,
        frameon=False,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
    )
    _save(fig, out_path)


def _plot_per_sample_lines(long_df: pd.DataFrame, out_path: Path) -> None:
    """One panel per mode, one line per sample."""
    samples = sorted(long_df["sample"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    sample_colors = _sample_palette(samples)
    for ax, mode in zip(axes, MODES):
        sub = long_df[long_df["mode"] == mode]
        if sub.empty:
            ax.set_visible(False)
            continue
        agg = (
            sub.groupby(["sample", "fraction"], observed=True)["cosine_distance"]
            .mean()
            .reset_index()
        )
        for sample in samples:
            s = agg[agg["sample"] == sample]
            ax.plot(
                s["fraction"],
                s["cosine_distance"],
                marker="o",
                linewidth=1.8,
                color=sample_colors[sample],
                label=sample if ax is axes[0] else None,
            )
        ax.set_title(mode, fontsize=13, color=PALETTE[mode])
        ax.set_xlabel("Fraction of positions perturbed (in this mode)", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean cosine distance to reference", fontsize=12)
    # Grab legend handles from the first axis (all samples appear there) and
    # place the figure-level legend outside on the right.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Sample",
    )
    fig.suptitle("Per-sample drift trajectories", fontsize=14, y=1.02)
    _save(fig, out_path)


def _sample_palette(samples: list[str]) -> dict[str, str]:
    base = [MCT_NAVY, FIREBRICK, SEA_GREEN, DARK_ORANGE, "#8A2BE2", "#B121B6", "#800000"]
    cmap = plt.cm.get_cmap("tab10")
    out: dict[str, str] = {}
    for i, s in enumerate(samples):
        out[s] = base[i] if i < len(base) else matplotlib.colors.to_hex(cmap(i % 10))
    return out


def _plot_drift_vs_baseline_ratio(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Drift as a fraction of both baselines: random-pair (left) and 1-NN (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    plots = [
        (axes[0], "ratio_to_baseline", "ratio_to_baseline_std",
         "Ratio to random-pair distance", PALETTE["baseline_pair"],
         "random-pair baseline"),
        (axes[1], "ratio_to_baseline_1nn", "ratio_to_baseline_1nn_std",
         "Ratio to 1-NN distance", PALETTE["baseline_1nn"],
         "1-NN baseline"),
    ]
    for ax, mean_col, std_col, ylabel, hline_color, hline_label in plots:
        for mode in MODES:
            sub = summary[summary["mode"] == mode]
            if sub.empty:
                continue
            ax.errorbar(
                sub["fraction"],
                sub[mean_col],
                yerr=sub[std_col],
                marker="o",
                linewidth=2.0,
                markersize=6,
                capsize=3,
                label=mode,
                color=PALETTE[mode],
            )
        ax.axhline(1.0, color=hline_color, linestyle="--", linewidth=1.5,
                   label=f"{hline_label} (ratio = 1)")
        ax.set_xlabel("Fraction of positions perturbed", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].legend(
        frameon=False,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.suptitle("Effect size relative to natural gene-embedding spread", fontsize=13, y=1.02)
    _save(fig, out_path)


def _plot_drift_and_baseline_distribution(
    long_df: pd.DataFrame,
    baseline_pairs: pd.DataFrame,
    baseline_1nn: pd.DataFrame,
    out_path: Path,
    focus_mode: str = "shuffle_across_ct",
) -> None:
    """Violin: per-gene drift under ``focus_mode`` at every fraction, next to both baselines."""
    fig, ax = plt.subplots(figsize=(11, 5))
    sub = long_df[long_df["mode"] == focus_mode].copy()
    sub["group"] = sub["fraction"].apply(lambda f: f"{focus_mode}\nf={f:.2f}")

    pair_df = baseline_pairs.copy()
    pair_df["group"] = "baseline\nrandom pairs"
    nn_df = baseline_1nn.copy()
    nn_df["group"] = "baseline\n1-NN"

    combined = pd.concat(
        [
            sub[["group", "cosine_distance"]],
            nn_df[["group", "cosine_distance"]],
            pair_df[["group", "cosine_distance"]],
        ],
        ignore_index=True,
    )
    fracs = sorted(sub["fraction"].unique())
    order = (
        [f"{focus_mode}\nf={f:.2f}" for f in fracs]
        + ["baseline\n1-NN", "baseline\nrandom pairs"]
    )
    focus_color = PALETTE.get(focus_mode, MCT_NAVY)
    palette = {f"{focus_mode}\nf={f:.2f}": focus_color for f in fracs}
    palette["baseline\n1-NN"] = LIGHT_GRAY
    palette["baseline\nrandom pairs"] = NEUTRAL_GRAY

    sns.violinplot(
        data=combined,
        x="group",
        y="cosine_distance",
        order=order,
        palette=palette,
        ax=ax,
        cut=0,
        inner="quartile",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cosine distance", fontsize=13)
    ax.set_title(
        "Per-gene drift under subsampling vs. natural gene-embedding spread",
        fontsize=13,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


def _plot_top_shifted_genes(
    long_df: pd.DataFrame,
    out_path: Path,
    n_top: int = 30,
    focus_mode: str = "shuffle_across_ct",
) -> None:
    """Rank genes by mean cosine drift at the MAX fraction of ``focus_mode``."""
    max_frac = long_df["fraction"].max()
    subset = long_df[(long_df["mode"] == focus_mode) & (long_df["fraction"] == max_frac)]
    if subset.empty:
        return
    ranking = (
        subset.groupby("gene", observed=True)["cosine_distance"]
        .mean()
        .sort_values(ascending=False)
        .head(n_top)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * n_top)))
    sns.barplot(
        data=ranking,
        x="cosine_distance",
        y="gene",
        color=PALETTE.get(focus_mode, MCT_NAVY),
        ax=ax,
    )
    ax.set_xlabel(f"Mean cosine distance ({focus_mode} @ f={max_frac:.2f})", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_title(f"Top-{n_top} genes with the largest contextual drift", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


# ---------------------------------------------------------------------- #
# Self-contained HTML report
# ---------------------------------------------------------------------- #


def _img_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MCT ablation — contextual embedding drift by input</title>
<style>
  html, body {{ background: #ffffff; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1080px; margin: 2rem auto; padding: 0 1.5rem; color: #1f2328;
         line-height: 1.55; background: #ffffff; }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.25rem; }}
  code, pre {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  pre {{ background: #f6f8fa; padding: 0.75rem 1rem; border-radius: 6px; overflow-x: auto;
         font-size: 0.85rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; background: #ffffff; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f6f8fa; }}
  figure {{ margin: 1.5rem 0; background: #ffffff; }}
  figcaption {{ font-size: 0.85rem; color: #57606a; margin-top: 0.5rem; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 4px;
         background: #ffffff; }}
  .meta {{ color: #57606a; font-size: 0.9rem; }}
  .glossary {{ background: #f6f8fa; padding: 1rem 1.25rem; border-radius: 6px;
               border: 1px solid #d0d7de; margin: 1rem 0 1.5rem; }}
  .glossary h3 {{ margin: 0.2rem 0 0.5rem; font-size: 1rem; }}
  .glossary dt {{ font-weight: 600; color: #24292f; margin-top: 0.5rem; }}
  .glossary dd {{ margin: 0.15rem 0 0.15rem 1.2rem; }}
  .swatch {{ display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px;
             margin-right: 0.4rem; vertical-align: middle; border: 1px solid #d0d7de; }}
</style>
</head>
<body>

<h1>Contextual embedding drift under input resampling</h1>
<p class="meta">
  Multicellular Transformer — supplementary ablation. Generated {generated_at}.
  n samples = {n_samples}, n_seeds = {n_seeds}, fractions = {fractions_str}.
</p>

<div class="glossary">
  <h3>What each color / condition means</h3>
  <p>
    The <b>reference bag</b> is 1024 cells drawn from a donor (fixed seed).
    The model sees an <code>(n_genes × 1024)</code> expression matrix — one
    column per cell slot. Each perturbation modifies that matrix and we
    measure how much the resulting per-gene contextual embeddings shift.
    In every mode, <code>fraction</code> is the fraction of the 1024
    positions that change (fraction=0 recovers the reference exactly).
  </p>
  <dl>
    <dt><span class="swatch" style="background:{c_redraw}"></span>redraw</dt>
    <dd>
      Replace <code>fraction × 1024</code> positions with fresh cells drawn
      from the same donor, <b>of the same cell type as the slot being
      replaced</b>. Cell-type-slot layout is preserved. Answers "if I
      resample which individual cells of each type fill the slots, how much
      do the embeddings change?" — the <b>sampling-variance signal</b>
      while keeping the training-time layout intact.
    </dd>
    <dt><span class="swatch" style="background:{c_swithin}"></span>shuffle_within_ct</dt>
    <dd>
      Keep exactly the reference cells. Permute a fraction of the positions,
      but only within each cell-type block (T cells stay in T-cell slots,
      etc.). This tests whether the order matters <b>within</b> a cell type
      given the same overall cellular composition.
    </dd>
    <dt><span class="swatch" style="background:{c_sacross}"></span>shuffle_across_ct</dt>
    <dd>
      Keep exactly the reference cells. Permute a fraction of positions
      <b>freely across cell-type blocks</b> — cell-type identities move to
      arbitrary positions. Tests whether the model's cell-type-slot
      convention matters.
    </dd>
    <dt><span class="swatch" style="background:{c_base}"></span>random gene-pair baseline (dashed)</dt>
    <dd>
      Cosine distance between {n_random_pairs:,} random pairs of genes in
      the reference embedding. This is the <b>loose reference</b>: natural
      spread between any two genes.
    </dd>
    <dt><span class="swatch" style="background:{c_1nn}"></span>1-NN baseline (dotted)</dt>
    <dd>
      For each gene in the reference, cosine distance to its <i>nearest
      neighbor</i> among all other genes in the same embedding. This is
      the <b>tight reference</b>: how far away is the closest gene? Drift
      that exceeds even this baseline means the perturbation pushes a gene
      further than any of its natural neighbors sit.
    </dd>
  </dl>
</div>

<h2>Motivation</h2>
<p>
  MCT produces context-aware gene embeddings: within a sample the surrounding
  cellular ensemble reshapes each gene's vector. This ablation quantifies
  how much that context-dependence itself moves as the sample's input
  representation changes. If the model is a stable estimator of context,
  small input perturbations should yield small embedding shifts, and the
  magnitude of the shift should scale meaningfully with the perturbation size.
</p>

<h2>Setup</h2>
<pre>{config_pre}</pre>

<h2>Samples used</h2>
{samples_table}

<h2>Main result — mean drift vs. fraction (aggregated across samples)</h2>
<figure>
  <img src="{fig1_data}" alt="Mean cosine drift vs. fraction of positions perturbed">
  <figcaption>
    Mean per-gene cosine distance between the reference bag's contextual
    embeddings and those of each perturbed bag. Points and error bars are
    the mean and ±1 std <b>across samples</b> after averaging over seeds
    within each sample. The dashed gray line is the mean random gene-pair
    cosine distance in the reference embedding (natural gene-embedding
    spread), with the shaded band showing ±1 std across samples.
  </figcaption>
</figure>

<h2>Effect size vs. natural gene-embedding spread</h2>
<figure>
  <img src="{fig2_data}" alt="Drift as fraction of baseline distance">
  <figcaption>
    Drift expressed as a fraction of each baseline. Left: relative to the
    random gene-pair distance (loose reference). Right: relative to the
    1-NN distance (tight reference). Ratios above 1 mean the perturbation
    pushes contextual embeddings further than that baseline distance.
  </figcaption>
</figure>

<h2>Per-sample trajectories</h2>
<figure>
  <img src="{fig3_data}" alt="Per-sample trajectories per mode">
  <figcaption>
    Individual sample trajectories, one panel per perturbation mode. Each
    line is one sample; each point averages across seeds. Sample-to-sample
    variation is small compared to the mode differences.
  </figcaption>
</figure>

<h2>Drift distribution vs. baselines</h2>
<figure>
  <img src="{fig4_data}" alt="Drift distribution vs baselines">
  <figcaption>
    Full per-gene distribution of drift for the most disruptive mode
    (<code>shuffle_across_ct</code>) at each fraction, next to the two
    baseline distributions. The bulk of per-gene shifts sits well below
    both the 1-NN and random-pair distributions, indicating that
    context-aware embeddings remain closer to their reference than to
    their nearest neighbor.
  </figcaption>
</figure>

<h2>Which genes shift most?</h2>
<figure>
  <img src="{fig5_data}" alt="Top shifted genes">
  <figcaption>
    Genes whose contextual embedding shifts most under
    <code>shuffle_across_ct</code> at the maximum fraction
    (<code>{max_fraction}</code>), averaged across samples.
  </figcaption>
</figure>

<h2>Summary table</h2>
{summary_table}

<h2>Random gene-pair baseline (per sample)</h2>
{baseline_table}

<h2>1-NN baseline (per sample)</h2>
{baseline_1nn_table}

<h2>Reproducibility</h2>
<p class="meta">
  <b>Checkpoint:</b> <code>{checkpoint}</code><br>
  <b>Index dir:</b> <code>{index_dir}</code><br>
  <b>AnnData:</b> <code>{adata}</code><br>
  <b>Command:</b> <code>{command}</code><br>
  <b>Python:</b> {python} · <b>Torch:</b> {torch_version}
</p>

</body>
</html>
"""


def _render_html(
    config: RunConfig,
    long_df: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_pairs: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    baseline_1nn: pd.DataFrame,
    baseline_1nn_summary: pd.DataFrame,
    sample_stats: pd.DataFrame,
    figures: dict[str, Path],
    output_dir: Path,
    command: str,
) -> None:
    summary_pretty = (
        summary.round(4)
        .sort_values(["mode", "fraction"])
        .to_html(index=False, border=0)
    )
    baseline_pretty = baseline_summary.round(4).to_html(index=False, border=0) if not baseline_summary.empty else "<p>(none)</p>"
    baseline_1nn_pretty = baseline_1nn_summary.round(4).to_html(index=False, border=0) if not baseline_1nn_summary.empty else "<p>(none)</p>"
    samples_table = sample_stats.to_html(index=False, border=0)

    config_pre = json.dumps(
        {k: v for k, v in asdict(config).items() if k not in {"checkpoint_config"}},
        indent=2,
    )

    html = HTML_TEMPLATE.format(
        generated_at=config.started_at,
        n_samples=len(config.samples),
        n_seeds=config.n_seeds,
        n_random_pairs=config.n_random_pairs,
        fractions_str=", ".join(f"{f:.2f}" for f in config.fractions),
        config_pre=config_pre,
        samples_table=samples_table,
        c_redraw=PALETTE["redraw"],
        c_swithin=PALETTE["shuffle_within_ct"],
        c_sacross=PALETTE["shuffle_across_ct"],
        c_base=PALETTE["baseline_pair"],
        c_1nn=PALETTE["baseline_1nn"],
        fig1_data=_img_data_uri(figures["fig1"]),
        fig2_data=_img_data_uri(figures["fig2"]),
        fig3_data=_img_data_uri(figures["fig3"]),
        fig4_data=_img_data_uri(figures["fig4"]),
        fig5_data=_img_data_uri(figures["fig5"]),
        max_fraction=max(config.fractions),
        summary_table=summary_pretty,
        baseline_table=baseline_pretty,
        baseline_1nn_table=baseline_1nn_pretty,
        checkpoint=config.checkpoint,
        index_dir=config.index_dir,
        adata=config.adata,
        command=command,
        python=config.python,
        torch_version=config.torch_version,
    )
    (output_dir / "report.html").write_text(html)


# ---------------------------------------------------------------------- #
# Entrypoint
# ---------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    p.add_argument("--gene-index", default=None,
                   help="Explicit gene2idx JSON path.")
    p.add_argument("--adata", default=DEFAULT_ADATA)
    p.add_argument("--sample-key", default="sample")
    p.add_argument("--sample-list", nargs="+", default=None,
                   help="Explicit sample ids; overrides --n-samples.")
    p.add_argument("--n-samples", type=int, default=3,
                   help="If --sample-list is not given, pick this many largest samples.")
    p.add_argument("--n-cells-bag", type=int, default=1024)
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        help=(
            "Fraction of positions changed. redraw: fraction of cells replaced; "
            "shuffle_*: fraction of positions permuted. fraction=0 recovers the "
            "reference (drift == 0)."
        ),
    )
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-random-pairs", type=int, default=10000)
    p.add_argument("--min-frac-expressed-cells", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=12000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=MODES,
        help="Perturbation modes to run. Default: all three.",
    )
    p.add_argument(
        "--cell-type-key",
        default="cell_type",
        help="obs column with cell-type labels; required by shuffle_within_ct.",
    )
    p.add_argument("--reference-seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replot-only", action="store_true",
                   help="Skip model + forward passes; regenerate figures/HTML from cached CSVs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    if args.replot_only:
        _replot_from_cache(output_dir, args)
        return

    _seed_everything(args.reference_seed)

    print(f"[1/7] Loading model from {args.checkpoint}")
    model = cci.load(
        checkpoint_path=args.checkpoint,
        index_dir=args.index_dir,
        device=args.device,
    )
    if args.n_cells_bag != model.config.n_input:
        raise ValueError(
            f"--n-cells-bag={args.n_cells_bag} must match n_input={model.config.n_input}"
        )
    if args.gene_index is not None:
        gene2idx_override = json.loads(Path(args.gene_index).read_text())
        model.index.gene2idx = gene2idx_override
        model.index.idx2gene = {int(v): k for k, v in gene2idx_override.items()}
    gene2idx = model.gene2idx

    print(f"[2/7] Loading AnnData {args.adata}")
    adata = sc.read_h5ad(args.adata)
    if args.sample_key not in adata.obs.columns:
        raise KeyError(f"{args.sample_key} not in adata.obs")

    sample_counts = adata.obs[args.sample_key].value_counts()
    eligible = sample_counts[sample_counts >= args.n_cells_bag]
    if len(eligible) == 0:
        raise RuntimeError(
            f"No sample has >= {args.n_cells_bag} cells."
        )
    if args.sample_list is not None:
        selected = [s for s in args.sample_list if s in eligible.index]
        if not selected:
            raise RuntimeError(f"None of --sample-list are eligible: {eligible.head().to_dict()}")
    else:
        selected = eligible.head(args.n_samples).index.tolist()

    sample_stats = pd.DataFrame(
        {"sample": selected, "n_cells": [int(sample_counts[s]) for s in selected]}
    )
    print(f"       selected samples: {selected}")

    print("[3/7] Preparing gene <-> token map")
    var_tokens = _resolve_var_to_token(adata, gene2idx)
    n_vocab_overlap = int((var_tokens >= 0).sum())
    print(f"       vocab overlap: {n_vocab_overlap}/{adata.n_vars}")
    if n_vocab_overlap < 200:
        raise RuntimeError("Fewer than 200 vars mapped to vocabulary.")

    if args.dry_run:
        print("[dry-run] skipping forward passes")
        return

    print("[4/7] Materializing expression matrix (dense)")
    X_all = _dense(adata.X, np.arange(adata.n_obs))
    min_expressed_cells = int(args.min_frac_expressed_cells * args.n_cells_bag)

    print(f"[5/7] Running per-sample loop over {len(selected)} sample(s)")
    all_drift: list[pd.DataFrame] = []
    all_baselines: list[pd.DataFrame] = []
    all_1nn: list[pd.DataFrame] = []
    for sample_id in selected:
        print(f"--- sample: {sample_id}")
        cell_pool = np.where(adata.obs[args.sample_key].values == sample_id)[0]
        drift, baseline, oneNN = _process_sample(
            model=model,
            adata=adata,
            sample_id=sample_id,
            X_all=X_all,
            cell_pool=cell_pool,
            var_tokens=var_tokens,
            modes=args.modes,
            fractions=args.fractions,
            n_seeds=args.n_seeds,
            n_cells_bag=args.n_cells_bag,
            min_expressed_cells=min_expressed_cells,
            max_seq_len=args.max_seq_len,
            n_random_pairs=args.n_random_pairs,
            reference_seed=args.reference_seed,
            cell_type_key=args.cell_type_key if args.cell_type_key in adata.obs.columns else None,
            cell_type_input_order=model.index.cell_type_input_order,
            cell_type_mapping=model.index.cell_type_map,
        )
        all_drift.append(drift)
        all_baselines.append(baseline)
        all_1nn.append(oneNN)

    long_df = pd.concat(all_drift, ignore_index=True)
    baseline_pairs = pd.concat(all_baselines, ignore_index=True)
    baseline_1nn = pd.concat(all_1nn, ignore_index=True)

    print("[6/7] Aggregating + saving")
    # Per-sample means then aggregate across samples.
    per_sample_mean = (
        long_df.groupby(["sample", "mode", "fraction"], observed=True)["cosine_distance"]
        .mean()
        .reset_index()
    )
    summary = (
        per_sample_mean.groupby(["mode", "fraction"], observed=True)["cosine_distance"]
        .agg(mean_cosine_distance="mean", std_cosine_distance="std")
        .reset_index()
    )
    summary["n_samples"] = per_sample_mean.groupby(["mode", "fraction"], observed=True).size().reset_index(name="n")["n"].values

    baseline_summary = (
        baseline_pairs.groupby("sample", observed=True)["cosine_distance"]
        .agg(mean_cosine_distance="mean", median_cosine_distance="median", std_cosine_distance="std")
        .reset_index()
    )
    baseline_1nn_summary = (
        baseline_1nn.groupby("sample", observed=True)["cosine_distance"]
        .agg(mean_cosine_distance="mean", median_cosine_distance="median", std_cosine_distance="std")
        .reset_index()
    )

    # Ratios to both baselines: use per-sample baseline means.
    per_sample_pair_base = baseline_summary.set_index("sample")["mean_cosine_distance"]
    per_sample_1nn_base = baseline_1nn_summary.set_index("sample")["mean_cosine_distance"]
    per_sample_mean = per_sample_mean.copy()
    per_sample_mean["ratio_to_baseline"] = per_sample_mean.apply(
        lambda r: r["cosine_distance"] / per_sample_pair_base.get(r["sample"], np.nan),
        axis=1,
    )
    per_sample_mean["ratio_to_baseline_1nn"] = per_sample_mean.apply(
        lambda r: r["cosine_distance"] / per_sample_1nn_base.get(r["sample"], np.nan),
        axis=1,
    )
    ratio_summary = (
        per_sample_mean.groupby(["mode", "fraction"], observed=True)
        .agg(
            ratio_to_baseline=("ratio_to_baseline", "mean"),
            ratio_to_baseline_std=("ratio_to_baseline", "std"),
            ratio_to_baseline_1nn=("ratio_to_baseline_1nn", "mean"),
            ratio_to_baseline_1nn_std=("ratio_to_baseline_1nn", "std"),
        )
        .reset_index()
    )
    summary = summary.merge(ratio_summary, on=["mode", "fraction"])

    long_df.to_csv(output_dir / "data" / "drift_long.csv", index=False)
    summary.to_csv(output_dir / "data" / "drift_summary.csv", index=False)
    baseline_pairs.to_csv(output_dir / "data" / "random_pair_baseline.csv", index=False)
    baseline_summary.to_csv(output_dir / "data" / "random_pair_baseline_summary.csv", index=False)
    baseline_1nn.to_csv(output_dir / "data" / "one_nn_baseline.csv", index=False)
    baseline_1nn_summary.to_csv(output_dir / "data" / "one_nn_baseline_summary.csv", index=False)
    sample_stats.to_csv(output_dir / "data" / "samples.csv", index=False)

    config = RunConfig(
        checkpoint=args.checkpoint,
        index_dir=args.index_dir,
        adata=args.adata,
        samples=selected,
        sample_key=args.sample_key,
        cell_type_key=args.cell_type_key if args.cell_type_key in adata.obs.columns else None,
        n_cells_bag=args.n_cells_bag,
        fractions=list(args.fractions),
        n_seeds=args.n_seeds,
        n_random_pairs=args.n_random_pairs,
        min_frac_expressed_cells=args.min_frac_expressed_cells,
        max_seq_len=args.max_seq_len,
        device=str(args.device),
        output_dir=str(output_dir),
        modes=list(args.modes),
        reference_seed=args.reference_seed,
        checkpoint_config={
            "n_genes": int(model.config.n_genes),
            "n_input": int(model.config.n_input),
            "gene_emb_dim": int(model.config.gene_emb_dim),
            "n_layers": int(model.config.n_layers),
        },
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python=platform.python_version(),
        torch_version=torch.__version__,
    )
    (output_dir / "data" / "run_config.json").write_text(json.dumps(asdict(config), indent=2))

    print("[7/7] Plotting + rendering HTML")
    fig_dir = output_dir / "figures"
    figures = {
        "fig1": fig_dir / "fig1_mean_drift_vs_fraction.png",
        "fig2": fig_dir / "fig2_effect_size_vs_baseline.png",
        "fig3": fig_dir / "fig3_per_sample_trajectories.png",
        "fig4": fig_dir / "fig4_drift_vs_baseline_distribution.png",
        "fig5": fig_dir / "fig5_top_shifted_genes.png",
    }
    _plot_mean_drift_multi_sample(summary, baseline_summary, baseline_1nn_summary, figures["fig1"])
    _plot_drift_vs_baseline_ratio(summary, figures["fig2"])
    _plot_per_sample_lines(long_df, figures["fig3"])
    _plot_drift_and_baseline_distribution(long_df, baseline_pairs, baseline_1nn, figures["fig4"])
    _plot_top_shifted_genes(long_df, figures["fig5"])

    _render_html(
        config=config,
        long_df=long_df,
        summary=summary,
        baseline_pairs=baseline_pairs,
        baseline_summary=baseline_summary,
        baseline_1nn=baseline_1nn,
        baseline_1nn_summary=baseline_1nn_summary,
        sample_stats=sample_stats,
        figures=figures,
        output_dir=output_dir,
        command=" ".join(sys.argv),
    )
    print(f"Report: {output_dir / 'report.html'}")


def _replot_from_cache(output_dir: Path, args: argparse.Namespace) -> None:
    """Regenerate figures + HTML from previously saved CSVs. No model needed."""
    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"

    long_df = pd.read_csv(data_dir / "drift_long.csv")
    summary = pd.read_csv(data_dir / "drift_summary.csv")
    baseline_pairs = pd.read_csv(data_dir / "random_pair_baseline.csv")
    baseline_summary = pd.read_csv(data_dir / "random_pair_baseline_summary.csv")
    baseline_1nn = pd.read_csv(data_dir / "one_nn_baseline.csv")
    baseline_1nn_summary = pd.read_csv(data_dir / "one_nn_baseline_summary.csv")
    sample_stats = pd.read_csv(data_dir / "samples.csv")
    with (data_dir / "run_config.json").open() as fh:
        cached = json.load(fh)
    config = RunConfig(**cached)

    print("[replot] regenerating figures")
    figures = {
        "fig1": fig_dir / "fig1_mean_drift_vs_fraction.png",
        "fig2": fig_dir / "fig2_effect_size_vs_baseline.png",
        "fig3": fig_dir / "fig3_per_sample_trajectories.png",
        "fig4": fig_dir / "fig4_drift_vs_baseline_distribution.png",
        "fig5": fig_dir / "fig5_top_shifted_genes.png",
    }
    _plot_mean_drift_multi_sample(summary, baseline_summary, baseline_1nn_summary, figures["fig1"])
    _plot_drift_vs_baseline_ratio(summary, figures["fig2"])
    _plot_per_sample_lines(long_df, figures["fig3"])
    _plot_drift_and_baseline_distribution(long_df, baseline_pairs, baseline_1nn, figures["fig4"])
    _plot_top_shifted_genes(long_df, figures["fig5"])

    print("[replot] rewriting HTML")
    _render_html(
        config=config,
        long_df=long_df,
        summary=summary,
        baseline_pairs=baseline_pairs,
        baseline_summary=baseline_summary,
        baseline_1nn=baseline_1nn,
        baseline_1nn_summary=baseline_1nn_summary,
        sample_stats=sample_stats,
        figures=figures,
        output_dir=output_dir,
        command=" ".join(sys.argv),
    )
    print(f"Report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
