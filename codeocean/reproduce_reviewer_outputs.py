"""Generate reviewer-facing CodeOcean results for Multicellular Transformer.

The script is intentionally headless and CPU-compatible. It uses bundled
tutorial resources plus a pretrained checkpoint to write concrete outputs to
CodeOcean's /results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score

from cell_cell_interactions.tl.condition_specific_genes import jaccard_similarity


REQUIRED_DATA_FILES = [
    "gene_symbol2idx.json",
    "receptor_ligand_with_negatives.csv",
    "receptors_and_ligands.csv",
    "tumor_microenvironment_hvg_genes.csv",
    "immune_across_organs_condition_specific_genes.parquet",
    "lupus_disease_condition_specific_genes.parquet",
    "covid_stephenson_disease_condition_specific_genes.parquet",
    "immune_across_organs_neighbors_sample.parquet",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tutorial-data-dir", type=Path, default=Path("/data/tutorials/data"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("/data/checkpoints/multicellular_transformer.ckpt"))
    parser.add_argument("--results-dir", type=Path, default=Path("/results"))
    parser.add_argument("--code-dir", type=Path, default=Path("/code"))
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="Permit local dry-runs that validate data outputs but skip checkpoint-derived analyses.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file is missing: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_checkpoint_embeddings(checkpoint_path: Path, allow_missing: bool) -> torch.Tensor | None:
    if not checkpoint_path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(
            "Pretrained checkpoint not found. Upload it to "
            f"{checkpoint_path} or set CODEOCEAN_DATA_DIR so run.sh can locate it."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    key = "model.gene_id_classifier.decoder.weight"
    if key not in state_dict:
        available = ", ".join(sorted(state_dict)[:20])
        raise KeyError(f"Checkpoint does not contain {key!r}. First available keys: {available}")

    return state_dict[key].detach().cpu().float()


def normalized_cosine_scores(
    embeddings: torch.Tensor,
    gene_symbol2idx: dict[str, int],
    left_genes: pd.Series,
    right_genes: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    max_idx = embeddings.shape[0] - 1
    left_idx = left_genes.map(gene_symbol2idx)
    right_idx = right_genes.map(gene_symbol2idx)
    valid = left_idx.notna() & right_idx.notna() & (left_idx <= max_idx) & (right_idx <= max_idx)

    scores = pd.Series(np.nan, index=left_genes.index, dtype=float)
    if not valid.any():
        return scores, valid

    left_tensor = embeddings[torch.tensor(left_idx[valid].astype(int).to_numpy())]
    right_tensor = embeddings[torch.tensor(right_idx[valid].astype(int).to_numpy())]
    cosine = torch.nn.functional.cosine_similarity(left_tensor, right_tensor, dim=1)
    scores.loc[valid] = ((cosine + 1.0) / 2.0).numpy()
    return scores, valid


def write_data_inventory(data_dir: Path, tables_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        row: dict[str, Any] = {"file": path.name, "size_bytes": path.stat().st_size, "suffix": path.suffix}
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
            row["columns"] = ", ".join(df.columns.astype(str).tolist())
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            row["rows"] = len(df)
            row["columns"] = ", ".join(df.columns.astype(str).tolist())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            row["entries"] = len(data)
        rows.append(row)

    pd.DataFrame(rows).to_csv(tables_dir / "data_inventory.csv", index=False)
    return rows


def summarize_condition_specific_genes(data_dir: Path, results_dir: Path) -> dict[str, Any]:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    files = {
        "immune_across_organs": "immune_across_organs_condition_specific_genes.parquet",
        "lupus": "lupus_disease_condition_specific_genes.parquet",
        "covid_stephenson": "covid_stephenson_disease_condition_specific_genes.parquet",
    }

    summary_rows = []
    for dataset, filename in files.items():
        df = pd.read_parquet(data_dir / filename)
        top = df.head(25).copy()
        top.insert(0, "dataset", dataset)
        top.to_csv(tables_dir / f"top_condition_specific_genes_{dataset}.csv", index=False)
        summary_rows.append(
            {
                "dataset": dataset,
                "n_genes": len(df),
                "top_gene": df.iloc[0]["gene_name"],
                "top_condition_specificity_score": float(df.iloc[0]["condition_specificity_score"]),
                "median_condition_specificity_score": float(df["condition_specificity_score"].median()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "condition_specific_gene_summary.csv", index=False)

    immune_df = pd.read_parquet(data_dir / files["immune_across_organs"]).copy()
    immune_df["one_minus_overlap_across_conditions"] = 1.0 - immune_df["overlap_across_conditions"]
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=immune_df,
        x="overlap_within_condition",
        y="one_minus_overlap_across_conditions",
        s=18,
        color="#003399",
        alpha=0.45,
        edgecolor=None,
    )
    plt.xlabel("Jaccard similarity within condition")
    plt.ylabel("1 - Jaccard similarity across conditions")
    plt.tight_layout()
    plt.savefig(figures_dir / "immune_across_organs_condition_specificity_scatter.png", dpi=200)
    plt.close()

    neighbors_df = pd.read_parquet(data_dir / "immune_across_organs_neighbors_sample.parquet")
    gene_df = neighbors_df[neighbors_df["gene_name"] == "OIT3"].copy()
    heatmap_rows = len(gene_df)
    if heatmap_rows >= 2:
        matrix = np.stack(gene_df["neighbors"].tolist())[:, :20]
        similarity = jaccard_similarity(matrix)
        plt.figure(figsize=(8, 7))
        sns.heatmap(similarity, cmap="rocket_r", cbar_kws={"label": "Jaccard similarity"})
        plt.title("OIT3 sample-neighborhood similarity")
        plt.xlabel("Sample")
        plt.ylabel("Sample")
        plt.tight_layout()
        plt.savefig(figures_dir / "oit3_neighbor_similarity_heatmap.png", dpi=200)
        plt.close()

    return {
        "condition_specific_datasets": summary_rows,
        "oit3_neighbor_rows": int(heatmap_rows),
    }


def write_receptor_ligand_outputs(
    data_dir: Path,
    results_dir: Path,
    embeddings: torch.Tensor | None,
) -> dict[str, Any]:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    if embeddings is None:
        message = (
            "Checkpoint was not available, so checkpoint-derived receptor-ligand "
            "and gene-neighbor outputs were skipped."
        )
        (results_dir / "logs" / "checkpoint_missing.txt").write_text(message + "\n")
        return {"checkpoint_available": False}

    gene_symbol2idx = json.loads((data_dir / "gene_symbol2idx.json").read_text())
    rl_df = pd.read_csv(data_dir / "receptor_ligand_with_negatives.csv")
    rl_df["mct_cosine_similarity"], valid = normalized_cosine_scores(
        embeddings,
        gene_symbol2idx,
        rl_df["ligand"],
        rl_df["receptor"],
    )
    scored_df = rl_df.dropna(subset=["mct_cosine_similarity"]).copy()
    scored_df.to_csv(tables_dir / "receptor_ligand_cosine_scores.csv", index=False)

    if scored_df["label"].nunique() < 2:
        raise ValueError("Receptor-ligand benchmark needs both positive and negative labels after filtering.")

    auroc = roc_auc_score(scored_df["label"], scored_df["mct_cosine_similarity"])
    auprc = average_precision_score(scored_df["label"], scored_df["mct_cosine_similarity"])
    metrics = {
        "checkpoint_available": True,
        "embedding_shape": list(embeddings.shape),
        "pairs_total": int(len(rl_df)),
        "pairs_scored": int(len(scored_df)),
        "pairs_missing_embedding": int((~valid).sum()),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }
    write_json(results_dir / "logs" / "receptor_ligand_metrics.json", metrics)

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(
        scored_df["label"],
        scored_df["mct_cosine_similarity"],
        ax=ax,
        name=f"Multicellular Transformer (AUROC={auroc:.3f})",
        color="#003399",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_title("Receptor-ligand ROC")
    fig.tight_layout()
    fig.savefig(figures_dir / "receptor_ligand_roc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(
        scored_df["label"],
        scored_df["mct_cosine_similarity"],
        ax=ax,
        name=f"Multicellular Transformer (AUPRC={auprc:.3f})",
        color="#003399",
    )
    ax.set_title("Receptor-ligand precision-recall")
    fig.tight_layout()
    fig.savefig(figures_dir / "receptor_ligand_precision_recall.png", dpi=200)
    plt.close(fig)

    idx2gene_symbol = {idx: gene for gene, idx in gene_symbol2idx.items() if idx < embeddings.shape[0]}
    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    receptor = "GPR84"
    ligand_candidates = pd.read_csv(data_dir / "receptors_and_ligands.csv")["gene_name"].drop_duplicates()
    ligand_candidates = [gene for gene in ligand_candidates if gene in gene_symbol2idx and gene_symbol2idx[gene] < embeddings.shape[0]]
    if receptor in gene_symbol2idx and ligand_candidates:
        receptor_embedding = normalized_embeddings[gene_symbol2idx[receptor]]
        ligand_indices = torch.tensor([gene_symbol2idx[gene] for gene in ligand_candidates])
        ligand_scores = torch.matmul(normalized_embeddings[ligand_indices], receptor_embedding)
        values, indices = torch.topk(ligand_scores, k=min(25, len(ligand_candidates)))
        pd.DataFrame(
            {
                "receptor": receptor,
                "ligand": [ligand_candidates[i] for i in indices.tolist()],
                "cosine_similarity": values.numpy(),
            }
        ).to_csv(tables_dir / "top_ligands_for_gpr84.csv", index=False)

    query_gene = "CD14"
    if query_gene in gene_symbol2idx and gene_symbol2idx[query_gene] < embeddings.shape[0]:
        gene_embedding = normalized_embeddings[gene_symbol2idx[query_gene]]
        scores = torch.matmul(normalized_embeddings, gene_embedding)
        values, indices = torch.topk(scores, k=min(26, embeddings.shape[0]))
        rows = []
        for value, idx in zip(values.tolist(), indices.tolist()):
            gene = idx2gene_symbol.get(idx)
            if gene is None or gene == query_gene:
                continue
            rows.append({"query_gene": query_gene, "gene_name": gene, "cosine_similarity": value})
            if len(rows) == 25:
                break
        pd.DataFrame(rows).to_csv(tables_dir / "top_similar_genes_to_cd14.csv", index=False)

    hvg_df = pd.read_csv(data_dir / "tumor_microenvironment_hvg_genes.csv")
    hvg_present = hvg_df[hvg_df["gene_name"].isin(gene_symbol2idx)].copy()
    hvg_present["embedding_index"] = hvg_present["gene_name"].map(gene_symbol2idx)
    hvg_present = hvg_present[hvg_present["embedding_index"] < embeddings.shape[0]]
    hvg_present.head(100).to_csv(tables_dir / "top_hvg_genes_with_embeddings.csv", index=False)

    return metrics


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    logs_dir = results_dir / "logs"
    for directory in [results_dir, tables_dir, figures_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_DATA_FILES:
        require_file(args.tutorial_data_dir / filename)

    inventory = write_data_inventory(args.tutorial_data_dir, tables_dir)
    condition_summary = summarize_condition_specific_genes(args.tutorial_data_dir, results_dir)
    embeddings = load_checkpoint_embeddings(args.checkpoint_path, args.allow_missing_checkpoint)
    receptor_ligand_summary = write_receptor_ligand_outputs(args.tutorial_data_dir, results_dir, embeddings)

    run_summary = {
        "code_dir": str(args.code_dir),
        "tutorial_data_dir": str(args.tutorial_data_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "allow_missing_checkpoint": args.allow_missing_checkpoint,
        "data_files": inventory,
        "condition_specific_outputs": condition_summary,
        "receptor_ligand_outputs": receptor_ligand_summary,
    }
    write_json(logs_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

