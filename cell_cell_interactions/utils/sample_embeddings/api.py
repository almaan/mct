"""Library-form wrappers for the sample-embedding collection scripts.

These expose the same computation as the ``run_collect_sample_embeddings.py``
and ``run_collect_sample_neighbors.py`` CLI scripts, but as normal Python
functions that take an already-loaded model + dataloader. The scripts remain
as CLI shims so existing ``python -m ...`` invocations keep working.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from cell_cell_interactions.api.model import MCTModel
from cell_cell_interactions.utils.sample_embeddings.run_collect_sample_neighbors import (
    calculate_neighbors,
    extract_number,
    save_data,
)

__all__ = [
    "calculate_neighbors",
    "collect_sample_embeddings",
    "collect_sample_neighbors",
    "save_data",
]


def _idx2tissue_from(datamodule) -> dict[int, str]:
    return {idx: tissue for tissue, idx in datamodule.tissue2idx.items()}


def collect_sample_embeddings(
    model: MCTModel,
    dataloader,
    output_dir: str | os.PathLike,
    max_steps: int | None = None,
    save_every_n_batches: int = 5000,
    idx2tissue: dict[int, str] | None = None,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Iterate ``dataloader``, run the model, save mean-pooled sample embeddings.

    Writes ``agg_sample_embeddings.pt`` and ``metadata.csv`` under
    ``output_dir``, checkpointing every ``save_every_n_batches`` batches so
    long runs are resumable via file inspection.

    Returns the stacked embedding tensor and the metadata DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeds: list[torch.Tensor] = []
    metadata: dict[str, list] = defaultdict(list)

    model.eval()
    for i, batch in enumerate(tqdm(dataloader, mininterval=10)):
        if max_steps is not None and i >= max_steps:
            break
        output = model.embed_batch(batch)
        embeds += list(output.last_hidden_state.mean(dim=1).cpu())

        if batch.study_label is not None:
            metadata["dataset"] += [s.split("/")[-1] for s in batch.study_label]
        if batch.sample_label is not None:
            metadata["sample"] += list(batch.sample_label)
        if batch.disease is not None:
            metadata["disease"] += list(batch.disease)
        if batch.tissue_idx is not None and idx2tissue is not None:
            metadata["tissue"] += [idx2tissue[int(idx)] for idx in batch.tissue_idx]

        if (i + 1) % save_every_n_batches == 0:
            torch.save(torch.stack(embeds), output_dir / "agg_sample_embeddings.pt")
            pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)

    stacked = torch.stack(embeds) if embeds else torch.empty(0)
    torch.save(stacked, output_dir / "agg_sample_embeddings.pt")
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    return stacked, meta_df


def collect_sample_neighbors(
    model: MCTModel,
    dataloader,
    output_dir: str | os.PathLike,
    n_neighbors: int = 50,
    max_steps: int | None = None,
    save_every_n_batches: int = 1000,
    idx2tissue: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute per-sample gene-embedding neighbor tables.

    Writes ``neighbors_{i}.parquet`` shards, ``metadata.csv``, and a
    consolidated ``neighbors_all.parquet`` under ``output_dir``. Returns the
    consolidated DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices: list[int] = []
    gene_ids_arr: list[torch.Tensor] = []
    neighbors_arr: list[torch.Tensor] = []
    metadata: dict[str, list] = defaultdict(list)

    model.eval()
    last_i = -1
    for i, batch in enumerate(tqdm(dataloader)):
        if max_steps is not None and i >= max_steps:
            break
        last_i = i
        output = model.embed_batch(batch)

        gene_ids = batch.gene_ids.squeeze(0).cpu()
        sample_indices += [i] * len(gene_ids)
        gene_ids_arr.append(gene_ids)

        embeddings = output.last_hidden_state.squeeze(0).cpu()
        neighbors = calculate_neighbors(embeddings, n_neighbors=n_neighbors)
        neighbors_arr.append(gene_ids[neighbors])

        if batch.study_label is not None:
            metadata["dataset"] += [s.split("/")[-1] for s in batch.study_label]
        if batch.sample_label is not None:
            metadata["sample"] += list(batch.sample_label)
        if batch.disease is not None:
            metadata["disease"] += list(batch.disease)
        if batch.tissue_idx is not None and idx2tissue is not None:
            metadata["tissue"] += [idx2tissue[int(idx)] for idx in batch.tissue_idx]

        if (i + 1) % save_every_n_batches == 0:
            save_data(sample_indices, gene_ids_arr, neighbors_arr, str(output_dir), i)
            pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)
            sample_indices = []
            gene_ids_arr = []
            neighbors_arr = []

    if gene_ids_arr:
        save_data(sample_indices, gene_ids_arr, neighbors_arr, str(output_dir), last_i)
    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)

    filepaths = sorted(
        (str(output_dir / f) for f in os.listdir(output_dir) if f.endswith(".parquet")),
        key=extract_number,
    )
    df = pd.concat([pd.read_parquet(f) for f in filepaths]) if filepaths else pd.DataFrame()
    if not df.empty:
        df.to_parquet(output_dir / "neighbors_all.parquet")
    return df
