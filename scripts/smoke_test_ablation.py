"""Smoke test: instantiate CellarrDataModule + model with new args, run
a few train/val/test batches on 1 GPU. Verifies:
  - split_file parsed, 3-way split materialized
  - collate sparse fix returns tensors of expected shape
  - per-sample scalars serve __getitem__ without pandas .loc
  - torch.compile + bf16-true forward/backward runs
  - fork-safe TileDB re-open under num_workers>0
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO = Path("/cv/data/braid/andera29/projs/mct/mct_pub")
sys.path.insert(0, str(REPO))

CELLARR = Path("/cv/scratch/u/andera29/mct_pretraining/cellarr")
SPLIT = CELLARR / "ablation_split.json"

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule


def main() -> None:
    assert torch.cuda.is_available(), "smoke test expects a GPU"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    import json
    cell_type_map = json.loads((CELLARR / "cell_type_map.json").read_text())

    print(f"[dm] building CellarrDataModule from {CELLARR}")
    t0 = time.time()
    dm = CellarrDataModule(
        cell_tdb_uri=str(CELLARR / "cell_metadata"),
        gene_tdb_uri=str(CELLARR / "gene_annotation"),
        counts_tdb_uri=str(CELLARR / "counts"),
        val_samples=None,
        label_column="cell_type",
        study_column="study",
        sample_column="sample",
        tissue_column="tissue",
        batch_size=1,
        sample_size=256,
        num_workers=2,
        min_sample_size=256,
        min_cell_type_size=0.01,
        agg_cell_type_gex=False,
        sample_sampling="weighted",
        cell_sampling="stratified",
        pad_value_expression=0.0,
        mask_prob=0.15,
        mask_tissue_prob=0.25,
        perc_min_expressed_cells_per_gene=0.01,
        max_seq_len=2500,
        lognorm=True,
        target_sum=1e4,
        random_seed=32,
        cell_type_mapping=cell_type_map,
        split_file=str(SPLIT),
        train_fraction=0.10,
        prefetch_factor=2,
        persistent_workers=True,
    )
    print(f"[dm] built in {time.time() - t0:.1f}s")
    print(f"[dm] train samples: {len(dm.train_dataset.samples)}")
    assert dm.val_dataset is not None, "val dataset missing"
    assert dm.test_dataset is not None, "test dataset missing (split_file didn't wire test_df)"
    print(f"[dm] val   samples: {len(dm.val_dataset.samples)}")
    print(f"[dm] test  samples: {len(dm.test_dataset.samples)}")

    # sanity: test/val/train sample sets disjoint
    train_s = set(dm.train_dataset.samples)
    val_s = set(dm.val_dataset.samples)
    test_s = set(dm.test_dataset.samples)
    assert train_s.isdisjoint(val_s), "train/val overlap"
    assert train_s.isdisjoint(test_s), "train/test overlap"
    assert val_s.isdisjoint(test_s), "val/test overlap"
    print("[dm] disjoint train / val / test: OK")

    # build model
    config = MulticellularGraphConfig(
        n_cells_bag=256,
        n_input=256,
        n_genes=len(dm.gene2idx),
        n_tissues=len(dm.tissue2idx),
        gene_emb_dim=64,
        hidden_dim=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=2500,
        dropout=0.2,
        transformer_dropout=0.1,
        lr=1e-3,
        weight_decay=0.02,
        alpha_reconstruction_loss_weight=0.01,
        gamma_tissue_loss_weight=0.05,
        agg_cell_type_gex=False,
        compile_model=True,
        precision="bf16-true",
        compute_pretraining_metrics=False,
        batch_size=1,
    )
    device = torch.device("cuda")
    print("[model] instantiating + compiling")
    pl_model = MulticellularTransformerPLModel(config).to(device)
    pl_model.train()
    n_params = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params:,}")

    # bf16-true: put params in bf16 to match forward dtype
    pl_model = pl_model.to(torch.bfloat16)
    opt = torch.optim.AdamW([p for p in pl_model.parameters() if p.requires_grad], lr=1e-3)

    def iterate(loader, name, n_batches=3):
        print(f"[loader:{name}] fetching {n_batches} batches")
        it = iter(loader)
        for i in range(n_batches):
            t0 = time.time()
            batch = next(it)
            batch = _to_device(batch, device, dtype=torch.bfloat16)
            output = pl_model.model(
                expression_matrix=batch.expression_matrix,
                gene_ids=batch.gene_ids,
                is_masked_gene_expression=batch.is_masked_gene_expression,
                padding_mask=batch.padding_mask,
                tissue=batch.tissue_idx,
            )
            loss_dict = pl_model.compute_loss(batch, output)
            loss = loss_dict["total_loss"]
            if name == "train":
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            dt = time.time() - t0
            gex_shape = tuple(batch.expression_matrix.shape)
            gid_shape = tuple(batch.gene_ids.shape)
            print(
                f"[loader:{name}] batch {i}: gex={gex_shape} gene_ids={gid_shape} "
                f"loss={loss.item():.4f} dt={dt:.2f}s"
            )
        print(f"[loader:{name}] OK")

    iterate(dm.train_dataloader(), "train")
    iterate(dm.val_dataloader(), "val")
    iterate(dm.test_dataloader(), "test")

    print("[smoke] all steps passed")


def _to_device(batch, device, dtype):
    """Move tensor fields of a BagOfCellsBatch to device + dtype (bf16 for float)."""
    for field in (
        "expression_matrix",
        "gene_ids",
        "labels",
        "is_masked_gene_id",
        "is_masked_gene_expression",
        "padding_mask",
        "tissue_idx",
        "tissue_labels",
        "n_expressed_genes",
    ):
        val = getattr(batch, field, None)
        if not isinstance(val, torch.Tensor):
            continue
        if val.is_floating_point():
            val = val.to(device=device, dtype=dtype)
        else:
            val = val.to(device=device)
        setattr(batch, field, val)
    return batch


if __name__ == "__main__":
    main()
