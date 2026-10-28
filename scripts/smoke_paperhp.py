"""Quick smoke: paper-scale config, 3 train batches, timing per batch."""

import os
import sys
import time
from pathlib import Path
import json

import torch

REPO = Path("/cv/data/braid/andera29/projs/mct/mct_pub")
sys.path.insert(0, str(REPO))

CELLARR = Path("/cv/scratch/u/andera29/mct_pretraining/cellarr")
SPLIT = CELLARR / "ablation_split.json"

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule


def main() -> None:
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    cell_type_map = json.loads((CELLARR / "cell_type_map.json").read_text())

    print("[dm] build")
    t0 = time.time()
    dm = CellarrDataModule(
        cell_tdb_uri=str(CELLARR / "cell_metadata"),
        gene_tdb_uri=str(CELLARR / "gene_annotation"),
        counts_tdb_uri=str(CELLARR / "counts"),
        label_column="cell_type",
        study_column="study",
        sample_column="sample",
        tissue_column="tissue",
        batch_size=4,
        sample_size=1024,
        num_workers=4,
        min_sample_size=256,
        min_cell_type_size=0.01,
        cell_sampling="stratified",
        mask_prob=0.15,
        mask_tissue_prob=0.25,
        perc_min_expressed_cells_per_gene=0.01,
        max_seq_len=12000,
        lognorm=True,
        target_sum=1e4,
        random_seed=32,
        cell_type_mapping=cell_type_map,
        split_file=str(SPLIT),
        train_fraction=0.10,
        prefetch_factor=2,
        persistent_workers=True,
    )
    print(f"[dm] built in {time.time()-t0:.1f}s")

    config = MulticellularGraphConfig(
        n_cells_bag=1024,
        n_input=1024,
        n_genes=len(dm.gene2idx),
        n_tissues=len(dm.tissue2idx),
        gene_emb_dim=512,
        hidden_dim=768,
        n_heads=8,
        n_layers=12,
        max_seq_len=12000,
        dropout=0.2,
        transformer_dropout=0.1,
        lr=1.5e-4,
        weight_decay=0.02,
        alpha_reconstruction_loss_weight=0.01,
        gamma_tissue_loss_weight=0.05,
        compile_model=False,
        precision="bf16-true",
        batch_size=4,
    )
    device = torch.device("cuda")
    print("[model] instantiate")
    pl_model = MulticellularTransformerPLModel(config).to(device).to(torch.bfloat16)
    pl_model.train()
    n = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    print(f"[model] params: {n:,}")

    opt = torch.optim.AdamW([p for p in pl_model.parameters() if p.requires_grad], lr=1.5e-4)

    loader = dm.train_dataloader()
    it = iter(loader)
    print("[fetch] pulling batches")
    for i in range(4):
        t0 = time.time()
        batch = next(it)
        t_fetch = time.time() - t0

        t0 = time.time()
        for f in ("expression_matrix", "gene_ids", "labels", "is_masked_gene_id",
                  "is_masked_gene_expression", "padding_mask", "tissue_idx",
                  "tissue_labels", "n_expressed_genes"):
            v = getattr(batch, f, None)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device=device)
                setattr(batch, f, v)

        out = pl_model.model(
            expression_matrix=batch.expression_matrix,
            gene_ids=batch.gene_ids,
            is_masked_gene_expression=batch.is_masked_gene_expression,
            padding_mask=batch.padding_mask,
            tissue=batch.tissue_idx,
        )
        losses = pl_model.compute_loss(batch, out)
        loss = losses["total_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t_step = time.time() - t0

        gex = tuple(batch.expression_matrix.shape)
        gid = tuple(batch.gene_ids.shape)
        print(f"[batch {i}] fetch={t_fetch:.2f}s step={t_step:.2f}s "
              f"gex={gex} gene_ids={gid} loss={loss.item():.3f}")

    print("[done]")


if __name__ == "__main__":
    main()
