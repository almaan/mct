import json
import os

import numpy as np
import torch
import torch.cuda

# PyTorch >=2.6 defaults `torch.load(..., weights_only=True)`, which blocks
# Lightning's checkpoint resume because saved schedulers/optimizer state
# contain numpy scalars. Trust our own checkpoints; force weights_only=False
# before Lightning is imported.
_orig_torch_load = torch.load
def _torch_load_with_pickle(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_with_pickle

from lightning import seed_everything

from cell_cell_interactions.modeling.argparser import MulticellularTransformerArgparser
from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.modeling.trainer import get_pl_trainer
from cell_cell_interactions.modeling.utils import save_dicts
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule


def train(
    args: MulticellularTransformerArgparser,
):
    """Train script for the MulticellularTransformer model."""
    seed_everything(args.random_seed)
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if torch.cuda.is_available():
        # Prefer Flash / mem-efficient SDPA kernels over the math fallback.
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except AttributeError:
            pass  # older torch, keep defaults

    # get config
    config = MulticellularGraphConfig.from_dict(args.as_dict())

    cell_type_mapping = None
    if args.cell_type_mapping_file_path is not None:
        with open(args.cell_type_mapping_file_path) as f:
            cell_type_mapping = json.load(f)

    # get the data
    datamodule = CellarrDataModule(
        cell_tdb_uri=os.path.join(args.input_dir, "cell_metadata"),
        gene_tdb_uri=os.path.join(args.input_dir, "gene_annotation"),
        counts_tdb_uri=os.path.join(args.input_dir, "counts"),
        val_samples=None,
        val_set_proportion=1 - args.train_split_prop,
        label_column=args.cell_type_key,
        study_column="study",
        sample_column=args.sample_key,
        tissue_column="tissue",
        batch_size=args.batch_size,
        sample_size=args.n_cells_bag,
        num_workers=args.num_workers,
        min_sample_size=args.min_n_cells_per_sample,
        min_cell_type_size=args.min_cell_type_size,
        agg_cell_type_gex=args.agg_cell_type_gex,
        sample_sampling=args.sample_sampling,
        cell_sampling=args.cell_sampling,
        pad_value_expression=args.pad_value_expression,
        mask_prob=args.mask_prob,
        mask_tissue_prob=args.mask_tissue_prob,
        cell_type_sqrt_sampling=False,
        perc_min_expressed_cells_per_gene=args.perc_min_expressed_cells_per_gene,
        max_seq_len=args.max_seq_len,
        lognorm=True,
        target_sum=1e4,
        random_seed=args.random_seed,
        cell_type_mapping=cell_type_mapping,
        split_file=args.split_file,
        train_fraction=args.train_fraction,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    save_dicts(
        output_dir=output_dir,
        dicts=[datamodule.cell_type_input_order, datamodule.gene2idx, datamodule.tissue2idx],
        filenames=["cell_type_input_order", "gene2idx", "tissue2idx"],
    )

    # update config with the data from the data reader
    config.n_input = config.n_cells_bag if not config.agg_cell_type_gex else len(datamodule.cell_type_input_order)
    config.n_genes = len(datamodule.gene2idx)
    config.n_tissues = len(datamodule.tissue2idx)
    # scale the lr with the nr of gpus
    n_gpus = torch.cuda.device_count()
    config.lr = config.lr * np.sqrt(max(n_gpus, 1) * args.num_nodes)

    # get model
    model = MulticellularTransformerPLModel(config)
    print("Nr of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # fetch the trainer
    trainer = get_pl_trainer(config, output_dir, args.wandb_api_key, args.devices, args.sample_sampling)

    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = MulticellularTransformerArgparser().parse_args()
    print("Training with args:", args)
    train(args)
