import json
import os
from collections import OrderedDict

import numpy as np
import torch.cuda
from lightning import seed_everything

from cell_cell_interactions.modeling.argparser import MulticellularTransformerArgparser
from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.modeling.trainer import get_pl_trainer
from cell_cell_interactions.modeling.utils import save_dicts
from cell_cell_interactions.pp.adata_rnaseq_datamodule import AdataRnaseqDataModule


def adjust_state_dict(state_dict: dict) -> OrderedDict:
    """Adjust the state dict."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # we always want to finetune the gene expression encoder, tissue emb and gex decoder, so we remove it
        # from pretraining state dict
        if "tissue" in k or "gene_expression_encoder" in k or "gex_decoder" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict


def train(
    args: MulticellularTransformerArgparser,
):
    """Train script for the MulticellularTransformer model."""
    seed_everything(args.random_seed)
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get config
    config = MulticellularGraphConfig.from_dict(args.as_dict())

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    with open(args.gene2idx_file_path) as f:
        gene2idx = json.load(f)

    # get the data
    datamodule = AdataRnaseqDataModule(
        adata_file_path=args.adata_file_path,
        gene2idx=gene2idx,
        val_samples=None,
        val_set_proportion=1 - args.train_split_prop,
        sample_size=args.n_cells_bag,
        sample_column=args.sample_key,
        label_column=args.cell_type_key,
        tissue_column="tissue",
        min_sample_size=args.min_n_cells_per_sample,
        min_cell_type_size=args.min_cell_type_size,
        sample_sampling=args.sample_sampling,
        cell_sampling=args.cell_sampling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        agg_cell_type_gex=args.agg_cell_type_gex,
        pad_value_expression=args.pad_value_expression,
        mask_prob=args.mask_prob,
        mask_tissue_prob=args.mask_tissue_prob,
        perc_min_expressed_cells_per_gene=args.perc_min_expressed_cells_per_gene,
        max_seq_len=args.max_seq_len,
        random_seed=args.random_seed,
        backed=args.backed,
    )
    save_dicts(
        output_dir=output_dir,
        dicts=[datamodule.cell_type_input_order, datamodule.gene2idx],
        filenames=["cell_type_input_order", "gene2idx"],
    )

    # update config with the data from the data reader
    config.n_input = config.n_cells_bag if not config.agg_cell_type_gex else len(datamodule.cell_type_input_order)
    config.n_genes = len(datamodule.gene2idx)
    config.n_tissues = len(datamodule.tissue2idx)
    # scale the lr with the nr of gpus
    n_gpus = torch.cuda.device_count()
    config.lr = config.lr * np.sqrt(max(n_gpus, 1) * args.num_nodes)

    # get model
    model = MulticellularTransformerPLModel(
        config,
        pretrained_ckpt_path=args.pretrained_ckpt_path,
    )

    # load pretrained weights
    if args.pretrained_ckpt_path is not None:
        state_dict = torch.load(args.pretrained_ckpt_path)["state_dict"]
        state_dict = adjust_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)

    print("Nr of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # fetch the trainer
    trainer = get_pl_trainer(config, output_dir, args.wandb_api_key, args.devices, args.sample_sampling)

    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = MulticellularTransformerArgparser().parse_args()
    print("Training with args:", args)
    train(args)
