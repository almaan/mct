import json
import os
from collections import defaultdict

import pandas as pd
import torch.cuda
from lightning import seed_everything
from tqdm import tqdm

from cell_cell_interactions.modeling.argparser import MulticellularTransformerArgparser
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule


def run(
    args: MulticellularTransformerArgparser,
):
    """Train script for the MulticellularTransformer model."""
    seed_everything(args.random_seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cell_type_mapping = None
    if args.cell_type_mapping_file_path is not None:
        with open(args.cell_type_mapping_file_path) as f:
            cell_type_mapping = json.load(f)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # get the data
    datamodule = CellarrDataModule(
        cell_tdb_uri=os.path.join(args.input_dir, "cell_metadata"),
        gene_tdb_uri=os.path.join(args.input_dir, "gene_annotation"),
        counts_tdb_uri=os.path.join(args.input_dir, "counts"),
        val_samples=None,
        val_set_proportion=0.0,
        label_column=args.cell_type_key,
        study_column="study",
        sample_column=args.sample_key,
        tissue_column="tissue",
        batch_size=1,
        sample_size=args.n_cells_bag,
        num_workers=args.num_workers,
        min_sample_size=args.min_n_cells_per_sample,
        min_cell_type_size=args.min_cell_type_size,
        agg_cell_type_gex=args.agg_cell_type_gex,
        sample_sampling=args.sample_sampling,
        cell_sampling=args.cell_sampling,
        pad_value_expression=args.pad_value_expression,
        mask_prob=0.0,
        mask_tissue_prob=0.0,
        cell_type_sqrt_sampling=False,
        perc_min_expressed_cells_per_gene=args.perc_min_expressed_cells_per_gene,
        max_seq_len=args.max_seq_len,
        lognorm=True,
        target_sum=1e4,
        random_seed=args.random_seed,
        cell_type_mapping=cell_type_mapping,
        # restrict_to_conditions={
        #     "study": "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/hlca/hlca_preprocessed_all_genes.h5ad"
        # },
    )
    idx2tissue = {idx: tissue for tissue, idx in datamodule.tissue2idx.items()}

    # load trained model
    model = MulticellularTransformerPLModel.load_from_checkpoint(args.ckpt_path).cuda()
    model.eval()

    dataloader = datamodule.train_dataloader()

    embeds = []
    metadata = defaultdict(list)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), mininterval=10):
            if i == args.max_steps:
                break
            output = model.model(
                expression_matrix=batch.expression_matrix.to(model.device),
                gene_ids=batch.gene_ids.to(model.device),
                is_masked_gene_expression=batch.is_masked_gene_expression.to(model.device),
                padding_mask=batch.padding_mask.to(model.device),
                tissue=batch.tissue_idx.to(model.device),
            )
            # take the mean of gene embeddings in the sample to get agg embedding
            embeds += list(output.last_hidden_state.mean(dim=1).cpu())
            metadata["dataset"] += [i.split("/")[-1] for i in batch.study_label]
            metadata["sample"] += batch.sample_label
            metadata["disease"] += batch.disease
            metadata["tissue"] += [idx2tissue[i.item()] for i in batch.tissue_idx]

            if (i + 1) % 5000 == 0:
                torch.save(torch.stack(embeds), os.path.join(args.output_dir, "agg_sample_embeddings.pt"))
                pd.DataFrame(metadata).to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)

    torch.save(torch.stack(embeds), os.path.join(args.output_dir, "agg_sample_embeddings.pt"))
    pd.DataFrame(metadata).to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)


if __name__ == "__main__":
    args = MulticellularTransformerArgparser().parse_args()
    print("Training with args:", args)
    run(args)
