import json
import os
import re
from collections import defaultdict

import pandas as pd
import torch.cuda
from lightning import seed_everything
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from cell_cell_interactions.modeling.argparser import MulticellularTransformerArgparser
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.cellarr_datamodule import CellarrDataModule


def calculate_neighbors(embeddings, n_neighbors=20, metric="cosine"):
    """Calculate the nearest neighbors."""
    # + 1 to account for the query point
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    # remove the first element, which is the query point itself
    return indices[:, 1:]


def save_data(
    sample_indices: list[int],
    gene_ids_arr: list[torch.Tensor],
    neighbors_arr: list[torch.Tensor],
    # metadata: dict,
    output_dir: str,
    idx: int,
):
    """Save the data."""
    gene_ids_arr = torch.cat(gene_ids_arr).tolist()
    neighbors_arr = [i.tolist() for i in torch.cat(neighbors_arr)]
    neighbors_df = pd.DataFrame({"sample_idx": sample_indices, "gene_id": gene_ids_arr, "neighbors": neighbors_arr})

    # metadata = pd.DataFrame(metadata)
    #
    # neighbors_df = pd.merge(neighbors_df, metadata, left_on="sample_idx", right_on=metadata.index, how="left")
    neighbors_df.to_parquet(os.path.join(output_dir, f"neighbors_{idx}.parquet"))
    del neighbors_df


# Function to extract the number from the file name
def extract_number(file_path):
    """Extract the number from the file path."""
    # Use regex to find the number after 'neighbors_' and before '.parquet'
    match = re.search(r"neighbors_(\d+).parquet", file_path)
    # Return the number as an integer, or 0 if no match is found
    return int(match.group(1)) if match else 0


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
        batch_size=args.batch_size,
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

    sample_indices = []
    gene_ids_arr = []
    neighbors_arr = []
    metadata = defaultdict(list)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            if i == args.max_steps:
                break
            output = model.model(
                expression_matrix=batch.expression_matrix.to(model.device),
                gene_ids=batch.gene_ids.to(model.device),
                is_masked_gene_expression=batch.is_masked_gene_expression.to(model.device),
                padding_mask=batch.padding_mask.to(model.device),
                tissue=batch.tissue_idx.to(model.device),
            )
            # get gene ids
            gene_ids = batch.gene_ids.squeeze(0).cpu()
            sample_indices += [i] * len(gene_ids)
            gene_ids_arr.append(gene_ids)

            # get neighbors
            embeddings = output.last_hidden_state.squeeze(0).cpu()
            neighbors = calculate_neighbors(embeddings, n_neighbors=50)
            neighbors = gene_ids[neighbors]
            neighbors_arr.append(neighbors)

            metadata["dataset"] += [i.split("/")[-1] for i in batch.study_label]
            metadata["sample"] += batch.sample_label
            metadata["disease"] += batch.disease
            metadata["tissue"] += [idx2tissue[i.item()] for i in batch.tissue_idx]

            if (i + 1) % 1000 == 0:
                save_data(sample_indices, gene_ids_arr, neighbors_arr, args.output_dir, i)
                pd.DataFrame(metadata).to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)

                sample_indices = []
                gene_ids_arr = []
                neighbors_arr = []

    save_data(sample_indices, gene_ids_arr, neighbors_arr, args.output_dir, i)
    pd.DataFrame(metadata).to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)

    # save all the files to one
    filepaths = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith(".parquet")]
    filepaths = sorted(filepaths, key=extract_number)
    df = pd.concat([pd.read_parquet(f) for f in filepaths])
    df.to_parquet(os.path.join(args.output_dir, "neighbors_all.parquet"))


if __name__ == "__main__":
    args = MulticellularTransformerArgparser().parse_args()
    print("Running with args:", args)
    run(args)
