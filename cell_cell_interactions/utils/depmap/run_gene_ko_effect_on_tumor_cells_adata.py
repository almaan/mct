import json
import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from cell_cell_interactions.modeling.argparser import MulticellularTransformerArgparser
from cell_cell_interactions.modeling.model import MulticellularTransformerPLModel
from cell_cell_interactions.pp.adata_rnaseq_datamodule import AdataRnaseqDataModule

DEPMAP_DISEASE_MAP = {
    "lung_adenocarcinoma": "lung adenocarcinoma",
    "squamous_lung_carcinoma": "squamous cell lung carcinoma",
    "non_small_lung_carcinoma": "non-small cell lung carcinoma",
}


def run(
    args: MulticellularTransformerArgparser,
):
    """Run in silico gene knock-out on genes from DepMap using Adata"""
    output_dir = os.path.join(args.output_dir, args.depmap_disease_model, args.depmap_crispr_genes)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load trained model
    model = MulticellularTransformerPLModel.load_from_checkpoint(args.ckpt_path, strict=True).cuda()
    model.eval()

    # select genes to knock-out
    depmap_genes_df = pd.read_parquet(args.depmap_crispr_genes_file_path)
    depmap_genes = (
        depmap_genes_df[
            (depmap_genes_df["gene_selection"] == args.depmap_crispr_genes)
            & (depmap_genes_df["disease_model"] == args.depmap_disease_model)
        ]
        .iloc[0]
        .genes
    )
    print(f"Running gene KO with {args.depmap_crispr_genes} genes in {args.depmap_disease_model} disease model.")

    # get the data
    with open(args.gene2idx_file_path) as f:
        gene2idx = json.load(f)

    # get gene name to idx file path
    gene_df = pd.read_csv(args.gene_info_file_path)
    gene2name = {row["cellarr_gene_index"]: row["feature_name"] for _, row in gene_df.iterrows()}
    name2idx = {gene2name[gene]: idx for gene, idx in gene2idx.items() if gene in gene2name}

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # get the data
    datamodule = AdataRnaseqDataModule(
        adata_file_path=args.adata_file_path,
        gene2idx=gene2idx,
        val_samples=None,
        # this is completely unseen scenario, so we perform perturbations
        # across all available samples due to low nr of samples
        val_set_proportion=0.0,
        sample_size=args.n_cells_bag,
        sample_column=args.sample_key,
        label_column=args.cell_type_key,
        tissue_column="tissue",
        min_sample_size=args.min_n_cells_per_sample,
        min_cell_type_size=args.min_cell_type_size,
        sample_sampling=args.sample_sampling,
        cell_sampling=args.cell_sampling,
        batch_size=1,  # hardcoding it as downstream code depends on it
        num_workers=args.num_workers,
        agg_cell_type_gex=True,
        pad_value_expression=args.pad_value_expression,
        mask_prob=0.0,
        mask_tissue_prob=0.0,
        perc_min_expressed_cells_per_gene=args.perc_min_expressed_cells_per_gene,
        max_seq_len=args.max_seq_len,
        random_seed=args.random_seed,
        backed=args.backed,
        restrict_to_conditions={
            "disease": DEPMAP_DISEASE_MAP[args.depmap_disease_model],
        },
    )

    adata_gene_ids = datamodule.train_dataset.global_gene_ids.tolist()
    genes_to_knock_out = [g for g in depmap_genes if name2idx.get(g, None) in adata_gene_ids]
    print("Nr of genes to knock out", len(genes_to_knock_out))

    output_expression = defaultdict(list)
    output_rel_expression = defaultdict(list)
    ct2idx = {ct: idx for idx, ct in enumerate(datamodule.cell_type_input_order)}
    tumor_ct_idx = ct2idx["Tumor cells"]

    dataloader = datamodule.train_dataloader()
    i = 0
    with torch.no_grad():
        for _ in range(1, 2):
            for batch in tqdm(dataloader):
                i += 1
                if i == args.max_steps:
                    break
                baseline = model.model(
                    expression_matrix=batch.expression_matrix.to(model.device),
                    gene_ids=batch.gene_ids.to(model.device),
                    is_masked_gene_expression=batch.is_masked_gene_expression.to(model.device),
                    padding_mask=batch.padding_mask.to(model.device),
                    tissue=batch.tissue_idx.to(model.device),
                )

                baseline_rel_gex = (
                    torch.nn.functional.softmax(baseline.gex_reconstruction, dim=-1).sum(dim=1).squeeze(0).cpu()
                )

                baseline_gex = baseline.gex_reconstruction.sum(dim=1).squeeze(0).cpu()

                for gene in genes_to_knock_out:
                    gene_idx = name2idx[gene]
                    # skip if gene not present in the sample
                    if gene_idx not in batch.gene_ids:
                        continue

                    idx = torch.nonzero(batch.gene_ids == gene_idx, as_tuple=True)[-1][0]

                    # in silico knock-out the gene by removing it
                    new_expression = torch.cat(
                        (batch.expression_matrix[:, :idx, :], batch.expression_matrix[:, idx + 1 :, :]), dim=1
                    )
                    new_gene_ids = torch.cat((batch.gene_ids[:, :idx], batch.gene_ids[:, idx + 1 :]), dim=1)
                    new_masked_gex = torch.cat(
                        (batch.is_masked_gene_expression[:, :idx], batch.is_masked_gene_expression[:, idx + 1 :]), dim=1
                    )
                    new_padding_mask = torch.cat((batch.padding_mask[:, :idx], batch.padding_mask[:, idx + 1 :]), dim=1)

                    knock_out = model.model(
                        expression_matrix=new_expression.to(model.device),
                        gene_ids=new_gene_ids.to(model.device),
                        is_masked_gene_expression=new_masked_gex.to(model.device),
                        padding_mask=new_padding_mask.to(model.device),
                        tissue=batch.tissue_idx.to(model.device),
                    )

                    ko_rel_gex = (
                        torch.nn.functional.softmax(knock_out.gex_reconstruction, dim=-1).sum(dim=1).squeeze(0).cpu()
                    )

                    ko_gex = knock_out.gex_reconstruction.sum(dim=1).squeeze(0).cpu()

                    if not args.collect_all_gex:
                        output_rel_expression[gene].append(
                            torch.tensor([ko_rel_gex[tumor_ct_idx], baseline_rel_gex[tumor_ct_idx]])
                        )
                        output_expression[gene].append(torch.tensor([ko_gex[tumor_ct_idx], baseline_gex[tumor_ct_idx]]))
                    else:
                        output_rel_expression[gene].append(torch.stack([ko_rel_gex, baseline_rel_gex]))
                        output_expression[gene].append(torch.stack([ko_gex, baseline_gex]))

                # save the data every N iterations
                if i % 10 == 0:
                    torch.save(
                        {gene: torch.stack(expr) for gene, expr in output_expression.items()},
                        os.path.join(output_dir, "gene_ko_expression.pt"),
                    )
                    torch.save(
                        {gene: torch.stack(expr) for gene, expr in output_rel_expression.items()},
                        os.path.join(output_dir, "gene_ko_relative_expression.pt"),
                    )
    # save everything at the end as well
    torch.save(
        {gene: torch.stack(expr) for gene, expr in output_expression.items()},
        os.path.join(output_dir, "gene_ko_expression.pt"),
    )
    torch.save(
        {gene: torch.stack(expr) for gene, expr in output_rel_expression.items()},
        os.path.join(output_dir, "gene_ko_relative_expression.pt"),
    )


if __name__ == "__main__":
    args = MulticellularTransformerArgparser().parse_args()
    print("Running with args:", args)
    run(args)
