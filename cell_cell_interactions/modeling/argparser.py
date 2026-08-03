from typing import Literal

from tap import Tap


class MulticellularTransformerArgparser(Tap):
    """Arguments for the running MulticellularTransformer."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # data
    # path to the anndata file
    adata_file_path: str = (
        "/Users/wiatrakm/Downloads/lung_cancer_atlas_processed_all_genes_log1pnorm_subset.h5ad"  # joint_subset.h5ad"
    )
    # path to the input directory of cellarr data
    input_dir: str = "/scratch/site/u/wiatrakm/cell-cell-interactions/data/input-data/datasets/corpus-14M-cells"
    output_dir: str = "/tmp/multicellular-graph-output/"  # path to the output directory
    n_cells_bag: int = 256  # number of cells in a bag
    train_split_prop: float = 0.8  # proportion of training samples
    batch_size: int = 1  # batch size
    sample_key: str = "sample"  # key for sample in anndata object
    cell_type_key: str = "cell_type_major"  # "Majority_voting_CellTypist"  # key for cell type in anndata object
    sample_sampling: Literal["random", "weighted"] = "weighted"  # sample sampling method
    cell_sampling: Literal["random", "stratified", "all_cells"] = "stratified"  # cell sampling method
    pad_value_expression: float = 0.0  # value to pad the output with if there are not enough cells of a given type
    test: bool = False  # whether to include the test set
    random_seed: int = 32  # random seed
    num_workers: int = 0  # number of workers for the data loader
    split_by_sample: bool = False
    min_n_cells_per_sample: int = 256  # minimum number of cells per sample
    min_cell_type_size: float = 0.01
    perc_min_expressed_cells_per_gene: float = 0.01
    cell_type_mapping_file_path: str = None
    compute_pretraining_metrics: bool = False
    gene2idx_file_path: str = "/Users/wiatrakm/Desktop/mcp/gene2idx.json"
    backed: bool = False

    # model parameters
    gene_emb_dim: int = 16
    hidden_dim: int = 256
    n_genes: int = None
    max_seq_len: int = 2500
    dropout: float = 0.2
    transformer_dropout: float = 0.1
    n_heads: int = 4
    n_layers: int = 3
    padding_token_idx: int = 0
    mask_token_idx: int = 1
    mask_prob: float = 0.15
    mask_tissue_prob: float = 0.25
    lr: float = 0.01
    weight_decay: float = 0.02
    layer_norm_eps: float = 1e-12
    latent_dim: int = 8
    # value between 0 and 1 with which to weight the reconstruction loss, (1 - alpha) will be used
    # for masked gene prediction
    alpha_reconstruction_loss_weight: float = 0.01
    gamma_tissue_loss_weight: float = 0.0
    agg_cell_type_gex: bool = False
    num_warmup_steps: int = 1000

    # trainer parameters
    num_nodes: int = 1
    max_steps: int = 100000
    limit_val_batches: int = 1000
    val_check_interval: int = 4000
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    monitor_metric: str = "val_total_loss"
    patience: int = 5
    save_checkpoint_every_n_steps: int = 10000
    accelerator: Literal["ddp", "cpu", "auto"] = "cpu"
    experiment_name: str = "initial_test_runs"
    wandb_api_key: str = None
    strategy: Literal["auto", "ddp", "ddp_spawn"] = "auto"
    devices: str = "auto"
    ckpt_path: str = None
    pretrained_ckpt_path: str = None  # "/tmp/multicellular-graph-output/initial_test_runs/checkpoints/last.ckpt"

    # parameters for running in silico gene KO on DepMap
    depmap_crispr_genes_file_path: str = None
    depmap_crispr_genes: Literal["random", "largest_effect", "all"] = "random"
    depmap_disease_model: Literal[
        "lung_adenocarcinoma",
        "squamous_lung_carcinoma",
        "non_small_lung_carcinoma",
        "glioblastoma",
    ] = "lung_adenocarcinoma"
    gene_info_file_path: str = None
    collect_all_gex: bool = False
