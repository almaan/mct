from dataclasses import dataclass
from typing import Literal

import torch
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MulticellularTransformerOutput:
    """Dataclass for MulticellularTransformerOutput."""

    last_hidden_state: torch.Tensor
    gene_ids_w_covariates: torch.Tensor = None
    gene_id_logits: torch.Tensor = None
    tissue_logits: torch.Tensor = None
    gex_reconstruction: torch.Tensor = None


@dataclass_json
@dataclass
class MulticellularGraphConfig:
    """Dataclass for the configuration of the multicellular transformer model."""

    # model parameters
    gene_emb_dim: int = 64
    hidden_dim: int = 256
    n_genes: int = None
    max_seq_len: int = 2500
    dropout: float = 0.2
    transformer_dropout: float = 0.1
    n_heads: int = 4
    n_layers: int = 3
    n_cells_bag: int = 256
    n_input: int = None  # this is either n_cells bag or nr of unique cell types
    batch_size: int = 1
    padding_token_idx: int = 0
    mask_token_idx: int = 1
    lr: float = 0.001
    layer_norm_eps: float = 1e-12
    weight_decay: float = 0.01
    alpha_reconstruction_loss_weight: float = 0.01
    latent_dim: int = 32
    agg_cell_type_gex: bool = False
    num_warmup_steps: int = 1000
    n_tissues: int = 0
    compute_pretraining_metrics: bool = False
    gamma_tissue_loss_weight: float = 0.05

    # data parameters
    mask_prob_gene_id: float = 0.15
    mask_prob_gene_expression: float = 0.0

    # trainer parameters
    num_nodes: int = 1
    max_steps: int = 1000000
    limit_val_batches: int = 10000
    val_check_interval: int = 10000
    log_every_n_steps: int = 100
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    monitor_metric: str = "val_total_loss"
    patience: int = 20
    save_checkpoint_every_n_steps: int = 10000
    accelerator: Literal["ddp", "cpu", "auto"] = "auto"
    experiment_name: str = "initial_test_runs"
    strategy: Literal["auto", "ddp"] = "ddp"
    precision: Literal["bf16-mixed", "bf16-true", "16-mixed", "32-true"] = "bf16-true"
    compile_model: bool = False
