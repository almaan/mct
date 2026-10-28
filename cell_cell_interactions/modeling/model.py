from typing import Any

import torch
from lightning import LightningModule
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig, MulticellularTransformerOutput
from cell_cell_interactions.modeling.loss import compute_masked_gene_id_loss
from cell_cell_interactions.modeling.modeling import (
    MulticellularTransformerModule,
)
from cell_cell_interactions.modeling.utils import compute_classification_metrics
from cell_cell_interactions.pp.data_types import BagOfCellsBatch


class MulticellularTransformerPLModel(LightningModule):
    """PyTorch Lightning model for the multicellular graph pretraining."""

    def __init__(self, config: MulticellularGraphConfig, pretrained_ckpt_path: str = None):
        super().__init__()

        self.config = config

        self.model = MulticellularTransformerModule(config, pretrained_ckpt_path)

        if getattr(config, "compile_model", False) and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, dynamic=True)
                print("torch.compile enabled on model")
            except Exception as exc:
                print(f"torch.compile failed, falling back to eager: {exc}")

        self.save_hyperparameters(logger=False)

    def training_step(self, batch: BagOfCellsBatch, batch_idx: int):
        """Training step."""
        output = self.model(
            expression_matrix=batch.expression_matrix,
            gene_ids=batch.gene_ids,
            is_masked_gene_expression=batch.is_masked_gene_expression,
            padding_mask=batch.padding_mask,
            tissue=batch.tissue_idx,
        )
        losses = self.compute_loss(batch, output)
        losses = {f"train_{k}": v for k, v in losses.items()}
        self.log_dict(
            losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.batch_size
        )
        self.log(
            "lr",
            self.scheduler.get_last_lr()[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config.batch_size,
        )
        if losses["train_total_loss"] > 0:
            return losses["train_total_loss"]
        # account for NaNs
        return None

    def eval_step(self, batch: BagOfCellsBatch, prefix: str) -> dict[str, Any]:
        """Evaluation step."""
        output = self.model(
            expression_matrix=batch.expression_matrix,
            gene_ids=batch.gene_ids,
            is_masked_gene_expression=batch.is_masked_gene_expression,
            padding_mask=batch.padding_mask,
            tissue=batch.tissue_idx,
        )
        losses = self.compute_loss(batch, output)
        metrics = {}
        if self.config.compute_pretraining_metrics:
            metrics = compute_classification_metrics(output.gene_id_logits, batch.gene_ids, batch.is_masked_gene_id)
        metrics.update(losses)
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        return metrics

    def validation_step(self, batch: BagOfCellsBatch, batch_idx: int):
        """Validation step."""
        metrics = self.eval_step(batch, prefix="val")
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config.batch_size)
        return metrics

    def test_step(self, batch: BagOfCellsBatch, batch_idx: int):
        """Test step."""
        metrics = self.eval_step(batch, prefix="test")
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config.batch_size)
        return metrics

    def configure_optimizers(self):
        """Configure the optimizer for the model.

        Uses the same LR schedule as the reference `12k_no_pseudobulk.ckpt`:
        linear warmup for `num_warmup_steps`, then CosineAnnealingLR with
        `T_max = max_steps - num_warmup_steps` and `eta_min = 0`. Assembled
        as a SequentialLR to match the saved scheduler state exactly. Both
        pieces are torch built-ins; no external dep.
        """
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        warmup_steps = max(1, int(self.config.num_warmup_steps))
        total_steps = max(warmup_steps + 1, int(self.config.max_steps))
        cosine_steps = total_steps - warmup_steps

        def _linear_warmup(step: int) -> float:
            return float(step) / float(warmup_steps)

        warmup_sched = LambdaLR(self.optimizer, lr_lambda=_linear_warmup)
        cosine_sched = CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=0.0)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps],
        )

        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "train_total_loss_step",
                }
            ],
        )

    def compute_loss(self, batch: BagOfCellsBatch, output: MulticellularTransformerOutput) -> dict[str, torch.Tensor]:
        """Compute the loss for the model."""
        # masked gene prediction loss
        gene_id_loss = compute_masked_gene_id_loss(output.gene_id_logits, batch.labels, batch.is_masked_gene_id)

        # compute loss as weighted sum of gene_id_loss and elbo_loss
        total_loss = (1 - self.config.alpha_reconstruction_loss_weight) * gene_id_loss

        tissue_loss = torch.tensor(0.0, device=gene_id_loss.device)
        if output.tissue_logits is not None and self.config.gamma_tissue_loss_weight > 0:
            tissue_loss = torch.nn.functional.cross_entropy(output.tissue_logits, batch.tissue_labels)
            if tissue_loss > 0:
                total_loss += self.config.gamma_tissue_loss_weight * tissue_loss
            else:
                tissue_loss = torch.tensor(0.0, device=gene_id_loss.device)

        if self.config.alpha_reconstruction_loss_weight > 0:
            rec_loss = mse_loss(output.gex_reconstruction, batch.expression_matrix, reduction="none").sum(-1)
            rec_loss_masked = (rec_loss * batch.is_masked_gene_expression).sum().mean()
            rec_loss = rec_loss.mean()
            if rec_loss > 0:  # account for NANs
                total_loss += self.config.alpha_reconstruction_loss_weight * rec_loss
            else:
                rec_loss = torch.tensor(0.0, device=gene_id_loss.device)
                rec_loss_masked = torch.tensor(0.0, device=gene_id_loss.device)
        else:
            rec_loss = torch.tensor(0.0, device=gene_id_loss.device)
            rec_loss_masked = torch.tensor(0.0, device=gene_id_loss.device)

        return {
            "total_loss": total_loss,
            "gene_id_loss": gene_id_loss,
            "tissue_loss": tissue_loss,
            "rec_loss": rec_loss,
            "rec_loss_masked": rec_loss_masked,
        }
