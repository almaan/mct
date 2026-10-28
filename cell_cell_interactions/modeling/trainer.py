import os
from typing import Literal

import torch.cuda
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from cell_cell_interactions.modeling.data_types import MulticellularGraphConfig


def get_pl_trainer(
    config: MulticellularGraphConfig,
    output_dir: str,
    wandb_api_key: str | None,
    devices: list[int] | str | int = "auto",
    sample_sampling: Literal["weighted", "random"] = "weighted",
):
    """Get the lightning trainer."""
    mode = "min" if "loss" in config.monitor_metric else "max"

    # create ckpt dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create logger dir
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logger = True
    if wandb_api_key:
        import wandb
        from lightning.pytorch.loggers import WandbLogger

        wandb.login(host="https://genentech.wandb.io/", key=wandb_api_key)
        logger = WandbLogger(
            name=config.experiment_name,
            project="cell-cell-interactions",
            save_dir=logs_dir,
        )

    if sample_sampling == "random":
        val_check_interval = None
        limit_val_batches = None
        check_val_every_n_epoch = 10
    else:
        val_check_interval = config.val_check_interval
        limit_val_batches = config.limit_val_batches
        check_val_every_n_epoch = None

    callbacks = [
        EarlyStopping(monitor=config.monitor_metric, patience=config.patience, mode=mode),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=config.monitor_metric,
            mode=mode,
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{step}-{val_total_loss:.2f}",
            every_n_train_steps=config.save_checkpoint_every_n_steps,
        ),
        TQDMProgressBar(refresh_rate=config.log_every_n_steps),
        LearningRateMonitor(logging_interval="step"),
    ]

    return Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        num_nodes=config.num_nodes,
        max_steps=config.max_steps,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
        precision=(getattr(config, "precision", "bf16-true") if torch.cuda.is_available() else "32-true"),
        strategy=config.strategy,
    )
