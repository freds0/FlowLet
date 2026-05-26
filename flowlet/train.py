"""
FlowLet Training Script.

Train FlowLet models for 3D brain MRI synthesis using Hydra configuration.

Usage:
    # Train with default config
    python flowlet/train.py experiment=flowlet_synthetic

    # Train on OASIS dataset
    python flowlet/train.py experiment=flowlet_oasis

    # Override parameters
    python flowlet/train.py experiment=flowlet_synthetic model.learning_rate=2e-4 trainer.max_epochs=200
"""

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from flowlet import utils

log = utils.get_pylogger(__name__)


def load_initial_checkpoint(model: LightningModule, cfg: DictConfig) -> Optional[str]:
    """Load pretrained weights without restoring optimizer or trainer state."""
    checkpoint_cfg = cfg.get("initial_checkpoint")
    if not checkpoint_cfg or not checkpoint_cfg.get("enabled", False):
        return None
    if cfg.get("ckpt_path"):
        log.info("Skipping initial checkpoint because ckpt_path resumes an existing training run.")
        return None

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=checkpoint_cfg.repo_id,
        filename=checkpoint_cfg.filename,
        revision=checkpoint_cfg.get("revision"),
        cache_dir=checkpoint_cfg.get("cache_dir"),
        token=checkpoint_cfg.get("token"),
    )
    log.info(f"Loading initial weights from <{path}>")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=checkpoint_cfg.get("strict", True))
    return path


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train FlowLet model.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (metrics_dict, objects_dict)
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Print model info
    log.info(f"Model: {model}")
    log.info(f"Number of parameters: {model.num_parameters:,}")

    # Allow loading checkpoints without weights_only restriction (needed for PyTorch >= 2.6)
    import typing
    from omegaconf import ListConfig, DictConfig as OmegaDictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, OmegaDictConfig, ContainerMetadata, typing.Any])

    initial_checkpoint_path = load_initial_checkpoint(model, cfg)

    # Instantiate callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Store objects for logging
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "initial_checkpoint_path": initial_checkpoint_path,
    }

    # Log hyperparameters
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Training
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    train_metrics = trainer.callback_metrics

    # Testing
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best checkpoint path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.

    Args:
        cfg: Hydra configuration

    Returns:
        Optimized metric value (for hyperparameter search)
    """
    # Apply extra utilities
    utils.extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Get metric value for hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )

    return metric_value


if __name__ == "__main__":
    main()
