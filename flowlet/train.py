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
    }

    # Log hyperparameters
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Allow omegaconf types in checkpoint loading (needed for PyTorch >= 2.6)
    from omegaconf import ListConfig, DictConfig as OmegaDictConfig
    torch.serialization.add_safe_globals([ListConfig, OmegaDictConfig])

    # Training
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # Testing
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
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
