"""Logging utilities for FlowLet training."""

from typing import Any, Dict

from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from flowlet.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """
    Log hyperparameters to all loggers.

    Args:
        object_dict: Dictionary containing cfg, model, datamodule, etc.
    """
    hparams = {}

    cfg: DictConfig = object_dict.get("cfg", {})
    model = object_dict.get("model")
    trainer = object_dict.get("trainer")

    # Add basic config info
    if cfg:
        hparams["cfg"] = OmegaConf.to_container(cfg, resolve=True)

    # Add model info
    if model is not None:
        hparams["model"] = {
            "class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }

    # Add trainer info
    if trainer is not None:
        hparams["trainer"] = {
            "max_epochs": trainer.max_epochs,
            "accelerator": str(trainer.accelerator),
            "devices": trainer.num_devices,
        }

    # Log to all loggers
    loggers = object_dict.get("logger", [])
    if loggers:
        for logger in loggers:
            if hasattr(logger, "log_hyperparams"):
                # Flatten nested dict for some loggers
                try:
                    logger.log_hyperparams(hparams)
                except Exception as e:
                    log.warning(f"Failed to log hyperparameters to {logger}: {e}")
