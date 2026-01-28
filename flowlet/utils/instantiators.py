"""Instantiator utilities for callbacks and loggers."""

from typing import List, Optional

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from flowlet.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: Optional[DictConfig]) -> List[Callback]:
    """
    Instantiate callbacks from config.

    Args:
        callbacks_cfg: Hydra config for callbacks

    Returns:
        List of instantiated callbacks
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Using empty list.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_name, cb_conf in callbacks_cfg.items():
        if cb_conf is not None and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: Optional[DictConfig]) -> List[Logger]:
    """
    Instantiate loggers from config.

    Args:
        logger_cfg: Hydra config for loggers

    Returns:
        List of instantiated loggers
    """
    loggers: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Using empty list.")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for lg_name, lg_conf in logger_cfg.items():
        if lg_conf is not None and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers
