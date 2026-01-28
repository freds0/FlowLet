"""General utility functions for FlowLet training."""

import warnings
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from flowlet.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """
    Decorator for training tasks that handles exceptions and logging.

    Args:
        task_func: Function to wrap

    Returns:
        Wrapped function
    """
    @wraps(task_func)
    def wrapper(cfg: DictConfig, *args, **kwargs):
        # Try to execute the task
        try:
            metric_dict, object_dict = task_func(cfg, *args, **kwargs)

        # Handle keyboard interrupt
        except KeyboardInterrupt:
            log.warning("Interrupted by user!")
            raise

        # Handle other exceptions
        except Exception as e:
            log.exception(f"Exception during training: {e}")
            raise

        return metric_dict, object_dict

    return wrapper


def extras(cfg: DictConfig) -> None:
    """
    Apply optional utilities before training.

    Args:
        cfg: Hydra configuration
    """
    # Disable warnings if requested
    if cfg.get("ignore_warnings"):
        log.info("Disabling Python warnings!")
        warnings.filterwarnings("ignore")

    # Print config tree if requested
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich!")
        print_config_tree(cfg, resolve=True)


def get_metric_value(
    metric_dict: Dict[str, Any],
    metric_name: Optional[str],
) -> Optional[float]:
    """
    Get metric value from metrics dictionary.

    Args:
        metric_dict: Dictionary containing metrics
        metric_name: Name of the metric to retrieve

    Returns:
        Metric value as float, or None
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        log.warning(
            f"Metric value not found in metric_dict! "
            f"Available metrics: {list(metric_dict.keys())}"
        )
        return None

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value: {metric_name}={metric_value}")

    return metric_value


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
) -> None:
    """
    Print config tree using Rich.

    Args:
        cfg: Hydra configuration
        print_order: Order to print config groups
        resolve: Whether to resolve interpolations
    """
    try:
        from rich import print as rprint
        from rich.tree import Tree

        tree = Tree("CONFIG", style="bold")

        # Add config groups in order
        for field in print_order:
            if field in cfg:
                branch = tree.add(field, style="bold blue")
                config_str = OmegaConf.to_yaml(cfg[field], resolve=resolve)
                branch.add(config_str)

        # Add remaining fields
        for field in cfg:
            if field not in print_order:
                branch = tree.add(field, style="bold magenta")
                config_str = OmegaConf.to_yaml(cfg[field], resolve=resolve)
                branch.add(config_str)

        rprint(tree)

    except ImportError:
        # Fall back to simple print if Rich is not available
        log.info(OmegaConf.to_yaml(cfg, resolve=resolve))
