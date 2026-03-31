#!/usr/bin/env python3
"""
Train FlowLet using pre-cached dataset for much faster data loading.
"""
import sys
sys.path.insert(0, '/root/experiments/FlowLet')

import hydra
from omegaconf import DictConfig
from flowlet.train import train as original_train


# Monkey-patch the datamodule to use cached version
def patched_train(cfg: DictConfig):
    """Wrapper that uses CachedOpenBHBDataModule instead of OpenBHBDataModule"""
    # Change datamodule config to use cached version
    if cfg.data._target_ == "flowlet.data.mri_datamodule.OpenBHBDataModule":
        # Update to use cached data
        cfg.data._target_ = "flowlet.data.cached_datamodule.CachedOpenBHBDataModule"
        # Change data_dir to cache_dir
        cfg.data.cache_dir = "/root/DATASETS/openbhb_data/train_cache_128"
        # Remove data_dir since CachedDataModule uses cache_dir instead
        if "data_dir" in cfg.data:
            del cfg.data.data_dir
        print(f"✓ Using cached dataset from: {cfg.data.cache_dir}")

    return original_train(cfg)


if __name__ == "__main__":
    # Use hydra decorator
    train_with_hydra = hydra.main(
        version_base="1.3",
        config_path="configs",
        config_name="train.yaml"
    )(patched_train)

    train_with_hydra()
