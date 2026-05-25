"""Lightning DataModule for FOMO300K brain-age training."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split

from flowlet.data.fomo300k_dataset import FOMO300KDataset


class MRIBatchCollate:
    """Stack MRI volumes and ages into the batch contract used by FlowLet."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "volume": torch.stack([item["volume"] for item in batch]),
            "age": torch.stack([item["age"] for item in batch]),
        }


class FOMO300KSubset(Subset):
    """Apply augmentation per split without mutating the shared dataset."""

    def __init__(self, dataset: FOMO300KDataset, indices, augment: bool = False):
        super().__init__(dataset, indices)
        self.augment = augment

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.augment:
            data = data.copy()
            data["volume"] = self.dataset._apply_augmentation(data["volume"])
        return data


class FOMO300KDataModule(LightningDataModule):
    """Provide train/validation/test splits for FOMO300K T1w acquisitions."""

    def __init__(
        self,
        data_dir: str = "/root/DATASETS/FOMO300K_brain_age",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Optional[Tuple[float, float]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.0,
        augment_train: bool = True,
        augment_val: bool = False,
        cache_data: bool = False,
        include_site: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.age_range = age_range
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.augment_train = augment_train
        self.augment_val = augment_val
        self.cache_data = cache_data
        self.include_site = include_site
        self.max_samples = max_samples
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset = FOMO300KDataset(
            data_dir=self.data_dir,
            target_shape=self.target_shape,
            age_range=self.age_range,
            normalize=True,
            augment=False,
            cache_data=self.cache_data,
            include_site=self.include_site,
            max_samples=self.max_samples,
        )
        if self.age_range is None:
            self.age_range = full_dataset.age_range

        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test
        print(f"FOMO300K Dataset: {n_total} scans -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

        train, val, test = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.train_dataset = FOMO300KSubset(full_dataset, train.indices, self.augment_train)
        self.val_dataset = FOMO300KSubset(full_dataset, val.indices, self.augment_val)
        self.test_dataset = FOMO300KSubset(full_dataset, test.indices, False)

        stats = full_dataset.get_age_statistics()
        print(
            f"Age statistics: min={stats['min']:.1f}, max={stats['max']:.1f}, "
            f"mean={stats['mean']:.1f}, std={stats['std']:.1f}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MRIBatchCollate(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MRIBatchCollate(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MRIBatchCollate(),
        )

    def get_data_statistics(self) -> Dict[str, Any]:
        if self.train_dataset is None:
            self.setup("fit")
        ages = [self.train_dataset[i]["age"].item() for i in range(len(self.train_dataset))]
        return {
            "num_train": len(self.train_dataset),
            "num_val": len(self.val_dataset),
            "num_test": len(self.test_dataset),
            "age_min": min(ages),
            "age_max": max(ages),
            "age_mean": np.mean(ages),
            "age_std": np.std(ages),
        }
