"""
MRI DataModule for FlowLet.

PyTorch Lightning DataModule for loading and managing MRI datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from flowlet.data.openbhb_dataset import OpenBHBDataset


class MRIBatchCollate:
    """
    Collate function for MRI batches.

    Handles variable-sized volumes by padding to the maximum size in the batch
    (though typically volumes should be pre-resized to the same shape).
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        volumes = torch.stack([item['volume'] for item in batch])
        ages = torch.stack([item['age'] for item in batch])

        return {
            'volume': volumes,  # (B, 1, D, H, W)
            'age': ages,        # (B, 1)
        }


class OpenBHBDataModule(LightningDataModule):
    """
    DataModule for OpenBHB dataset.

    Specialized DataModule for the OpenBHB brain MRI dataset from Hugging Face.
    Default settings follow the FlowLet paper: batch_size=4, 80/20 split,
    CN-only subjects.

    Args:
        data_dir: Root directory containing OpenBHB data
        metadata_file: CSV/parquet file with participant metadata
        target_shape: Target volume shape (D, H, W)
        age_range: (min_age, max_age) for normalization (None = auto-infer)
        batch_size: Batch size for training (paper: 4)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        val_split: Fraction for validation split (paper: 0.2)
        test_split: Fraction for test split (paper: 0.0)
        augment_train: Apply augmentation to training data
        augment_val: Apply augmentation to validation data
        cache_data: Cache loaded volumes in memory
        include_site: Include site information in batches
        filter_diagnosis: Filter by diagnosis (paper: 'control' for CN subjects)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
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
        filter_diagnosis: Optional[str] = "control",
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.metadata_file = metadata_file
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
        self.filter_diagnosis = filter_diagnosis
        self.seed = seed

        # Datasets (set during setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing."""
        # Create full dataset
        full_dataset = OpenBHBDataset(
            data_dir=self.data_dir,
            metadata_file=self.metadata_file,
            target_shape=self.target_shape,
            age_range=self.age_range,
            normalize=True,
            augment=False,  # Augmentation applied per-split
            cache_data=self.cache_data,
            include_site=self.include_site,
            filter_diagnosis=self.filter_diagnosis,
        )

        # Update age_range if auto-inferred
        if self.age_range is None:
            self.age_range = full_dataset.age_range

        # Split dataset
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test

        print(f"OpenBHB Dataset: {n_total} samples -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # Random split
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Wrap train dataset with augmentation
        if self.augment_train:
            # Access underlying dataset and enable augmentation
            train_dataset.dataset.augment = True

        if self.augment_val:
            val_dataset.dataset.augment = True

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Print statistics
        stats = full_dataset.get_age_statistics()
        print(f"Age statistics: min={stats['min']:.1f}, max={stats['max']:.1f}, "
              f"mean={stats['mean']:.1f}, std={stats['std']:.1f}")

        if self.include_site:
            site_dist = full_dataset.get_site_distribution()
            print(f"Site distribution: {len(site_dist)} sites")

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
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
        """Validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MRIBatchCollate(),
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=MRIBatchCollate(),
        )

    def get_data_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if self.train_dataset is None:
            self.setup('fit')

        ages = []
        for i in range(len(self.train_dataset)):
            sample = self.train_dataset[i]
            ages.append(sample['age'].item())

        import numpy as np
        return {
            'num_train': len(self.train_dataset),
            'num_val': len(self.val_dataset) if self.val_dataset else 0,
            'num_test': len(self.test_dataset) if self.test_dataset else 0,
            'age_min': min(ages),
            'age_max': max(ages),
            'age_mean': np.mean(ages),
            'age_std': np.std(ages),
        }
