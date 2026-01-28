"""
MRI DataModule for FlowLet.

PyTorch Lightning DataModule for loading and managing MRI datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from flowlet.data.mri_dataset import (
    MRIDataset,
    MRIDatasetFromDirectory,
    NumpyMRIDataset,
    SyntheticMRIDataset,
)


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


class MRIDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for MRI data.

    Handles data loading, splitting, and batching for training FlowLet.

    Args:
        # Data paths
        data_dir: Directory containing NIfTI files
        train_metadata: CSV file for training data
        val_metadata: Optional CSV file for validation data
        test_metadata: Optional CSV file for test data

        # Data parameters
        target_shape: Target volume shape (D, H, W)
        age_range: (min_age, max_age) for normalization

        # DataLoader parameters
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

        # Split parameters (if val_metadata not provided)
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing

        # Augmentation
        augment_train: Apply augmentation to training data
        augment_val: Apply augmentation to validation data

        # Caching
        cache_data: Cache loaded data in memory
    """

    def __init__(
        self,
        # Data paths
        name: str = "mri",
        data_dir: Optional[str] = None,
        train_metadata: Optional[str] = None,
        val_metadata: Optional[str] = None,
        test_metadata: Optional[str] = None,
        # Data parameters
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Tuple[float, float] = (18.0, 90.0),
        # DataLoader parameters
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        # Split parameters
        val_split: float = 0.1,
        test_split: float = 0.0,
        # Augmentation
        augment_train: bool = True,
        augment_val: bool = False,
        # Caching
        cache_data: bool = False,
        # Seed
        seed: int = 42,
        # Use synthetic data for testing
        use_synthetic: bool = False,
        synthetic_num_samples: int = 100,
        # Use numpy files (.npy) instead of NIfTI
        use_numpy: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.test_metadata = test_metadata
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
        self.seed = seed
        self.use_synthetic = use_synthetic
        self.synthetic_num_samples = synthetic_num_samples
        self.use_numpy = use_numpy

        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing."""
        if self.use_synthetic:
            self._setup_synthetic()
        elif self.use_numpy:
            self._setup_numpy()
        else:
            self._setup_real()

    def _setup_synthetic(self) -> None:
        """Set up synthetic datasets for testing."""
        # Create synthetic datasets
        full_dataset = SyntheticMRIDataset(
            num_samples=self.synthetic_num_samples,
            volume_shape=self.target_shape,
            age_range=self.age_range,
            seed=self.seed,
        )

        # Split
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def _setup_numpy(self) -> None:
        """Set up NumPy-based datasets (e.g., FOMO60k)."""
        if self.data_dir is None:
            raise ValueError("data_dir must be provided for numpy datasets")

        # Create dataset from numpy files
        full_dataset = NumpyMRIDataset(
            data_dir=self.data_dir,
            metadata_file=self.train_metadata,
            target_shape=self.target_shape,
            age_range=self.age_range,
            normalize=True,
            augment=self.augment_train,
            cache_data=self.cache_data,
            use_random_age=True,
            seed=self.seed,
        )

        # Split
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test

        print(f"Dataset: {n_total} samples -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def _setup_real(self) -> None:
        """Set up real MRI datasets."""
        # Training dataset
        if self.train_metadata is not None:
            train_full = MRIDataset(
                data_dir=self.data_dir,
                metadata_file=self.train_metadata,
                target_shape=self.target_shape,
                age_range=self.age_range,
                augment=self.augment_train,
                cache_data=self.cache_data,
            )

            # Split if validation metadata not provided
            if self.val_metadata is None:
                n_total = len(train_full)
                n_val = int(n_total * self.val_split)
                n_test = int(n_total * self.test_split)
                n_train = n_total - n_val - n_test

                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    train_full,
                    [n_train, n_val, n_test],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                self.train_dataset = train_full
        elif self.data_dir is not None:
            # Discover files in directory
            train_full = MRIDatasetFromDirectory(
                data_dir=self.data_dir,
                target_shape=self.target_shape,
                age_range=self.age_range,
                augment=self.augment_train,
                cache_data=self.cache_data,
            )

            n_total = len(train_full)
            n_val = int(n_total * self.val_split)
            n_test = int(n_total * self.test_split)
            n_train = n_total - n_val - n_test

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                train_full,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Separate validation dataset
        if self.val_metadata is not None:
            self.val_dataset = MRIDataset(
                data_dir=self.data_dir,
                metadata_file=self.val_metadata,
                target_shape=self.target_shape,
                age_range=self.age_range,
                augment=self.augment_val,
                cache_data=self.cache_data,
            )

        # Separate test dataset
        if self.test_metadata is not None:
            self.test_dataset = MRIDataset(
                data_dir=self.data_dir,
                metadata_file=self.test_metadata,
                target_shape=self.target_shape,
                age_range=self.age_range,
                augment=False,
                cache_data=self.cache_data,
            )

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
        """Compute statistics from the training data."""
        if self.train_dataset is None:
            self.setup('fit')

        # Get age statistics
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


class OASISDataModule(MRIDataModule):
    """
    DataModule for OASIS dataset.

    Preconfigured for the OASIS-3 dataset structure.
    """

    def __init__(
        self,
        data_dir: str,
        participants_file: str,
        **kwargs
    ):
        # Default parameters for OASIS
        kwargs.setdefault('target_shape', (128, 128, 128))
        kwargs.setdefault('age_range', (42.0, 95.0))  # OASIS age range

        super().__init__(
            data_dir=data_dir,
            train_metadata=participants_file,
            **kwargs
        )


class IXIDataModule(MRIDataModule):
    """
    DataModule for IXI dataset.

    Preconfigured for the IXI dataset structure.
    """

    def __init__(
        self,
        data_dir: str,
        demographics_file: str,
        **kwargs
    ):
        # Default parameters for IXI
        kwargs.setdefault('target_shape', (128, 128, 128))
        kwargs.setdefault('age_range', (20.0, 86.0))  # IXI age range

        super().__init__(
            data_dir=data_dir,
            train_metadata=demographics_file,
            **kwargs
        )


class ADNIDataModule(MRIDataModule):
    """
    DataModule for ADNI dataset.

    Preconfigured for the ADNI dataset structure.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        **kwargs
    ):
        # Default parameters for ADNI
        kwargs.setdefault('target_shape', (128, 128, 128))
        kwargs.setdefault('age_range', (55.0, 96.0))  # ADNI age range

        super().__init__(
            data_dir=data_dir,
            train_metadata=metadata_file,
            **kwargs
        )
