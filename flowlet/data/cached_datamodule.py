"""
LightningDataModule for cached OpenBHB dataset.
"""
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from flowlet.data.cached_dataset import CachedOpenBHBDataset


class CachedOpenBHBDataModule(LightningDataModule):
    """
    DataModule for pre-cached OpenBHB volumes.

    Args:
        cache_dir: Directory with cached .pt files
        metadata_file: TSV file with metadata
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Validation split fraction
        test_split: Test split fraction
        augment_train: Apply augmentation to training data
        augment_val: Apply augmentation to validation data
        filter_diagnosis: Filter samples by diagnosis
        include_site: Include site information
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for splits
    """

    def __init__(
        self,
        cache_dir: str,
        metadata_file: str,
        batch_size: int = 4,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.0,
        augment_train: bool = True,
        augment_val: bool = False,
        filter_diagnosis: Optional[str] = "control",
        include_site: bool = False,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cache_dir = cache_dir
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.augment_train = augment_train
        self.augment_val = augment_val
        self.filter_diagnosis = filter_diagnosis
        self.include_site = include_site
        self.pin_memory = pin_memory
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Load full dataset
            full_dataset = CachedOpenBHBDataset(
                cache_dir=self.cache_dir,
                metadata_file=self.metadata_file,
                augment=False,  # Will be set per split
                include_site=self.include_site,
                filter_diagnosis=self.filter_diagnosis,
            )

            # Calculate split sizes
            total_size = len(full_dataset)
            test_size = int(total_size * self.test_split)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size - test_size

            # Split dataset
            from torch import Generator
            generator = Generator().manual_seed(self.seed)
            splits = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator,
            )

            self.train_dataset, self.val_dataset, self.test_dataset = splits

            # Set augmentation flags
            self.train_dataset.dataset.augment = self.augment_train
            self.val_dataset.dataset.augment = self.augment_val

            print(f"CachedOpenBHB splits: Train={train_size}, Val={val_size}, Test={test_size}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_split > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
            )
        return None
