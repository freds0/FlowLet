"""
OpenBHB Dataset for FlowLet Fine-tuning.

Loads 3D brain MRI volumes from the OpenBHB dataset (Hugging Face).
Dataset contains preprocessed .npy volumes with metadata including age and site.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OpenBHBDataset(Dataset):
    """
    Dataset for OpenBHB 3D brain MRI volumes.

    OpenBHB is a neuroimaging dataset focused on brain age prediction tasks.
    Each sample includes:
    - quasiraw_3d_path: Path to .npy volume (shape: 182x218x182)
    - age: Participant age (target variable)
    - site: Scan site identifier (for domain adaptation)
    - participant_id: Unique identifier

    Args:
        data_dir: Root directory containing OpenBHB data
        metadata_file: CSV/parquet file with columns:
                      - participant_id
                      - age
                      - site
                      - quasiraw_3d_path (relative path to .npy file)
        target_shape: Target volume shape (D, H, W) - default (128, 128, 128)
        age_range: (min_age, max_age) for normalization - will be inferred if None
        normalize: Whether to apply intensity normalization
        augment: Whether to apply data augmentation
        cache_data: Whether to cache loaded volumes in memory
        include_site: Whether to include site information in output
        transform: Optional custom transform function
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        augment: bool = False,
        cache_data: bool = False,
        include_site: bool = False,
        transform=None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        self.include_site = include_site
        self.transform = transform

        # Load metadata
        self.metadata = self._load_metadata(metadata_file)

        # Infer age range if not provided
        if age_range is None:
            ages = self.metadata['age'].values
            self.age_range = (float(ages.min()), float(ages.max()))
            print(f"Inferred age range: {self.age_range}")
        else:
            self.age_range = age_range

        # Cache for loaded volumes
        self._cache: Dict[int, Dict] = {} if cache_data else None

        print(f"Loaded OpenBHB dataset with {len(self)} samples")
        print(f"Age range: {self.age_range}")
        if include_site:
            sites = self.metadata['site'].unique()
            print(f"Number of sites: {len(sites)}")

    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata CSV, TSV, or parquet file."""
        metadata_path = Path(metadata_file)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load based on file extension
        if metadata_path.suffix == '.csv':
            df = pd.read_csv(metadata_file)
        elif metadata_path.suffix == '.tsv':
            df = pd.read_csv(metadata_file, sep='\t')
        elif metadata_path.suffix == '.parquet':
            df = pd.read_parquet(metadata_file)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

        # Check if we need to construct the quasiraw_3d_path
        if 'quasiraw_3d_path' not in df.columns and 'participant_id' in df.columns:
            # OpenBHB format: construct path from participant_id
            # Format: quasiraw_3d/{participant_id}_quasiraw_3d.npy
            df['quasiraw_3d_path'] = df['participant_id'].astype(str).apply(
                lambda x: f"quasiraw_3d/{x}_quasiraw_3d.npy"
            )
            print("  ℹ Constructed quasiraw_3d_path from participant_id")

        # Validate required columns
        required_cols = ['participant_id', 'age', 'site', 'quasiraw_3d_path']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns in metadata: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # Filter out samples with missing values
        df = df.dropna(subset=['age', 'quasiraw_3d_path'])

        # Filter out files that don't exist
        original_count = len(df)
        valid_indices = []

        print(f"  ℹ Checking file existence for {original_count} samples...")
        for idx, row in df.iterrows():
            volume_path = self.data_dir / row['quasiraw_3d_path']
            if volume_path.exists():
                valid_indices.append(idx)

        df = df.loc[valid_indices].reset_index(drop=True)

        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"  ⚠ Removed {removed_count} samples with missing files")
            print(f"  ✓ {len(df)} valid samples remaining")
        else:
            print(f"  ✓ All {len(df)} samples have valid files")

        if len(df) == 0:
            raise ValueError("No valid samples found after filtering missing files!")

        return df

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self._cache is not None and idx in self._cache:
            data = self._cache[idx].copy()
            if self.augment:
                data['volume'] = self._apply_augmentation(data['volume'])
            return data

        # Get metadata for this sample
        row = self.metadata.iloc[idx]

        # Construct full path to .npy file
        relative_path = row['quasiraw_3d_path']
        volume_path = self.data_dir / relative_path

        # Load volume (file existence already verified during metadata loading)
        volume = np.load(volume_path, mmap_mode='r').astype(np.float32)

        # Preprocess
        volume = self._preprocess(volume)

        # Convert to tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
        age = torch.tensor([row['age']]).float()

        # Build output dictionary
        data = {
            'volume': volume,
            'age': age,
            'participant_id': row['participant_id'],
        }

        # Include site information if requested
        if self.include_site:
            data['site'] = row['site']

        # Apply custom transform
        if self.transform is not None:
            data = self.transform(data)

        # Cache if enabled (before augmentation)
        if self._cache is not None:
            self._cache[idx] = {
                'volume': data['volume'].clone(),
                'age': data['age'].clone(),
                'participant_id': data['participant_id'],
            }
            if self.include_site:
                self._cache[idx]['site'] = data['site']

        # Apply augmentation (after caching)
        if self.augment:
            data['volume'] = self._apply_augmentation(data['volume'])

        return data

    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocess volume: resize and normalize.

        OpenBHB volumes are typically (182, 218, 182).
        We resize to target_shape and apply z-score normalization.
        """
        from scipy.ndimage import zoom

        # Resize to target shape if needed
        if volume.shape != self.target_shape:
            factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
            volume = zoom(volume, factors, order=1)

        # Intensity normalization
        if self.normalize:
            # Create brain mask (non-background voxels)
            # Use percentile to avoid background
            threshold = np.percentile(volume, 5)
            mask = volume > threshold

            if mask.sum() > 0:
                # Z-score normalization within brain mask
                brain_mean = volume[mask].mean()
                brain_std = volume[mask].std() + 1e-8
                volume = (volume - brain_mean) / brain_std

                # Clip extreme values to prevent outliers
                volume = np.clip(volume, -5, 5)

        return volume

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations for training.

        Includes:
        - Random flipping (left-right, anterior-posterior)
        - Random intensity scaling
        - Random noise addition
        """
        # Random left-right flip
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip along depth

        # Random anterior-posterior flip
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2])  # Flip along height

        # Random intensity scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = volume * scale

        # Random Gaussian noise
        if random.random() > 0.7:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise

        return volume

    def get_age_statistics(self) -> Dict[str, float]:
        """Compute age statistics for the dataset."""
        ages = self.metadata['age'].values
        return {
            'min': float(ages.min()),
            'max': float(ages.max()),
            'mean': float(ages.mean()),
            'std': float(ages.std()),
            'count': len(ages),
        }

    def get_site_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across sites."""
        return self.metadata['site'].value_counts().to_dict()

    def filter_by_age_range(
        self,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None
    ) -> 'OpenBHBDataset':
        """
        Create a filtered dataset with specified age range.

        Useful for creating age-specific subsets or stratified splits.
        """
        df = self.metadata.copy()

        if min_age is not None:
            df = df[df['age'] >= min_age]
        if max_age is not None:
            df = df[df['age'] <= max_age]

        # Create new instance with filtered metadata
        new_dataset = OpenBHBDataset.__new__(OpenBHBDataset)
        new_dataset.data_dir = self.data_dir
        new_dataset.target_shape = self.target_shape
        new_dataset.age_range = self.age_range
        new_dataset.normalize = self.normalize
        new_dataset.augment = self.augment
        new_dataset.cache_data = False  # Don't share cache
        new_dataset.include_site = self.include_site
        new_dataset.transform = self.transform
        new_dataset.metadata = df.reset_index(drop=True)
        new_dataset._cache = {} if self.cache_data else None

        return new_dataset

    def filter_by_sites(self, sites: List[str]) -> 'OpenBHBDataset':
        """
        Create a filtered dataset with only specified sites.

        Useful for domain adaptation experiments.
        """
        df = self.metadata[self.metadata['site'].isin(sites)].copy()

        # Create new instance with filtered metadata
        new_dataset = OpenBHBDataset.__new__(OpenBHBDataset)
        new_dataset.data_dir = self.data_dir
        new_dataset.target_shape = self.target_shape
        new_dataset.age_range = self.age_range
        new_dataset.normalize = self.normalize
        new_dataset.augment = self.augment
        new_dataset.cache_data = False
        new_dataset.include_site = self.include_site
        new_dataset.transform = self.transform
        new_dataset.metadata = df.reset_index(drop=True)
        new_dataset._cache = {} if self.cache_data else None

        return new_dataset
