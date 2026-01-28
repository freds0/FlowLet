"""
MRI Dataset for FlowLet.

Loads 3D MRI volumes (NIfTI format) with age conditioning for brain image synthesis.
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    """
    Dataset for 3D MRI volumes with age conditioning.

    Supports loading from:
    - Directory with NIfTI files + CSV metadata
    - List of file paths + ages

    Args:
        data_dir: Directory containing NIfTI files
        metadata_file: CSV file with columns 'filename' and 'age'
                      OR None if using file_list parameter
        file_list: Optional list of (filepath, age) tuples
        target_shape: Target volume shape (D, H, W)
        age_range: (min_age, max_age) for normalization
        normalize: Whether to apply intensity normalization
        augment: Whether to apply augmentation
        cache_data: Whether to cache loaded data in memory
        transform: Optional custom transform function
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        metadata_file: Optional[str] = None,
        file_list: Optional[List[Tuple[str, float]]] = None,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Tuple[float, float] = (18.0, 90.0),
        normalize: bool = True,
        augment: bool = False,
        cache_data: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir) if data_dir else None
        self.target_shape = target_shape
        self.age_range = age_range
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        self.transform = transform

        # Load file list
        if file_list is not None:
            self.samples = file_list
        elif metadata_file is not None:
            self.samples = self._load_metadata(metadata_file)
        else:
            raise ValueError("Must provide either file_list or metadata_file")

        # Cache for loaded data
        self._cache: Dict[int, Dict] = {} if cache_data else None

    def _load_metadata(self, metadata_file: str) -> List[Tuple[str, float]]:
        """Load metadata CSV file."""
        import pandas as pd

        df = pd.read_csv(metadata_file)

        # Handle various column naming conventions
        filename_col = None
        age_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['filename', 'file', 'filepath', 'path', 'subject', 'subject_id']:
                filename_col = col
            if col_lower in ['age', 'age_at_scan', 'years']:
                age_col = col

        if filename_col is None:
            raise ValueError(f"Could not find filename column in metadata. Columns: {df.columns.tolist()}")
        if age_col is None:
            raise ValueError(f"Could not find age column in metadata. Columns: {df.columns.tolist()}")

        samples = []
        for _, row in df.iterrows():
            filename = row[filename_col]
            age = float(row[age_col])

            # Construct full path
            if self.data_dir is not None:
                filepath = str(self.data_dir / filename)
            else:
                filepath = filename

            samples.append((filepath, age))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self._cache is not None and idx in self._cache:
            data = self._cache[idx].copy()
            if self.augment:
                data['volume'] = self._apply_augmentation(data['volume'])
            return data

        filepath, age = self.samples[idx]

        # Load volume
        volume = self._load_volume(filepath)

        # Preprocess
        volume = self._preprocess(volume)

        # Convert to tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
        age = torch.tensor([age]).float()

        data = {
            'volume': volume,
            'age': age,
            'filepath': filepath,
        }

        # Apply custom transform
        if self.transform is not None:
            data = self.transform(data)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = {
                'volume': data['volume'].clone(),
                'age': data['age'].clone(),
                'filepath': data['filepath'],
            }

        # Apply augmentation
        if self.augment:
            data['volume'] = self._apply_augmentation(data['volume'])

        return data

    def _load_volume(self, filepath: str) -> np.ndarray:
        """Load NIfTI volume."""
        import nibabel as nib

        nii = nib.load(filepath)
        volume = nii.get_fdata().astype(np.float32)

        return volume

    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess volume: resize, normalize."""
        from scipy.ndimage import zoom

        # Resize to target shape
        if volume.shape != self.target_shape:
            factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
            volume = zoom(volume, factors, order=1)

        # Intensity normalization
        if self.normalize:
            # Create brain mask (non-zero voxels)
            mask = volume > np.percentile(volume, 5)

            if mask.sum() > 0:
                # Z-score normalization within mask
                brain_mean = volume[mask].mean()
                brain_std = volume[mask].std() + 1e-8
                volume = (volume - brain_mean) / brain_std

                # Clip extreme values
                volume = np.clip(volume, -5, 5)

        return volume

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random flipping
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip along depth
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2])  # Flip along height

        # Random intensity scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = volume * scale

        # Random noise
        if random.random() > 0.7:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise

        return volume

    def get_age_statistics(self) -> Dict[str, float]:
        """Compute age statistics for the dataset."""
        ages = [age for _, age in self.samples]
        return {
            'min': min(ages),
            'max': max(ages),
            'mean': np.mean(ages),
            'std': np.std(ages),
        }


class MRIDatasetFromDirectory(MRIDataset):
    """
    MRI Dataset that automatically discovers NIfTI files in a directory.

    Expects age to be encoded in filename or a separate age file.

    Args:
        data_dir: Directory containing NIfTI files
        age_file: Optional file with age information
        pattern: Glob pattern for NIfTI files
        age_from_filename: Function to extract age from filename
    """

    def __init__(
        self,
        data_dir: str,
        age_file: Optional[str] = None,
        pattern: str = "**/*.nii*",
        age_from_filename: Optional[Callable[[str], float]] = None,
        **kwargs
    ):
        data_dir = Path(data_dir)

        # Find all NIfTI files
        files = sorted(data_dir.glob(pattern))

        # Load ages
        if age_file is not None:
            # Load ages from separate file
            import pandas as pd
            age_df = pd.read_csv(age_file)
            age_dict = dict(zip(age_df['filename'], age_df['age']))
            file_list = [
                (str(f), age_dict.get(f.name, age_dict.get(f.stem, 50.0)))
                for f in files
            ]
        elif age_from_filename is not None:
            # Extract age from filename
            file_list = [(str(f), age_from_filename(f.name)) for f in files]
        else:
            # Use placeholder age
            print("Warning: No age information provided. Using placeholder age=50.")
            file_list = [(str(f), 50.0) for f in files]

        super().__init__(
            data_dir=str(data_dir),
            file_list=file_list,
            **kwargs
        )


class NumpyMRIDataset(Dataset):
    """
    Dataset for 3D MRI volumes stored as NumPy files (.npy).

    Supports FOMO60k and similar preprocessed datasets.

    Args:
        data_dir: Directory containing .npy files
        metadata_file: Optional CSV file with columns 'filename' and 'age'
        file_list: Optional list of (filepath, age) tuples
        target_shape: Target volume shape (D, H, W)
        age_range: (min_age, max_age) for normalization
        normalize: Whether to apply intensity normalization
        augment: Whether to apply augmentation
        cache_data: Whether to cache loaded data in memory
        use_random_age: If True, use random ages when no age info available
    """

    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        file_list: Optional[List[Tuple[str, float]]] = None,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        age_range: Tuple[float, float] = (18.0, 90.0),
        normalize: bool = True,
        augment: bool = False,
        cache_data: bool = False,
        use_random_age: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.age_range = age_range
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        self.use_random_age = use_random_age
        self.seed = seed

        # Load file list
        if file_list is not None:
            self.samples = file_list
        elif metadata_file is not None:
            self.samples = self._load_metadata(metadata_file)
        else:
            self.samples = self._discover_files()

        # Cache for loaded data
        self._cache: Dict[int, Dict] = {} if cache_data else None

    def _load_metadata(self, metadata_file: str) -> List[Tuple[str, float]]:
        """Load metadata CSV file."""
        import pandas as pd

        df = pd.read_csv(metadata_file)

        # Handle various column naming conventions
        filename_col = None
        age_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['filename', 'file', 'filepath', 'path', 'subject', 'subject_id']:
                filename_col = col
            if col_lower in ['age', 'age_at_scan', 'years']:
                age_col = col

        if filename_col is None:
            raise ValueError(f"Could not find filename column in metadata. Columns: {df.columns.tolist()}")
        if age_col is None:
            raise ValueError(f"Could not find age column in metadata. Columns: {df.columns.tolist()}")

        samples = []
        for _, row in df.iterrows():
            filename = row[filename_col]
            age = float(row[age_col])
            filepath = str(self.data_dir / filename)
            samples.append((filepath, age))

        return samples

    def _discover_files(self) -> List[Tuple[str, float]]:
        """Discover .npy files in directory."""
        npy_files = sorted(self.data_dir.glob("*.npy"))

        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in {self.data_dir}")

        # Generate ages
        if self.use_random_age:
            np.random.seed(self.seed)
            ages = np.random.uniform(
                self.age_range[0], self.age_range[1], len(npy_files)
            )
            samples = [(str(f), float(ages[i])) for i, f in enumerate(npy_files)]
        else:
            # Use middle of age range as placeholder
            default_age = (self.age_range[0] + self.age_range[1]) / 2
            print(f"Warning: No age metadata. Using random ages for training.")
            np.random.seed(self.seed)
            ages = np.random.uniform(
                self.age_range[0], self.age_range[1], len(npy_files)
            )
            samples = [(str(f), float(ages[i])) for i, f in enumerate(npy_files)]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self._cache is not None and idx in self._cache:
            data = self._cache[idx].copy()
            if self.augment:
                data['volume'] = self._apply_augmentation(data['volume'])
            return data

        filepath, age = self.samples[idx]

        # Load volume
        volume = np.load(filepath).astype(np.float32)

        # Preprocess
        volume = self._preprocess(volume)

        # Convert to tensor
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
        age = torch.tensor([age]).float()

        data = {
            'volume': volume,
            'age': age,
            'filepath': filepath,
        }

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = {
                'volume': data['volume'].clone(),
                'age': data['age'].clone(),
                'filepath': data['filepath'],
            }

        # Apply augmentation
        if self.augment:
            data['volume'] = self._apply_augmentation(data['volume'])

        return data

    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess volume: resize, normalize."""
        from scipy.ndimage import zoom

        # Resize to target shape
        if volume.shape != self.target_shape:
            factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
            volume = zoom(volume, factors, order=1)

        # Intensity normalization
        if self.normalize:
            # Data is already normalized 0-1, convert to z-score
            mask = volume > 0.05
            if mask.sum() > 0:
                brain_mean = volume[mask].mean()
                brain_std = volume[mask].std() + 1e-8
                volume = (volume - brain_mean) / brain_std
                volume = np.clip(volume, -5, 5)

        return volume

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random flipping
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[2])

        # Random intensity scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = volume * scale

        # Random noise
        if random.random() > 0.7:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise

        return volume

    def get_age_statistics(self) -> Dict[str, float]:
        """Compute age statistics for the dataset."""
        ages = [age for _, age in self.samples]
        return {
            'min': min(ages),
            'max': max(ages),
            'mean': np.mean(ages),
            'std': np.std(ages),
        }


class SyntheticMRIDataset(Dataset):
    """
    Synthetic MRI dataset for testing and debugging.

    Generates random volumes with age-dependent patterns.
    """

    def __init__(
        self,
        num_samples: int = 100,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        age_range: Tuple[float, float] = (18.0, 90.0),
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.volume_shape = volume_shape
        self.age_range = age_range
        self.seed = seed

        # Generate fixed ages
        np.random.seed(seed)
        self.ages = np.random.uniform(age_range[0], age_range[1], num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Deterministic generation based on index
        np.random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)

        age = self.ages[idx]

        # Generate synthetic volume with age-dependent features
        volume = self._generate_synthetic_volume(age)

        volume = torch.from_numpy(volume).float().unsqueeze(0)
        age_tensor = torch.tensor([age]).float()

        return {
            'volume': volume,
            'age': age_tensor,
            'filepath': f'synthetic_{idx}',
        }

    def _generate_synthetic_volume(self, age: float) -> np.ndarray:
        """Generate synthetic brain-like volume."""
        D, H, W = self.volume_shape

        # Create coordinate grids
        z, y, x = np.ogrid[:D, :H, :W]
        center = np.array([D/2, H/2, W/2])

        # Base ellipsoid (brain-like shape)
        dz = (z - center[0]) / (D * 0.4)
        dy = (y - center[1]) / (H * 0.35)
        dx = (x - center[2]) / (W * 0.35)
        ellipsoid = dz**2 + dy**2 + dx**2

        # Brain mask
        brain_mask = ellipsoid < 1.0

        # Age-dependent features
        age_norm = (age - self.age_range[0]) / (self.age_range[1] - self.age_range[0])

        # Simulate atrophy with age (smaller ventricles, less volume)
        ventricle_size = 0.15 + 0.1 * age_norm
        ventricles = ellipsoid < ventricle_size

        # Create volume
        volume = np.zeros(self.volume_shape, dtype=np.float32)

        # White matter (age decreases intensity slightly)
        volume[brain_mask] = 0.7 - 0.1 * age_norm

        # Gray matter (cortex)
        cortex = brain_mask & (ellipsoid > 0.7)
        volume[cortex] = 0.9 - 0.15 * age_norm

        # Ventricles (CSF)
        volume[ventricles] = 0.3

        # Add noise
        volume += np.random.randn(*self.volume_shape) * 0.05

        # Normalize
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)

        return volume
