"""
Cached OpenBHB Dataset that loads pre-processed volumes from disk cache.
Much faster than on-the-fly preprocessing with scipy.zoom.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedOpenBHBDataset(Dataset):
    """
    OpenBHB Dataset that loads from preprocessed cache files.

    Args:
        cache_dir: Directory containing preprocessed .pt files
        metadata_file: TSV file with participant metadata
        augment: Whether to apply data augmentation
        include_site: Whether to include site information
        filter_diagnosis: Filter by diagnosis (e.g., 'control')
    """

    def __init__(
        self,
        cache_dir: str,
        metadata_file: str,
        augment: bool = False,
        include_site: bool = False,
        filter_diagnosis: Optional[str] = "control",
    ):
        super().__init__()

        self.cache_dir = Path(cache_dir)
        self.augment = augment
        self.include_site = include_site

        # Load metadata
        self.metadata = self._load_metadata(metadata_file, filter_diagnosis)

        print(f"Loaded CachedOpenBHBDataset with {len(self)} samples from {cache_dir}")

    def _load_metadata(self, metadata_file: str, filter_diagnosis: Optional[str]) -> pd.DataFrame:
        """Load and filter metadata."""
        df = pd.read_csv(metadata_file, sep='\t')

        # Filter by diagnosis
        if filter_diagnosis and 'diagnosis' in df.columns:
            original_count = len(df)
            df = df[df['diagnosis'] == filter_diagnosis].copy()
            print(f"  Filtered by diagnosis='{filter_diagnosis}': {original_count} -> {len(df)}")

        # Verify cache files exist
        valid_indices = []
        for idx, row in df.iterrows():
            cache_path = self.cache_dir / f"{row['participant_id']}.pt"
            if cache_path.exists():
                valid_indices.append(idx)

        df = df.loc[valid_indices].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No valid cached files found in {self.cache_dir}")

        print(f"  Found {len(df)} cached files")

        return df

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]

        # Load from cache
        cache_path = self.cache_dir / f"{row['participant_id']}.pt"
        cached_data = torch.load(cache_path, weights_only=False)

        # Build output
        data = {
            'volume': cached_data['volume'],
            'age': torch.tensor([cached_data['age']]).float(),
        }

        if self.include_site and 'site' in row:
            data['site'] = row['site']

        # Apply augmentation if enabled
        if self.augment:
            data['volume'] = self._apply_augmentation(data['volume'])

        return data

    def _apply_augmentation(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation (placeholder - reuse from original dataset if needed)."""
        # For now, return as-is
        # TODO: Add augmentation if self.augment=True
        return volume
