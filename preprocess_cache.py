#!/usr/bin/env python3
"""
Pre-process and cache all training volumes to avoid slow scipy.zoom during training.
"""
import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import pandas as pd

# Configuration
DATA_DIR = Path("/root/DATASETS/openbhb_data/train/")
METADATA_FILE = "/root/DATASETS/openbhb_data/train.tsv"
CACHE_DIR = Path("/root/DATASETS/openbhb_data/train_cache_128")
TARGET_SHAPE = (128, 128, 128)

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)

# Load metadata
print("Loading metadata...")
df = pd.read_csv(METADATA_FILE, sep='\t')
df = df[df['diagnosis'] == 'control'].copy()
df['quasiraw_3d_path'] = df['participant_id'].astype(str).apply(
    lambda x: f"quasiraw_3d/{x}_quasiraw_3d.npy"
)
print(f"Found {len(df)} samples")

# Process each volume
print(f"\nPreprocessing volumes to {TARGET_SHAPE}...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    participant_id = row['participant_id']
    cache_path = CACHE_DIR / f"{participant_id}.pt"

    # Skip if already cached
    if cache_path.exists():
        continue

    # Load volume
    volume_path = DATA_DIR / row['quasiraw_3d_path']
    volume = np.load(volume_path).astype(np.float32)

    # Resize
    if volume.shape != TARGET_SHAPE:
        factors = [t / s for t, s in zip(TARGET_SHAPE, volume.shape)]
        volume = zoom(volume, factors, order=1)

    # Normalize: clip to [0.5th, 99.5th] percentiles + scale to [-1, 1]
    p_low = np.percentile(volume, 0.5)
    p_high = np.percentile(volume, 99.5)
    volume = np.clip(volume, p_low, p_high)
    volume = 2.0 * (volume - p_low) / (p_high - p_low + 1e-8) - 1.0

    # Convert to tensor and save
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
    torch.save({
        'volume': volume_tensor,
        'age': float(row['age']),
        'participant_id': participant_id,
    }, cache_path)

print(f"\n✓ Done! Cached {len(df)} volumes to {CACHE_DIR}")
