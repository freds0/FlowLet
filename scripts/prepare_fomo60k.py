#!/usr/bin/env python3
"""
Script to prepare FOMO60k dataset for training.

This script:
1. Scans the FOMO60k directory for .npy files
2. Generates a metadata CSV with filenames and random ages
3. Optionally validates the data format

Usage:
    python scripts/prepare_fomo60k.py --data_dir /path/to/FOMO60k --output metadata.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def scan_fomo60k(data_dir: str, output_csv: str, age_range: tuple = (18.0, 90.0), seed: int = 42):
    """
    Scan FOMO60k directory and create metadata CSV.

    Args:
        data_dir: Path to FOMO60k directory
        output_csv: Output CSV file path
        age_range: (min_age, max_age) for random age generation
        seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)

    # Find all .npy files
    npy_files = sorted(data_dir.glob("*.npy"))

    if len(npy_files) == 0:
        print(f"Error: No .npy files found in {data_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")

    # Generate random ages
    np.random.seed(seed)
    ages = np.random.uniform(age_range[0], age_range[1], len(npy_files))

    # Create DataFrame
    data = []
    for i, npy_file in enumerate(npy_files):
        data.append({
            'filename': npy_file.name,
            'age': round(ages[i], 1),
            'subject_id': npy_file.stem.split('_')[1] if '_' in npy_file.stem else i,
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"Metadata saved to: {output_csv}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    print(f"  - Mean age: {df['age'].mean():.1f}")

    return df


def validate_data(data_dir: str, n_samples: int = 5):
    """
    Validate FOMO60k data format.

    Args:
        data_dir: Path to FOMO60k directory
        n_samples: Number of samples to validate
    """
    data_dir = Path(data_dir)
    npy_files = sorted(data_dir.glob("*.npy"))[:n_samples]

    print(f"\nValidating {len(npy_files)} samples...")

    for npy_file in npy_files:
        volume = np.load(npy_file)
        print(f"  {npy_file.name}:")
        print(f"    Shape: {volume.shape}")
        print(f"    Dtype: {volume.dtype}")
        print(f"    Range: [{volume.min():.4f}, {volume.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Prepare FOMO60k dataset for FlowLet training")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/fred/Projetos/FlowLet/FOMO60k",
        help="Path to FOMO60k directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data_dir/fomo60k_metadata.csv)"
    )
    parser.add_argument(
        "--age_min",
        type=float,
        default=18.0,
        help="Minimum age for random generation"
    )
    parser.add_argument(
        "--age_max",
        type=float,
        default=90.0,
        help="Maximum age for random generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data format"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = os.path.join(args.data_dir, "fomo60k_metadata.csv")

    # Scan and create metadata
    scan_fomo60k(
        data_dir=args.data_dir,
        output_csv=args.output,
        age_range=(args.age_min, args.age_max),
        seed=args.seed,
    )

    # Validate if requested
    if args.validate:
        validate_data(args.data_dir)


if __name__ == "__main__":
    main()
