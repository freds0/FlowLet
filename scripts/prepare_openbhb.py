"""
Prepare OpenBHB dataset for FlowLet training.

This script:
1. Validates the OpenBHB metadata file
2. Checks that all .npy files exist and are readable
3. Computes dataset statistics
4. Creates train/val/test splits (optional)
5. Generates a summary report

Usage:
    python scripts/prepare_openbhb.py \
        --data_dir /path/to/openbhb/data \
        --metadata_file /path/to/metadata.csv \
        --validate \
        --stats

    # Create stratified splits
    python scripts/prepare_openbhb.py \
        --data_dir /path/to/openbhb/data \
        --metadata_file /path/to/metadata.csv \
        --create_splits \
        --val_split 0.1 \
        --test_split 0.1
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_metadata(metadata_file: str) -> pd.DataFrame:
    """Load metadata file (CSV, TSV, or parquet)."""
    metadata_path = Path(metadata_file)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

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
        df['quasiraw_3d_path'] = df['participant_id'].astype(str).apply(
            lambda x: f"train/quasiraw_3d/{x}_quasiraw_3d.npy"
        )
        print("ℹ Constructed quasiraw_3d_path from participant_id")

    return df


def validate_metadata(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate metadata structure."""
    # Minimum required columns
    min_required = ['participant_id', 'age', 'site']
    missing_cols = [col for col in min_required if col not in df.columns]

    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # quasiraw_3d_path should be present (either original or constructed)
    if 'quasiraw_3d_path' not in df.columns:
        return False, "quasiraw_3d_path column missing (should be constructed from participant_id)"

    # Check for missing values in essential columns
    essential_cols = ['participant_id', 'age', 'site', 'quasiraw_3d_path']
    missing_counts = df[essential_cols].isnull().sum()
    if missing_counts.any():
        return False, f"Missing values found:\n{missing_counts[missing_counts > 0]}"

    return True, "Metadata validation passed"


def validate_files(data_dir: Path, df: pd.DataFrame, sample_size: int = 10) -> Dict:
    """Validate that files exist and are readable."""
    print("\nValidating files...")

    results = {
        'total': len(df),
        'missing': 0,
        'invalid': 0,
        'valid': 0,
        'missing_files': [],
        'invalid_files': [],
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking files"):
        file_path = data_dir / row['quasiraw_3d_path']

        if not file_path.exists():
            results['missing'] += 1
            results['missing_files'].append(str(file_path))
            continue

        # Validate a sample of files
        if idx < sample_size or idx % (len(df) // sample_size) == 0:
            try:
                volume = np.load(file_path, mmap_mode='r')

                # Check shape
                if len(volume.shape) != 3:
                    results['invalid'] += 1
                    results['invalid_files'].append(f"{file_path}: wrong dims {volume.shape}")
                    continue

                # Check dtype
                if not np.issubdtype(volume.dtype, np.floating):
                    results['invalid'] += 1
                    results['invalid_files'].append(f"{file_path}: wrong dtype {volume.dtype}")
                    continue

                results['valid'] += 1

            except Exception as e:
                results['invalid'] += 1
                results['invalid_files'].append(f"{file_path}: {str(e)}")
        else:
            results['valid'] += 1

    return results


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute dataset statistics."""
    stats = {
        'total_samples': len(df),
        'age_min': df['age'].min(),
        'age_max': df['age'].max(),
        'age_mean': df['age'].mean(),
        'age_std': df['age'].std(),
        'age_median': df['age'].median(),
        'num_sites': df['site'].nunique(),
        'site_distribution': df['site'].value_counts().to_dict(),
    }

    # Age distribution by decade
    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    stats['age_distribution'] = df['age_group'].value_counts().sort_index().to_dict()

    return stats


def create_stratified_splits(
    df: pd.DataFrame,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits by age groups."""
    from sklearn.model_selection import train_test_split

    # Create age groups for stratification
    age_bins = [0, 30, 45, 60, 75, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=False)

    # First split: train + val, test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split,
        stratify=df['age_group'],
        random_state=seed
    )

    # Second split: train, val
    val_size_adjusted = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['age_group'],
        random_state=seed
    )

    # Remove temporary age_group column
    for split_df in [train_df, val_df, test_df]:
        if 'age_group' in split_df.columns:
            split_df = split_df.drop(columns=['age_group'])

    return train_df, val_df, test_df


def print_report(stats: Dict, validation_results: Dict = None):
    """Print dataset report."""
    print("\n" + "="*60)
    print("OpenBHB Dataset Report")
    print("="*60)

    print(f"\nTotal samples: {stats['total_samples']}")

    print(f"\nAge statistics:")
    print(f"  Min:    {stats['age_min']:.1f}")
    print(f"  Max:    {stats['age_max']:.1f}")
    print(f"  Mean:   {stats['age_mean']:.1f}")
    print(f"  Median: {stats['age_median']:.1f}")
    print(f"  Std:    {stats['age_std']:.1f}")

    print(f"\nAge distribution:")
    for age_group, count in sorted(stats['age_distribution'].items()):
        pct = 100 * count / stats['total_samples']
        print(f"  {age_group:>6}: {count:5d} ({pct:5.1f}%)")

    print(f"\nSite distribution ({stats['num_sites']} sites):")
    for site, count in sorted(stats['site_distribution'].items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / stats['total_samples']
        print(f"  {site:>10}: {count:5d} ({pct:5.1f}%)")
    if len(stats['site_distribution']) > 10:
        print(f"  ... and {len(stats['site_distribution']) - 10} more sites")

    if validation_results:
        print(f"\nFile validation:")
        print(f"  Total:   {validation_results['total']}")
        print(f"  Valid:   {validation_results['valid']}")
        print(f"  Missing: {validation_results['missing']}")
        print(f"  Invalid: {validation_results['invalid']}")

        if validation_results['missing_files']:
            print(f"\n  First 5 missing files:")
            for file in validation_results['missing_files'][:5]:
                print(f"    - {file}")

        if validation_results['invalid_files']:
            print(f"\n  First 5 invalid files:")
            for file in validation_results['invalid_files'][:5]:
                print(f"    - {file}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenBHB dataset for FlowLet")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing OpenBHB data")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to metadata CSV/parquet file")
    parser.add_argument("--validate", action="store_true",
                        help="Validate that all files exist and are readable")
    parser.add_argument("--stats", action="store_true",
                        help="Compute and display dataset statistics")
    parser.add_argument("--create_splits", action="store_true",
                        help="Create train/val/test splits")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split fraction (default: 0.1)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test split fraction (default: 0.1)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for split files (default: same as metadata)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits (default: 42)")

    args = parser.parse_args()

    # Convert paths
    data_dir = Path(args.data_dir)
    metadata_file = Path(args.metadata_file)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading metadata from: {metadata_file}")
    df = load_metadata(str(metadata_file))

    # Validate metadata structure
    is_valid, message = validate_metadata(df)
    if not is_valid:
        print(f"ERROR: {message}")
        return
    print(f"✓ {message}")

    # Compute statistics
    if args.stats or args.validate:
        print("\nComputing statistics...")
        stats = compute_statistics(df)

    # Validate files
    validation_results = None
    if args.validate:
        validation_results = validate_files(data_dir, df)

    # Print report
    if args.stats or args.validate:
        print_report(stats, validation_results)

    # Create splits
    if args.create_splits:
        print(f"\nCreating stratified splits (val={args.val_split}, test={args.test_split})...")
        train_df, val_df, test_df = create_stratified_splits(
            df,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed
        )

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = metadata_file.parent

        # Save splits
        train_file = output_dir / "openbhb_train.csv"
        val_file = output_dir / "openbhb_val.csv"
        test_file = output_dir / "openbhb_test.csv"

        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(f"\n✓ Splits saved:")
        print(f"  Train: {train_file} ({len(train_df)} samples)")
        print(f"  Val:   {val_file} ({len(val_df)} samples)")
        print(f"  Test:  {test_file} ({len(test_df)} samples)")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
