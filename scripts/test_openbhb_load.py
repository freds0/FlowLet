#!/usr/bin/env python
"""
Quick test to verify OpenBHB dataset loads correctly from TSV.

Usage:
    python scripts/test_openbhb_load.py
"""

import sys
from pathlib import Path

# Add FlowLet to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from module to avoid Lightning import issues
from flowlet.data.openbhb_dataset import OpenBHBDataset


def main():
    print("="*60)
    print("Testing OpenBHB Dataset Loading")
    print("="*60)

    # Paths
    data_dir = "/home/fred/Projetos/Einstein/openbhb_train_sample"
    metadata_file = "/home/fred/Projetos/Einstein/openbhb_train_sample/train.tsv"

    print(f"\nData directory: {data_dir}")
    print(f"Metadata file: {metadata_file}")

    # Check files exist
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    if not Path(metadata_file).exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return

    print("\n" + "-"*60)
    print("Creating dataset...")
    print("-"*60)

    try:
        dataset = OpenBHBDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            target_shape=(128, 128, 128),
            age_range=None,  # Auto-detect
            normalize=True,
            augment=False,
            cache_data=False,
            include_site=True,
        )

        print("\n✅ Dataset created successfully!")

        # Print statistics
        print("\n" + "-"*60)
        print("Dataset Statistics")
        print("-"*60)

        stats = dataset.get_age_statistics()
        print(f"\nTotal samples: {stats['count']}")
        print(f"Age range: {stats['min']:.1f} - {stats['max']:.1f} years")
        print(f"Age mean: {stats['mean']:.1f} ± {stats['std']:.1f} years")

        site_dist = dataset.get_site_distribution()
        print(f"\nNumber of sites: {len(site_dist)}")
        print("\nTop 10 sites by sample count:")
        for i, (site, count) in enumerate(sorted(site_dist.items(), key=lambda x: -x[1])[:10], 1):
            print(f"  {i:2d}. Site {site:>3}: {count:>4} samples ({100*count/stats['count']:5.1f}%)")

        # Test loading a sample
        print("\n" + "-"*60)
        print("Testing Sample Loading")
        print("-"*60)

        print("\nLoading first 3 samples...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  Participant ID: {sample['participant_id']}")
            print(f"  Age: {sample['age'].item():.1f} years")
            print(f"  Site: {sample['site']}")
            print(f"  Volume shape: {sample['volume'].shape}")
            print(f"  Volume dtype: {sample['volume'].dtype}")
            print(f"  Volume range: [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

        print("\nYou can now proceed with training:")
        print(f"\npython flowlet/train.py \\")
        print(f"  experiment=flowlet_openbhb_finetune \\")
        print(f"  ckpt_path=/path/to/fomo60k_checkpoint.ckpt")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
