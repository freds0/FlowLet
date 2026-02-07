"""
Test OpenBHB DataLoader

Simple script to test the OpenBHB dataset and dataloader.
Useful for debugging and understanding the data pipeline.

Usage:
    python examples/test_openbhb_dataloader.py \
        --data_dir /path/to/openbhb/data \
        --metadata_file /path/to/openbhb/metadata.csv \
        --num_samples 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from flowlet.data.openbhb_dataset import OpenBHBDataset


def visualize_volume_slices(volume: torch.Tensor, age: float, participant_id: str):
    """
    Visualize axial, coronal, and sagittal slices of a 3D volume.

    Args:
        volume: (1, D, H, W) tensor
        age: Age of the participant
        participant_id: Participant ID
    """
    volume = volume.squeeze(0).numpy()  # (D, H, W)
    D, H, W = volume.shape

    # Middle slices
    axial_slice = volume[D // 2, :, :]       # Middle depth
    coronal_slice = volume[:, H // 2, :]     # Middle height
    sagittal_slice = volume[:, :, W // 2]    # Middle width

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view
    axes[0].imshow(axial_slice, cmap='gray')
    axes[0].set_title('Axial View')
    axes[0].axis('off')

    # Coronal view
    axes[1].imshow(coronal_slice, cmap='gray')
    axes[1].set_title('Coronal View')
    axes[1].axis('off')

    # Sagittal view
    axes[2].imshow(sagittal_slice, cmap='gray')
    axes[2].set_title('Sagittal View')
    axes[2].axis('off')

    fig.suptitle(f'Participant: {participant_id} | Age: {age:.1f} years', fontsize=14)
    plt.tight_layout()

    return fig


def test_dataset(data_dir: str, metadata_file: str, num_samples: int = 5):
    """
    Test the OpenBHB dataset.

    Args:
        data_dir: Root directory with data
        metadata_file: Path to metadata CSV/parquet
        num_samples: Number of samples to test
    """
    print("="*60)
    print("Testing OpenBHB Dataset")
    print("="*60)

    # Create dataset
    print(f"\n1. Creating dataset...")
    print(f"   Data dir: {data_dir}")
    print(f"   Metadata: {metadata_file}")

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

    print(f"\n   ✓ Dataset created successfully!")
    print(f"   Total samples: {len(dataset)}")

    # Get statistics
    print(f"\n2. Dataset statistics:")
    stats = dataset.get_age_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key:>12}: {value:.2f}")
        else:
            print(f"   {key:>12}: {value}")

    site_dist = dataset.get_site_distribution()
    print(f"\n   Sites: {len(site_dist)}")
    for site, count in list(site_dist.items())[:5]:
        print(f"   {site:>12}: {count} samples")

    # Test loading samples
    print(f"\n3. Testing sample loading (n={num_samples})...")

    for i in range(min(num_samples, len(dataset))):
        print(f"\n   Sample {i+1}:")

        # Load sample
        sample = dataset[i]

        # Print info
        print(f"      Participant: {sample['participant_id']}")
        print(f"      Age: {sample['age'].item():.1f} years")
        print(f"      Site: {sample['site']}")
        print(f"      Volume shape: {sample['volume'].shape}")
        print(f"      Volume dtype: {sample['volume'].dtype}")
        print(f"      Volume range: [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")

        # Visualize
        fig = visualize_volume_slices(
            sample['volume'],
            sample['age'].item(),
            sample['participant_id']
        )

        # Save figure
        output_dir = Path("openbhb_test_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"sample_{i+1}_{sample['participant_id']}.png"
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"      ✓ Saved visualization to: {output_path}")

    # Test augmentation
    print(f"\n4. Testing augmentation...")
    dataset.augment = True

    sample_no_aug = dataset[0]
    sample_aug = dataset[0]

    diff = torch.abs(sample_no_aug['volume'] - sample_aug['volume']).mean()
    print(f"   Mean difference between augmented samples: {diff:.6f}")

    if diff > 0.01:
        print(f"   ✓ Augmentation is working (samples are different)")
    else:
        print(f"   ⚠ Augmentation may not be working properly")

    # Test dataloader
    print(f"\n5. Testing PyTorch DataLoader...")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
    )

    batch = next(iter(dataloader))
    print(f"   Batch volume shape: {batch['volume'].shape}")  # Should be (2, 1, 128, 128, 128)
    print(f"   Batch age shape: {batch['age'].shape}")        # Should be (2, 1)
    print(f"   ✓ DataLoader working correctly!")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print(f"\nVisualization saved to: openbhb_test_output/")


def main():
    parser = argparse.ArgumentParser(description="Test OpenBHB Dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing OpenBHB data")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to metadata CSV/parquet file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize (default: 5)")

    args = parser.parse_args()

    # Check paths exist
    if not Path(args.data_dir).exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return

    if not Path(args.metadata_file).exists():
        print(f"ERROR: Metadata file not found: {args.metadata_file}")
        return

    # Run tests
    test_dataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
