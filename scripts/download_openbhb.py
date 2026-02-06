"""
Download OpenBHB dataset from Hugging Face.

This script demonstrates how to download the OpenBHB dataset
using the Hugging Face datasets library.

Installation:
    pip install datasets huggingface_hub

Usage:
    # Basic download
    python scripts/download_openbhb.py --output_dir ./data/openbhb

    # Download with authentication (if required)
    python scripts/download_openbhb.py \
        --output_dir ./data/openbhb \
        --token YOUR_HF_TOKEN

    # Download only metadata
    python scripts/download_openbhb.py \
        --output_dir ./data/openbhb \
        --metadata_only

Note:
    This is a template script. You'll need to:
    1. Replace DATASET_NAME with the actual OpenBHB dataset identifier
    2. Adapt the data extraction logic to match the actual dataset structure
"""

import argparse
from pathlib import Path

try:
    from datasets import load_dataset
    from huggingface_hub import login
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please run: pip install datasets huggingface_hub")
    exit(1)

import pandas as pd
import numpy as np
from tqdm import tqdm


# TODO: Replace with actual OpenBHB dataset name
# Example: "your-org/openbhb-dataset"
DATASET_NAME = "REPLACE_WITH_ACTUAL_DATASET_NAME"


def download_openbhb(
    output_dir: str,
    token: str = None,
    metadata_only: bool = False,
    split: str = "train"
):
    """
    Download OpenBHB dataset from Hugging Face.

    Args:
        output_dir: Directory to save downloaded data
        token: Hugging Face API token (if dataset is private)
        metadata_only: Only download metadata, skip volumes
        split: Dataset split to download (train/validation/test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Login if token provided
    if token:
        print("Logging in to Hugging Face...")
        login(token=token)

    # Load dataset
    print(f"\nLoading dataset: {DATASET_NAME}")
    print(f"Split: {split}")
    print("This may take a while depending on dataset size...")

    try:
        dataset = load_dataset(DATASET_NAME, split=split)
    except Exception as e:
        print(f"\nERROR: Failed to load dataset.")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify the dataset name is correct")
        print("2. Check if you need authentication (use --token)")
        print("3. Ensure you have accepted the dataset terms on Hugging Face")
        return

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Extract metadata
    print("\nExtracting metadata...")
    metadata_list = []

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        # TODO: Adapt these field names to match actual dataset
        # This is a template based on expected OpenBHB structure
        metadata_list.append({
            'participant_id': sample.get('participant_id', f'sub_{idx:05d}'),
            'age': sample.get('age', 0.0),
            'site': sample.get('site', 'unknown'),
            'quasiraw_3d_path': f"sub_{sample.get('participant_id', f'{idx:05d}')}/quasiraw_3d.npy",
            # Add other fields as needed
        })

    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n✓ Metadata saved to: {metadata_path}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"  Total samples: {len(metadata_df)}")
    print(f"  Age range: {metadata_df['age'].min():.1f} - {metadata_df['age'].max():.1f}")
    print(f"  Number of sites: {metadata_df['site'].nunique()}")

    # Download volumes (if not metadata_only)
    if not metadata_only:
        print("\nDownloading volumes...")
        print("This will take significant time and disk space!")

        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        for idx, sample in enumerate(tqdm(dataset, desc="Saving volumes")):
            # Get participant ID
            participant_id = sample.get('participant_id', f'sub_{idx:05d}')

            # Create participant directory
            participant_dir = data_dir / f"sub_{participant_id}"
            participant_dir.mkdir(exist_ok=True)

            # TODO: Adapt this to match actual dataset structure
            # Example assumes volume is in 'volume' or 'image' field
            if 'volume' in sample:
                volume = np.array(sample['volume'])
            elif 'image' in sample:
                volume = np.array(sample['image'])
            else:
                print(f"Warning: No volume found for sample {idx}")
                continue

            # Save as .npy
            volume_path = participant_dir / "quasiraw_3d.npy"
            np.save(volume_path, volume)

        print(f"\n✓ Volumes saved to: {data_dir}")

    # Summary
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Metadata: {metadata_path}")
    if not metadata_only:
        print(f"Volumes: {data_dir}")

    print("\nNext steps:")
    print("1. Validate the dataset:")
    print(f"   python scripts/prepare_openbhb.py \\")
    print(f"     --data_dir {data_dir} \\")
    print(f"     --metadata_file {metadata_path} \\")
    print(f"     --validate --stats")
    print("\n2. Test the dataloader:")
    print(f"   python examples/test_openbhb_dataloader.py \\")
    print(f"     --data_dir {data_dir} \\")
    print(f"     --metadata_file {metadata_path}")
    print("\n3. Start fine-tuning:")
    print(f"   python flowlet/train.py experiment=flowlet_openbhb_finetune \\")
    print(f"     ckpt_path=/path/to/fomo60k_checkpoint.ckpt \\")
    print(f"     data.data_dir={data_dir} \\")
    print(f"     data.metadata_file={metadata_path}")


def download_from_url(url: str, output_dir: str):
    """
    Alternative method: Download from direct URL.

    Use this if the dataset is hosted elsewhere (not Hugging Face).
    """
    import urllib.request
    import zipfile
    import tarfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {url}")

    # Determine filename from URL
    filename = url.split("/")[-1]
    download_path = output_dir / filename

    # Download
    print("Downloading... This may take a while.")
    urllib.request.urlretrieve(url, download_path)
    print(f"✓ Downloaded to: {download_path}")

    # Extract if compressed
    if filename.endswith('.zip'):
        print("Extracting zip file...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Extraction complete")

    elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        print("Extracting tar.gz file...")
        with tarfile.open(download_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
        print("✓ Extraction complete")

    print(f"\nData extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download OpenBHB dataset")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save downloaded data"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (required if dataset is private)"
    )

    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Only download metadata, skip volume files"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test", "all"],
        help="Dataset split to download"
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Alternative: Direct download URL (if not using Hugging Face)"
    )

    args = parser.parse_args()

    # Check if dataset name is configured
    if DATASET_NAME == "REPLACE_WITH_ACTUAL_DATASET_NAME" and not args.url:
        print("ERROR: Dataset name not configured!")
        print("\nPlease edit this script and replace DATASET_NAME with the actual")
        print("OpenBHB dataset identifier from Hugging Face.")
        print("\nAlternatively, use --url to download from a direct URL.")
        return

    # Download
    if args.url:
        # Direct URL download
        download_from_url(args.url, args.output_dir)
    else:
        # Hugging Face download
        download_openbhb(
            output_dir=args.output_dir,
            token=args.token,
            metadata_only=args.metadata_only,
            split=args.split
        )


if __name__ == "__main__":
    main()
