"""
FlowLet Sampling Script.

Generate 3D brain MRI volumes conditioned on age using a trained FlowLet model.

Usage:
    # Generate samples for specific ages
    python scripts/sample.py --checkpoint path/to/checkpoint.ckpt --ages 25 45 65 85 --output_dir ./samples

    # Generate with more sampling steps for better quality
    python scripts/sample.py --checkpoint path/to/checkpoint.ckpt --ages 50 --n_steps 64 --output_dir ./samples

    # Generate age progression
    python scripts/sample.py --checkpoint path/to/checkpoint.ckpt --interpolate 20 90 10 --output_dir ./samples
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowlet import FlowLet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D brain MRI volumes with FlowLet"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )

    # Generation options (mutually exclusive)
    age_group = parser.add_mutually_exclusive_group(required=True)
    age_group.add_argument(
        "--ages",
        type=float,
        nargs="+",
        help="List of ages to generate (e.g., 25 45 65 85)",
    )
    age_group.add_argument(
        "--interpolate",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEPS"),
        help="Interpolate between ages (start, end, num_steps)",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory to save generated volumes",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sample",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--save_nifti",
        action="store_true",
        default=True,
        help="Save as NIfTI files (default)",
    )
    parser.add_argument(
        "--save_numpy",
        action="store_true",
        help="Also save as numpy arrays",
    )
    parser.add_argument(
        "--save_slices",
        action="store_true",
        help="Save middle slices as PNG images",
    )

    # Sampling options
    parser.add_argument(
        "--n_steps",
        type=int,
        default=32,
        help="Number of sampling steps (more = better quality)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more diverse)",
    )

    # Hardware options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )

    # Misc options
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> FlowLet:
    """Load FlowLet model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = FlowLet.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model}")
    return model


def generate_samples(
    model: FlowLet,
    ages: List[float],
    n_steps: int,
    temperature: float,
    batch_size: int,
    device: str,
) -> List[torch.Tensor]:
    """Generate volumes for given ages."""
    volumes = []

    # Process in batches
    for i in range(0, len(ages), batch_size):
        batch_ages = ages[i:i + batch_size]
        age_tensor = torch.tensor(batch_ages, device=device).float()

        print(f"Generating batch {i // batch_size + 1}, ages: {batch_ages}")

        with torch.inference_mode():
            batch_volumes = model.synthesize(
                age_tensor,
                n_steps=n_steps,
                temperature=temperature,
            )

        # Move to CPU and store
        for j, vol in enumerate(batch_volumes):
            volumes.append(vol.cpu())
            print(f"  Generated volume for age {batch_ages[j]:.1f}")

    return volumes


def save_nifti(volume: np.ndarray, filepath: Path, affine: Optional[np.ndarray] = None):
    """Save volume as NIfTI file."""
    import nibabel as nib

    if affine is None:
        # Default affine (1mm isotropic)
        affine = np.eye(4)

    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, str(filepath))


def save_slices(volume: np.ndarray, filepath: Path):
    """Save middle slices as PNG image."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get middle indices
    d, h, w = volume.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

    # Axial, Coronal, Sagittal views
    axes[0].imshow(volume[mid_d, :, :], cmap='gray')
    axes[0].set_title('Axial')
    axes[0].axis('off')

    axes[1].imshow(volume[:, mid_h, :], cmap='gray')
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    axes[2].imshow(volume[:, :, mid_w], cmap='gray')
    axes[2].set_title('Sagittal')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(filepath), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Determine ages to generate
    if args.ages:
        ages = args.ages
    else:
        # Interpolation mode
        start, end, n_steps = args.interpolate
        ages = np.linspace(start, end, int(n_steps)).tolist()

    print(f"Will generate {len(ages)} volumes for ages: {ages}")

    # Generate samples
    volumes = generate_samples(
        model,
        ages,
        args.n_steps,
        args.temperature,
        args.batch_size,
        args.device,
    )

    # Save outputs
    print(f"\nSaving outputs to {output_dir}...")

    for i, (vol, age) in enumerate(zip(volumes, ages)):
        # Remove batch and channel dimensions
        vol_np = vol.squeeze().numpy()

        # Base filename
        base_name = f"{args.prefix}_age{age:.0f}_{i:04d}"

        # Save NIfTI
        if args.save_nifti:
            nifti_path = output_dir / f"{base_name}.nii.gz"
            save_nifti(vol_np, nifti_path)
            print(f"  Saved: {nifti_path}")

        # Save numpy
        if args.save_numpy:
            numpy_path = output_dir / f"{base_name}.npy"
            np.save(str(numpy_path), vol_np)
            print(f"  Saved: {numpy_path}")

        # Save slices
        if args.save_slices:
            slices_path = output_dir / f"{base_name}_slices.png"
            save_slices(vol_np, slices_path)
            print(f"  Saved: {slices_path}")

    print(f"\nDone! Generated {len(volumes)} volumes.")


if __name__ == "__main__":
    main()
