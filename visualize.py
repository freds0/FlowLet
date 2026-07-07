import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="FlowLet Visualization Script: Display 3D Brain MRI Volumes")
    
    # Required arguments
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the 3D volume file (.nii.gz, .nii, or .npy)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save the plotted slice figures (Default: ./visualizations)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the pop-up window on screen, only save the image to disk",
    )
    
    return parser.parse_args()

def load_volume(file_path: str) -> np.ndarray:
    """Loads a 3D volume from a NumPy array or a NIfTI medical file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"The input file was not found: {file_path}")
        
    print(f"-> Reading volume: {path.name}...")
    
    if path.suffix == ".npy":
        volume = np.load(str(path))
    elif path.name.endswith(".nii.gz") or path.suffix == ".nii":
        try:
            import nibabel as nib
            nii_img = nib.load(str(path))
            volume = nii_img.get_fdata()
        except ImportError:
            raise ImportError(
                "The 'nibabel' package is required to read NIfTI files.\n"
                "Please run: pip install nibabel"
            )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .npy, .nii, or .nii.gz")
        
    # Remove single-dimensional entries if present (e.g., batch or channel dimensions)
    volume = np.squeeze(volume)
    
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D matrix, but loaded array has shape: {volume.shape}")
        
    return volume

def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalizes 2D slice intensities to [0, 1] range for optimal visualization contrast."""
    min_val = slice_2d.min()
    max_val = slice_2d.max()
    if max_val - min_val > 1e-8:
        return (slice_2d - min_val) / (max_val - min_val)
    return slice_2d - min_val

def plot_mri_slices(volume: np.ndarray, file_name: str, output_dir: Path, show_plot: bool):
    """Extracts mid-slices across all 3 anatomical planes and renders them using matplotlib."""
    # Retrieve matrix dimensions
    d, h, w = volume.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    # Extract the central slices
    axial_slice = normalize_slice(volume[mid_d, :, :])
    coronal_slice = normalize_slice(volume[:, mid_h, :])
    sagittal_slice = normalize_slice(volume[:, :, mid_w])
    
    # Create the figure subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Anatomical Views: {file_name} (Shape: {d}x{h}x{w})", fontsize=14, fontweight='bold')
    
    # 1. Axial View (Transverse Plane)
    axes[0].imshow(np.rot90(axial_slice), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Axial (Slice {mid_d})", fontsize=12)
    axes[0].axis('off')
    
    # 2. Coronal View (Frontal Plane)
    axes[1].imshow(np.rot90(coronal_slice), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Coronal (Slice {mid_h})", fontsize=12)
    axes[1].axis('off')
    
    # 3. Sagittal View (Median Plane)
    axes[2].imshow(np.rot90(sagittal_slice), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Sagittal (Slice {mid_w})", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save image to file
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"vis_{Path(file_name).stem}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"✓ Visualization image saved to disk: {save_path}")
    
    # Render interface on screen if allowed
    if show_plot:
        print("-> Opening interactive display window...")
        plt.show()
        
    plt.close()

def main():
    args = parse_args()
    
    # 1. Load the 3D data array
    try:
        volume = load_volume(args.input_file)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    print(f"✓ Volume successfully loaded. Matrix shape: {volume.shape}")
    
    # Determine if pop-up window should trigger
    show_plot = not args.no_show
    
    # 2. Generate and save plots
    plot_mri_slices(
        volume=volume,
        file_name=Path(args.input_file).name,
        output_dir=Path(args.output_dir),
        show_plot=show_plot
    )

if __name__ == "__main__":
    main()