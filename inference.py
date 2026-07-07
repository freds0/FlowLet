import argparse
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Ensure the repository root is in the system path to import the flowlet package
sys.path.insert(0, str(Path(__file__).parent))

# Import the official model class from the freds0/flowlet architecture
from flowlet import FlowLet

def parse_args():
    parser = argparse.ArgumentParser(description="FlowLet Inference: Synthesize 3D Brain MRI conditioned on Age")
    
    # Required parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the folder containing the model checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--age",
        type=float,
        required=True,
        help="Desired age for the synthesized brain MRI (e.g., 25, 45, 65, 85)",
    )
    
    # Optional parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory where the generated image will be saved (Default: ./inference_results)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=10,
        help="Number of ODE sampling steps. The paper demonstrates that 10 steps offer the optimal balance (Default: 10)",
    )
    
    return parser.parse_args()

def load_model_from_dir(checkpoint_dir: str, device: str) -> FlowLet:
    dir_path = Path(checkpoint_dir)
    
    # Search for any .ckpt file inside the provided directory
    ckpt_files = list(dir_path.glob("*.ckpt"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint file (.ckpt) was found in the folder: {checkpoint_dir}")
    
    # Select the first checkpoint found in the directory
    checkpoint_path = ckpt_files[0]
    print(f"-> Loading checkpoint from: {checkpoint_path}...")
    
    # Load model using the official PyTorch Lightning method defined in the project
    model = FlowLet.load_from_checkpoint(str(checkpoint_path), map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    
    return model

def save_nifti(volume: np.ndarray, filepath: Path):
    """Saves the generated 3D volume into the standard medical NIfTI format (.nii.gz)."""
    try:
        import nibabel as nib
        affine = np.eye(4)  # Default affine matrix (1mm isotropic)
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, str(filepath))
        print(f"✓ 3D MRI image successfully saved in NIfTI format: {filepath}")
    except ImportError:
        print("Warning: The 'nibabel' package is not installed. Could not save as NIfTI.")
        print("To fix this, please run: pip install nibabel")

def main():
    args = parse_args()
    
    # Hardware device setup (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware device detected for inference: {device.upper()}")
    
    # Create the output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the model by searching inside the specified folder
    try:
        model = load_model_from_dir(args.checkpoint_dir, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 2. Prepare the age tensor in the shape expected by the model (B, 1)
    age_tensor = torch.tensor([args.age], device=device).float()
    
    print(f"Synthesizing 3D Brain MRI volume for age {args.age} using {args.n_steps} ODE steps...")
    
    # 3. Run generative sampling using the official integration method
    with torch.inference_mode():
        # Generates a tensor with shape (B, 1, D, H, W)
        generated_mri = model.synthesize(
            age=age_tensor,
            n_steps=args.n_steps
        )
    
    # Remove Batch and Channel dimensions to get a clean 3D matrix (D, H, W)
    volume_np = generated_mri.squeeze().cpu().numpy()
    
    # Generate a unique timestamp to prevent file collision
    timestamp = int(time.time())
    
    # 4. Export the final files including the age and unique timestamp
    file_name = f"synthesized_mri_age_{args.age:.2f}_{timestamp}.nii.gz"
    save_path = output_dir / file_name
    save_nifti(volume_np, save_path)
    
    # Also save a compressed NumPy version using the same name structure
    numpy_path = output_dir / f"synthesized_mri_age_{args.age:.2f}_{timestamp}.npy"
    np.save(str(numpy_path), volume_np)
    print(f"✓ Copy of the volume also saved in NumPy format: {numpy_path}")

if __name__ == "__main__":
    main()