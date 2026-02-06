#!/usr/bin/env python
"""
Quick setup checker for OpenBHB fine-tuning.

Verifies that all components are correctly installed and configured.

Usage:
    python scripts/check_openbhb_setup.py
"""

import sys
from pathlib import Path


def check_imports():
    """Check that all required packages are installed."""
    print("Checking Python packages...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('lightning', 'PyTorch Lightning'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('pywt', 'PyWavelets'),
        ('hydra', 'Hydra'),
    ]

    missing = []
    errors = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)
        except Exception as e:
            # Package exists but has runtime errors (like version conflicts)
            print(f"  ⚠ {name} - IMPORT ERROR (may still work)")
            errors.append((name, str(e)))

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    if errors:
        print(f"\n⚠ Some packages have import warnings:")
        for name, error in errors:
            print(f"  {name}: {error[:80]}...")
        print("These warnings may not prevent training.")

    print("✓ Required packages are available\n")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("Checking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
                # Get memory info
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"      Memory: {mem_total:.1f} GB")
        else:
            print("  ⚠ CUDA not available (CPU mode)")
            print("  Fine-tuning will be very slow on CPU")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}\n")
        return False


def check_flowlet_structure():
    """Check FlowLet project structure."""
    print("Checking FlowLet structure...")

    required_files = [
        'flowlet/__init__.py',
        'flowlet/flowlet.py',
        'flowlet/train.py',
        'flowlet/data/__init__.py',
        'flowlet/data/mri_dataset.py',
        'flowlet/data/mri_datamodule.py',
        'flowlet/data/openbhb_dataset.py',
        'flowlet/models/unet3d.py',
        'flowlet/models/flow_matching.py',
        'flowlet/utils/wavelet3d.py',
        'configs/train.yaml',
        'configs/data/openbhb.yaml',
        'configs/experiment/flowlet_openbhb_finetune.yaml',
        'scripts/prepare_openbhb.py',
    ]

    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            missing.append(file)

    if missing:
        print(f"\n❌ Missing files: {len(missing)}")
        return False

    print("✓ All required files present\n")
    return True


def check_openbhb_classes():
    """Check that OpenBHB classes can be imported."""
    print("Checking OpenBHB classes...")

    try:
        from flowlet.data import OpenBHBDataset, OpenBHBDataModule
        print("  ✓ OpenBHBDataset")
        print("  ✓ OpenBHBDataModule")
        print("✓ OpenBHB classes importable\n")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}\n")
        return False


def check_configs():
    """Check Hydra configs."""
    print("Checking Hydra configs...")

    try:
        from omegaconf import OmegaConf

        # Check data config
        data_config = OmegaConf.load('configs/data/openbhb.yaml')
        print("  ✓ configs/data/openbhb.yaml")

        # Check experiment config
        exp_config = OmegaConf.load('configs/experiment/flowlet_openbhb_finetune.yaml')
        print("  ✓ configs/experiment/flowlet_openbhb_finetune.yaml")

        print("✓ Configs valid\n")
        return True

    except Exception as e:
        print(f"  ✗ Config error: {e}\n")
        return False


def print_summary(checks):
    """Print summary of checks."""
    print("="*60)
    print("SETUP CHECK SUMMARY")
    print("="*60)

    passed = sum(checks.values())
    total = len(checks)

    for check_name, passed_check in checks.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        print(f"{check_name:.<40} {status}")

    print("="*60)

    if passed == total:
        print("✓ All checks passed! Ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Prepare your OpenBHB dataset")
        print("2. Update configs/data/openbhb.yaml with your paths")
        print("3. Run: python flowlet/train.py experiment=flowlet_openbhb_finetune \\")
        print("         ckpt_path=/path/to/fomo60k.ckpt \\")
        print("         data.data_dir=/path/to/openbhb/data \\")
        print("         data.metadata_file=/path/to/metadata.csv")
    else:
        print(f"✗ {total - passed}/{total} checks failed. Please fix issues above.")
        print("\nCommon fixes:")
        print("- Install packages: pip install -r requirements.txt")
        print("- Ensure you're in the FlowLet root directory")
        print("- Re-run the OpenBHB setup if files are missing")

    print("="*60)

    return passed == total


def main():
    print("="*60)
    print("OpenBHB Fine-tuning Setup Checker")
    print("="*60)
    print()

    # Run checks
    checks = {
        'Python Packages': check_imports(),
        'CUDA': check_cuda(),
        'Project Structure': check_flowlet_structure(),
        'OpenBHB Classes': check_openbhb_classes(),
        'Hydra Configs': check_configs(),
    }

    print()

    # Print summary
    all_passed = print_summary(checks)

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
