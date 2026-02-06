#!/usr/bin/env python
"""
Test script for FlowLet visualization updates.

Tests that the new comparison visualizations work correctly.

Usage:
    python scripts/test_visualization.py
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Add FlowLet to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_compute_metrics():
    """Test the _compute_comparison_metrics method."""
    print("="*60)
    print("Testing Comparison Metrics")
    print("="*60)

    from flowlet.flowlet import FlowLet

    # Create a small FlowLet model for testing
    model = FlowLet(
        volume_shape=(32, 32, 32),  # Small for testing
        hidden_dims=[16, 32],
        wavelet_level=2,
    )

    # Create fake volumes
    batch_size = 2
    real_vols = torch.randn(batch_size, 1, 32, 32, 32)
    gen_vols = real_vols + torch.randn_like(real_vols) * 0.1  # Similar to real

    # Compute metrics
    print("\nComputing metrics...")
    metrics = model._compute_comparison_metrics(real_vols, gen_vols)

    print("\nMetrics computed:")
    for name, value in metrics.items():
        print(f"  {name:>12}: {value:.6f}")

    # Validate metrics
    assert 'mse' in metrics, "MSE metric missing"
    assert 'mae' in metrics, "MAE metric missing"
    assert 'psnr' in metrics, "PSNR metric missing"
    assert 'correlation' in metrics, "Correlation metric missing"

    # Check reasonable ranges
    assert metrics['mse'] >= 0, "MSE should be non-negative"
    assert metrics['mae'] >= 0, "MAE should be non-negative"
    assert metrics['psnr'] > 0, "PSNR should be positive"
    assert -1 <= metrics['correlation'] <= 1, "Correlation should be in [-1, 1]"

    print("\n✅ All metric tests passed!")
    return True


def test_normalize_slice():
    """Test the _normalize_slice method."""
    print("\n" + "="*60)
    print("Testing Slice Normalization")
    print("="*60)

    from flowlet.flowlet import FlowLet

    model = FlowLet(volume_shape=(32, 32, 32))

    # Test with regular slice
    slice_tensor = torch.randn(128, 128)
    normalized = model._normalize_slice(slice_tensor)

    print("\nOriginal slice:")
    print(f"  Min: {slice_tensor.min():.4f}")
    print(f"  Max: {slice_tensor.max():.4f}")

    print("\nNormalized slice:")
    print(f"  Min: {normalized.min():.4f}")
    print(f"  Max: {normalized.max():.4f}")

    # Check normalization
    assert normalized.min() >= 0, "Normalized min should be >= 0"
    assert normalized.max() <= 1, "Normalized max should be <= 1"
    assert torch.isclose(normalized.min(), torch.tensor(0.0), atol=1e-6), "Min should be close to 0"
    assert torch.isclose(normalized.max(), torch.tensor(1.0), atol=1e-6), "Max should be close to 1"

    # Test with constant slice
    constant_slice = torch.ones(128, 128) * 5.0
    normalized_constant = model._normalize_slice(constant_slice)

    print("\nConstant slice (all 5.0):")
    print(f"  Normalized min: {normalized_constant.min():.4f}")
    print(f"  Normalized max: {normalized_constant.max():.4f}")

    assert torch.allclose(normalized_constant, torch.zeros_like(normalized_constant)), \
        "Constant slice should normalize to 0"

    print("\n✅ All normalization tests passed!")
    return True


def test_visualization_methods_exist():
    """Test that all visualization methods exist and are callable."""
    print("\n" + "="*60)
    print("Testing Method Existence")
    print("="*60)

    from flowlet.flowlet import FlowLet

    model = FlowLet(volume_shape=(32, 32, 32))

    required_methods = [
        '_log_sample_images',
        '_compute_comparison_metrics',
        '_normalize_slice',
        '_log_images_to_loggers',
    ]

    print("\nChecking required methods:")
    for method_name in required_methods:
        if hasattr(model, method_name):
            method = getattr(model, method_name)
            if callable(method):
                print(f"  ✓ {method_name}")
            else:
                print(f"  ✗ {method_name} exists but is not callable")
                return False
        else:
            print(f"  ✗ {method_name} not found")
            return False

    print("\n✅ All required methods exist!")
    return True


def test_fake_validation_step():
    """Test a fake validation step with the new visualization code."""
    print("\n" + "="*60)
    print("Testing Fake Validation Step")
    print("="*60)

    from flowlet.flowlet import FlowLet

    # Create model
    model = FlowLet(
        volume_shape=(32, 32, 32),
        hidden_dims=[16, 32],
        wavelet_level=2,
    )
    model.eval()

    # Create fake batch
    batch = {
        'volume': torch.randn(4, 1, 32, 32, 32),
        'age': torch.tensor([[25.0], [45.0], [65.0], [85.0]]),
    }

    print("\nCreated fake batch:")
    print(f"  Volume shape: {batch['volume'].shape}")
    print(f"  Ages: {batch['age'].squeeze().tolist()}")

    # Test that we can compute metrics
    print("\nTesting metric computation...")
    generated = model.synthesize(batch['age'].squeeze(), n_steps=4)

    print(f"  Generated shape: {generated.shape}")

    metrics = model._compute_comparison_metrics(
        batch['volume'],
        generated
    )

    print("\n  Computed metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.6f}")

    # Test normalization on slices
    print("\nTesting slice normalization...")
    D, H, W = batch['volume'].shape[2], batch['volume'].shape[3], batch['volume'].shape[4]

    real_slice = batch['volume'][0, 0, D // 2, :, :]
    gen_slice = generated[0, 0, D // 2, :, :]

    normalized_real = model._normalize_slice(real_slice)
    normalized_gen = model._normalize_slice(gen_slice)

    print(f"  Real slice normalized: min={normalized_real.min():.4f}, max={normalized_real.max():.4f}")
    print(f"  Gen slice normalized: min={normalized_gen.min():.4f}, max={normalized_gen.max():.4f}")

    print("\n✅ Fake validation step test passed!")
    return True


def main():
    """Run all tests."""
    print("╔" + "="*58 + "╗")
    print("║" + " "*18 + "FlowLet Visualization Tests" + " "*13 + "║")
    print("╚" + "="*58 + "╝")
    print()

    tests = [
        ("Visualization Methods Exist", test_visualization_methods_exist),
        ("Slice Normalization", test_normalize_slice),
        ("Comparison Metrics", test_compute_metrics),
        ("Fake Validation Step", test_fake_validation_step),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not success:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nVisualization updates are working correctly.")
        print("You can now train with automatic comparison visualizations!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        print("Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
