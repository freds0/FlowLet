"""
Standalone test script for evaluating a trained FlowLet model on a dedicated test dataset.

Evaluates the model on a separate test set (not a re-split of training data) and saves
metrics to disk.

Metrics computed:
- CFM loss (flow matching loss on test data)
- MSE, MAE, PSNR, Pearson correlation between real and generated volumes
- SSIM (if scikit-image is available)
- Per-sample and aggregate statistics
- Per-age-bin analysis

Usage:
    python scripts/test.py \
        --checkpoint checkpoints/last.ckpt \
        --data_dir /path/to/test_data \
        --metadata_file /path/to/test.tsv \
        --output_dir ./test_results

    # With visualization and more sampling steps
    python scripts/test.py \
        --checkpoint checkpoints/last.ckpt \
        --data_dir /path/to/test_data \
        --metadata_file /path/to/test.tsv \
        --output_dir ./test_results \
        --save_visualizations \
        --n_steps 64

    # Limit number of generation samples (faster)
    python scripts/test.py \
        --checkpoint checkpoints/last.ckpt \
        --data_dir /path/to/test_data \
        --metadata_file /path/to/test.tsv \
        --output_dir ./test_results \
        --max_gen_samples 50
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowlet import FlowLet
from flowlet.data.openbhb_dataset import OpenBHBDataset
from flowlet.data.mri_datamodule import MRIBatchCollate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FlowLet model on a dedicated test dataset"
    )

    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing test data",
    )
    parser.add_argument(
        "--metadata_file", type=str, required=True,
        help="Path to test metadata TSV/CSV/parquet file",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./test_results",
        help="Directory to save test results",
    )

    # Data
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # Generation
    parser.add_argument(
        "--max_gen_samples", type=int, default=None,
        help="Max number of samples for generation metrics (None = all test samples)",
    )
    parser.add_argument(
        "--n_steps", type=int, default=32,
        help="Number of ODE sampling steps for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature",
    )

    # Visualization
    parser.add_argument(
        "--save_visualizations", action="store_true",
        help="Save comparison slice images",
    )
    parser.add_argument(
        "--n_vis_samples", type=int, default=8,
        help="Number of samples to visualize",
    )

    # Hardware
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> FlowLet:
    """Load FlowLet model from checkpoint with weights_only=False for PyTorch >= 2.6."""
    print(f"Loading model from {checkpoint_path}...")
    model = FlowLet.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  Volume shape: {model.volume_shape}")
    print(f"  Age range: [{model.age_min}, {model.age_max}]")
    return model


def build_test_dataloader(
    data_dir: str,
    metadata_file: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, OpenBHBDataset]:
    """Build a test DataLoader using the entire provided dataset (no splitting)."""
    dataset = OpenBHBDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        target_shape=(128, 128, 128),
        age_range=None,
        normalize=True,
        augment=False,
        cache_data=False,
        include_site=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=MRIBatchCollate(),
    )

    return loader, dataset


def compute_ssim_3d(
    real: torch.Tensor, gen: torch.Tensor, data_range: float = 10.0
) -> float:
    """Compute mean SSIM between two 3D volumes.

    Falls back to a simplified implementation if skimage is not available.
    Expects inputs of shape (D, H, W).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        real_np = real.cpu().numpy()
        gen_np = gen.cpu().numpy()
        score = ssim(real_np, gen_np, data_range=data_range)
        return float(score)
    except ImportError:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu_x = real.mean()
        mu_y = gen.mean()
        sigma_x_sq = real.var()
        sigma_y_sq = gen.var()
        sigma_xy = ((real - mu_x) * (gen - mu_y)).mean()

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        return float(numerator / denominator)


def compute_sample_metrics(
    real_vol: torch.Tensor, gen_vol: torch.Tensor
) -> Dict[str, float]:
    """Compute all metrics for a single real vs generated volume pair.

    Args:
        real_vol: (1, D, H, W) real volume
        gen_vol: (1, D, H, W) generated volume

    Returns:
        Dictionary of metric values.
    """
    metrics = {}

    # MSE
    mse = F.mse_loss(gen_vol, real_vol).item()
    metrics["mse"] = mse

    # MAE
    mae = F.l1_loss(gen_vol, real_vol).item()
    metrics["mae"] = mae

    # PSNR
    max_val = max(real_vol.max().item(), gen_vol.max().item())
    if mse > 0:
        metrics["psnr"] = 10 * math.log10((max_val ** 2) / mse)
    else:
        metrics["psnr"] = 100.0

    # Pearson correlation
    real_flat = real_vol.flatten()
    gen_flat = gen_vol.flatten()
    real_mean = real_flat.mean()
    gen_mean = gen_flat.mean()
    numerator = ((real_flat - real_mean) * (gen_flat - gen_mean)).sum()
    denominator = torch.sqrt(
        ((real_flat - real_mean) ** 2).sum() * ((gen_flat - gen_mean) ** 2).sum()
    )
    if denominator > 1e-8:
        metrics["pearson_r"] = (numerator / denominator).item()
    else:
        metrics["pearson_r"] = 0.0

    # SSIM
    real_3d = real_vol.squeeze(0)
    gen_3d = gen_vol.squeeze(0)
    data_range = real_3d.max().item() - real_3d.min().item()
    if data_range < 1e-8:
        data_range = 10.0
    metrics["ssim"] = compute_ssim_3d(real_3d, gen_3d, data_range=data_range)

    return metrics


@torch.no_grad()
def evaluate_cfm_loss(
    model: FlowLet, test_loader: DataLoader, device: str
) -> Tuple[float, List[float]]:
    """Compute the CFM loss on the test set.

    Returns:
        (mean_loss, list of per-batch losses)
    """
    model.eval()
    losses = []

    for batch_idx, batch in enumerate(test_loader):
        volume = batch["volume"].to(device)
        age = batch["age"].to(device)
        loss, _ = model(volume, age)
        losses.append(loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  CFM loss batch {batch_idx + 1}/{len(test_loader)}: {loss.item():.6f}")

    mean_loss = sum(losses) / len(losses)
    return mean_loss, losses


@torch.no_grad()
def evaluate_generation(
    model: FlowLet,
    test_loader: DataLoader,
    device: str,
    n_steps: int,
    temperature: float,
    max_samples: Optional[int],
) -> List[Dict]:
    """Generate volumes for test samples and compute per-sample metrics.

    Returns:
        List of dicts with per-sample metrics and metadata.
    """
    model.eval()
    results = []
    sample_count = 0

    for batch_idx, batch in enumerate(test_loader):
        volume = batch["volume"].to(device)  # (B, 1, D, H, W)
        age = batch["age"].to(device)  # (B, 1)

        # Generate volumes conditioned on the same ages
        ages_1d = age.squeeze(1)  # (B,)
        gen_volume = model.synthesize(ages_1d, n_steps=n_steps, temperature=temperature)

        bs = volume.shape[0]
        for i in range(bs):
            if max_samples is not None and sample_count >= max_samples:
                return results

            real_v = volume[i]  # (1, D, H, W)
            gen_v = gen_volume[i]  # (1, D, H, W)
            age_val = age[i].item()

            metrics = compute_sample_metrics(real_v, gen_v)
            metrics["age"] = age_val
            metrics["sample_idx"] = sample_count
            results.append(metrics)

            sample_count += 1
            if sample_count % 10 == 0:
                print(
                    f"  Generated {sample_count} samples | "
                    f"MSE={metrics['mse']:.6f} MAE={metrics['mae']:.4f} "
                    f"PSNR={metrics['psnr']:.2f} r={metrics['pearson_r']:.4f}"
                )

    return results


def compute_aggregate_metrics(sample_results: List[Dict]) -> Dict[str, float]:
    """Compute aggregate statistics from per-sample results."""
    df = pd.DataFrame(sample_results)

    agg = {}
    for col in ["mse", "mae", "psnr", "pearson_r", "ssim"]:
        if col in df.columns:
            agg[f"{col}_mean"] = float(df[col].mean())
            agg[f"{col}_std"] = float(df[col].std())
            agg[f"{col}_median"] = float(df[col].median())
            agg[f"{col}_min"] = float(df[col].min())
            agg[f"{col}_max"] = float(df[col].max())

    agg["n_samples"] = len(df)
    return agg


def compute_age_bin_metrics(
    sample_results: List[Dict], bin_edges: Optional[List[float]] = None
) -> pd.DataFrame:
    """Compute metrics grouped by age bins."""
    df = pd.DataFrame(sample_results)

    if bin_edges is None:
        bin_edges = [0, 20, 30, 40, 50, 60, 70, 80, 100]

    df["age_bin"] = pd.cut(df["age"], bins=bin_edges)

    agg_funcs = {
        "mse": ["mean", "std", "count"],
        "mae": ["mean", "std"],
        "psnr": ["mean", "std"],
        "pearson_r": ["mean", "std"],
        "ssim": ["mean", "std"],
    }

    # Only aggregate columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    binned = df.groupby("age_bin", observed=False).agg(agg_funcs)

    # Flatten multi-level columns
    binned.columns = ["_".join(col).strip() for col in binned.columns]
    binned = binned.reset_index()
    binned["age_bin"] = binned["age_bin"].astype(str)

    return binned


def save_visualizations(
    model: FlowLet,
    test_loader: DataLoader,
    device: str,
    output_dir: Path,
    n_steps: int,
    n_samples: int = 8,
):
    """Save comparison visualizations (real vs generated slices)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    for batch in test_loader:
        volume = batch["volume"].to(device)
        age = batch["age"].to(device)

        ages_1d = age.squeeze(1)
        with torch.inference_mode():
            gen_volume = model.synthesize(ages_1d, n_steps=n_steps)

        bs = volume.shape[0]
        for i in range(bs):
            if sample_count >= n_samples:
                _save_summary_grid(vis_dir, n_samples)
                return

            real_v = volume[i, 0].cpu().numpy()  # (D, H, W)
            gen_v = gen_volume[i, 0].cpu().numpy()
            age_val = age[i].item()
            D, H, W = real_v.shape

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Sample {sample_count} | Age: {age_val:.1f}", fontsize=14)

            def normalize(s):
                smin, smax = s.min(), s.max()
                if smax - smin > 1e-8:
                    return (s - smin) / (smax - smin)
                return s - smin

            views = [
                ("Axial", real_v[D // 2, :, :], gen_v[D // 2, :, :]),
                ("Coronal", real_v[:, H // 2, :], gen_v[:, H // 2, :]),
                ("Sagittal", real_v[:, :, W // 2], gen_v[:, :, W // 2]),
            ]

            for j, (name, real_s, gen_s) in enumerate(views):
                axes[0, j].imshow(normalize(real_s), cmap="gray", vmin=0, vmax=1)
                axes[0, j].set_title(f"Real - {name}")
                axes[0, j].axis("off")

                axes[1, j].imshow(normalize(gen_s), cmap="gray", vmin=0, vmax=1)
                axes[1, j].set_title(f"Generated - {name}")
                axes[1, j].axis("off")

            plt.tight_layout()
            fig.savefig(vis_dir / f"comparison_{sample_count:03d}_age{age_val:.0f}.png", dpi=150)
            plt.close(fig)

            sample_count += 1

    if sample_count > 0:
        _save_summary_grid(vis_dir, sample_count)


def _save_summary_grid(vis_dir: Path, n_samples: int):
    """Create a summary grid image from individual comparison images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    images = sorted(vis_dir.glob("comparison_*.png"))[:n_samples]
    if not images:
        return

    ncols = min(4, len(images))
    nrows = math.ceil(len(images) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, img_path in enumerate(images):
        r, c = divmod(idx, ncols)
        img = mpimg.imread(str(img_path))
        axes[r, c].imshow(img)
        axes[r, c].axis("off")

    # Hide empty subplots
    for idx in range(len(images), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(vis_dir / "summary_grid.png", dpi=100)
    plt.close(fig)
    print(f"  Saved summary grid to {vis_dir / 'summary_grid.png'}")


def save_metrics_plots(
    sample_results: List[Dict], output_dir: Path
):
    """Save metric distribution plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(sample_results)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["mse", "mae", "psnr", "pearson_r", "ssim"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # 1. Metric distributions
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        ax.hist(df[col], bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"mean={df[col].mean():.4f}")
        ax.set_title(col.upper())
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    fig.savefig(plots_dir / "metric_distributions.png", dpi=150)
    plt.close(fig)

    # 2. Metrics vs age scatter
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols):
        ax.scatter(df["age"], df[col], alpha=0.5, s=15)
        # Linear trend
        z = np.polyfit(df["age"], df[col], 1)
        p = np.poly1d(z)
        age_sorted = np.sort(df["age"].values)
        ax.plot(age_sorted, p(age_sorted), "r--", linewidth=2, label="trend")
        ax.set_title(f"{col.upper()} vs Age")
        ax.set_xlabel("Age")
        ax.set_ylabel(col)
        ax.legend()

    plt.tight_layout()
    fig.savefig(plots_dir / "metrics_vs_age.png", dpi=150)
    plt.close(fig)

    # 3. Age distribution of test set
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["age"], bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_title("Test Set Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "test_age_distribution.png", dpi=150)
    plt.close(fig)

    print(f"  Saved plots to {plots_dir}")


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("FlowLet - Model Evaluation on Dedicated Test Set")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    model = load_model(args.checkpoint, args.device)

    # ------------------------------------------------------------------ #
    # 2. Build test dataloader (uses entire dataset, no splitting)
    # ------------------------------------------------------------------ #
    print("\nPreparing test dataset...")
    test_loader, test_dataset = build_test_dataloader(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Test set size: {len(test_dataset)}")

    all_results = {}

    # ------------------------------------------------------------------ #
    # 3. Evaluate CFM loss on test set
    # ------------------------------------------------------------------ #
    print("\n--- Evaluating CFM loss on test set ---")
    t0 = time.time()
    mean_cfm_loss, batch_losses = evaluate_cfm_loss(model, test_loader, args.device)
    cfm_time = time.time() - t0
    print(f"  Mean CFM loss: {mean_cfm_loss:.6f}  ({cfm_time:.1f}s)")

    all_results["cfm_loss"] = {
        "mean": mean_cfm_loss,
        "std": float(np.std(batch_losses)),
        "min": float(np.min(batch_losses)),
        "max": float(np.max(batch_losses)),
        "n_batches": len(batch_losses),
    }

    # ------------------------------------------------------------------ #
    # 4. Generate volumes and compute comparison metrics
    # ------------------------------------------------------------------ #
    print(f"\n--- Evaluating generation quality (n_steps={args.n_steps}) ---")
    t0 = time.time()
    sample_results = evaluate_generation(
        model=model,
        test_loader=test_loader,
        device=args.device,
        n_steps=args.n_steps,
        temperature=args.temperature,
        max_samples=args.max_gen_samples,
    )
    gen_time = time.time() - t0
    n_gen = len(sample_results)
    print(f"  Evaluated {n_gen} samples ({gen_time:.1f}s)")

    # Aggregate metrics
    agg_metrics = compute_aggregate_metrics(sample_results)
    all_results["generation_metrics"] = agg_metrics

    print("\n--- Aggregate Generation Metrics ---")
    for k, v in agg_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-age-bin metrics
    age_bin_df = compute_age_bin_metrics(sample_results)
    all_results["age_bin_metrics"] = age_bin_df.to_dict(orient="records")

    print("\n--- Per-Age-Bin Metrics ---")
    print(age_bin_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 5. Save results
    # ------------------------------------------------------------------ #
    print(f"\n--- Saving results to {output_dir} ---")

    # Save full config + results as JSON
    all_results["config"] = config
    all_results["config"]["checkpoint"] = str(args.checkpoint)

    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Save per-sample results as CSV
    per_sample_csv = output_dir / "per_sample_metrics.csv"
    pd.DataFrame(sample_results).to_csv(per_sample_csv, index=False)
    print(f"  Saved: {per_sample_csv}")

    # Save age-bin results as CSV
    age_bin_csv = output_dir / "age_bin_metrics.csv"
    age_bin_df.to_csv(age_bin_csv, index=False)
    print(f"  Saved: {age_bin_csv}")

    # Save summary text
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("FlowLet - Model Evaluation on Dedicated Test Set\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data dir:   {args.data_dir}\n")
        f.write(f"Metadata:   {args.metadata_file}\n")
        f.write(f"Timestamp:  {config['timestamp']}\n")
        f.write(f"Device:     {args.device}\n")
        f.write(f"Seed:       {args.seed}\n\n")

        f.write(f"Test samples:       {len(test_dataset)}\n")
        f.write(f"Generation samples: {n_gen}\n")
        f.write(f"Sampling steps:     {args.n_steps}\n")
        f.write(f"Temperature:        {args.temperature}\n\n")

        f.write("--- CFM Loss ---\n")
        f.write(f"  Mean: {mean_cfm_loss:.6f}\n")
        f.write(f"  Std:  {all_results['cfm_loss']['std']:.6f}\n\n")

        f.write("--- Generation Metrics (mean +/- std) ---\n")
        for metric in ["mse", "mae", "psnr", "pearson_r", "ssim"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in agg_metrics:
                f.write(
                    f"  {metric.upper():12s}: "
                    f"{agg_metrics[mean_key]:.6f} +/- {agg_metrics[std_key]:.6f}\n"
                )

        f.write(f"\n--- Per-Age-Bin Metrics ---\n")
        f.write(age_bin_df.to_string(index=False))
        f.write("\n")

    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------ #
    # 6. Visualizations (optional)
    # ------------------------------------------------------------------ #
    if args.save_visualizations:
        print("\n--- Generating visualizations ---")
        save_visualizations(
            model=model,
            test_loader=test_loader,
            device=args.device,
            output_dir=output_dir,
            n_steps=args.n_steps,
            n_samples=args.n_vis_samples,
        )

    # Save metric plots (always, if matplotlib is available)
    if sample_results:
        try:
            save_metrics_plots(sample_results, output_dir)
        except ImportError:
            print("  Skipping plots (matplotlib not available)")

    print("\n" + "=" * 70)
    print("Test complete!")
    print(f"  CFM Loss:    {mean_cfm_loss:.6f}")
    print(f"  MSE:         {agg_metrics['mse_mean']:.6f} +/- {agg_metrics['mse_std']:.6f}")
    print(f"  MAE:         {agg_metrics['mae_mean']:.6f} +/- {agg_metrics['mae_std']:.6f}")
    print(f"  PSNR:        {agg_metrics['psnr_mean']:.2f} +/- {agg_metrics['psnr_std']:.2f}")
    print(f"  Pearson r:   {agg_metrics['pearson_r_mean']:.4f} +/- {agg_metrics['pearson_r_std']:.4f}")
    print(f"  SSIM:        {agg_metrics['ssim_mean']:.4f} +/- {agg_metrics['ssim_std']:.4f}")
    print(f"  Results at:  {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
