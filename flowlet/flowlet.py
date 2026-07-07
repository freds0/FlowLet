"""
FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching.

Main model class that combines:
- 3D Wavelet Transform for domain conversion
- U-Net 3D for velocity field prediction
- Conditional Flow Matching for generation

Adapted from the FlowMAC architecture for audio to 3D medical imaging.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from flowlet.models.unet3d import UNet3D
from flowlet.models.flow_matching import ConditionalFlowMatching, RectifiedFlowMatching
from flowlet.utils.wavelet3d import Wavelet3DTransform


class FlowLet(LightningModule):
    """
    FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching.

    This model generates 3D brain MRI volumes conditioned on subject age.
    It operates in the wavelet domain for computational efficiency and
    uses Conditional Flow Matching for high-quality generation.

    Architecture:
        MRI Volume → Wavelet Transform → CFM in wavelet domain → Inverse Wavelet → MRI Volume

    Args:
        # Wavelet parameters
        wavelet: Wavelet type (default: 'db4')
        wavelet_level: Number of decomposition levels (default: 3)

        # U-Net parameters
        hidden_dims: Channel dimensions for U-Net levels
        time_embed_dim: Timestep embedding dimension
        cond_embed_dim: Condition (age) embedding dimension
        num_res_blocks: Residual blocks per level
        attention_levels: Which levels use attention
        dropout: Dropout rate

        # CFM parameters
        sigma_min: Minimum noise level
        use_rectified_flow: Use rectified flow instead of OT-CFM

        # Training parameters
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_steps: Linear warmup steps

        # Sampling parameters
        n_sample_steps: Default number of sampling steps
    """

    def __init__(
        self,
        # Wavelet parameters
        wavelet: str = 'haar',
        wavelet_level: int = 1,
        # Volume parameters
        volume_shape: Tuple[int, int, int] = (128, 128, 128),
        # U-Net parameters
        hidden_dims: List[int] = [64, 128, 256, 512],
        time_embed_dim: int = 256,
        cond_embed_dim: int = 64,
        num_res_blocks: int = 2,
        attention_levels: List[int] = [2, 3],
        dropout: float = 0.0,
        # CFM parameters
        sigma_min: float = 1e-4,
        use_rectified_flow: bool = True,
        # Training parameters
        learning_rate: float = 3e-6,
        weight_decay: float = 1e-5,
        warmup_steps: int = 0,
        # Scheduler parameters
        scheduler_type: str = "cosine",  # "cosine" or "constant"
        scheduler_eta_min: float = 1e-7,
        # Sampling parameters
        n_sample_steps: int = 10,
        # Data statistics
        data_mean: float = 0.0,
        data_std: float = 1.0,
        # Age range for normalization (official condition_ranges.json range)
        age_min: float = 5.90,
        age_max: float = 95.46,
        # Optimizer
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Store parameters
        self.volume_shape = volume_shape
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.scheduler_type = scheduler_type
        self.scheduler_eta_min = scheduler_eta_min
        self.n_sample_steps = n_sample_steps
        self.age_min = age_min
        self.age_max = age_max

        # Data statistics (for normalization)
        self.register_buffer('data_mean', torch.tensor(data_mean))
        self.register_buffer('data_std', torch.tensor(data_std))

        # Initialize wavelet transform
        self.wavelet = Wavelet3DTransform(
            wavelet=wavelet,
            level=wavelet_level,
        )

        # Compute wavelet output shape by running a dummy forward pass
        # (simple division by 2**level is inaccurate for most wavelet modes/filters)
        wavelet_channels = self.wavelet.num_channels
        dummy = torch.zeros(1, 1, *volume_shape)
        dummy_coeffs = self.wavelet.forward(dummy)
        self.wavelet_shape = tuple(dummy_coeffs.shape[1:])
        self.wavelet.reset_cache()

        # Initialize U-Net
        self.unet = UNet3D(
            in_channels=wavelet_channels,
            out_channels=wavelet_channels,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
            cond_embed_dim=cond_embed_dim,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            dropout=dropout,
        )

        # Initialize CFM
        if use_rectified_flow:
            self.cfm = RectifiedFlowMatching(self.unet)
        else:
            self.cfm = ConditionalFlowMatching(self.unet, sigma_min=sigma_min)

    def forward(
        self,
        volume: torch.Tensor,
        age: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass: compute CFM loss.

        Args:
            volume: (B, 1, D, H, W) - MRI volume
            age: (B,) or (B, 1) - Subject age in years

        Returns:
            loss: Scalar CFM loss
            x_t: Noisy wavelet coefficients (for visualization)
        """
        # Normalize age to [0, 1]
        if age.dim() == 1:
            age = age.unsqueeze(1)
        age_norm = (age - self.age_min) / (self.age_max - self.age_min)
        age_norm = torch.clamp(age_norm, 0, 1)

        # Transform to wavelet domain
        coeffs = self.wavelet.forward(volume)

        # Compute CFM loss
        loss, x_t = self.cfm.compute_loss(coeffs, age_norm)

        return loss, x_t

    @torch.inference_mode()
    def synthesize(
        self,
        age: torch.Tensor,
        n_steps: Optional[int] = None,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate MRI volume conditioned on age.

        Args:
            age: (B,) or (B, 1) - Target age(s) in years
            n_steps: Number of sampling steps (default: self.n_sample_steps)
            temperature: Noise temperature
            return_trajectory: Return intermediate samples

        Returns:
            volume: (B, 1, D, H, W) - Generated MRI volume
            trajectory: (optional) List of intermediate wavelet coefficients
        """
        n_steps = n_steps or self.n_sample_steps

        # Normalize age
        if age.dim() == 1:
            age = age.unsqueeze(1)
        age_norm = (age - self.age_min) / (self.age_max - self.age_min)
        age_norm = torch.clamp(age_norm, 0, 1)

        # Initialize wavelet cache for reconstruction
        # We need to run a dummy forward pass to set up the reconstruction shapes
        # and determine the actual wavelet coefficient spatial dimensions
        if self.wavelet._original_shape is None:
            dummy_vol = torch.zeros(1, 1, *self.volume_shape, device=self.device)
            dummy_coeffs = self.wavelet.forward(dummy_vol)
            self._actual_wavelet_shape = tuple(dummy_coeffs.shape[1:])
        if not hasattr(self, '_actual_wavelet_shape'):
            dummy_vol = torch.zeros(1, 1, *self.volume_shape, device=self.device)
            dummy_coeffs = self.wavelet.forward(dummy_vol)
            self._actual_wavelet_shape = tuple(dummy_coeffs.shape[1:])

        # Sample in wavelet domain using actual wavelet output shape
        sample_shape = self._actual_wavelet_shape
        if return_trajectory:
            coeffs, trajectory = self.cfm.sample_with_trajectory(
                age_norm,
                sample_shape,
                n_steps=n_steps,
                temperature=temperature,
                device=self.device,
            )
            # Reconstruct volume
            volume = self.wavelet.inverse(coeffs)
            return volume, trajectory
        else:
            coeffs = self.cfm.sample(
                age_norm,
                sample_shape,
                n_steps=n_steps,
                temperature=temperature,
                device=self.device,
            )
            # Reconstruct volume
            volume = self.wavelet.inverse(coeffs)
            return volume

    @torch.inference_mode()
    def interpolate(
        self,
        age1: float,
        age2: float,
        n_interp: int = 5,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate volumes interpolating between two ages.

        Args:
            age1: Starting age
            age2: Ending age
            n_interp: Number of interpolation points
            n_steps: Sampling steps

        Returns:
            volumes: (n_interp, 1, D, H, W) - Interpolated volumes
        """
        ages = torch.linspace(age1, age2, n_interp, device=self.device)
        return self.synthesize(ages, n_steps=n_steps)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning training step."""
        volume = batch['volume']
        age = batch['age']

        loss, _ = self(volume, age)

        # Logging
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning validation step."""
        volume = batch['volume']
        age = batch['age']

        loss, _ = self(volume, age)

        # Logging
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning test step."""
        volume = batch['volume']
        age = batch['age']

        loss, _ = self(volume, age)

        # Logging
        self.log('test/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Log sample images at the end of validation."""
        if not self.trainer.is_global_zero:
            return

        try:
            self._log_sample_images()
        except Exception as e:
            # Don't crash training on visualization errors
            print(f"Warning: Failed to log validation samples: {e}")

    def _log_sample_images(self) -> None:
        """Generate and log sample MRI images with ground-truth comparison.

        Creates matplotlib figures with per-sample age labels for three
        anatomical views (axial, sagittal, coronal) plus fixed-age samples,
        and logs them to TensorBoard and WandB.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Get a sample batch for comparison
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))

        # Move batch to device
        real_volumes = batch['volume'].to(self.device)  # (B, 1, D, H, W)
        real_ages = batch['age'].to(self.device)  # (B, 1)

        batch_size = min(4, real_volumes.shape[0])

        # Get the actual ages from the batch
        sample_ages = real_ages[:batch_size].squeeze(1)  # (batch_size,)
        real_vols_subset = real_volumes[:batch_size]  # (batch_size, 1, D, H, W)

        # Generate samples with the same ages as ground-truth
        with torch.inference_mode():
            generated_vols = self.synthesize(sample_ages, n_steps=min(16, self.n_sample_steps))

        # generated_vols shape: (batch_size, 1, D, H, W)
        if generated_vols.shape[2] == 0:
            return

        D = generated_vols.shape[2]
        H = generated_vols.shape[3]
        W = generated_vols.shape[4]
        ages_list = [sample_ages[i].item() for i in range(batch_size)]

        # Helper: build a 2-row comparison figure (Real top, Generated bottom)
        def make_comparison_fig(view_name, real_slices, gen_slices):
            n = len(real_slices)
            fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
            if n == 1:
                axes = axes.reshape(2, 1)
            fig.suptitle(
                f"{view_name} -- Real (top) vs Generated (bottom) | Epoch {self.current_epoch}",
                fontsize=13,
            )
            for i in range(n):
                r = self._normalize_slice(real_slices[i]).cpu().numpy()
                g = self._normalize_slice(gen_slices[i]).cpu().numpy()
                axes[0, i].imshow(r, cmap="gray", vmin=0, vmax=1)
                axes[0, i].set_title(f"Real | age={ages_list[i]:.1f}", fontsize=10)
                axes[0, i].axis("off")
                axes[1, i].imshow(g, cmap="gray", vmin=0, vmax=1)
                axes[1, i].set_title(f"Gen | age={ages_list[i]:.1f}", fontsize=10)
                axes[1, i].axis("off")
            fig.tight_layout()
            return fig

        # Build comparison figures for three anatomical views
        axial_fig = make_comparison_fig(
            "Axial",
            [real_vols_subset[i, 0, D // 2, :, :] for i in range(batch_size)],
            [generated_vols[i, 0, D // 2, :, :] for i in range(batch_size)],
        )
        sagittal_fig = make_comparison_fig(
            "Sagittal",
            [real_vols_subset[i, 0, :, :, W // 2] for i in range(batch_size)],
            [generated_vols[i, 0, :, :, W // 2] for i in range(batch_size)],
        )
        coronal_fig = make_comparison_fig(
            "Coronal",
            [real_vols_subset[i, 0, :, H // 2, :] for i in range(batch_size)],
            [generated_vols[i, 0, :, H // 2, :] for i in range(batch_size)],
        )

        # Compute comparison metrics
        metrics = self._compute_comparison_metrics(real_vols_subset, generated_vols)

        # Generate samples for fixed ages (for consistency across epochs)
        fixed_ages_values = [25.0, 45.0, 65.0, 85.0]
        fixed_ages = torch.tensor(fixed_ages_values, device=self.device)
        with torch.inference_mode():
            fixed_samples = self.synthesize(fixed_ages, n_steps=min(16, self.n_sample_steps))

        fixed_fig, fixed_axes = plt.subplots(1, 4, figsize=(16, 4))
        fixed_fig.suptitle(
            f"Fixed Ages -- Axial Slices | Epoch {self.current_epoch}", fontsize=13,
        )
        for i in range(4):
            s = self._normalize_slice(fixed_samples[i, 0, D // 2, :, :]).cpu().numpy()
            fixed_axes[i].imshow(s, cmap="gray", vmin=0, vmax=1)
            fixed_axes[i].set_title(f"Age {fixed_ages_values[i]:.0f}", fontsize=10)
            fixed_axes[i].axis("off")
        fixed_fig.tight_layout()

        # Log to all available loggers
        self._log_figures_to_loggers(
            axial_fig=axial_fig,
            sagittal_fig=sagittal_fig,
            coronal_fig=coronal_fig,
            fixed_ages_fig=fixed_fig,
            metrics=metrics,
        )

        # Free memory
        plt.close(axial_fig)
        plt.close(sagittal_fig)
        plt.close(coronal_fig)
        plt.close(fixed_fig)

    def _log_figures_to_loggers(
        self,
        axial_fig,
        sagittal_fig,
        coronal_fig,
        fixed_ages_fig,
        metrics: Dict[str, float],
    ) -> None:
        """Log matplotlib figures and metrics to all available loggers."""
        loggers = self.loggers if hasattr(self, 'loggers') and self.loggers else []
        if not loggers and self.logger is not None:
            loggers = [self.logger]

        step = self.global_step

        for logger in loggers:
            logger_cls = type(logger).__name__

            # --- TensorBoard ---
            if "TensorBoard" in logger_cls:
                exp = logger.experiment
                # close=False because we manage figure lifecycle ourselves
                exp.add_figure(
                    "comparison/axial_real_vs_gen", axial_fig, step, close=False,
                )
                exp.add_figure(
                    "comparison/sagittal_real_vs_gen", sagittal_fig, step, close=False,
                )
                exp.add_figure(
                    "comparison/coronal_real_vs_gen", coronal_fig, step, close=False,
                )
                exp.add_figure(
                    "samples/fixed_ages_[25_45_65_85]", fixed_ages_fig, step, close=False,
                )
                for metric_name, metric_value in metrics.items():
                    exp.add_scalar(
                        f"comparison_metrics/{metric_name}", metric_value, step,
                    )

            # --- WandB ---
            if "Wandb" in logger_cls or "WandB" in logger_cls:
                try:
                    import wandb

                    log_dict = {
                        "comparison/axial_real_vs_gen": wandb.Image(
                            axial_fig,
                            caption="Axial: Real (top) vs Generated (bottom)",
                        ),
                        "comparison/sagittal_real_vs_gen": wandb.Image(
                            sagittal_fig,
                            caption="Sagittal: Real (top) vs Generated (bottom)",
                        ),
                        "comparison/coronal_real_vs_gen": wandb.Image(
                            coronal_fig,
                            caption="Coronal: Real (top) vs Generated (bottom)",
                        ),
                        "samples/fixed_ages": wandb.Image(
                            fixed_ages_fig,
                            caption="Fixed ages: 25, 45, 65, 85 years",
                        ),
                    }
                    for metric_name, metric_value in metrics.items():
                        log_dict[f"comparison_metrics/{metric_name}"] = metric_value

                    logger.experiment.log(log_dict, commit=False)
                except (ImportError, Exception) as e:
                    print(f"Warning: Failed to log to WandB: {e}")

        # Also log metrics via Lightning
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/comparison_{metric_name}",
                metric_value,
                prog_bar=False,
                sync_dist=True,
            )

    def _normalize_slice(self, slice_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a 2D slice to [0, 1] range."""
        min_val = slice_tensor.min()
        max_val = slice_tensor.max()
        if max_val - min_val > 1e-8:
            return (slice_tensor - min_val) / (max_val - min_val)
        return slice_tensor - min_val

    def _compute_comparison_metrics(
        self,
        real_vols: torch.Tensor,
        gen_vols: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute comparison metrics between real and generated volumes.

        Args:
            real_vols: (B, 1, D, H, W) - Real volumes
            gen_vols: (B, 1, D, H, W) - Generated volumes

        Returns:
            Dictionary with metric values
        """
        metrics = {}

        # MSE (Mean Squared Error)
        mse = F.mse_loss(gen_vols, real_vols)
        metrics['mse'] = mse.item()

        # MAE (Mean Absolute Error)
        mae = F.l1_loss(gen_vols, real_vols)
        metrics['mae'] = mae.item()

        # PSNR (Peak Signal-to-Noise Ratio)
        # PSNR = 10 * log10(MAX^2 / MSE)
        # Assuming normalized data with max value around 5 (after z-score)
        max_val = max(real_vols.max().item(), gen_vols.max().item())
        if mse.item() > 0:
            psnr = 10 * math.log10((max_val ** 2) / mse.item())
            metrics['psnr'] = psnr
        else:
            metrics['psnr'] = 100.0  # Perfect match

        # Structural Similarity (simplified version without external library)
        # Compute correlation coefficient as a proxy
        real_flat = real_vols.flatten()
        gen_flat = gen_vols.flatten()

        # Pearson correlation
        real_mean = real_flat.mean()
        gen_mean = gen_flat.mean()

        numerator = ((real_flat - real_mean) * (gen_flat - gen_mean)).sum()
        denominator = torch.sqrt(
            ((real_flat - real_mean) ** 2).sum() *
            ((gen_flat - gen_mean) ** 2).sum()
        )

        if denominator > 1e-8:
            correlation = (numerator / denominator).item()
            metrics['correlation'] = correlation
        else:
            metrics['correlation'] = 0.0

        return metrics

    def configure_optimizers(self) -> Dict:
        """Configure optimizer and learning rate scheduler."""
        # Use Hydra-instantiated optimizer if provided
        if self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.parameters())
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )

        # Learning rate scheduler
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }
        elif self.scheduler_type == "cosine":
            # Cosine annealing (paper: eta_min=1e-7)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 200,
                eta_min=self.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "learning_rate",
                },
            }
        else:
            return {"optimizer": optimizer}

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log gradient norms."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict({f"grad_norm/{k}": v for k, v in norms.items()})

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle checkpoint loading."""
        if 'epoch' in checkpoint:
            self.ckpt_loaded_epoch = checkpoint['epoch']

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FlowLet(\n"
            f"  wavelet_shape={self.wavelet_shape},\n"
            f"  volume_shape={self.volume_shape},\n"
            f"  num_parameters={self.num_parameters:,},\n"
            f")"
        )


class FlowLetSmall(FlowLet):
    """Small FlowLet variant for limited GPU memory."""

    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_dims', [32, 64, 128, 256])
        kwargs.setdefault('wavelet_level', 2)
        kwargs.setdefault('time_embed_dim', 128)
        kwargs.setdefault('cond_embed_dim', 32)
        kwargs.setdefault('num_res_blocks', 1)
        kwargs.setdefault('attention_levels', [])
        super().__init__(**kwargs)


class FlowLetLarge(FlowLet):
    """Large FlowLet variant for high-quality generation."""

    def __init__(self, **kwargs):
        kwargs.setdefault('hidden_dims', [64, 128, 256, 512, 512])
        kwargs.setdefault('wavelet_level', 3)
        kwargs.setdefault('num_res_blocks', 3)
        kwargs.setdefault('time_embed_dim', 512)
        kwargs.setdefault('cond_embed_dim', 128)
        kwargs.setdefault('attention_levels', [2, 3, 4])
        kwargs.setdefault('dropout', 0.1)
        super().__init__(**kwargs)
