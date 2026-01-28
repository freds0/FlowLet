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

from flowlet.models.unet3d import UNet3D, UNet3DSmall, UNet3DLarge
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
        wavelet: str = 'db4',
        wavelet_level: int = 3,
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
        use_rectified_flow: bool = False,
        # Training parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        # Scheduler parameters
        scheduler_type: str = "cosine",  # "cosine" or "constant"
        # Sampling parameters
        n_sample_steps: int = 32,
        # Data statistics
        data_mean: float = 0.0,
        data_std: float = 1.0,
        # Age range for normalization
        age_min: float = 18.0,
        age_max: float = 90.0,
        # Model size variant
        model_size: str = "base",  # "small", "base", or "large"
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

        # Compute wavelet output shape
        wavelet_channels = self.wavelet.num_channels
        factor = 2 ** wavelet_level
        wavelet_spatial = tuple(s // factor for s in volume_shape)
        self.wavelet_shape = (wavelet_channels, *wavelet_spatial)

        # Select U-Net variant
        if model_size == "small":
            UNetClass = UNet3DSmall
        elif model_size == "large":
            UNetClass = UNet3DLarge
        else:
            UNetClass = UNet3D

        # Initialize U-Net
        unet_kwargs = {
            'in_channels': wavelet_channels,
            'out_channels': wavelet_channels,
            'hidden_dims': hidden_dims,
            'time_embed_dim': time_embed_dim,
            'cond_embed_dim': cond_embed_dim,
            'num_res_blocks': num_res_blocks,
            'attention_levels': attention_levels,
            'dropout': dropout,
        }

        # Only pass arguments that aren't overridden by the variant
        if model_size == "base":
            self.unet = UNet3D(**unet_kwargs)
        else:
            # Variants have their own defaults
            self.unet = UNetClass(
                in_channels=wavelet_channels,
                out_channels=wavelet_channels,
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
        if self.wavelet._original_shape is None:
            dummy_vol = torch.zeros(1, 1, *self.volume_shape, device=self.device)
            _ = self.wavelet.forward(dummy_vol)

        # Sample in wavelet domain
        if return_trajectory:
            coeffs, trajectory = self.cfm.sample_with_trajectory(
                age_norm,
                self.wavelet_shape,
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
                self.wavelet_shape,
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
        """Generate and log sample MRI images to tensorboard/wandb."""
        import torchvision.utils as vutils

        # Get a sample batch for comparison
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))

        # Generate samples for 4 different ages
        sample_ages = torch.tensor([25.0, 45.0, 65.0, 85.0], device=self.device)

        with torch.inference_mode():
            samples = self.synthesize(sample_ages, n_steps=min(16, self.n_sample_steps))

        # samples shape: (4, 1, D, H, W)
        if samples.shape[2] == 0:
            return

        D, H, W = samples.shape[2], samples.shape[3], samples.shape[4]

        # Create grid of middle slices (axial view)
        axial_slices = []
        for i in range(4):
            slice_img = samples[i, 0, D // 2, :, :]  # (H, W)
            slice_img = self._normalize_slice(slice_img)
            axial_slices.append(slice_img.unsqueeze(0))  # (1, H, W)

        # Stack and create 2x2 grid
        axial_grid = vutils.make_grid(
            torch.stack(axial_slices, dim=0),  # (4, 1, H, W)
            nrow=2,
            normalize=False,
            padding=2,
            pad_value=1.0
        )

        # Create grid of sagittal slices
        sagittal_slices = []
        for i in range(4):
            slice_img = samples[i, 0, :, H // 2, :]  # (D, W)
            slice_img = self._normalize_slice(slice_img)
            sagittal_slices.append(slice_img.unsqueeze(0))

        sagittal_grid = vutils.make_grid(
            torch.stack(sagittal_slices, dim=0),
            nrow=2,
            normalize=False,
            padding=2,
            pad_value=1.0
        )

        # Create grid of coronal slices
        coronal_slices = []
        for i in range(4):
            slice_img = samples[i, 0, :, :, W // 2]  # (D, H)
            slice_img = self._normalize_slice(slice_img)
            coronal_slices.append(slice_img.unsqueeze(0))

        coronal_grid = vutils.make_grid(
            torch.stack(coronal_slices, dim=0),
            nrow=2,
            normalize=False,
            padding=2,
            pad_value=1.0
        )

        # Get real sample for comparison
        real_axial = None
        if 'volume' in batch and batch['volume'].shape[0] > 0:
            real_vol = batch['volume'][0].to(self.device)  # (1, D, H, W)
            if real_vol.shape[1] > 0:
                real_axial = self._normalize_slice(real_vol[0, D // 2, :, :])

        # Log to all available loggers
        self._log_images_to_loggers(
            axial_grid=axial_grid,
            sagittal_grid=sagittal_grid,
            coronal_grid=coronal_grid,
            axial_slices=axial_slices,
            sample_ages=sample_ages,
            real_axial=real_axial,
        )

    def _log_images_to_loggers(
        self,
        axial_grid: torch.Tensor,
        sagittal_grid: torch.Tensor,
        coronal_grid: torch.Tensor,
        axial_slices: list,
        sample_ages: torch.Tensor,
        real_axial: Optional[torch.Tensor] = None,
    ) -> None:
        """Log images to all available loggers (TensorBoard, WandB, etc.)."""
        # Get list of loggers
        loggers = self.loggers if hasattr(self, 'loggers') and self.loggers else []
        if not loggers and self.logger is not None:
            loggers = [self.logger]

        for logger in loggers:
            # TensorBoard logging
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_image'):
                exp = logger.experiment
                exp.add_image('samples/axial_grid', axial_grid.cpu(), self.current_epoch)
                exp.add_image('samples/sagittal_grid', sagittal_grid.cpu(), self.current_epoch)
                exp.add_image('samples/coronal_grid', coronal_grid.cpu(), self.current_epoch)

                # Log individual samples with age labels
                for i, age in enumerate(sample_ages):
                    exp.add_image(
                        f'samples/age_{int(age.item())}_axial',
                        axial_slices[i].cpu(),
                        self.current_epoch
                    )

                # Log real sample
                if real_axial is not None:
                    exp.add_image(
                        'samples/real_axial',
                        real_axial.unsqueeze(0).cpu(),
                        self.current_epoch
                    )

            # WandB logging
            elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                try:
                    import wandb
                    log_dict = {
                        'samples/axial_grid': wandb.Image(axial_grid.cpu(), caption="Ages: 25, 45, 65, 85"),
                        'samples/sagittal_grid': wandb.Image(sagittal_grid.cpu(), caption="Sagittal view"),
                        'samples/coronal_grid': wandb.Image(coronal_grid.cpu(), caption="Coronal view"),
                    }
                    # Add individual samples
                    for i, age in enumerate(sample_ages):
                        log_dict[f'samples/age_{int(age.item())}'] = wandb.Image(
                            axial_slices[i].cpu(),
                            caption=f"Age {int(age.item())} years"
                        )
                    # Add real sample
                    if real_axial is not None:
                        log_dict['samples/real'] = wandb.Image(real_axial.cpu(), caption="Real sample")

                    logger.experiment.log(log_dict)
                except (ImportError, AttributeError, Exception) as e:
                    print(f"Warning: Failed to log to WandB: {e}")

    def _normalize_slice(self, slice_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a 2D slice to [0, 1] range."""
        min_val = slice_tensor.min()
        max_val = slice_tensor.max()
        if max_val - min_val > 1e-8:
            return (slice_tensor - min_val) / (max_val - min_val)
        return slice_tensor - min_val

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
            # Cosine annealing with warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 500,
                eta_min=1e-6,
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
        kwargs.setdefault('model_size', 'small')
        kwargs.setdefault('hidden_dims', [32, 64, 128, 256])
        kwargs.setdefault('wavelet_level', 2)
        super().__init__(**kwargs)


class FlowLetLarge(FlowLet):
    """Large FlowLet variant for high-quality generation."""

    def __init__(self, **kwargs):
        kwargs.setdefault('model_size', 'large')
        kwargs.setdefault('hidden_dims', [64, 128, 256, 512, 512])
        kwargs.setdefault('wavelet_level', 3)
        kwargs.setdefault('num_res_blocks', 3)
        super().__init__(**kwargs)
