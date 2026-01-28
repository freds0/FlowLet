"""
Conditional Flow Matching (CFM) for 3D MRI volumes.

Adapted from FlowMAC's flow_matching.py for 3D wavelet domain generation.
Implements Optimal Transport CFM for efficient training and sampling.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching module for 3D volume generation.

    Implements the OT-CFM (Optimal Transport Conditional Flow Matching) formulation:
    - Training: Learn velocity field v(x_t, t, c) that transforms noise to data
    - Inference: Integrate ODE from noise to generate samples

    The flow interpolation is:
        x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1

    where x_0 ~ N(0, I) and x_1 is the target data.

    Args:
        vector_field: Neural network that predicts velocity v(x_t, t, c)
        sigma_min: Minimum noise level (default: 1e-4)
    """

    def __init__(
        self,
        vector_field: nn.Module,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.vector_field = vector_field
        self.sigma_min = sigma_min

    def forward(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFM training loss.

        Args:
            x1: (B, C, D, H, W) - Target data (wavelet coefficients)
            cond: (B, 1) - Condition (normalized age)

        Returns:
            loss: Scalar CFM loss
            x_t: (B, C, D, H, W) - Noisy samples (for visualization)
        """
        return self.compute_loss(x1, cond)

    def compute_loss(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFM loss for training.

        Uses the optimal transport formulation where the target velocity is:
            u_t = x_1 - (1 - sigma_min) * x_0

        And the interpolated sample is:
            x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1

        Args:
            x1: (B, C, D, H, W) - Target data
            cond: (B, 1) - Condition

        Returns:
            loss: Scalar MSE loss
            x_t: Noisy samples for visualization
        """
        B = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, device=device, dtype=dtype)

        # Sample noise x_0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # Interpolate: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        t_expanded = t[:, None, None, None, None]
        x_t = (1 - (1 - self.sigma_min) * t_expanded) * x0 + t_expanded * x1

        # Target velocity: u_t = x_1 - (1 - sigma_min) * x_0
        u_t = x1 - (1 - self.sigma_min) * x0

        # Predict velocity
        v_t = self.vector_field(x_t, t, cond)

        # MSE loss
        loss = F.mse_loss(v_t, u_t)

        return loss, x_t

    @torch.inference_mode()
    def sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        n_steps: int = 32,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples via ODE integration (Euler method).

        Args:
            cond: (B, 1) - Condition (normalized age)
            shape: (C, D, H, W) - Shape of output (without batch)
            n_steps: Number of Euler steps
            temperature: Noise temperature (scales initial noise)
            device: Device for computation

        Returns:
            x1: (B, C, D, H, W) - Generated samples
        """
        if device is None:
            device = cond.device

        B = cond.shape[0]

        # Start from noise
        x = torch.randn(B, *shape, device=device, dtype=cond.dtype) * temperature

        # Time steps: 0 -> 1
        t_span = torch.linspace(0, 1, n_steps + 1, device=device)

        # Euler integration
        x = self.solve_euler(x, t_span, cond)

        return x

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve ODE using Euler method.

        Args:
            x: (B, C, D, H, W) - Initial state (noise)
            t_span: (n_steps + 1,) - Time points from 0 to 1
            cond: (B, 1) - Condition

        Returns:
            x: (B, C, D, H, W) - Final state (generated sample)
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Predict velocity at current time
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
            v = self.vector_field(x, t_batch, cond)

            # Euler step
            x = x + dt * v

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x

    @torch.inference_mode()
    def sample_with_trajectory(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        n_steps: int = 32,
        save_every: int = 4,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Generate samples and return intermediate trajectory.

        Useful for visualization and debugging.

        Args:
            cond: (B, 1) - Condition
            shape: (C, D, H, W) - Output shape (without batch)
            n_steps: Number of Euler steps
            save_every: Save trajectory every N steps
            temperature: Noise temperature
            device: Device for computation

        Returns:
            x1: Final samples
            trajectory: List of intermediate samples
        """
        if device is None:
            device = cond.device

        B = cond.shape[0]

        # Start from noise
        x = torch.randn(B, *shape, device=device, dtype=cond.dtype) * temperature

        # Time steps
        t_span = torch.linspace(0, 1, n_steps + 1, device=device)

        trajectory = [x.clone()]
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Predict velocity
            t_batch = torch.full((B,), t.item(), device=device, dtype=x.dtype)
            v = self.vector_field(x, t_batch, cond)

            # Euler step
            x = x + dt * v

            # Save trajectory
            if step % save_every == 0:
                trajectory.append(x.clone())

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x, trajectory


class ConditionalFlowMatchingWithMu(nn.Module):
    """
    CFM variant that conditions on a mean signal (mu).

    This is closer to the original FlowMAC formulation where the decoder
    receives both the noisy sample and an encoder output (mu) as input.
    The velocity field learns to refine mu into a high-quality sample.

    For FlowLet, this could be used if we want to condition on a
    coarse MRI reconstruction or atlas template.

    Args:
        vector_field: Neural network that takes concatenated [x_t, mu]
        sigma_min: Minimum noise level
    """

    def __init__(
        self,
        vector_field: nn.Module,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.vector_field = vector_field
        self.sigma_min = sigma_min

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CFM loss with mean conditioning.

        Args:
            x1: (B, C, D, H, W) - Target data
            mu: (B, C, D, H, W) - Mean/template to refine
            cond: (B, 1) - Condition

        Returns:
            loss: Scalar loss
            x_t: Noisy samples
        """
        B = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        # Sample random timestep
        t = torch.rand(B, device=device, dtype=dtype)

        # Sample noise
        z = torch.randn_like(x1)

        # Interpolate
        t_expanded = t[:, None, None, None, None]
        x_t = (1 - (1 - self.sigma_min) * t_expanded) * z + t_expanded * x1

        # Target velocity
        u_t = x1 - (1 - self.sigma_min) * z

        # Concatenate x_t and mu for conditioning
        x_mu = torch.cat([x_t, mu], dim=1)

        # Predict velocity
        v_t = self.vector_field(x_mu, t, cond)

        # MSE loss
        loss = F.mse_loss(v_t, u_t)

        return loss, x_t

    @torch.inference_mode()
    def sample(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        n_steps: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples conditioned on mean.

        Args:
            mu: (B, C, D, H, W) - Mean/template
            cond: (B, 1) - Condition
            n_steps: Number of steps

        Returns:
            x1: Generated samples
        """
        B = mu.shape[0]
        device = mu.device

        # Start from noise
        x = torch.randn_like(mu) * temperature

        # Time steps
        t_span = torch.linspace(0, 1, n_steps + 1, device=device)
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Concatenate with mu
            x_mu = torch.cat([x, mu], dim=1)

            # Predict velocity
            t_batch = torch.full((B,), t.item(), device=device, dtype=x.dtype)
            v = self.vector_field(x_mu, t_batch, cond)

            # Euler step
            x = x + dt * v

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x


class RectifiedFlowMatching(nn.Module):
    """
    Rectified Flow Matching variant.

    Uses straight-line interpolation without sigma_min:
        x_t = (1 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

    This can lead to straighter flow paths and potentially faster sampling.

    Args:
        vector_field: Neural network for velocity prediction
    """

    def __init__(self, vector_field: nn.Module):
        super().__init__()
        self.vector_field = vector_field

    def compute_loss(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss with straight-line interpolation."""
        B = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        # Sample timestep
        t = torch.rand(B, device=device, dtype=dtype)

        # Sample noise
        x0 = torch.randn_like(x1)

        # Straight-line interpolation
        t_expanded = t[:, None, None, None, None]
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        # Target velocity (straight line)
        u_t = x1 - x0

        # Predict velocity
        v_t = self.vector_field(x_t, t, cond)

        # MSE loss
        loss = F.mse_loss(v_t, u_t)

        return loss, x_t

    @torch.inference_mode()
    def sample(
        self,
        cond: torch.Tensor,
        shape: Tuple[int, ...],
        n_steps: int = 32,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples using rectified flow."""
        if device is None:
            device = cond.device

        B = cond.shape[0]

        # Start from noise
        x = torch.randn(B, *shape, device=device, dtype=cond.dtype) * temperature

        # Time steps
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            v = self.vector_field(x, t, cond)
            x = x + dt * v

        return x
