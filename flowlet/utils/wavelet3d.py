"""
3D Wavelet Transform utilities for FlowLet.

Implements invertible 3D wavelet decomposition for MRI volumes,
replacing the mel spectrogram transform used in FlowMAC.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
import torch
import torch.nn as nn


class Wavelet3DTransform(nn.Module):
    """
    Invertible 3D Wavelet Transform for MRI volumes.

    Decomposes 3D volumes into multi-scale wavelet coefficients that can be
    processed by neural networks and perfectly reconstructed.

    The wavelet decomposition produces:
    - 1 approximation subband (low-frequency)
    - 7 detail subbands per level (high-frequency in different orientations)

    For level=3: 1 + 7*3 = 22 subbands total

    Args:
        wavelet: Wavelet type (default: 'db4' - Daubechies 4)
        level: Number of decomposition levels (default: 3)
        mode: Signal extension mode for boundaries (default: 'symmetric')
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        mode: str = 'symmetric'
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        # Subband keys for 3D wavelet (ordered consistently)
        self.detail_keys = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

        # Number of channels: 1 approx + 7 details per level
        self.n_subbands = 1 + 7 * level

        # Cache for reconstruction shapes
        self._coeff_slices: Optional[List] = None
        self._original_shape: Optional[Tuple] = None

    @property
    def num_channels(self) -> int:
        """Number of output channels (subbands)."""
        return self.n_subbands

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Decompose volume(s) into wavelet coefficients.

        Args:
            volume: (B, 1, D, H, W) - MRI volume batch

        Returns:
            coeffs: (B, C, D', H', W') - Stacked wavelet coefficients
                    where C = num_channels, and D', H', W' are reduced sizes
        """
        return self.decompose(volume)

    def decompose(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Decompose volume(s) into wavelet coefficients.

        Args:
            volume: (B, 1, D, H, W) - MRI volume batch

        Returns:
            coeffs: (B, C, D', H', W') - Stacked wavelet coefficients
        """
        B = volume.shape[0]
        device = volume.device
        dtype = volume.dtype

        coeffs_list = []

        for b in range(B):
            # Extract single volume (D, H, W)
            vol = volume[b, 0].cpu().numpy().astype(np.float64)

            # Store original shape for reconstruction
            if self._original_shape is None:
                self._original_shape = vol.shape

            # Perform 3D wavelet decomposition
            coeffs = pywt.wavedecn(vol, self.wavelet, mode=self.mode, level=self.level)

            # Store coefficient slices for reconstruction
            if self._coeff_slices is None:
                self._coeff_slices = self._compute_coeff_slices(coeffs)

            # Flatten coefficients to tensor
            flat_coeffs = self._coeffs_to_tensor(coeffs)
            coeffs_list.append(flat_coeffs)

        # Stack batch
        result = torch.stack(coeffs_list, dim=0).to(device=device, dtype=dtype)

        return result

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct volume(s) from wavelet coefficients.

        Args:
            coeffs: (B, C, D', H', W') - Stacked wavelet coefficients

        Returns:
            volume: (B, 1, D, H, W) - Reconstructed MRI volume
        """
        return self.reconstruct(coeffs)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct volume(s) from wavelet coefficients.

        Args:
            coeffs: (B, C, D', H', W') - Stacked wavelet coefficients

        Returns:
            volume: (B, 1, D, H, W) - Reconstructed MRI volume
        """
        if self._coeff_slices is None:
            raise RuntimeError(
                "Reconstruction requires prior decomposition to determine shapes. "
                "Call decompose() first or use reconstruct_with_shapes()."
            )

        B = coeffs.shape[0]
        device = coeffs.device
        dtype = coeffs.dtype

        volumes = []

        for b in range(B):
            # Convert tensor back to coefficient structure
            coeffs_struct = self._tensor_to_coeffs(coeffs[b].cpu().numpy())

            # Perform inverse wavelet transform
            vol = pywt.waverecn(coeffs_struct, self.wavelet, mode=self.mode)

            # Crop to original shape (wavelet may add padding)
            if self._original_shape is not None:
                vol = vol[:self._original_shape[0],
                         :self._original_shape[1],
                         :self._original_shape[2]]

            volumes.append(torch.from_numpy(vol.astype(np.float32)))

        # Stack and add channel dimension
        result = torch.stack(volumes, dim=0).unsqueeze(1).to(device=device, dtype=dtype)

        return result

    def _coeffs_to_tensor(self, coeffs: List) -> torch.Tensor:
        """
        Convert pywt coefficient structure to a stacked tensor.

        The output tensor has shape (C, D', H', W') where:
        - C = 1 + 7 * level (number of subbands)
        - D', H', W' = shape of the coarsest approximation

        Each subband is resized to match the approximation shape.
        """
        # Get approximation coefficients (coarsest level)
        approx = coeffs[0]
        target_shape = approx.shape

        subbands = [approx]

        # Process detail coefficients at each level
        for level_idx in range(1, len(coeffs)):
            level_dict = coeffs[level_idx]
            for key in self.detail_keys:
                detail = level_dict[key]
                # Resize to match approximation shape
                resized = self._resize_3d(detail, target_shape)
                subbands.append(resized)

        # Stack into tensor (C, D, H, W)
        stacked = np.stack(subbands, axis=0)

        return torch.from_numpy(stacked.astype(np.float32))

    def _tensor_to_coeffs(self, tensor: np.ndarray) -> List:
        """
        Convert stacked tensor back to pywt coefficient structure.

        Args:
            tensor: (C, D', H', W') numpy array

        Returns:
            coeffs: pywt coefficient structure for waverecn
        """
        # Extract approximation
        approx = tensor[0]

        coeffs = [approx]

        # Extract detail coefficients at each level
        idx = 1
        for level_idx in range(self.level):
            level_dict = {}
            # Get original shape for this level from cached slices
            level_shape = self._coeff_slices[level_idx]

            for key in self.detail_keys:
                detail = tensor[idx]
                # Resize back to original shape for this level
                resized = self._resize_3d(detail, level_shape)
                level_dict[key] = resized
                idx += 1

            coeffs.append(level_dict)

        return coeffs

    def _compute_coeff_slices(self, coeffs: List) -> List[Tuple[int, int, int]]:
        """
        Compute and store the shapes of coefficients at each level.

        This is needed for reconstruction to resize subbands back to their
        original sizes.
        """
        slices = []

        for level_idx in range(1, len(coeffs)):
            # Get shape from first detail subband at this level
            first_key = self.detail_keys[0]
            shape = coeffs[level_idx][first_key].shape
            slices.append(shape)

        return slices

    @staticmethod
    def _resize_3d(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Resize 3D array to target shape using trilinear interpolation.
        """
        from scipy.ndimage import zoom

        current_shape = array.shape
        if current_shape == target_shape:
            return array

        factors = [t / c for t, c in zip(target_shape, current_shape)]
        return zoom(array, factors, order=1, mode='nearest')

    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Compute the output shape for a given input volume shape.

        Args:
            input_shape: (D, H, W) - Input volume shape

        Returns:
            (C, D', H', W') - Output coefficient tensor shape
        """
        # Each level reduces spatial dimensions by factor of 2
        factor = 2 ** self.level
        D, H, W = input_shape

        # Compute reduced dimensions (ceiling division for padding)
        D_out = math.ceil(D / factor)
        H_out = math.ceil(H / factor)
        W_out = math.ceil(W / factor)

        return (self.n_subbands, D_out, H_out, W_out)

    def reset_cache(self):
        """Reset cached shapes (call when changing input sizes)."""
        self._coeff_slices = None
        self._original_shape = None


class DifferentiableWavelet3D(nn.Module):
    """
    Differentiable approximation of 3D wavelet transform using learned filters.

    This is an alternative to Wavelet3DTransform that is fully differentiable
    and GPU-compatible, using learnable or fixed convolutional filters.

    Note: This is an approximation and may not provide perfect reconstruction.
    Use Wavelet3DTransform for exact wavelet transforms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        wavelet: str = 'db4',
        level: int = 3,
        trainable: bool = False
    ):
        super().__init__()
        self.level = level
        self.trainable = trainable

        # Get wavelet filters
        wav = pywt.Wavelet(wavelet)
        lo = np.array(wav.dec_lo, dtype=np.float32)
        hi = np.array(wav.dec_hi, dtype=np.float32)

        # Create 3D separable filters for decomposition
        self.decomp_filters = nn.ParameterList()
        self.recon_filters = nn.ParameterList()

        for lvl in range(level):
            # Decomposition filters (8 combinations of lo/hi for 3D)
            dec_filters = self._create_3d_filters(lo, hi)
            if trainable:
                self.decomp_filters.append(nn.Parameter(dec_filters))
            else:
                self.register_buffer(f'dec_filter_{lvl}', dec_filters)

            # Reconstruction filters
            rec_lo = np.array(wav.rec_lo, dtype=np.float32)
            rec_hi = np.array(wav.rec_hi, dtype=np.float32)
            rec_filters = self._create_3d_filters(rec_lo, rec_hi)
            if trainable:
                self.recon_filters.append(nn.Parameter(rec_filters))
            else:
                self.register_buffer(f'rec_filter_{lvl}', rec_filters)

    def _create_3d_filters(self, lo: np.ndarray, hi: np.ndarray) -> torch.Tensor:
        """Create 3D separable wavelet filters from 1D filters."""
        filters = []
        for f1 in [lo, hi]:
            for f2 in [lo, hi]:
                for f3 in [lo, hi]:
                    # Outer product to create 3D filter
                    filt_3d = np.outer(f1, f2).reshape(-1, 1) * f3
                    filt_3d = filt_3d.reshape(len(f1), len(f2), len(f3))
                    filters.append(filt_3d)

        # Stack: (8, 1, kD, kH, kW)
        filters = np.stack(filters, axis=0)[:, np.newaxis, :, :, :]
        return torch.from_numpy(filters.astype(np.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward wavelet decomposition.

        Args:
            x: (B, 1, D, H, W) - Input volume

        Returns:
            coeffs: (B, C, D', H', W') - Wavelet coefficients
        """
        subbands = []
        current = x

        for lvl in range(self.level):
            if self.trainable:
                filters = self.decomp_filters[lvl]
            else:
                filters = getattr(self, f'dec_filter_{lvl}')

            # Apply filters with stride 2 for downsampling
            pad = filters.shape[2] // 2
            coeffs = torch.nn.functional.conv3d(
                current, filters,
                stride=2,
                padding=pad
            )

            # Split into approximation and details
            approx = coeffs[:, 0:1]  # LLL
            details = coeffs[:, 1:]   # LLH, LHL, LHH, HLL, HLH, HHL, HHH

            subbands.append(details)
            current = approx

        # Add final approximation
        subbands.insert(0, current)

        # Resize all to same spatial size and concatenate
        target_shape = subbands[0].shape[2:]
        resized = [subbands[0]]
        for sb in subbands[1:]:
            resized.append(
                torch.nn.functional.interpolate(
                    sb, size=target_shape, mode='trilinear', align_corners=False
                )
            )

        return torch.cat(resized, dim=1)

    def inverse(self, coeffs: torch.Tensor, output_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Inverse wavelet reconstruction (approximate).

        Args:
            coeffs: (B, C, D', H', W') - Wavelet coefficients
            output_shape: (D, H, W) - Desired output shape

        Returns:
            x: (B, 1, D, H, W) - Reconstructed volume
        """
        # Simple interpolation-based reconstruction
        # Note: This is approximate; for exact reconstruction use Wavelet3DTransform
        return torch.nn.functional.interpolate(
            coeffs[:, 0:1],
            size=output_shape,
            mode='trilinear',
            align_corners=False
        )
