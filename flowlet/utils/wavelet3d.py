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


