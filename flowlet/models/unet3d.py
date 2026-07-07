"""
3D U-Net architecture for FlowLet vector field prediction.

Adapted from the FlowMAC 1D decoder architecture for 3D MRI volumes.
Uses residual blocks with time/condition embeddings for flow matching.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.

    Identical to the FlowMAC implementation - encodes scalar timestep
    into a high-dimensional embedding using sine/cosine functions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "SinusoidalPositionEmbeddings requires dim to be even"

    def forward(self, t: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        """
        Args:
            t: (B,) - Timestep values in [0, 1]
            scale: Scaling factor for timesteps

        Returns:
            emb: (B, dim) - Sinusoidal embeddings
        """
        if t.ndim < 1:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        # Frequency bands
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)

        # Scale and compute embeddings
        emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    """
    MLP for processing timestep embeddings.

    Optionally incorporates condition embeddings (e.g., age).
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        cond_dim: Optional[int] = None,
        act_fn: str = "silu",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(
        self,
        sample: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, in_channels) - Timestep embedding
            condition: (B, cond_dim) - Optional condition embedding

        Returns:
            (B, time_embed_dim) - Processed embedding
        """
        if condition is not None and self.cond_proj is not None:
            sample = sample + self.cond_proj(condition)

        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class Block3D(nn.Module):
    """
    Basic 3D convolution block with GroupNorm and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        # Ensure groups divides out_channels
        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock3D(nn.Module):
    """
    3D Residual block with time/condition embedding injection.

    Adapted from FlowMAC's ResnetBlock1D for 3D volumes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        # Ensure groups divides channels
        groups_1 = min(groups, out_channels)
        while out_channels % groups_1 != 0:
            groups_1 -= 1

        # First convolution block
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups_1, out_channels)

        # Time embedding projection (additive)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

        # Condition (age) FiLM projection -> per-channel (scale, shift).
        # Mirrors the official ResBlock, which modulates every block with the
        # condition embedding instead of adding it once globally.
        if cond_dim is not None:
            self.cond_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * out_channels)
            )
        else:
            self.cond_film = None

        # Second convolution block
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_1, out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - Input feature map
            emb: (B, emb_dim) - Time (+ condition) embedding, added
            cond_emb: (B, cond_dim) - Condition embedding for FiLM modulation

        Returns:
            (B, out_channels, D, H, W) - Output feature map
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding (broadcast to spatial dimensions)
        emb_out = self.emb_proj(emb)[:, :, None, None, None]
        h = h + emb_out

        # FiLM condition modulation: h = h * (1 + scale) + shift
        if self.cond_film is not None and cond_emb is not None:
            scale, shift = self.cond_film(cond_emb)[:, :, None, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)


class AttentionBlock3D(nn.Module):
    """
    Self-attention block for 3D feature maps.

    Uses efficient attention implementation for memory efficiency.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 32,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv3d(channels, self.inner_dim * 3, 1)
        self.to_out = nn.Conv3d(self.inner_dim, channels, 1)

        self.scale = head_dim ** -0.5

        # Optional cross-attention to the condition embedding, mirroring the
        # official SpatialTransformerConditional. The condition is a single token,
        # so spatial queries attend to it as a learned global modulation.
        self.context_dim = context_dim
        if context_dim is not None:
            self.norm_cross = nn.GroupNorm(8, channels)
            self.norm_context = nn.LayerNorm(context_dim)
            self.to_q_cross = nn.Conv3d(channels, self.inner_dim, 1)
            self.to_kv_cross = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
            self.to_out_cross = nn.Conv3d(self.inner_dim, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - Input feature map
            context: (B, context_dim) - Condition embedding for cross-attention

        Returns:
            (B, C, D, H, W) - Output feature map with attention
        """
        B, C, D, H, W = x.shape
        N = D * H * W

        # --- Self-attention ---
        residual = x
        h = self.norm(x)
        qkv = self.to_qkv(h).reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, self.inner_dim, D, H, W)
        x = residual + self.to_out(out)

        # --- Cross-attention to condition ---
        if self.context_dim is not None and context is not None:
            residual2 = x
            hq = self.norm_cross(x)
            q_c = self.to_q_cross(hq).reshape(B, self.num_heads, self.head_dim, N)
            q_c = q_c.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

            ctx = self.norm_context(context)  # (B, context_dim)
            kv = self.to_kv_cross(ctx)        # (B, 2 * inner_dim)
            k_c, v_c = kv.chunk(2, dim=-1)
            k_c = k_c.reshape(B, self.num_heads, 1, self.head_dim)
            v_c = v_c.reshape(B, self.num_heads, 1, self.head_dim)

            attn_c = torch.matmul(q_c, k_c.transpose(-1, -2)) * self.scale  # (B, heads, N, 1)
            attn_c = F.softmax(attn_c, dim=-1)
            out_c = torch.matmul(attn_c, v_c)  # (B, heads, N, head_dim)
            out_c = out_c.permute(0, 1, 3, 2).reshape(B, self.inner_dim, D, H, W)
            x = residual2 + self.to_out_cross(out_c)

        return x


class Downsample3D(nn.Module):
    """3D downsampling with strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling with transposed convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for flow matching velocity field prediction.

    Adapted from FlowMAC's 1D decoder architecture for 3D MRI volumes.
    Takes noisy wavelet coefficients and predicts the velocity field.

    Architecture:
    - Encoder: ResBlocks + Downsample
    - Bottleneck: ResBlocks with optional attention
    - Decoder: ResBlocks + Upsample with skip connections

    Args:
        in_channels: Number of input channels (wavelet subbands)
        out_channels: Number of output channels (same as input for velocity)
        hidden_dims: List of channel dimensions for each encoder/decoder level
        time_embed_dim: Dimension of timestep embedding
        cond_embed_dim: Dimension of condition embedding (e.g., age)
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels to add attention (0-indexed from bottom)
        dropout: Dropout rate
        groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        in_channels: int = 22,
        out_channels: int = 22,
        hidden_dims: List[int] = [64, 128, 256, 512],
        time_embed_dim: int = 256,
        cond_embed_dim: int = 64,
        num_res_blocks: int = 2,
        attention_levels: List[int] = [2, 3],
        dropout: float = 0.0,
        groups: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.num_levels = len(hidden_dims)

        # Time embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)
        total_emb_dim = time_embed_dim + cond_embed_dim
        self.time_mlp = TimestepEmbedding(
            in_channels=time_embed_dim,
            time_embed_dim=total_emb_dim,
            act_fn="silu"
        )

        # Condition embedding (for age) - output same dim as time_mlp
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, total_emb_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv3d(in_channels, hidden_dims[0], 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        prev_ch = hidden_dims[0]
        for i, dim in enumerate(hidden_dims):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(prev_ch, dim, total_emb_dim, groups, dropout, cond_dim=total_emb_dim))
                prev_ch = dim

            # Add attention at specified levels
            if i in attention_levels:
                blocks.append(AttentionBlock3D(dim, context_dim=total_emb_dim))

            self.encoders.append(blocks)

            # Downsample (except at last level)
            if i < len(hidden_dims) - 1:
                self.downsamplers.append(Downsample3D(dim))
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock3D(hidden_dims[-1], hidden_dims[-1], total_emb_dim, groups, dropout, cond_dim=total_emb_dim),
            AttentionBlock3D(hidden_dims[-1], context_dim=total_emb_dim),
            ResBlock3D(hidden_dims[-1], hidden_dims[-1], total_emb_dim, groups, dropout, cond_dim=total_emb_dim),
        ])

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_dims = list(reversed(hidden_dims))
        for i, dim in enumerate(reversed_dims):
            # Upsampler (except at first level)
            if i > 0:
                self.upsamplers.append(Upsample3D(prev_ch))
            else:
                self.upsamplers.append(nn.Identity())

            # Skip connection doubles channels
            skip_ch = reversed_dims[i]
            total_ch = prev_ch + skip_ch if i > 0 else prev_ch

            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = total_ch if j == 0 else dim
                blocks.append(ResBlock3D(in_ch, dim, total_emb_dim, groups, dropout, cond_dim=total_emb_dim))

            # Add attention at specified levels (mirrored)
            level_idx = len(hidden_dims) - 1 - i
            if level_idx in attention_levels:
                blocks.append(AttentionBlock3D(dim, context_dim=total_emb_dim))

            self.decoders.append(blocks)
            prev_ch = dim

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(groups, hidden_dims[0]), hidden_dims[0]),
            nn.SiLU(),
            nn.Conv3d(hidden_dims[0], out_channels, 3, padding=1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights.

        Uses PyTorch default init for conv/linear (the previous blanket
        kaiming_normal with nonlinearity='relu' was mismatched for a SiLU/attention
        network) and zero-inits the residual/output convolutions so ResBlocks and
        attention blocks start near identity -- the flow-matching-friendly
        initialization the official model gets from `zero_module`.
        """
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init the final output conv.
        nn.init.zeros_(self.output_conv[-1].weight)
        if self.output_conv[-1].bias is not None:
            nn.init.zeros_(self.output_conv[-1].bias)

        # Zero-init each ResBlock's second conv and each attention block's output
        # projection(s) so they contribute nothing at init (identity residual).
        for m in self.modules():
            if isinstance(m, ResBlock3D):
                nn.init.zeros_(m.conv2.weight)
                if m.conv2.bias is not None:
                    nn.init.zeros_(m.conv2.bias)
            elif isinstance(m, AttentionBlock3D):
                nn.init.zeros_(m.to_out.weight)
                if m.to_out.bias is not None:
                    nn.init.zeros_(m.to_out.bias)
                if getattr(m, 'to_out_cross', None) is not None:
                    nn.init.zeros_(m.to_out_cross.weight)
                    if m.to_out_cross.bias is not None:
                        nn.init.zeros_(m.to_out_cross.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict velocity field.

        Args:
            x: (B, C, D, H, W) - Noisy wavelet coefficients
            t: (B,) - Timestep in [0, 1]
            cond: (B, 1) - Condition (normalized age)

        Returns:
            v: (B, C, D, H, W) - Predicted velocity field
        """
        # Compute embeddings
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb  # Additive time+condition (existing global path)
        cond_ctx = c_emb     # Dedicated condition signal for FiLM / cross-attention

        # Initial convolution
        h = self.input_conv(x)

        # Encoder path with skip connections
        skips = []
        for encoder, downsampler in zip(self.encoders, self.downsamplers):
            for block in encoder:
                h = self._apply_block(block, h, emb, cond_ctx)
            skips.append(h)
            h = downsampler(h)

        # Remove last skip (bottleneck level)
        skips = skips[:-1]

        # Bottleneck
        for block in self.bottleneck:
            h = self._apply_block(block, h, emb, cond_ctx)

        # Decoder path with skip connections
        for i, (upsampler, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            h = upsampler(h)

            # Concatenate skip connection (except first decoder level)
            if i > 0 and skips:
                skip = skips.pop()
                # Handle potential size mismatch from downsampling
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)

            for block in decoder:
                h = self._apply_block(block, h, emb, cond_ctx)

        # Output
        v = self.output_conv(h)

        return v

    @staticmethod
    def _apply_block(block, h, emb, cond_ctx):
        """Dispatch a block with the embeddings it consumes."""
        if isinstance(block, ResBlock3D):
            return block(h, emb, cond_ctx)
        if isinstance(block, AttentionBlock3D):
            return block(h, context=cond_ctx)
        return block(h)


