# FlowLet — Alignment Plan with the Official Implementation

Comparative analysis of this working version (`FlowLet/`, Lightning + Hydra, FlowMAC
lineage) against the reference (`FlowLet_Official/`, pure PyTorch, WDM-3D lineage),
plus the concrete changes applied to close the highest-impact gaps.

## Context

The two codebases are independent re-implementations, not a drifted copy. Several
things already match: the rectified-flow objective (`xt=(1-t)·x0+t·x1`, `v=x1-x0`,
MSE), Haar level-1 → 8 channels for the paper config, `lr=3e-6`, cosine→`1e-7`,
grad-clip 1.0, AMP, batch 4, 80/20 CN-only split, 10 ODE sampling steps.

The parts that most affect a *conditional, convergent* generator diverge. The single
most important gap: **age barely conditions the working model** — it enters as one
additive global vector, whereas the official model injects it as FiLM in every
ResBlock and as cross-attention context in every attention block.

## Discrepancies and priority

| # | Discrepancy | Official | Working (before) | Priority |
|---|---|---|---|---|
| A | Age normalization | min–max to [0,1] from data range, in dataset | hardcoded `age_min=18, age_max=90` in model, clamped; inferred range never propagated | Critical |
| C1 | Age conditioning | FiLM scale-shift in **every** ResBlock | single additive vector `emb=t_emb+c_emb` | Critical |
| C2 | Attention conditioning | self-attn **+ cross-attn** to condition | self-attn only | Critical |
| G | Wavelet invertibility | Haar matrix DWT, perfect reconstruction | `pywt.wavedecn` + `zoom` resize → non-invertible for level>1 | High |
| B1 | Weight decay | `1e-5` | `0.01` | Low (negligible at this lr, align for fidelity) |
| B2 | Weight init | default + targeted `zero_module` | blanket `kaiming_normal_(relu)`, only final conv zeroed | Medium |

Not changed here (documented for a later pass): intensity preprocessing uses
`zoom`-resize to 128³ vs the official replicate-pad to 112³; `ConvTranspose3d`
upsampling vs interpolate+conv; base width 64 vs 128; head config 4×32 vs 8×full.

## Changes applied

### Fix A — age normalization (`flowlet.py`, `configs/model/flowlet.yaml`, `configs/experiment/*`)
`FlowLet` already normalizes age consistently in both `forward` and `synthesize`
using `age_min`/`age_max`, keeping years as the external interface (visualization
passes ages like 25/45/65/85). The bug was only in the *values*: hardcoded
`[18, 90]` clips real subjects (OpenBHB / the official range is 5.90–95.46) to 0 or
1, collapsing the conditioning signal at the extremes. Fix: set `age_min=5.90`,
`age_max=95.46` (the official `condition_ranges.json` range) everywhere, and drop the
misleading "auto-inferred from data" comment. Normalization stays centralized in the
model so no caller changes. The datamodule still logs the dataset's own inferred
range so the constants can be verified against the actual training set.

### Fix B — optimizer + init (`configs/model/flowlet.yaml`, `models/unet3d.py`)
`weight_decay: 1e-5`. Drop the blanket `kaiming_normal_(nonlinearity='relu')` init
(wrong nonlinearity for a SiLU/attention network); keep GroupNorm at (1,0) and
zero-init the output conv **and** each ResBlock's second conv so residual blocks
start near identity — the flow-matching-friendly initialization the official model
gets from `zero_module`.

### Fix C — FiLM + cross-attention conditioning (`models/unet3d.py`)
- `ResBlock3D` gains a `cond_dim` and a `cond_film` head producing per-channel
  `(scale, shift)`; applied as `h = h·(1+scale)+shift` after the time embedding.
- `AttentionBlock3D` gains an optional `context_dim`; when a condition context is
  supplied it runs a second, cross-attention pass (spatial queries attend to the
  condition embedding) with its own norm and zero-init output.
- `UNet3D.forward` now threads a dedicated condition embedding (`cond_ctx`) into
  every ResBlock and attention block, in addition to the existing additive path.

### Fix G — wavelet guardrail (`utils/wavelet3d.py`)
`Wavelet3DTransform` warns that `level > 1` reconstruction is lossy (the `zoom`
resize of differently-shaped subbands is not invertible), so `FlowLetLarge`
(`wavelet_level=3`) does not silently produce unreconstructable coefficients.

## Verification
CPU smoke test: build `UNet3D` and `FlowLet`, run a forward/`compute_loss` pass with
dummy tensors, and confirm the velocity output shape matches the input and that
changing the condition changes the output (conditioning is now live).

## Not done / recommended next
1. Port the official matrix-based Haar DWT (GPU, differentiable, exact) and move the
   transform into the datamodule (currently CPU/numpy per sample every step).
2. Replicate-pad to 112³ instead of `zoom`-resize to 128³ to match paper data.
3. Replace `ConvTranspose3d` upsampling with interpolate+conv; base width 64→128.
