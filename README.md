<div align="center">

# FlowLet: Conditional 3D Brain MRI Synthesis Using Wavelet Flow Matching

[![python](https://img.shields.io/badge/-Python_3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

> **Disclaimer:** This is an **unofficial implementation** of the paper *"FlowLet: Conditional 3D Brain MRI Synthesis Using Wavelet Flow Matching"*. This repository is not affiliated with the original authors.

## Overview

FlowLet is a generative model for conditional 3D brain MRI synthesis that combines **Wavelet Transform** with **Flow Matching**. By operating in the wavelet domain, FlowLet achieves efficient high-resolution 3D medical image generation while preserving fine anatomical details.

## Architecture

```
                            FlowLet Pipeline
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │   TRAINING                                                          │
 │   ────────                                                          │
 │                                                                     │
 │   MRI Volume (B,1,D,H,W)                                           │
 │        │                                                            │
 │        ▼                                                            │
 │   ┌─────────────────────┐     ┌──────────┐                         │
 │   │  3D Wavelet Transform│     │  Age (c) │                         │
 │   │  (db4, level=3)     │     └────┬─────┘                         │
 │   │                     │          │                                │
 │   │  1 approx subband   │          ▼                                │
 │   │  + 7 detail subbands │   ┌───────────┐                         │
 │   │    per level         │   │ Cond MLP   │                         │
 │   │  = 22 channels total │   │ age → emb  │                         │
 │   └────────┬────────────┘   └─────┬─────┘                         │
 │            │                      │                                 │
 │            ▼                      │                                 │
 │   Wavelet Coeffs x₁              │                                 │
 │   (B, 22, D', H', W')            │                                 │
 │            │                      │                                 │
 │            ▼                      │                                 │
 │   ┌────────────────────────────────────────────────────────┐       │
 │   │         Conditional Flow Matching (OT-CFM)             │       │
 │   │                                                        │       │
 │   │  x₀ ~ N(0,I)          t ~ U(0,1)                      │       │
 │   │       │                    │                           │       │
 │   │       ▼                    ▼                           │       │
 │   │  x_t = (1-(1-σ)t)·x₀ + t·x₁    target: u_t = x₁-(1-σ)·x₀   │
 │   │       │                                                │       │
 │   │       ▼                                                │       │
 │   │  ┌──────────────────────────────────────────────┐      │       │
 │   │  │              3D U-Net  v(x_t, t, c)          │      │       │
 │   │  │                                              │      │       │
 │   │  │  ┌─────────┐                                 │      │       │
 │   │  │  │ t ──────►│ SinEmbed ──► TimeMLP ──┐       │      │       │
 │   │  │  │ c ──────►│ CondMLP  ──────────────┤ + =emb│      │       │
 │   │  │  └─────────┘                         │       │      │       │
 │   │  │                                      ▼       │      │       │
 │   │  │  ENCODER              DECODER                │      │       │
 │   │  │  ┌──────────┐        ┌──────────┐            │      │       │
 │   │  │  │ ResBlock  │───────│ ResBlock  │            │      │       │
 │   │  │  │ +Attn?    │ skip  │ +Attn?    │            │      │       │
 │   │  │  │ ↓Down     │       │ ↑Up       │            │      │       │
 │   │  │  ├──────────┤        ├──────────┤            │      │       │
 │   │  │  │ ResBlock  │───────│ ResBlock  │            │      │       │
 │   │  │  │ +Attn?    │ skip  │ +Attn?    │            │      │       │
 │   │  │  │ ↓Down     │       │ ↑Up       │            │      │       │
 │   │  │  ├──────────┤        ├──────────┤            │      │       │
 │   │  │  │ ResBlock  │───────│ ResBlock  │            │      │       │
 │   │  │  │ +Attn     │ skip  │ +Attn     │            │      │       │
 │   │  │  │ ↓Down     │       │ ↑Up       │            │      │       │
 │   │  │  └─────┬────┘        └─────▲────┘            │      │       │
 │   │  │        │  ┌────────────┐   │                 │      │       │
 │   │  │        └──│ Bottleneck │───┘                 │      │       │
 │   │  │           │ Res+Attn+Res│                    │      │       │
 │   │  │           └────────────┘                     │      │       │
 │   │  │                                              │      │       │
 │   │  │  Output: predicted velocity v_t              │      │       │
 │   │  └──────────────────────────────────────────────┘      │       │
 │   │                                                        │       │
 │   │  Loss = MSE(v_t, u_t)                                  │       │
 │   └────────────────────────────────────────────────────────┘       │
 │                                                                     │
 ├─────────────────────────────────────────────────────────────────────┤
 │                                                                     │
 │   INFERENCE (Sampling)                                              │
 │   ────────────────────                                              │
 │                                                                     │
 │   x₀ ~ N(0,I)      Age condition (c)                               │
 │      │                    │                                         │
 │      ▼                    ▼                                         │
 │   ┌──────────────────────────────┐                                  │
 │   │  ODE Solver (Euler, N steps) │                                  │
 │   │                              │                                  │
 │   │  for t = 0 → 1:             │                                  │
 │   │    v = UNet(x_t, t, c)      │                                  │
 │   │    x_{t+dt} = x_t + dt · v  │                                  │
 │   └──────────────┬───────────────┘                                  │
 │                  │                                                  │
 │                  ▼                                                  │
 │   Wavelet Coeffs x₁ (generated)                                    │
 │                  │                                                  │
 │                  ▼                                                  │
 │   ┌──────────────────────────────┐                                  │
 │   │  Inverse 3D Wavelet Transform│                                  │
 │   └──────────────┬───────────────┘                                  │
 │                  │                                                  │
 │                  ▼                                                  │
 │   Generated MRI Volume (B,1,D,H,W)                                 │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘

                    3D U-Net Variants (Base)

    Input (22,D',H',W')
        │
        ▼
    InputConv 22→64
        │
   ┌────┴──────────────────────────────────────────────┐
   │ ENCODER                              DECODER      │
   │                                                   │
   │ Level 0: 64ch                   Level 0: 64ch     │
   │ [ResBlock ×2]──────────skip──────[ResBlock ×2]    │
   │     │ ↓ Downsample                  ↑ Upsample    │
   │                                                   │
   │ Level 1: 128ch                  Level 1: 128ch    │
   │ [ResBlock ×2]──────────skip──────[ResBlock ×2]    │
   │     │ ↓ Downsample                  ↑ Upsample    │
   │                                                   │
   │ Level 2: 256ch +Attn            Level 2: 256ch    │
   │ [ResBlock ×2, Attn]───skip───[ResBlock ×2, Attn]  │
   │     │ ↓ Downsample                  ↑ Upsample    │
   │                                                   │
   │ Level 3: 512ch +Attn            Level 3: 512ch    │
   │ [ResBlock ×2, Attn]───skip───[ResBlock ×2, Attn]  │
   │     │                               ↑             │
   │     └──► Bottleneck ────────────────┘              │
   │          [Res, Attn, Res]                          │
   └───────────────────────────────────────────────────┘
        │
        ▼
    OutputConv 64→22
        │
        ▼
    Velocity Field v_t (22,D',H',W')
```

## Key Features

- **Wavelet-based Latent Space:** Leverages wavelet decomposition to efficiently represent 3D brain volumes.
- **Conditional Flow Matching:** Uses CFM for stable and efficient training of generative models.
- **3D Brain MRI Synthesis:** Generates realistic, high-quality 3D brain MRI scans.
- **Conditional Generation:** Supports conditioning on various attributes (e.g., age, pathology).

## Installation

1. Create an environment

```bash
conda create -n flowlet python=3.10 -y
conda activate flowlet
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Install the package in editable mode

```bash
pip install -e .
```

## Model Variants

FlowLet offers three size variants with different trade-offs between generation quality and hardware requirements:

| Parameter | **FlowLetSmall** | **FlowLet (Base)** | **FlowLetLarge** |
|---|---|---|---|
| `hidden_dims` | `[32, 64, 128, 256]` | `[64, 128, 256, 512]` | `[64, 128, 256, 512, 512]` |
| `time_embed_dim` | 128 | 256 | 512 |
| `cond_embed_dim` | 32 | 64 | 128 |
| `num_res_blocks` | 1 | 2 | 3 |
| `attention_levels` | `[]` (none) | `[2, 3]` | `[2, 3, 4]` |
| `wavelet_level` | 2 | 3 | 3 |
| `dropout` | 0.0 | 0.0 | 0.1 |
| Volume shape | 96³ | 128³ | 128³ |
| VRAM required | **8–12 GB** | **16+ GB** | **24+ GB** |

### Choosing the Right Variant

- **FlowLetSmall** — For GPUs with limited memory (8–12 GB). No attention blocks, fewer parameters (~50–70% less than base). Ideal for prototyping and consumer-grade GPUs.
- **FlowLet (Base)** — Recommended default. Balanced quality and cost. Suitable for A100 40GB, RTX 3090/4090.
- **FlowLetLarge** — Maximum quality. 5-level U-Net, 3 residual blocks per level, attention at 3 levels. Requires high-end GPUs (A100 80GB, H100).

## Usage

### Training

Select the model variant via experiment configs:

```bash
# Small (limited memory)
python flowlet/train.py experiment=flowlet_min_memory

# Base (default)
python flowlet/train.py experiment=flowlet_oasis

# Base for FOMO60k
python flowlet/train.py experiment=flowlet_fomo60k
```

Or override directly on the command line:

```bash
# Switch to small
python flowlet/train.py model.model_size=small model.hidden_dims=[32,64,128,256]

# Use the class target directly
python flowlet/train.py model._target_=flowlet.flowlet.FlowLetSmall
python flowlet/train.py model._target_=flowlet.flowlet.FlowLetLarge
```

### In Python

```python
from flowlet.flowlet import FlowLet, FlowLetSmall, FlowLetLarge

model = FlowLetSmall()   # 8-12 GB GPU
model = FlowLet()        # 16+ GB GPU (default)
model = FlowLetLarge()   # 24+ GB GPU
```

### Available Experiment Configs

| Config file | Variant | Use case |
|---|---|---|
| `flowlet_synthetic.yaml` | small (64³) | Quick test/debugging |
| `flowlet_min_memory.yaml` | small (96³) | 8–12 GB GPUs |
| `flowlet_oasis.yaml` | base (128³) | Full OASIS training |
| `flowlet_fomo60k.yaml` | base (128³) | Full FOMO60k training |
| `flowlet_fomo60k_min_memory.yaml` | small (96³) | FOMO60k on limited GPU |

## Acknowledgements

This implementation is based on the paper:

> **FlowLet: Conditional 3D Brain MRI Synthesis Using Wavelet Flow Matching**

We also acknowledge the use of:

* [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) for the flow matching architecture.
* [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) for the project structure.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{flowlet2024,
  title={FlowLet: Conditional 3D Brain MRI Synthesis Using Wavelet Flow Matching},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is for research purposes only. Please refer to the original paper for terms of use.
