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

## Usage

*(Documentation coming soon)*

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
