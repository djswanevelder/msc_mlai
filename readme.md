# Functional Hypernetworks

Single-pass neural network weight generation from prototype images, with provably better initialisations.

**DJ Swanevelder** — MSc Applied Mathematics, Stellenbosch University (2026)

## Key Results

| Scale | Zero-shot (unseen classes) | Better minimum gap | p-value |
|-------|---------------------------|-------------------|---------|
| 50K (direct) | 90.3% | +0.68pp vs Kaiming | < 10⁻⁶ |
| 109K (SANE) | 81.4% | +0.41pp vs Kaiming | 0.00006 |
| 236K (SANE) | 62.0% | +0.33pp (seen combos) | 0.006 |

## What This Does

Given a few example images per class, the system generates all weights of a neural network classifier in a single forward pass — no training required at inference. The generated weights converge to statistically better local minima than Random, Xavier, or Kaiming initialisation when fine-tuned.

## Repository Structure

```
msc_mlai/
├── src/
│   ├── direct/           # Direct hypernetwork (50K params)
│   │   ├── config.py     # Target MLP + hypernetwork config
│   │   ├── zoo.py        # Model zoo generator (MNIST/EMNIST)
│   │   ├── models.py     # HyperNetwork + differentiable forward pass
│   │   └── train.py      # Training with functional loss
│   │
│   └── prototype/        # Prototype-conditioned hypernetwork (50K-236K)
│       ├── config.py     # EMNIST config (62 classes, seen/unseen split)
│       ├── zoo.py        # EMNIST model zoo generator
│       ├── models.py     # PrototypeEncoder + HyperNetwork
│       ├── train.py      # Training with prototype conditioning
│       ├── sane.py       # SANE tokenisation + Transformer AE
│       ├── sane_func.py  # SANE with functional AE (best 109K pipeline)
│       ├── sane_e2e.py   # End-to-end fine-tuning of SANE + hypernetwork
│       ├── ablations.py  # Cross-attention + task diversity ablations
│       └── scale_v5.py   # 236K scaling experiment
│
├── eval_zoo.py           # Evaluate zoo baseline accuracy
├── eval_meta.py          # Evaluate hypernetwork (zero-shot + fine-tuning)
├── verify_no_leakage.py  # Data leakage verification
├── docs/report/          # LaTeX report + PDF
└── pyproject.toml        # Dependencies (uv)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Generate model zoo (EMNIST, 200 tasks)
uv run python -m src.prototype.zoo

# Train prototype-conditioned hypernetwork
uv run python -m src.prototype.train

# Evaluate on unseen classes
uv run python eval_meta.py --split unseen --finetune 10 --compare-random

# Run ablation study (cross-attention + task diversity)
uv run python -m src.prototype.ablations

# Run SANE pipeline for 109K params
uv run python -m src.prototype.sane_func
```

## Method

1. **Model Zoo**: Train many small MLPs on random 3-class subsets of EMNIST
2. **Prototype Encoding**: Encode example images per class via shared MLP + optional cross-attention
3. **Hypernetwork**: Map prototype embeddings → target network weights (direct or via SANE latent space)
4. **Functional Loss**: Train by executing generated weights on task data and backpropagating CE loss

## Requirements

- Python 3.12+
- PyTorch 2.8+
- Apple Silicon (MPS) or CUDA GPU
- `uv` for dependency management
