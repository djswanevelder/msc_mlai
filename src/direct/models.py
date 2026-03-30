"""
V3 Models — Direct Hypernetwork
=================================
HyperNetwork: maps class-identity encoding → flat weights for target MLP.
No autoencoder, no contrastive loss. Direct weight generation.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.direct.config import Config, TargetMLPConfig


class HyperNetwork(nn.Module):
    """
    Maps a class-identity encoding (which 3 MNIST digits) to all weights
    of a target MLP in a single forward pass.

    Architecture:
        class indices → learned embeddings → concat → MLP → flat weights
    """

    def __init__(
        self,
        target_weight_dim: int,
        num_classes: int = 10,
        class_embed_dim: int = 64,
        num_classes_per_task: int = 3,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.target_weight_dim = target_weight_dim
        self.num_classes_per_task = num_classes_per_task

        # Learned embedding per digit class
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        # MLP: concat embeddings → hidden → target weights
        input_dim = class_embed_dim * num_classes_per_task
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_indices: (batch, num_classes_per_task) integer tensor of digit classes
        Returns:
            (batch, target_weight_dim) flat weight vectors for the target MLP
        """
        emb = self.class_embedding(class_indices)  # (B, 3, class_embed_dim)
        emb = emb.view(emb.size(0), -1)  # (B, 3 * class_embed_dim)
        return self.net(emb)


def differentiable_forward(
    flat_weights: torch.Tensor,
    inputs: torch.Tensor,
    target_cfg: TargetMLPConfig,
) -> torch.Tensor:
    """
    Run a forward pass through the target MLP using flat weights directly.
    Fully differentiable — gradients flow back to flat_weights.

    Args:
        flat_weights: (weight_dim,) flat parameter vector
        inputs: (N, input_dim) input images (flattened)
        target_cfg: target MLP architecture config
    Returns:
        (N, num_classes) logits
    """
    x = inputs
    idx = 0
    d = target_cfg.input_dim
    for h in target_cfg.hidden_dims:
        w_size = d * h
        b_size = h
        W = flat_weights[idx:idx + w_size].view(h, d)
        b = flat_weights[idx + w_size:idx + w_size + b_size]
        x = F.relu(F.linear(x, W, b))
        idx += w_size + b_size
        d = h
    # Final layer
    w_size = d * target_cfg.num_classes
    b_size = target_cfg.num_classes
    W = flat_weights[idx:idx + w_size].view(target_cfg.num_classes, d)
    b = flat_weights[idx + w_size:idx + w_size + b_size]
    x = F.linear(x, W, b)
    return x


def batched_differentiable_forward(
    all_weights: torch.Tensor,
    inputs: torch.Tensor,
    target_cfg: TargetMLPConfig,
) -> torch.Tensor:
    """
    Batched forward pass: run B models on the same inputs simultaneously.
    Fully differentiable.

    Args:
        all_weights: (B, weight_dim) flat parameter vectors
        inputs: (N, input_dim) shared input images
        target_cfg: target MLP architecture config
    Returns:
        (B, N, num_classes) logits
    """
    B = all_weights.size(0)
    x = inputs.unsqueeze(0).expand(B, -1, -1)  # (B, N, input_dim)

    idx = 0
    d = target_cfg.input_dim
    for h in target_cfg.hidden_dims:
        w_size = d * h
        b_size = h
        W = all_weights[:, idx:idx + w_size].view(B, h, d)  # (B, h, d)
        b = all_weights[:, idx + w_size:idx + w_size + b_size]  # (B, h)
        x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)  # (B, N, h)
        x = F.relu(x)
        idx += w_size + b_size
        d = h
    # Final layer
    w_size = d * target_cfg.num_classes
    b_size = target_cfg.num_classes
    W = all_weights[:, idx:idx + w_size].view(B, target_cfg.num_classes, d)
    b = all_weights[:, idx + w_size:idx + w_size + b_size]
    x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)  # (B, N, num_classes)
    return x
