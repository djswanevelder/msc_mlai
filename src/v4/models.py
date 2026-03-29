"""
V4 Models — Prototype-Conditioned Hypernetwork
================================================
Key difference from V3: the hypernetwork is conditioned on IMAGES (prototypes),
not class IDs. This enables generalisation to truly unseen classes.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v4.config import TargetMLPConfig


class PrototypeEncoder(nn.Module):
    """
    Encodes a set of example images (prototypes) for one class into a
    fixed-size vector. Uses a shared image encoder + mean pooling.

    For a 3-class task, we get 3 prototype vectors, one per class.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (K, input_dim) — K prototype images for one class
        Returns:
            (output_dim,) — mean-pooled prototype embedding
        """
        embeddings = self.encoder(images)  # (K, output_dim)
        return embeddings.mean(dim=0)  # (output_dim,)


class HyperNetwork(nn.Module):
    """
    Prototype-conditioned hypernetwork.

    Takes prototype images for each of the 3 classes in a task,
    encodes them, concatenates, and generates target MLP weights.
    """

    def __init__(
        self,
        target_weight_dim: int,
        num_classes_per_task: int = 3,
        input_dim: int = 784,
        prototype_encoder_hidden: int = 256,
        prototype_dim: int = 128,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.target_weight_dim = target_weight_dim
        self.num_classes_per_task = num_classes_per_task
        self.prototype_dim = prototype_dim

        self.prototype_encoder = PrototypeEncoder(
            input_dim=input_dim,
            hidden_dim=prototype_encoder_hidden,
            output_dim=prototype_dim,
        )

        # MLP: concat prototype embeddings -> target weights
        concat_dim = prototype_dim * num_classes_per_task
        layers = []
        d = concat_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.weight_generator = nn.Sequential(*layers)

    def encode_task(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a task from its prototype images.

        Args:
            prototypes: list of 3 tensors, each (K, input_dim) for one class
        Returns:
            (concat_dim,) task embedding
        """
        class_embeddings = []
        for proto_images in prototypes:
            class_embeddings.append(self.prototype_encoder(proto_images))
        return torch.cat(class_embeddings, dim=0)  # (num_classes * prototype_dim,)

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        """
        Generate target weights from prototype images.

        Args:
            prototypes: list of 3 tensors, each (K, input_dim)
        Returns:
            (target_weight_dim,) flat weight vector
        """
        task_emb = self.encode_task(prototypes)
        return self.weight_generator(task_emb)

    def forward_batch(self, batch_prototypes: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Generate weights for a batch of tasks.

        Args:
            batch_prototypes: list of B tasks, each a list of 3 tensors (K, input_dim)
        Returns:
            (B, target_weight_dim)
        """
        task_embeddings = []
        for prototypes in batch_prototypes:
            task_embeddings.append(self.encode_task(prototypes))
        task_emb_batch = torch.stack(task_embeddings)  # (B, concat_dim)
        return self.weight_generator(task_emb_batch)  # (B, target_weight_dim)


def differentiable_forward(
    flat_weights: torch.Tensor,
    inputs: torch.Tensor,
    target_cfg: TargetMLPConfig,
) -> torch.Tensor:
    """Differentiable forward pass through target MLP using flat weights."""
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
    w_size = d * target_cfg.num_classes
    b_size = target_cfg.num_classes
    W = flat_weights[idx:idx + w_size].view(target_cfg.num_classes, d)
    b = flat_weights[idx + w_size:idx + w_size + b_size]
    x = F.linear(x, W, b)
    return x
