"""
V3 MVP Configuration — Direct Hypernetwork
============================================
Meta-model M maps class-identity D → weights W for a small MLP.
Trained with functional loss (cross-entropy of generated weights on D's data).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TargetMLPConfig:
    """Architecture of the small MLP whose weights we generate."""
    input_dim: int = 784  # flattened 28x28 MNIST
    hidden_dims: List[int] = field(default_factory=lambda: [32])
    num_classes: int = 3  # 3-class subset


@dataclass
class ZooConfig:
    """Config for generating the model zoo."""
    dataset: str = "mnist"
    num_classes_per_task: int = 3
    num_total_classes: int = 10
    train_epochs: int = 30  # train each model to convergence
    lr: float = 1e-3
    batch_size: int = 128
    zoo_dir: str = "data/v3/zoo"
    num_train_subsets: int = 100
    num_test_subsets: int = 20  # held-out for evaluation
    device: str = "cpu"  # CPU for small MLPs (faster than MPS transfer overhead)


@dataclass
class HyperNetConfig:
    """Config for the hypernetwork that generates weights."""
    class_embed_dim: int = 64  # embedding dim per digit class
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16  # small batches — each requires forward pass through generated model
    num_epochs: int = 500
    patience: int = 80
    num_classes: int = 10  # total MNIST digit classes (for embedding table)
    num_classes_per_task: int = 3
    functional_loss_weight: float = 1.0  # primary loss
    weight_mse_weight: float = 0.1  # auxiliary: MSE to zoo weights
    num_train_samples: int = 256  # images per class subset for functional loss
    device: str = "mps"


@dataclass
class Config:
    target: TargetMLPConfig = field(default_factory=TargetMLPConfig)
    zoo: ZooConfig = field(default_factory=ZooConfig)
    hypernet: HyperNetConfig = field(default_factory=HyperNetConfig)

    def target_weight_dim(self) -> int:
        """Total number of parameters in the target MLP."""
        dims = [self.target.input_dim] + self.target.hidden_dims + [self.target.num_classes]
        total = 0
        for i in range(len(dims) - 1):
            total += dims[i] * dims[i + 1] + dims[i + 1]  # weight matrix + bias
        return total
