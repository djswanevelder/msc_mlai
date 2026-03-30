"""
V4 Configuration — Prototype-Conditioned Hypernetwork
======================================================
Key change from V3: conditioning on prototype images (not class IDs).
Uses EMNIST ByClass (47 classes). Seen classes 0-35, unseen 36-46.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TargetMLPConfig:
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    num_classes: int = 3


@dataclass
class ZooConfig:
    dataset: str = "emnist"
    emnist_split: str = "byclass"
    num_classes_per_task: int = 3
    seen_classes: List[int] = field(default_factory=lambda: list(range(10, 62)))  # letters only (A-Z, a-z)
    unseen_classes: List[int] = field(default_factory=lambda: list(range(10)))  # digits 0-9 (visually distinct from letters)
    num_train_subsets: int = 200
    num_seen_holdout_subsets: int = 50
    num_unseen_subsets: int = 50
    train_epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 128
    zoo_dir: str = "data/digits_unseen/zoo"
    device: str = "cpu"


@dataclass
class HyperNetConfig:
    prototypes_per_class: int = 20
    prototype_encoder_hidden: int = 256
    prototype_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512])
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 300
    patience: int = 60
    num_classes_per_task: int = 3
    functional_loss_weight: float = 1.0
    weight_mse_weight: float = 0.1
    num_train_samples: int = 200
    finetune_lr: float = 1e-2
    finetune_steps: int = 10
    device: str = "mps"


@dataclass
class Config:
    target: TargetMLPConfig = field(default_factory=TargetMLPConfig)
    zoo: ZooConfig = field(default_factory=ZooConfig)
    hypernet: HyperNetConfig = field(default_factory=HyperNetConfig)

    def target_weight_dim(self) -> int:
        dims = [self.target.input_dim] + self.target.hidden_dims + [self.target.num_classes]
        total = 0
        for i in range(len(dims) - 1):
            total += dims[i] * dims[i + 1] + dims[i + 1]
        return total
