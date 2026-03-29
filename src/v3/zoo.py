"""
V3 Model Zoo Generator
=======================
Trains one small MLP per 3-class MNIST subset to convergence.
All 120 = C(10,3) subsets are generated. Split into train/test for the hypernetwork.
"""

import itertools
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.v3.config import Config, TargetMLPConfig


class TargetMLP(nn.Module):
    """The small MLP whose weights the hypernetwork will generate."""

    def __init__(self, cfg: TargetMLPConfig):
        super().__init__()
        layers = []
        d = cfg.input_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def get_flat_weights(self) -> torch.Tensor:
        return torch.cat([p.detach().cpu().flatten() for p in self.parameters()])

    def set_flat_weights(self, flat: torch.Tensor) -> None:
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(flat[idx:idx + n].view_as(p))
                idx += n

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_mnist_data() -> Tuple[datasets.MNIST, datasets.MNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("data/v3/raw", train=True, download=True, transform=transform)
    test = datasets.MNIST("data/v3/raw", train=False, download=True, transform=transform)
    return train, test


def make_class_subset(
    dataset: datasets.MNIST,
    classes: List[int],
) -> Tuple[Subset, Dict[int, int]]:
    """Filter to specific classes, remap labels to 0..n-1."""
    class_set = set(classes)
    indices = [i for i, (_, label) in enumerate(dataset) if label in class_set]
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    return Subset(dataset, indices), label_map


def _train_one_subset(
    classes: List[int],
    input_dim: int,
    hidden_dims: List[int],
    num_classes: int,
    train_epochs: int,
    lr: float,
    batch_size: int,
) -> Dict:
    """Train a single model on a class subset. Runs in worker process."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("data/v3/raw", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("data/v3/raw", train=False, download=False, transform=transform)

    train_subset, label_map = make_class_subset(train_data, classes)
    test_subset, _ = make_class_subset(test_data, classes)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    cfg = TargetMLPConfig(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    model = TargetMLP(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train to convergence
    model.train()
    for epoch in range(1, train_epochs + 1):
        for images, labels in train_loader:
            labels = torch.tensor([label_map[l.item()] for l in labels])
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = torch.tensor([label_map[l.item()] for l in labels])
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {
        "classes": classes,
        "flat_weights": model.get_flat_weights(),
        "test_accuracy": correct / total,
        "test_loss": total_loss / total,
    }


def generate_zoo(cfg: Config) -> str:
    """Generate the model zoo: one converged model per 3-class subset."""
    zoo_dir = Path(cfg.zoo.zoo_dir)
    zoo_dir.mkdir(parents=True, exist_ok=True)

    # Download MNIST first (before spawning workers)
    print("Downloading MNIST...")
    get_mnist_data()

    # All C(10,3) = 120 subsets
    all_subsets = [list(c) for c in itertools.combinations(range(cfg.zoo.num_total_classes), cfg.zoo.num_classes_per_task)]
    assert len(all_subsets) == 120, f"Expected 120 subsets, got {len(all_subsets)}"

    # Deterministic random split (seeded for reproducibility)
    import random as _rng
    _rng_state = _rng.getstate()
    _rng.seed(42)
    _rng.shuffle(all_subsets)
    _rng.setstate(_rng_state)

    train_subsets = all_subsets[:cfg.zoo.num_train_subsets]
    test_subsets = all_subsets[cfg.zoo.num_train_subsets:cfg.zoo.num_train_subsets + cfg.zoo.num_test_subsets]

    ref = TargetMLP(cfg.target)
    total_params = ref.total_params()
    print(f"Target MLP: {cfg.target.input_dim} → {cfg.target.hidden_dims} → {cfg.target.num_classes}")
    print(f"Parameters per model: {total_params:,}")
    print(f"Training {len(all_subsets)} models ({len(train_subsets)} train / {len(test_subsets)} test)")

    worker_fn = partial(
        _train_one_subset,
        input_dim=cfg.target.input_dim,
        hidden_dims=cfg.target.hidden_dims,
        num_classes=cfg.target.num_classes,
        train_epochs=cfg.zoo.train_epochs,
        lr=cfg.zoo.lr,
        batch_size=cfg.zoo.batch_size,
    )

    num_workers = min(mp.cpu_count(), 8)
    print(f"Using {num_workers} parallel workers")

    results = []
    with mp.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(worker_fn, all_subsets),
            total=len(all_subsets),
            desc="Training zoo models",
        ):
            results.append(result)

    # Separate train/test
    train_results = results[:cfg.zoo.num_train_subsets]
    test_results = results[cfg.zoo.num_train_subsets:cfg.zoo.num_train_subsets + cfg.zoo.num_test_subsets]

    # Report accuracies
    train_accs = [r["test_accuracy"] for r in train_results]
    test_accs = [r["test_accuracy"] for r in test_results]
    print(f"\nTrain split accuracies: mean={sum(train_accs)/len(train_accs):.4f}, "
          f"min={min(train_accs):.4f}, max={max(train_accs):.4f}")
    print(f"Test split accuracies:  mean={sum(test_accs)/len(test_accs):.4f}, "
          f"min={min(test_accs):.4f}, max={max(test_accs):.4f}")

    # Save
    save_data = {
        "train": {
            "weights": torch.stack([r["flat_weights"] for r in train_results]),
            "classes": [r["classes"] for r in train_results],
            "accuracies": [r["test_accuracy"] for r in train_results],
        },
        "test": {
            "weights": torch.stack([r["flat_weights"] for r in test_results]),
            "classes": [r["classes"] for r in test_results],
            "accuracies": [r["test_accuracy"] for r in test_results],
        },
        "architecture": {
            "input_dim": cfg.target.input_dim,
            "hidden_dims": cfg.target.hidden_dims,
            "num_classes": cfg.target.num_classes,
            "total_params": total_params,
        },
    }

    save_path = zoo_dir / "zoo.pt"
    torch.save(save_data, save_path)
    print(f"\nZoo saved to {save_path}")
    print(f"  Train: {save_data['train']['weights'].shape}")
    print(f"  Test:  {save_data['test']['weights'].shape}")

    return str(zoo_dir)


if __name__ == "__main__":
    cfg = Config()
    generate_zoo(cfg)
