"""
V4 Model Zoo — EMNIST ByClass (47 classes)
============================================
Trains target MLPs on 3-class subsets from EMNIST.
Three splits:
  - train: subsets drawn only from seen classes (0-35)
  - seen_holdout: different subsets from seen classes (novel combos)
  - unseen: subsets containing at least one unseen class (36-46)
"""

import itertools
import json
import random
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

from src.v4.config import Config, TargetMLPConfig


class TargetMLP(nn.Module):
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


def get_emnist_data(split: str = "byclass") -> Tuple[datasets.EMNIST, datasets.EMNIST]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    train = datasets.EMNIST("data/v4/raw", split=split, train=True, download=True, transform=transform)
    test = datasets.EMNIST("data/v4/raw", split=split, train=False, download=True, transform=transform)
    return train, test


def make_class_subset(
    dataset: datasets.EMNIST,
    classes: List[int],
) -> Tuple[Subset, Dict[int, int]]:
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    train_data = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    test_data = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)

    train_subset, label_map = make_class_subset(train_data, classes)
    test_subset, _ = make_class_subset(test_data, classes)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    cfg = TargetMLPConfig(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    model = TargetMLP(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, train_epochs + 1):
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            labels = torch.tensor([label_map[l.item()] for l in labels])
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            labels = torch.tensor([label_map[l.item()] for l in labels])
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {
        "classes": classes,
        "flat_weights": model.get_flat_weights(),
        "test_accuracy": correct / total if total > 0 else 0,
    }


def generate_zoo(cfg: Config) -> str:
    zoo_dir = Path(cfg.zoo.zoo_dir)
    zoo_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading EMNIST...")
    get_emnist_data(cfg.zoo.emnist_split)

    seen = cfg.zoo.seen_classes
    unseen = cfg.zoo.unseen_classes

    # Generate subsets
    rng = random.Random(42)

    # Seen-only subsets (for train + seen_holdout)
    all_seen_subsets = list(itertools.combinations(seen, cfg.zoo.num_classes_per_task))
    rng.shuffle(all_seen_subsets)
    all_seen_subsets = [list(s) for s in all_seen_subsets]

    train_subsets = all_seen_subsets[:cfg.zoo.num_train_subsets]
    seen_holdout = all_seen_subsets[cfg.zoo.num_train_subsets:cfg.zoo.num_train_subsets + cfg.zoo.num_seen_holdout_subsets]

    # Unseen subsets: at least one class from unseen set
    unseen_subsets = []
    unseen_combos = []
    for u in unseen:
        for pair in itertools.combinations(seen, cfg.zoo.num_classes_per_task - 1):
            unseen_combos.append(sorted(list(pair) + [u]))
    rng.shuffle(unseen_combos)
    # Deduplicate
    seen_set = set()
    for combo in unseen_combos:
        key = tuple(combo)
        if key not in seen_set:
            seen_set.add(key)
            unseen_subsets.append(list(combo))
        if len(unseen_subsets) >= cfg.zoo.num_unseen_subsets:
            break

    ref = TargetMLP(cfg.target)
    total_params = ref.total_params()
    total_tasks = len(train_subsets) + len(seen_holdout) + len(unseen_subsets)
    print(f"Target MLP: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Parameters per model: {total_params:,}")
    print(f"Tasks: {len(train_subsets)} train, {len(seen_holdout)} seen_holdout, {len(unseen_subsets)} unseen = {total_tasks} total")

    all_tasks = train_subsets + seen_holdout + unseen_subsets

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
            pool.imap(worker_fn, all_tasks),
            total=len(all_tasks),
            desc="Training zoo models",
        ):
            results.append(result)

    n_train = len(train_subsets)
    n_seen = len(seen_holdout)
    n_unseen = len(unseen_subsets)

    train_results = results[:n_train]
    seen_results = results[n_train:n_train + n_seen]
    unseen_results = results[n_train + n_seen:]

    for name, res in [("Train", train_results), ("Seen holdout", seen_results), ("Unseen", unseen_results)]:
        accs = [r["test_accuracy"] for r in res]
        print(f"  {name}: mean={sum(accs)/len(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")

    def pack(res_list):
        return {
            "weights": torch.stack([r["flat_weights"] for r in res_list]),
            "classes": [r["classes"] for r in res_list],
            "accuracies": [r["test_accuracy"] for r in res_list],
        }

    save_data = {
        "train": pack(train_results),
        "seen_holdout": pack(seen_results),
        "unseen": pack(unseen_results),
        "architecture": {
            "input_dim": cfg.target.input_dim,
            "hidden_dims": cfg.target.hidden_dims,
            "num_classes": cfg.target.num_classes,
            "total_params": total_params,
        },
        "seen_classes": seen,
        "unseen_classes": unseen,
    }

    save_path = zoo_dir / "zoo.pt"
    torch.save(save_data, save_path)
    print(f"\nZoo saved to {save_path}")

    return str(zoo_dir)


if __name__ == "__main__":
    cfg = Config()
    generate_zoo(cfg)
