"""
CNN Target Experiment
======================
Tests whether the hypernetwork can generate weights for a CNN,
not just an MLP. CNNs have fundamentally different weight structure
(2D conv filters with spatial locality).

Target CNN: Conv(1→16, 5x5) → ReLU → MaxPool
          → Conv(16→32, 5x5) → ReLU → MaxPool
          → FC(512→64) → ReLU → FC(64→3)
Total: ~46K params (similar to 50K MLP)
"""

import itertools
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from torchvision import datasets, transforms
from tqdm import tqdm

from src.prototype.config import Config
from src.prototype.models import PrototypeEncoder
from src.prototype.train import build_class_image_index, sample_prototypes, sample_task_data


class TargetCNN(nn.Module):
    """Small CNN for 28x28 grayscale classification."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)    # 28→24
        self.conv2 = nn.Conv2d(16, 32, 5)   # 12→8
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)              # 24→12
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)              # 8→4
        x = x.view(x.size(0), -1)           # 32*4*4 = 512
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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


def differentiable_cnn_forward(flat_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiable forward pass through the CNN using flat weights."""
    if x.dim() == 2:
        x = x.view(-1, 1, 28, 28)
    idx = 0

    # Conv1: (16, 1, 5, 5) + (16,) = 400 + 16
    w1 = flat_weights[idx:idx+400].view(16, 1, 5, 5); idx += 400
    b1 = flat_weights[idx:idx+16]; idx += 16
    x = F.relu(F.conv2d(x, w1, b1))
    x = F.max_pool2d(x, 2)

    # Conv2: (32, 16, 5, 5) + (32,) = 12800 + 32
    w2 = flat_weights[idx:idx+12800].view(32, 16, 5, 5); idx += 12800
    b2 = flat_weights[idx:idx+32]; idx += 32
    x = F.relu(F.conv2d(x, w2, b2))
    x = F.max_pool2d(x, 2)

    x = x.view(x.size(0), -1)  # flatten

    # FC1: (64, 512) + (64,) = 32768 + 64
    w3 = flat_weights[idx:idx+32768].view(64, 512); idx += 32768
    b3 = flat_weights[idx:idx+64]; idx += 64
    x = F.relu(F.linear(x, w3, b3))

    # FC2: (3, 64) + (3,) = 192 + 3
    w4 = flat_weights[idx:idx+192].view(3, 64); idx += 192
    b4 = flat_weights[idx:idx+3]; idx += 3
    x = F.linear(x, w4, b4)

    return x


class CNNHyperNetwork(nn.Module):
    """Prototype-conditioned hypernetwork for CNN weight generation."""

    def __init__(self, target_weight_dim: int, prototype_dim: int = 128,
                 num_classes_per_task: int = 3, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        self.prototype_encoder = PrototypeEncoder(
            input_dim=784, hidden_dim=256, output_dim=prototype_dim
        )
        concat_dim = prototype_dim * num_classes_per_task
        layers = []
        d = concat_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.weight_generator = nn.Sequential(*layers)

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)
        return self.weight_generator(task_emb)


def make_class_subset(dataset, classes):
    class_set = set(classes)
    indices = [i for i, (_, label) in enumerate(dataset) if label in class_set]
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    return torch.utils.data.Subset(dataset, indices), label_map


def _train_one_cnn(classes, train_epochs=30, lr=1e-3, batch_size=128):
    """Train one CNN on a class subset."""
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))
    ])
    train_data = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    test_data = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)

    train_sub, label_map = make_class_subset(train_data, classes)
    test_sub, _ = make_class_subset(test_data, classes)
    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_sub, batch_size=batch_size, shuffle=False)

    model = TargetCNN(num_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(train_epochs):
        for images, labels in train_loader:
            labels = torch.tensor([label_map[l.item()] for l in labels])
            opt.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = torch.tensor([label_map[l.item()] for l in labels])
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {"classes": classes, "flat_weights": model.get_flat_weights(), "test_accuracy": correct / total}


def run():
    cfg = Config()
    device = cfg.hypernet.device

    ref = TargetCNN()
    weight_dim = ref.total_params()
    print(f"Target CNN params: {weight_dim:,}")
    print(f"Seen: letters (10-61). Unseen: digits (0-9).")

    seen = cfg.zoo.seen_classes
    unseen = cfg.zoo.unseen_classes

    # === STAGE 1: Generate CNN zoo ===
    zoo_path = Path("data/cnn_zoo/zoo.pt")
    if not zoo_path.exists():
        print("\n=== Generating CNN zoo ===")
        zoo_path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(42)
        all_seen = [list(s) for s in itertools.combinations(seen, 3)]
        rng.shuffle(all_seen)
        train_subsets = all_seen[:200]

        # Unseen subsets (digits)
        unseen_combos = []
        for combo in itertools.combinations(unseen, 3):
            unseen_combos.append(list(combo))
        rng.shuffle(unseen_combos)
        unseen_subsets = unseen_combos[:50]

        all_tasks = train_subsets + unseen_subsets
        print(f"Tasks: {len(train_subsets)} train, {len(unseen_subsets)} unseen = {len(all_tasks)} total")

        num_workers = min(mp.cpu_count(), 8)
        results = []
        with mp.Pool(num_workers) as pool:
            for r in tqdm(pool.imap(_train_one_cnn, all_tasks), total=len(all_tasks), desc="CNN Zoo"):
                results.append(r)

        train_results = results[:200]
        unseen_results = results[200:]

        for name, rl in [("Train", train_results), ("Unseen", unseen_results)]:
            accs = [r["test_accuracy"] for r in rl]
            print(f"  {name}: mean={sum(accs)/len(accs):.4f}, min={min(accs):.4f}")

        def pack(rl):
            return {
                "weights": torch.stack([r["flat_weights"] for r in rl]),
                "classes": [r["classes"] for r in rl],
                "accuracies": [r["test_accuracy"] for r in rl],
            }

        torch.save({
            "train": pack(train_results),
            "unseen": pack(unseen_results),
            "weight_dim": weight_dim,
        }, zoo_path)
        print(f"Saved to {zoo_path}")

    # === STAGE 2: Train CNN hypernetwork ===
    ckpt_path = Path("data/cnn_zoo/hypernet.pt")
    print("\n=== Training CNN HyperNetwork ===")

    zoo = torch.load(str(zoo_path), map_location=device, weights_only=False)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    w_mean = train_weights.mean(0, keepdim=True)
    w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_weights - w_mean) / w_std

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    hypernet = CNNHyperNetwork(target_weight_dim=weight_dim).to(device)
    print(f"CNN HyperNet params: {sum(p.numel() for p in hypernet.parameters()):,}")

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    n_train = len(train_classes)
    best_val = 0; best_state = None; no_improve = 0

    for epoch in range(1, 301):
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0; el = 0.0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
            optimizer.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                gen_w_norm = hypernet(protos)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                logits = differentiable_cnn_forward(gen_w, imgs)
                func = F.cross_entropy(logits, labs)
                mse = F.mse_loss(gen_w_norm, train_norm[idx])
                bl = bl + func + 0.1 * mse
                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            optimizer.step()
            el += bl.item(); nb += 1

        scheduler.step()
        ta = ec / et if et > 0 else 0

        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(n_train, 30)):
                classes = train_classes[idx]
                protos = sample_prototypes(ci_test, classes, 20, device)
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_cnn_forward(gen_w, imgs).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    hypernet.load_state_dict(best_state)
    hypernet.to(device).eval()
    print(f"  Best val: {best_val:.4f}")
    torch.save({"state": best_state, "w_mean": w_mean.cpu(), "w_std": w_std.cpu()}, ckpt_path)

    # === STAGE 3: Evaluate on unseen digits ===
    print(f"\n{'='*60}")
    print("CNN HyperNet: Digits-Unseen Evaluation (3/3 unseen)")
    print(f"{'='*60}")

    all_tasks = [list(c) for c in itertools.combinations(unseen, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_cnn_init():
        m = TargetCNN()
        return m.get_flat_weights().to(device)

    zs, ft_gen, ft_kai = [], [], []

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_cnn_forward(gen_w, imgs).argmax(1) == labs).float().mean().item())

        for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_cnn_init(), ft_kai)]:
            w = init_w.detach().requires_grad_(True)
            opt = torch.optim.SGD([w], lr=0.01)
            for _ in range(3000):
                i2, l2 = sample_task_data(ci_train, classes, 50, device)
                opt.zero_grad()
                F.cross_entropy(differentiable_cnn_forward(w, i2), l2).backward()
                opt.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                store.append((differentiable_cnn_forward(w.detach(), imgs, ).argmax(1) == labs).float().mean().item())

        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    g = np.array(ft_gen); k = np.array(ft_kai)
    gap = g - k; wins = int(np.sum(gap > 0))
    t_stat, t_p = stats.ttest_rel(g, k)
    sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"

    print(f"\nCNN TARGET — DIGITS UNSEEN (trained on letters, n=50)")
    print(f"{'='*60}")
    print(f"Zero-shot:           {np.mean(zs):.4f} +- {np.std(zs):.4f}")
    print(f"Generated +3000FT:   {g.mean():.4f} +- {g.std():.4f}")
    print(f"Kaiming +3000FT:     {k.mean():.4f} +- {k.std():.4f}")
    print(f"Gap: {gap.mean()*100:+.2f}pp, wins={wins}/50, p={t_p:.6f} {sig}")


if __name__ == "__main__":
    run()
