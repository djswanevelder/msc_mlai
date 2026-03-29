"""
V4 Scaling Test — Larger Target Architecture
=============================================
Tests if the better-minimum finding holds with a deeper target MLP.
Uses a 2-hidden-layer MLP (784 -> 128 -> 64 -> 3) = 107,459 params.
Trains a new zoo + new hypernetwork, then runs the all-unseen comparison.
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
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.v4.config import Config, TargetMLPConfig, ZooConfig, HyperNetConfig
from src.v4.zoo import TargetMLP, get_emnist_data, make_class_subset, _train_one_subset
from src.v4.models import PrototypeEncoder, differentiable_forward
from src.v4.train import build_class_image_index, sample_prototypes, sample_task_data


# Override config for larger arch
def get_large_config() -> Config:
    cfg = Config()
    cfg.target = TargetMLPConfig(
        input_dim=784,
        hidden_dims=[128, 64],  # 2 hidden layers
        num_classes=3,
    )
    cfg.zoo.zoo_dir = "data/v4b/zoo"
    cfg.zoo.num_train_subsets = 150  # more training data for harder problem
    cfg.hypernet.hidden_dims = [1024, 1024]  # bigger hypernetwork
    cfg.hypernet.num_epochs = 200
    cfg.hypernet.patience = 50
    cfg.hypernet.batch_size = 4  # smaller batch — each task is more expensive
    return cfg


class LargeHyperNetwork(nn.Module):
    """Same as HyperNetwork but generates more weights."""

    def __init__(self, target_weight_dim, num_classes_per_task=3,
                 input_dim=784, prototype_encoder_hidden=256,
                 prototype_dim=128, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024]
        self.target_weight_dim = target_weight_dim
        self.prototype_encoder = PrototypeEncoder(input_dim, prototype_encoder_hidden, prototype_dim)
        concat_dim = prototype_dim * num_classes_per_task
        layers = []
        d = concat_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.weight_generator = nn.Sequential(*layers)

    def forward(self, prototypes):
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)
        return self.weight_generator(task_emb)


def run_scale_test():
    cfg = get_large_config()
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    print(f"Target MLP: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Target params: {weight_dim:,}")

    zoo_path = Path(cfg.zoo.zoo_dir) / "zoo.pt"

    # === STAGE 1: Generate zoo ===
    if not zoo_path.exists():
        print("\n=== Generating larger zoo ===")
        zoo_path.parent.mkdir(parents=True, exist_ok=True)
        get_emnist_data()

        rng = random.Random(42)
        seen = cfg.zoo.seen_classes
        unseen = cfg.zoo.unseen_classes

        all_seen = [list(s) for s in itertools.combinations(seen, 3)]
        rng.shuffle(all_seen)
        train_subsets = all_seen[:cfg.zoo.num_train_subsets]
        seen_holdout = all_seen[cfg.zoo.num_train_subsets:cfg.zoo.num_train_subsets + cfg.zoo.num_seen_holdout_subsets]

        unseen_combos = []
        for u in unseen:
            for pair in itertools.combinations(seen, 2):
                unseen_combos.append(sorted(list(pair) + [u]))
        rng.shuffle(unseen_combos)
        seen_set = set()
        unseen_subsets = []
        for combo in unseen_combos:
            key = tuple(combo)
            if key not in seen_set:
                seen_set.add(key)
                unseen_subsets.append(list(combo))
            if len(unseen_subsets) >= cfg.zoo.num_unseen_subsets:
                break

        all_tasks = train_subsets + seen_holdout + unseen_subsets
        print(f"Tasks: {len(train_subsets)} train, {len(seen_holdout)} holdout, {len(unseen_subsets)} unseen")

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
        results = []
        with mp.Pool(num_workers) as pool:
            for r in tqdm(pool.imap(worker_fn, all_tasks), total=len(all_tasks), desc="Zoo"):
                results.append(r)

        n1 = len(train_subsets)
        n2 = len(seen_holdout)
        def pack(rl):
            return {
                "weights": torch.stack([r["flat_weights"] for r in rl]),
                "classes": [r["classes"] for r in rl],
                "accuracies": [r["test_accuracy"] for r in rl],
            }
        torch.save({
            "train": pack(results[:n1]),
            "seen_holdout": pack(results[n1:n1+n2]),
            "unseen": pack(results[n1+n2:]),
            "architecture": {"input_dim": cfg.target.input_dim, "hidden_dims": cfg.target.hidden_dims,
                           "num_classes": cfg.target.num_classes, "total_params": weight_dim},
            "seen_classes": cfg.zoo.seen_classes, "unseen_classes": cfg.zoo.unseen_classes,
        }, zoo_path)
        for name, rl in [("Train", results[:n1]), ("Holdout", results[n1:n1+n2]), ("Unseen", results[n1+n2:])]:
            accs = [r["test_accuracy"] for r in rl]
            print(f"  {name}: mean={sum(accs)/len(accs):.4f}")

    # === STAGE 2: Train hypernetwork ===
    ckpt_path = Path("data/v4b/checkpoints/hypernet.pt")
    if not ckpt_path.exists():
        print("\n=== Training larger hypernetwork ===")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        zoo = torch.load(str(zoo_path), map_location=device, weights_only=False)
        train_weights = zoo["train"]["weights"].to(device)
        train_classes = zoo["train"]["classes"]
        w_mean = train_weights.mean(0, keepdim=True)
        w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
        train_norm = (train_weights - w_mean) / w_std

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
        emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
        emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
        ci_train = build_class_image_index(emnist_train)
        ci_test = build_class_image_index(emnist_test)

        hypernet = LargeHyperNetwork(
            target_weight_dim=weight_dim,
            hidden_dims=cfg.hypernet.hidden_dims,
        ).to(device)
        print(f"HyperNet params: {sum(p.numel() for p in hypernet.parameters()):,}")

        optimizer = torch.optim.Adam(hypernet.parameters(), lr=cfg.hypernet.lr, weight_decay=cfg.hypernet.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.hypernet.num_epochs)

        n_train = len(train_classes)
        best_val = 0
        best_state = None
        no_improve = 0

        for epoch in range(1, cfg.hypernet.num_epochs + 1):
            hypernet.train()
            perm = torch.randperm(n_train)
            epoch_correct = epoch_total = n_batches = 0
            epoch_loss = 0.0

            for start in range(0, n_train, cfg.hypernet.batch_size):
                batch_idx = perm[start:start + cfg.hypernet.batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device)
                for i in batch_idx:
                    idx = i.item()
                    classes = train_classes[idx]
                    protos = sample_prototypes(ci_train, classes, 20, device)
                    imgs, labs = sample_task_data(ci_train, classes, 60, device)
                    gen_w_norm = hypernet(protos)
                    gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                    logits = differentiable_forward(gen_w, imgs, cfg.target)
                    func = F.cross_entropy(logits, labs)
                    mse = F.mse_loss(gen_w_norm, train_norm[idx])
                    batch_loss = batch_loss + func + 0.1 * mse
                    with torch.no_grad():
                        epoch_correct += (logits.argmax(1) == labs).sum().item()
                        epoch_total += labs.size(0)
                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1
            scheduler.step()

            # Val
            hypernet.eval()
            vc = vt = 0
            with torch.no_grad():
                for idx in range(min(n_train, 40)):
                    classes = train_classes[idx]
                    protos = sample_prototypes(ci_test, classes, 20, device)
                    imgs, labs = sample_task_data(ci_test, classes, 50, device)
                    gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)
                    vc += (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).sum().item()
                    vt += labs.size(0)
            val_acc = vc / vt if vt > 0 else 0

            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {epoch_loss/n_batches:.4f} | Train: {epoch_correct/epoch_total:.4f} | Val: {val_acc:.4f} | Best: {best_val:.4f}", flush=True)
            if no_improve >= cfg.hypernet.patience:
                print(f"  Early stop at {epoch}")
                break

        hypernet.load_state_dict(best_state)
        hypernet.to(device)
        torch.save({
            "hypernet_state": hypernet.state_dict(),
            "config": {"target_weight_dim": weight_dim, "hidden_dims": cfg.hypernet.hidden_dims},
            "normalization": {"w_mean": w_mean.cpu(), "w_std": w_std.cpu()},
        }, ckpt_path)
        print(f"  Saved. Best val: {best_val:.4f}")

    # === STAGE 3: All-unseen eval ===
    print("\n=== All-unseen evaluation (larger arch) ===")
    zoo = torch.load(str(zoo_path), map_location='cpu', weights_only=False)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    hypernet = LargeHyperNetwork(
        target_weight_dim=ckpt["config"]["target_weight_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
    ).to(device)
    hypernet.load_state_dict(ckpt["hypernet_state"])
    hypernet.eval()
    w_mean = ckpt["normalization"]["w_mean"].to(device).squeeze(0)
    w_std = ckpt["normalization"]["w_std"].to(device).squeeze(0)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
    ci_train = build_class_image_index(emnist_train)
    ci_test = build_class_image_index(emnist_test)

    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = (2.0 / fan_in) ** 0.5
            w_size = dims[i] * dims[i+1]
            b_size = dims[i+1]
            w[idx:idx+w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    STEPS = 3000
    results = {'hypernet': [], 'kaiming': [], 'random': []}
    hypernet_zero = []

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            gen_w = hypernet(protos) * w_std + w_mean

        # 0-step eval
        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            hypernet_zero.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        for name, init_w in [('hypernet', gen_w.clone()), ('kaiming', kaiming_init()), ('random', torch.randn(weight_dim, device=device)*0.01)]:
            w = init_w.detach().requires_grad_(True)
            opt = torch.optim.SGD([w], lr=0.01)
            for _ in range(STEPS):
                ti2, tl2 = sample_task_data(ci_train, classes, 50, device)
                opt.zero_grad()
                F.cross_entropy(differentiable_forward(w, ti2, cfg.target), tl2).backward()
                opt.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                results[name].append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item())

        if (ti+1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    print(f"\nLARGER ARCH: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes} ({weight_dim:,} params)")
    print(f"All-unseen (3/3), n={len(tasks)}, {STEPS} steps")
    print(f"{'='*60}")
    print(f"HyperNet 0-step: {np.mean(hypernet_zero):.4f} +- {np.std(hypernet_zero):.4f}")
    for name in ['hypernet', 'kaiming', 'random']:
        a = np.array(results[name])
        print(f"{name:>12}: {a.mean():.4f} +- {a.std():.4f}")

    h = np.array(results['hypernet'])
    for base in ['kaiming', 'random']:
        b = np.array(results[base])
        gap = h - b
        wins = np.sum(gap > 0)
        t_stat, t_p = stats.ttest_rel(h, b)
        nz = gap[gap != 0]
        w_stat, w_p = stats.wilcoxon(nz) if len(nz) > 0 else (0, 1)
        print(f"\n  vs {base}: gap={gap.mean()*100:+.2f}pp, wins={wins}/{len(gap)}, t-test p={t_p:.8f}, Wilcoxon p={w_p:.8f}")


if __name__ == '__main__':
    run_scale_test()
