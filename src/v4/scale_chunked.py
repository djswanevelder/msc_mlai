"""
V4 Chunked Hypernetwork — Layer-wise Weight Generation
=======================================================
Instead of generating all 109K params at once, generate each layer's
weights separately. Each sub-generator outputs a smaller tensor.

Architecture:
  Shared prototype encoder -> task embedding (384D)
  Per-layer generators:
    Layer 1: 384 -> 512 -> 512 -> (784*128 + 128) = 100,480
    Layer 2: 384 -> 256 -> 256 -> (128*64 + 64)   = 8,256
    Layer 3: 384 -> 128 -> 128 -> (64*3 + 3)      = 195
"""

import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from torchvision import datasets, transforms
from tqdm import tqdm

from src.v4.config import Config, TargetMLPConfig
from src.v4.models import PrototypeEncoder, differentiable_forward
from src.v4.train import build_class_image_index, sample_prototypes, sample_task_data


class ChunkedHyperNetwork(nn.Module):
    """Generate weights layer-by-layer from prototype embeddings."""

    def __init__(self, target_cfg: TargetMLPConfig, prototype_dim: int = 128,
                 num_classes_per_task: int = 3):
        super().__init__()
        self.target_cfg = target_cfg
        self.prototype_encoder = PrototypeEncoder(
            input_dim=target_cfg.input_dim,
            hidden_dim=256,
            output_dim=prototype_dim,
        )

        concat_dim = prototype_dim * num_classes_per_task

        # Compute per-layer output sizes
        dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
        self.layer_sizes = []
        for i in range(len(dims) - 1):
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            self.layer_sizes.append(w_size + b_size)

        # Per-layer generators — sized proportionally
        self.generators = nn.ModuleList()
        for layer_size in self.layer_sizes:
            if layer_size > 50000:
                # Large layer: bigger generator
                gen = nn.Sequential(
                    nn.Linear(concat_dim, 512), nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, layer_size),
                )
            elif layer_size > 1000:
                gen = nn.Sequential(
                    nn.Linear(concat_dim, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, layer_size),
                )
            else:
                gen = nn.Sequential(
                    nn.Linear(concat_dim, 128), nn.ReLU(),
                    nn.Linear(128, layer_size),
                )
            self.generators.append(gen)

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)  # (concat_dim,)

        # Generate each layer's weights
        chunks = [gen(task_emb) for gen in self.generators]
        return torch.cat(chunks, dim=0)


def run_chunked_test():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Target params: {weight_dim:,}")

    # Load zoo from scale test (already generated)
    zoo_path = "data/v4b/zoo/zoo.pt"
    if not Path(zoo_path).exists():
        print("ERROR: Run scale_test.py first to generate the zoo")
        return
    zoo = torch.load(zoo_path, map_location=device, weights_only=False)
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

    # Build chunked hypernetwork
    hypernet = ChunkedHyperNetwork(cfg.target).to(device)
    total_params = sum(p.numel() for p in hypernet.parameters())
    print(f"Chunked HyperNet params: {total_params:,}")
    print(f"Layer sizes: {hypernet.layer_sizes}")

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    n_train = len(train_classes)
    batch_size = 4
    best_val = 0
    best_state = None
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training Chunked HyperNetwork (layer-wise generation)")
    print(f"{'='*60}\n")

    for epoch in range(1, 301):
        hypernet.train()
        perm = torch.randperm(n_train)
        epoch_correct = epoch_total = n_batches = 0
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
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
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        # Quick val
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
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss/n_batches:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Best: {best_val:.4f}", flush=True)

        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    if best_state:
        hypernet.load_state_dict(best_state)
    hypernet.to(device)
    print(f"\nBest val: {best_val:.4f}")

    # Save
    ckpt_path = Path("data/v4b/checkpoints/chunked_hypernet.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": best_state,
        "normalization": {"w_mean": w_mean.cpu(), "w_std": w_std.cpu()},
    }, ckpt_path)

    # === All-unseen eval ===
    print(f"\n{'='*60}")
    print(f"All-unseen evaluation (chunked, 109K params)")
    print(f"{'='*60}")

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
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    STEPS = 3000
    results = {'chunked': [], 'kaiming': [], 'random': []}
    zero_shot = []

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zero_shot.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        for name, init_w in [('chunked', gen_w.clone()), ('kaiming', kaiming_init()), ('random', torch.randn(weight_dim, device=device) * 0.01)]:
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

        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    print(f"\nCHUNKED HYPERNET — 109K params, all-unseen (3/3), n={len(tasks)}, {STEPS} steps")
    print(f"{'='*60}")
    print(f"Zero-shot: {np.mean(zero_shot):.4f} +- {np.std(zero_shot):.4f}")
    for name in ['chunked', 'kaiming', 'random']:
        a = np.array(results[name])
        print(f"  {name:>10}: {a.mean():.4f} +- {a.std():.4f}")

    h = np.array(results['chunked'])
    for base in ['kaiming', 'random']:
        b = np.array(results[base])
        gap = h - b
        wins = np.sum(gap > 0)
        t_stat, t_p = stats.ttest_rel(h, b)
        nz = gap[gap != 0]
        w_stat, w_p = stats.wilcoxon(nz) if len(nz) > 0 else (0, 1)
        print(f"\n  vs {base}: gap={gap.mean()*100:+.2f}pp, wins={wins}/{len(gap)}, t={t_stat:.3f}, p={t_p:.8f}")


if __name__ == "__main__":
    run_chunked_test()
