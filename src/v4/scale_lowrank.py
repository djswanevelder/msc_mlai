"""
V4 Low-Rank Hypernetwork
=========================
Instead of generating full weight matrices, generate low-rank factors:
  W = A @ B where A is (out, rank) and B is (rank, in)

For 784->128: instead of 100K params, generate:
  A: 128 x 32 = 4,096
  B: 32 x 784 = 25,088
  Total: 29,184 (vs 100,480) — 3.4x reduction

Then fine-tuning expands to full-rank. The hypothesis: low-rank init
captures the most important weight directions, fine-tuning fills in details.
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

from src.v4.config import Config, TargetMLPConfig
from src.v4.models import PrototypeEncoder, differentiable_forward
from src.v4.train import build_class_image_index, sample_prototypes, sample_task_data


class LowRankHyperNetwork(nn.Module):
    """
    Generates low-rank weight factors per layer, then assembles full weights.
    W_layer = A @ B (low-rank), bias generated directly.
    """

    def __init__(self, target_cfg: TargetMLPConfig, rank: int = 32,
                 prototype_dim: int = 128, num_classes_per_task: int = 3):
        super().__init__()
        self.target_cfg = target_cfg
        self.rank = rank

        self.prototype_encoder = PrototypeEncoder(
            input_dim=target_cfg.input_dim, hidden_dim=256, output_dim=prototype_dim
        )
        concat_dim = prototype_dim * num_classes_per_task

        # Per-layer: generate A, B factors and bias
        dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
        self.layer_generators = nn.ModuleList()

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            # For small layers, just generate directly
            if fan_in * fan_out <= 10000:
                out_size = fan_in * fan_out + fan_out
                gen = nn.Sequential(
                    nn.Linear(concat_dim, 256), nn.ReLU(),
                    nn.Linear(256, out_size),
                )
                self.layer_generators.append(nn.ModuleDict({
                    'gen': gen,
                }))
            else:
                # Low-rank: generate A (out x rank) and B (rank x in)
                a_size = fan_out * rank
                b_size = rank * fan_in
                bias_size = fan_out
                gen_a = nn.Sequential(
                    nn.Linear(concat_dim, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, a_size),
                )
                gen_b = nn.Sequential(
                    nn.Linear(concat_dim, 512), nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, b_size),
                )
                gen_bias = nn.Sequential(
                    nn.Linear(concat_dim, 64), nn.ReLU(),
                    nn.Linear(64, bias_size),
                )
                self.layer_generators.append(nn.ModuleDict({
                    'gen_a': gen_a, 'gen_b': gen_b, 'gen_bias': gen_bias,
                }))

        # Store dims for forward
        self.dims = dims

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)

        weight_chunks = []
        for i, gen_dict in enumerate(self.layer_generators):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]

            if 'gen' in gen_dict:
                # Direct generation (small layer)
                out = gen_dict['gen'](task_emb)
                weight_chunks.append(out)
            else:
                # Low-rank: W = A @ B
                a_flat = gen_dict['gen_a'](task_emb)  # (fan_out * rank)
                b_flat = gen_dict['gen_b'](task_emb)  # (rank * fan_in)
                bias = gen_dict['gen_bias'](task_emb)  # (fan_out)

                A = a_flat.view(fan_out, self.rank)
                B = b_flat.view(self.rank, fan_in)
                W = (A @ B).flatten()  # (fan_out * fan_in)

                weight_chunks.append(torch.cat([W, bias]))

        return torch.cat(weight_chunks)


def run_lowrank_test():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Target params: {weight_dim:,}")

    zoo_path = "data/v4b/zoo/zoo.pt"
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

    for rank in [16, 32, 64]:
        print(f"\n{'='*60}")
        print(f"RANK = {rank}")
        print(f"{'='*60}")

        hypernet = LowRankHyperNetwork(cfg.target, rank=rank).to(device)
        total_params = sum(p.numel() for p in hypernet.parameters())
        print(f"HyperNet params: {total_params:,}")

        optimizer = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        n_train = len(train_classes)
        batch_size = 4
        best_val = 0
        best_state = None
        no_improve = 0

        for epoch in range(1, 301):
            hypernet.train()
            perm = torch.randperm(n_train)
            ec = et = nb = 0
            el = 0.0

            for start in range(0, n_train, batch_size):
                batch_idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                bl = torch.tensor(0.0, device=device)

                for i in batch_idx:
                    idx = i.item()
                    classes = train_classes[idx]
                    protos = sample_prototypes(ci_train, classes, 20, device)
                    imgs, labs = sample_task_data(ci_train, classes, 60, device)
                    gen_w_norm = hypernet(protos)
                    gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                    logits = differentiable_forward(gen_w, imgs, cfg.target)
                    bl = bl + F.cross_entropy(logits, labs) + 0.1 * F.mse_loss(gen_w_norm, train_norm[idx])
                    with torch.no_grad():
                        ec += (logits.argmax(1) == labs).sum().item()
                        et += labs.size(0)

                bl = bl / len(batch_idx)
                bl.backward()
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
                optimizer.step()
                el += bl.item()
                nb += 1

            scheduler.step()
            ta = ec / et if et > 0 else 0

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
            va = vc / vt if vt > 0 else 0

            if va > best_val:
                best_val = va
                best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
            if no_improve >= 60:
                print(f"  Early stop at {epoch}")
                break

        if best_state:
            hypernet.load_state_dict(best_state)
        hypernet.to(device)
        print(f"  Best val: {best_val:.4f}")

        # Quick all-unseen eval (20 tasks for speed)
        unseen_classes = list(range(50, 62))
        all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
        rng = random.Random(99)
        tasks = rng.sample(all_tasks, 20)

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

        zs = []
        ft_gen = []
        ft_kai = []
        for ti, classes in enumerate(tasks):
            protos = sample_prototypes(ci_train, classes, 20, device)
            with torch.no_grad():
                gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)

            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

            for name, init_w, store in [('gen', gen_w.clone(), ft_gen), ('kai', kaiming_init(), ft_kai)]:
                w = init_w.detach().requires_grad_(True)
                opt = torch.optim.SGD([w], lr=0.01)
                for _ in range(100):  # Quick 100-step eval
                    ti2, tl2 = sample_task_data(ci_train, classes, 50, device)
                    opt.zero_grad()
                    F.cross_entropy(differentiable_forward(w, ti2, cfg.target), tl2).backward()
                    opt.step()
                imgs, labs = sample_task_data(ci_test, classes, 200, device)
                with torch.no_grad():
                    store.append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item())

        print(f"\n  Rank {rank} | Zero-shot: {np.mean(zs):.4f} | +100 FT: {np.mean(ft_gen):.4f} | Kaiming +100 FT: {np.mean(ft_kai):.4f} | Gap: {np.mean(np.array(ft_gen)-np.array(ft_kai))*100:+.2f}pp")


if __name__ == "__main__":
    run_lowrank_test()
