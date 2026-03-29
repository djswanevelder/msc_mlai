"""
V4 Warm-Start Scaling — MSE pre-training then functional loss
==============================================================
The 109K param target fails because functional loss gradients vanish
when generated weights are far from useful. Solution: pre-train with
MSE loss to get close to zoo weights, then switch to functional loss
for fine-tuning the generation.

Uses chunked (layer-wise) generation for the architecture.
"""

import itertools
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from torchvision import datasets, transforms

from src.v4.config import Config, TargetMLPConfig
from src.v4.models import PrototypeEncoder, differentiable_forward
from src.v4.train import build_class_image_index, sample_prototypes, sample_task_data


class WarmStartHyperNetwork(nn.Module):
    def __init__(self, target_cfg: TargetMLPConfig, prototype_dim: int = 128,
                 num_classes_per_task: int = 3):
        super().__init__()
        self.prototype_encoder = PrototypeEncoder(
            input_dim=target_cfg.input_dim, hidden_dim=256, output_dim=prototype_dim
        )
        concat_dim = prototype_dim * num_classes_per_task

        dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
        self.generators = nn.ModuleList()
        self.layer_sizes = []
        for i in range(len(dims) - 1):
            size = dims[i] * dims[i + 1] + dims[i + 1]
            self.layer_sizes.append(size)
            # Scale generator hidden size to layer size
            h = min(512, max(128, size // 4))
            self.generators.append(nn.Sequential(
                nn.Linear(concat_dim, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, size),
            ))

    def forward(self, prototypes):
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)
        return torch.cat([g(task_emb) for g in self.generators])


def run():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    print(f"Target: 784 -> [128, 64] -> 3, params: {weight_dim:,}")

    zoo = torch.load("data/v4b/zoo/zoo.pt", map_location=device, weights_only=False)
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

    hypernet = WarmStartHyperNetwork(cfg.target).to(device)
    print(f"HyperNet params: {sum(p.numel() for p in hypernet.parameters()):,}")
    print(f"Layer sizes: {hypernet.layer_sizes}")

    n_train = len(train_classes)
    BS = 8

    # === PHASE 1: MSE-only pre-training (fast, gets weights close to zoo) ===
    print(f"\n--- Phase 1: MSE pre-training (100 epochs) ---")
    opt = torch.optim.Adam(hypernet.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_mse = float('inf')
    best_state = None

    for epoch in range(1, 101):
        hypernet.train()
        perm = torch.randperm(n_train)
        el = nb = 0

        for start in range(0, n_train, BS):
            batch_idx = perm[start:start + BS]
            opt.zero_grad()
            bl = torch.tensor(0.0, device=device)
            for i in batch_idx:
                idx = i.item()
                protos = sample_prototypes(ci_train, train_classes[idx], 20, device)
                gen_w_norm = hypernet(protos)
                bl = bl + F.mse_loss(gen_w_norm, train_norm[idx])
            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            opt.step()
            el += bl.item()
            nb += 1

        # Val MSE
        hypernet.eval()
        vmse = 0
        vn = 0
        with torch.no_grad():
            for idx in range(min(n_train, 40)):
                protos = sample_prototypes(ci_test, train_classes[idx], 20, device)
                gen_w_norm = hypernet(protos)
                vmse += F.mse_loss(gen_w_norm, train_norm[idx]).item()
                vn += 1
        vmse /= vn

        if vmse < best_val_mse:
            best_val_mse = vmse
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train MSE: {el/nb:.6f} | Val MSE: {vmse:.6f}", flush=True)

    hypernet.load_state_dict(best_state)
    hypernet.to(device)

    # Check: what accuracy does MSE-only achieve?
    hypernet.eval()
    mse_accs = []
    with torch.no_grad():
        for idx in range(min(n_train, 40)):
            classes = train_classes[idx]
            protos = sample_prototypes(ci_test, classes, 20, device)
            gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)
            imgs, labs = sample_task_data(ci_test, classes, 100, device)
            acc = (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item()
            mse_accs.append(acc)
    print(f"  MSE-only val accuracy: {np.mean(mse_accs):.4f}")

    # === PHASE 2: Functional loss fine-tuning ===
    print(f"\n--- Phase 2: Functional loss fine-tuning (200 epochs) ---")
    opt = torch.optim.Adam(hypernet.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    best_val = 0
    no_improve = 0

    for epoch in range(1, 201):
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, BS):
            batch_idx = perm[start:start + BS]
            opt.zero_grad()
            bl = torch.tensor(0.0, device=device)
            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)
                gen_w_norm = hypernet(protos)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                logits = differentiable_forward(gen_w, imgs, cfg.target)
                bl = bl + F.cross_entropy(logits, labs) + 0.05 * F.mse_loss(gen_w_norm, train_norm[idx])
                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)
            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            opt.step()
            el += bl.item()
            nb += 1
        sched.step()

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
        va = vc / vt

        if va > best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ec/et:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    hypernet.load_state_dict(best_state)
    hypernet.to(device)

    # === EVAL: All-unseen ===
    print(f"\n--- All-unseen eval (109K params, warm-started) ---")
    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 30)

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

    zs, ft_gen, ft_kai = [], [], []
    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_init(), ft_kai)]:
            w = init_w.detach().requires_grad_(True)
            o = torch.optim.SGD([w], lr=0.01)
            for _ in range(500):
                ti2, tl2 = sample_task_data(ci_train, classes, 50, device)
                o.zero_grad()
                F.cross_entropy(differentiable_forward(w, ti2, cfg.target), tl2).backward()
                o.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                store.append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item())

        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    g = np.array(ft_gen)
    k = np.array(ft_kai)
    gap = g - k
    wins = np.sum(gap > 0)
    t_stat, t_p = stats.ttest_rel(g, k)

    print(f"\n109K PARAM TARGET — WARM-START RESULTS")
    print(f"{'='*50}")
    print(f"Zero-shot:        {np.mean(zs):.4f} +- {np.std(zs):.4f}")
    print(f"Generated +500FT: {g.mean():.4f} +- {g.std():.4f}")
    print(f"Kaiming +500FT:   {k.mean():.4f} +- {k.std():.4f}")
    print(f"Gap: {gap.mean()*100:+.2f}pp, Wins: {wins}/{len(gap)}, p={t_p:.6f}")


if __name__ == "__main__":
    run()
