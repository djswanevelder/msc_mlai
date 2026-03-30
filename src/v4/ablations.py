"""
V4 Ablation Study — Targeted improvements for zero-shot accuracy
=================================================================
Tests 4 improvements on the 50K direct model for fast iteration.
Each ablation trains + evals in ~20min on MPS.

Ablations:
  A: Baseline (current V4 direct, for fair comparison)
  B: Cross-attention prototype encoder (captures inter-class relationships)
  C: 5x more training tasks (1000 instead of 200)
  D: B + C combined
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
from src.v4.zoo import TargetMLP, get_emnist_data, make_class_subset, _train_one_subset

import multiprocessing as mp
from functools import partial


# ============================================================
# Improved Prototype Encoder: Cross-Attention
# ============================================================

class CrossAttentionProtoEncoder(nn.Module):
    """
    Instead of encoding each class independently and concatenating,
    encode all 3 classes jointly with cross-attention so the network
    understands inter-class relationships (which classes are similar/different).
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256,
                 proto_dim: int = 128, n_heads: int = 4):
        super().__init__()
        # Per-image encoder (shared)
        self.image_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proto_dim),
        )
        # Cross-attention between class prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proto_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(proto_dim)
        self.proto_dim = proto_dim

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prototypes: list of 3 tensors, each (K, input_dim)
        Returns:
            (3 * proto_dim,) task embedding with inter-class info
        """
        # Encode each class: mean-pool images, get (3, proto_dim)
        class_embs = []
        for proto_images in prototypes:
            img_embs = self.image_encoder(proto_images)  # (K, proto_dim)
            class_embs.append(img_embs.mean(dim=0))  # (proto_dim,)

        # Stack to (1, 3, proto_dim) for attention
        x = torch.stack(class_embs).unsqueeze(0)  # (1, 3, proto_dim)

        # Self-attention between the 3 class prototypes
        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm(x + attn_out)  # residual + norm

        return x.squeeze(0).flatten()  # (3 * proto_dim,)


class MeanPoolProtoWrapper(nn.Module):
    """Wraps PrototypeEncoder to handle list of 3 class prototype tensors."""

    def __init__(self, encoder: PrototypeEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.encoder(p) for p in prototypes]
        return torch.cat(embs, dim=0)


class ImprovedHyperNetwork(nn.Module):
    """HyperNetwork with pluggable prototype encoder."""

    def __init__(self, target_weight_dim: int, proto_encoder: nn.Module,
                 task_emb_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        self.proto_encoder = proto_encoder
        layers = []
        d = task_emb_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        task_emb = self.proto_encoder(prototypes)
        return self.net(task_emb)


# ============================================================
# Training + Eval helper
# ============================================================

def train_and_eval(
    hypernet: nn.Module,
    train_classes: List[List[int]],
    train_weights_norm: torch.Tensor,
    w_mean: torch.Tensor,
    w_std: torch.Tensor,
    ci_train: Dict,
    ci_test: Dict,
    cfg: Config,
    device: str,
    num_epochs: int = 300,
    patience: int = 60,
    lr: float = 5e-4,
) -> Tuple[nn.Module, float]:
    """Train hypernetwork, return best model + val accuracy."""
    n_train = len(train_classes)
    opt = torch.optim.Adam(hypernet.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    best_val = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = 0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
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
                func_loss = F.cross_entropy(logits, labs)
                mse_loss = F.mse_loss(gen_w_norm, train_weights_norm[idx])
                bl = bl + func_loss + 0.1 * mse_loss

                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            opt.step()

        sched.step()

        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(40, n_train)):
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

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | Train: {ec/et:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= patience:
            print(f"    Early stop at {epoch}")
            break

    hypernet.load_state_dict(best_state)
    hypernet.to(device)
    return hypernet, best_val


def eval_unseen(
    hypernet: nn.Module,
    w_mean: torch.Tensor,
    w_std: torch.Tensor,
    ci_train: Dict,
    ci_test: Dict,
    cfg: Config,
    device: str,
    n_tasks: int = 50,
) -> Dict:
    """Evaluate on all-unseen tasks. Returns zero-shot acc and FT comparison."""
    weight_dim = cfg.target_weight_dim()
    dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, n_tasks)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = (2.0 / fan_in) ** 0.5
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    hypernet.eval()
    zs, ft_gen, ft_kai = [], [], []
    STEPS = 3000

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_init(), ft_kai)]:
            w = init_w.detach().requires_grad_(True)
            opt = torch.optim.SGD([w], lr=0.01)
            for _ in range(STEPS):
                i2, l2 = sample_task_data(ci_train, classes, 50, device)
                opt.zero_grad()
                F.cross_entropy(differentiable_forward(w, i2, cfg.target), l2).backward()
                opt.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                store.append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item())

    g = np.array(ft_gen)
    k = np.array(ft_kai)
    gap = g - k
    wins = int(np.sum(gap > 0))
    t_stat, t_p = stats.ttest_rel(g, k)

    return {
        "zero_shot": np.mean(zs),
        "zero_shot_std": np.std(zs),
        "ft_gen": g.mean(),
        "ft_kai": k.mean(),
        "gap": gap.mean() * 100,
        "wins": wins,
        "n_tasks": n_tasks,
        "p_value": t_p,
    }


# ============================================================
# Main: Run all ablations
# ============================================================

def run():
    cfg = Config()
    # Use 50K target for fast iteration
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    proto_dim = 128

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Params: {weight_dim:,}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    seen = list(range(50))
    rng = random.Random(42)

    # ======== Generate zoo for each task count ========
    zoo_200_path = Path("data/v4/zoo/zoo.pt")
    zoo_1000_path = Path("data/v4_500/zoo/zoo.pt")

    # Load/generate 200-task zoo (already exists)
    zoo_200 = torch.load(str(zoo_200_path), map_location=device, weights_only=False)
    train_w_200 = zoo_200["train"]["weights"].to(device)
    train_c_200 = zoo_200["train"]["classes"]

    # Generate 1000-task zoo if needed
    if not zoo_1000_path.exists():
        print("\nGenerating 1000-task zoo...")
        zoo_1000_path.parent.mkdir(parents=True, exist_ok=True)
        get_emnist_data()

        all_seen = [list(s) for s in itertools.combinations(seen, 3)]
        rng_zoo = random.Random(42)
        rng_zoo.shuffle(all_seen)
        train_subsets_1k = all_seen[:500]

        worker_fn = partial(
            _train_one_subset,
            input_dim=cfg.target.input_dim,
            hidden_dims=cfg.target.hidden_dims,
            num_classes=cfg.target.num_classes,
            train_epochs=30,
            lr=1e-3,
            batch_size=128,
        )
        results = []
        num_workers = min(mp.cpu_count(), 8)
        print(f"  Training 500 models with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            from tqdm import tqdm
            for r in tqdm(pool.imap(worker_fn, train_subsets_1k), total=500, desc="Zoo 500"):
                results.append(r)

        accs = [r["test_accuracy"] for r in results]
        print(f"  1K zoo accuracy: mean={sum(accs)/len(accs):.4f}")

        torch.save({
            "train": {
                "weights": torch.stack([r["flat_weights"] for r in results]),
                "classes": [r["classes"] for r in results],
                "accuracies": accs,
            },
            "architecture": zoo_200["architecture"],
            "seen_classes": seen,
            "unseen_classes": list(range(50, 62)),
        }, zoo_1000_path)

    zoo_1000 = torch.load(str(zoo_1000_path), map_location=device, weights_only=False)
    train_w_1k = zoo_1000["train"]["weights"].to(device)
    train_c_1k = zoo_1000["train"]["classes"]

    # Normalize
    def norm(w):
        m = w.mean(0, keepdim=True)
        s = w.std(0, keepdim=True).clamp(min=1e-6)
        return (w - m) / s, m, s

    tw200_n, m200, s200 = norm(train_w_200)
    tw1k_n, m1k, s1k = norm(train_w_1k)

    # ======== ABLATIONS ========
    ablations = {}

    # A: Baseline (mean-pool encoder, 200 tasks)
    print(f"\n{'='*60}")
    print("ABLATION A: Baseline (mean-pool, 200 tasks)")
    print(f"{'='*60}")
    enc_a = MeanPoolProtoWrapper(PrototypeEncoder(input_dim=784, hidden_dim=256, output_dim=proto_dim))
    hn_a = ImprovedHyperNetwork(weight_dim, enc_a, 3 * proto_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in hn_a.parameters()):,}")
    hn_a, val_a = train_and_eval(hn_a, train_c_200, tw200_n, m200, s200, ci_train, ci_test, cfg, device)
    print(f"  Evaluating on unseen...")
    ablations["A"] = eval_unseen(hn_a, m200, s200, ci_train, ci_test, cfg, device, n_tasks=50)

    # B: Cross-attention encoder, 200 tasks
    print(f"\n{'='*60}")
    print("ABLATION B: Cross-attention encoder (200 tasks)")
    print(f"{'='*60}")
    enc_b = CrossAttentionProtoEncoder(input_dim=784, hidden_dim=256, proto_dim=proto_dim, n_heads=4)
    hn_b = ImprovedHyperNetwork(weight_dim, enc_b, 3 * proto_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in hn_b.parameters()):,}")
    hn_b, val_b = train_and_eval(hn_b, train_c_200, tw200_n, m200, s200, ci_train, ci_test, cfg, device)
    print(f"  Evaluating on unseen...")
    ablations["B"] = eval_unseen(hn_b, m200, s200, ci_train, ci_test, cfg, device, n_tasks=50)

    # C: Mean-pool encoder, 1000 tasks
    print(f"\n{'='*60}")
    print("ABLATION C: More training tasks (mean-pool, 500 tasks)")
    print(f"{'='*60}")
    enc_c = MeanPoolProtoWrapper(PrototypeEncoder(input_dim=784, hidden_dim=256, output_dim=proto_dim))
    hn_c = ImprovedHyperNetwork(weight_dim, enc_c, 3 * proto_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in hn_c.parameters()):,}")
    hn_c, val_c = train_and_eval(hn_c, train_c_1k, tw1k_n, m1k, s1k, ci_train, ci_test, cfg, device)
    print(f"  Evaluating on unseen...")
    ablations["C"] = eval_unseen(hn_c, m1k, s1k, ci_train, ci_test, cfg, device, n_tasks=50)

    # D: Cross-attention + 1000 tasks
    print(f"\n{'='*60}")
    print("ABLATION D: Cross-attention + 500 tasks")
    print(f"{'='*60}")
    enc_d = CrossAttentionProtoEncoder(input_dim=784, hidden_dim=256, proto_dim=proto_dim, n_heads=4)
    hn_d = ImprovedHyperNetwork(weight_dim, enc_d, 3 * proto_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in hn_d.parameters()):,}")
    hn_d, val_d = train_and_eval(hn_d, train_c_1k, tw1k_n, m1k, s1k, ci_train, ci_test, cfg, device)
    print(f"  Evaluating on unseen...")
    ablations["D"] = eval_unseen(hn_d, m1k, s1k, ci_train, ci_test, cfg, device, n_tasks=50)

    # ======== RESULTS ========
    print(f"\n{'='*60}")
    print("ABLATION RESULTS (50K direct, all-unseen 3/3, 3000 FT steps)")
    print(f"{'='*60}")
    print(f"{'Ablation':<35} {'Zero-shot':>10} {'+3000FT':>10} {'Gap':>8} {'Wins':>6} {'p':>10}")
    print("-" * 82)
    for name, label in [
        ("A", "A: Baseline (mean-pool, 200)"),
        ("B", "B: Cross-attn encoder (200)"),
        ("C", "C: Mean-pool (500 tasks)"),
        ("D", "D: Cross-attn + 500 tasks"),
    ]:
        r = ablations[name]
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{label:<35} {r['zero_shot']:>10.4f} {r['ft_gen']:>10.4f} {r['gap']:>+8.2f} {r['wins']:>4}/50 {r['p_value']:>10.6f} {sig}")

    # Pairwise: is D better than A?
    print(f"\n--- Pairwise comparison: D vs A ---")
    # Need to re-eval with paired tasks for fair comparison
    # The eval_unseen function uses the same random seed so tasks are identical
    za = ablations["A"]["zero_shot"]
    zd = ablations["D"]["zero_shot"]
    print(f"  Zero-shot: A={za:.4f}, D={zd:.4f}, improvement={zd-za:+.4f}")
    ga = ablations["A"]["gap"]
    gd = ablations["D"]["gap"]
    print(f"  Gap vs Kaiming: A={ga:+.2f}pp, D={gd:+.2f}pp")


if __name__ == "__main__":
    run()
