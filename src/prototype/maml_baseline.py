"""
MAML Baseline Comparison
=========================
Implements first-order MAML (FOMAML) on the same EMNIST tasks
and compares head-to-head with the hypernetwork.

MAML learns ONE shared initialisation θ optimised for fast adaptation.
At test time: θ' = θ - α∇L(θ) for k steps.

Our hypernetwork generates a TASK-SPECIFIC initialisation from prototypes.
At test time: W = M(proto(X)) then optionally W' = W - α∇L(W) for k steps.

Both get the same compute budget at test time (k gradient steps).
The question: which produces better initialisations?
"""

import itertools
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from torchvision import datasets, transforms

from src.prototype.config import Config
from src.prototype.models import HyperNetwork as ProtoHyperNetwork, differentiable_forward
from src.prototype.train import build_class_image_index, sample_prototypes, sample_task_data
from src.prototype.zoo import TargetMLP


def maml_inner_loop(
    flat_weights: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
    inner_lr: float,
    inner_steps: int,
) -> torch.Tensor:
    """FOMAML inner loop: k gradient steps on task data.
    Returns adapted weights with requires_grad=True for meta-gradient."""
    w = flat_weights.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([w], lr=inner_lr)
    for _ in range(inner_steps):
        opt.zero_grad()
        logits = differentiable_forward(w, images, cfg.target)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    # Return detached + re-enabled for meta gradient computation
    result = w.data.clone().requires_grad_(True)
    return result


def run():
    cfg = Config()
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Params: {weight_dim:,}")
    print(f"Seen: classes 10-61 (letters). Unseen: classes 0-9 (digits).")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    seen = cfg.zoo.seen_classes
    unseen = cfg.zoo.unseen_classes
    INNER_LR = 0.01
    INNER_STEPS = 10
    META_LR = 1e-3

    # ===================================================================
    # TRAIN MAML
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Training FOMAML (inner_lr={INNER_LR}, inner_steps={INNER_STEPS})")
    print(f"{'='*60}")

    # Kaiming init as starting point
    theta = torch.zeros(weight_dim, device=device)
    idx = 0
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        std = (2.0 / fan_in) ** 0.5
        w_size = dims[i] * dims[i + 1]
        b_size = dims[i + 1]
        theta[idx:idx + w_size] = torch.randn(w_size, device=device) * std
        idx += w_size + b_size
    rng = random.Random(42)
    TASKS_PER_BATCH = 8
    best_val = 0
    best_theta = None
    no_improve = 0

    for epoch in range(1, 301):
        epoch_correct = 0
        epoch_total = 0
        task_grads = []

        for _ in range(TASKS_PER_BATCH):
            classes = sorted(rng.sample(seen, 3))
            support_imgs, support_labs = sample_task_data(ci_train, classes, 50, device)
            query_imgs, query_labs = sample_task_data(ci_train, classes, 50, device)

            # FOMAML inner loop: adapt from theta
            adapted = maml_inner_loop(theta, support_imgs, support_labs, cfg, INNER_LR, INNER_STEPS)

            # Query loss at adapted point
            query_logits = differentiable_forward(adapted, query_imgs, cfg.target)
            query_loss = F.cross_entropy(query_logits, query_labs)

            # FOMAML: gradient of query loss w.r.t. adapted weights
            grad = torch.autograd.grad(query_loss, adapted)[0]
            task_grads.append(grad.detach())

            with torch.no_grad():
                epoch_correct += (query_logits.argmax(1) == query_labs).sum().item()
                epoch_total += query_labs.size(0)

        # Manual FOMAML meta-update (no optimizer — direct SGD on theta)
        avg_grad = torch.stack(task_grads).mean(0)
        grad_norm = avg_grad.norm()
        if grad_norm > 5.0:
            avg_grad = avg_grad * 5.0 / grad_norm
        with torch.no_grad():
            theta -= META_LR * avg_grad

        # Val: adapt to held-out seen tasks, measure accuracy
        if epoch % 10 == 0:
            val_acc_list = []
            for _ in range(20):
                classes = sorted(rng.sample(seen, 3))
                support_imgs, support_labs = sample_task_data(ci_test, classes, 50, device)
                query_imgs, query_labs = sample_task_data(ci_test, classes, 50, device)
                adapted = maml_inner_loop(theta, support_imgs, support_labs, cfg, INNER_LR, INNER_STEPS)
                with torch.no_grad():
                    logits = differentiable_forward(adapted, query_imgs, cfg.target)
                    acc = (logits.argmax(1) == query_labs).float().mean().item()
                val_acc_list.append(acc)
            va = np.mean(val_acc_list)

            if va > best_val:
                best_val = va
                best_theta = theta.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            print(f"  Epoch {epoch:3d} | Grad Norm: {avg_grad.norm().item():.4f} | "
                  f"Train: {epoch_correct/epoch_total:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)

            if no_improve >= 6:  # 60 epochs patience
                print(f"  Early stop at {epoch}")
                break

    theta_maml = best_theta.to(device)
    print(f"  Best MAML val: {best_val:.4f}")

    # ===================================================================
    # LOAD HYPERNETWORK
    # ===================================================================
    print(f"\n{'='*60}")
    print("Loading hypernetwork for comparison")
    print(f"{'='*60}")

    ckpt = torch.load("data/v4/checkpoints/hypernet.pt", map_location=device, weights_only=False)
    hypernet = ProtoHyperNetwork(
        target_weight_dim=ckpt["config"]["target_weight_dim"],
        num_classes_per_task=ckpt["config"]["num_classes_per_task"],
        input_dim=ckpt["config"]["input_dim"],
        prototype_encoder_hidden=ckpt["config"]["prototype_encoder_hidden"],
        prototype_dim=ckpt["config"]["prototype_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
    ).to(device)
    hypernet.load_state_dict(ckpt["hypernet_state"])
    hypernet.eval()
    w_mean = ckpt["normalization"]["w_mean"].to(device).squeeze(0)
    w_std = ckpt["normalization"]["w_std"].to(device).squeeze(0)

    # ===================================================================
    # EVALUATE: MAML vs HyperNet vs Kaiming
    # ===================================================================
    print(f"\n{'='*60}")
    print("Evaluation: MAML vs HyperNet vs Kaiming")
    print(f"{'='*60}")

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

    # Test on digits (all unseen)
    all_tasks = [list(c) for c in itertools.combinations(unseen, 3)]
    rng_eval = random.Random(99)
    tasks = rng_eval.sample(all_tasks, min(50, len(all_tasks)))

    step_counts = [0, 10, 100, 3000]
    results = {s: {"hypernet": [], "maml": [], "kaiming": []} for s in step_counts}

    for ti, classes in enumerate(tasks):
        # HyperNet init
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            hn_w = hypernet(protos) * w_std + w_mean

        for steps in step_counts:
            for name, init_w in [("hypernet", hn_w.clone()), ("maml", theta_maml.clone()), ("kaiming", kaiming_init())]:
                w = init_w.detach().requires_grad_(True)
                opt = torch.optim.SGD([w], lr=INNER_LR)
                for _ in range(steps):
                    imgs, labs = sample_task_data(ci_train, classes, 50, device)
                    opt.zero_grad()
                    F.cross_entropy(differentiable_forward(w, imgs, cfg.target), labs).backward()
                    opt.step()
                # Eval
                imgs, labs = sample_task_data(ci_test, classes, 200, device)
                with torch.no_grad():
                    acc = (differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item()
                results[steps][name].append(acc)

        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    # Report
    print(f"\n{'='*60}")
    print("MAML vs HYPERNETWORK vs KAIMING — DIGITS UNSEEN (3/3)")
    print(f"Trained on letters only. Tested on digits. n={len(tasks)} tasks.")
    print(f"{'='*60}")
    print(f"{'Steps':>6} | {'HyperNet':>10} | {'MAML':>10} | {'Kaiming':>10} | {'HN-MAML':>8} | {'p(HN>MAML)':>10}")
    print("-" * 70)
    for s in step_counts:
        h = np.array(results[s]["hypernet"])
        m = np.array(results[s]["maml"])
        k = np.array(results[s]["kaiming"])
        gap_hm = h - m
        t_stat, t_p = stats.ttest_rel(h, m)
        wins = int(np.sum(gap_hm > 0))
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"{s:>6} | {h.mean():>10.4f} | {m.mean():>10.4f} | {k.mean():>10.4f} | "
              f"{gap_hm.mean()*100:>+7.2f}pp | {t_p:>10.6f} {sig}")

    # Also report HN vs Kaiming for completeness
    print()
    for s in step_counts:
        h = np.array(results[s]["hypernet"])
        k = np.array(results[s]["kaiming"])
        gap = h - k
        t_stat, t_p = stats.ttest_rel(h, k)
        wins = int(np.sum(gap > 0))
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"  HN vs Kaiming @ {s} steps: {gap.mean()*100:+.2f}pp, wins={wins}/{len(tasks)}, p={t_p:.6f} {sig}")


if __name__ == "__main__":
    run()
