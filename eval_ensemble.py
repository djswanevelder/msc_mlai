"""
Ensemble Evaluation — Compose hypernetwork specialists to classify full EMNIST.

Uses a trained prototype-conditioned hypernetwork to generate 3-class specialist
models, then ensembles them to classify all 62 EMNIST ByClass classes.

Key insight: naive aggregation fails because 3-class specialists have no reject
option — they confidently predict one of their 3 classes even for OOD inputs.
We solve this with prototype-gated aggregation: use the learned prototype encoder
to detect which specialists are relevant for each input, suppressing OOD votes.

Comparisons:
  1. Hypernetwork ensemble (varying coverage: 1x, 2x, 3x)
  2. Baseline MLP with matched total params (trained on full EMNIST)
  3. Random-weight ensemble (same structure, no hypernetwork)
  4. SOTA reference numbers

Usage:
  python eval_ensemble.py
  python eval_ensemble.py --coverages 1 2 3 --device mps
  python eval_ensemble.py --finetune 10
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.prototype.config import Config, TargetMLPConfig
from src.prototype.models import HyperNetwork, differentiable_forward
from src.prototype.train import build_class_image_index, sample_prototypes

CACHE_DIR = Path("data/v4/cache")


# ---------------------------------------------------------------------------
# EMNIST class name mapping (for readable output)
# ---------------------------------------------------------------------------
EMNIST_BYCLASS_NAMES = {
    **{i: str(i) for i in range(10)},                          # 0-9: digits
    **{i + 10: chr(ord("A") + i) for i in range(26)},          # 10-35: A-Z
    **{i + 36: chr(ord("a") + i) for i in range(26)},          # 36-61: a-z
}


# ---------------------------------------------------------------------------
# Covering subset generation
# ---------------------------------------------------------------------------

def generate_covering_subsets(
    all_classes: List[int],
    coverage: int,
    num_per_task: int = 3,
    seed: int = 42,
) -> List[List[int]]:
    """Generate 3-class subsets covering each class at least `coverage` times."""
    rng = random.Random(seed)
    subsets = []
    for _ in range(coverage):
        classes = list(all_classes)
        rng.shuffle(classes)
        for i in range(0, len(classes), num_per_task):
            group = classes[i : i + num_per_task]
            if len(group) < num_per_task:
                pool = [c for c in all_classes if c not in group]
                group += rng.sample(pool, num_per_task - len(group))
            subsets.append(sorted(group))
    return subsets


def coverage_stats(subsets: List[List[int]], all_classes: List[int]) -> Dict:
    counts = {c: 0 for c in all_classes}
    for sub in subsets:
        for c in sub:
            counts[c] += 1
    vals = list(counts.values())
    return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


# ---------------------------------------------------------------------------
# Hypernetwork loading & specialist generation
# ---------------------------------------------------------------------------

def load_hypernet(checkpoint_path: str, device: str):
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hypernet = HyperNetwork(
        target_weight_dim=data["config"]["target_weight_dim"],
        num_classes_per_task=data["config"]["num_classes_per_task"],
        input_dim=data["config"]["input_dim"],
        prototype_encoder_hidden=data["config"]["prototype_encoder_hidden"],
        prototype_dim=data["config"]["prototype_dim"],
        hidden_dims=data["config"]["hidden_dims"],
    )
    hypernet.load_state_dict(data["hypernet_state"])
    hypernet.to(device)
    hypernet.eval()
    w_mean = data["normalization"]["w_mean"].to(device)
    w_std = data["normalization"]["w_std"].to(device)
    return hypernet, w_mean, w_std


def generate_specialists(
    hypernet: HyperNetwork,
    w_mean: torch.Tensor,
    w_std: torch.Tensor,
    subsets: List[List[int]],
    class_images: Dict[int, torch.Tensor],
    K: int,
    device: str,
    n_proto_samples: int = 5,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate specialist weight vectors via the hypernetwork."""
    specialists = []
    for classes in subsets:
        weights_accum = []
        for _ in range(n_proto_samples):
            prototypes = sample_prototypes(class_images, classes, K, device)
            with torch.no_grad():
                gen_w_norm = hypernet(prototypes)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
            weights_accum.append(gen_w)
        avg_w = torch.stack(weights_accum).mean(dim=0)
        specialists.append((classes, avg_w))
    return specialists


def generate_random_specialists(
    subsets: List[List[int]], weight_dim: int, device: str,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate specialists with Kaiming-scale random weights (control)."""
    specialists = []
    for classes in subsets:
        w = torch.randn(weight_dim, device=device) * (2.0 / 784) ** 0.5
        specialists.append((classes, w))
    return specialists


# ---------------------------------------------------------------------------
# Prototype embeddings for gating
# ---------------------------------------------------------------------------

def build_class_prototypes(
    proto_encoder: nn.Module,
    class_images: Dict[int, torch.Tensor],
    all_classes: List[int],
    K: int,
    device: str,
    n_samples: int = 5,
) -> Dict[int, torch.Tensor]:
    """Build prototype embeddings for each class using the hypernetwork's encoder.

    Averages over `n_samples` draws of K prototypes for stability.
    Returns: {class_id: (embed_dim,) tensor}
    """
    proto_encoder.eval()
    class_embeds = {}
    for c in all_classes:
        embeds = []
        for _ in range(n_samples):
            imgs = class_images[c]
            idx = torch.randperm(len(imgs))[:K]
            with torch.no_grad():
                emb = proto_encoder(imgs[idx].to(device))  # (embed_dim,)
            embeds.append(emb)
        class_embeds[c] = torch.stack(embeds).mean(dim=0)  # (embed_dim,)
    return class_embeds


def compute_image_embeddings(
    proto_encoder: nn.Module,
    images: torch.Tensor,
    device: str,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode all images through the prototype encoder (per-image, not pooled).

    The prototype encoder processes (K, 784) -> mean -> (embed_dim,).
    For single images, we feed (1, 784) -> (embed_dim,).
    """
    proto_encoder.eval()
    parts = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        with torch.no_grad():
            # Process each image individually through encoder layers
            emb = proto_encoder.encoder(batch)  # (B, embed_dim)
        parts.append(emb.cpu())
    return torch.cat(parts, dim=0)  # (N, embed_dim)


# ---------------------------------------------------------------------------
# Optional fine-tuning
# ---------------------------------------------------------------------------

def finetune_specialists(
    specialists: List[Tuple[List[int], torch.Tensor]],
    class_images: Dict[int, torch.Tensor],
    target_cfg: TargetMLPConfig,
    device: str,
    steps: int = 10,
    lr: float = 1e-2,
    n_per_class: int = 50,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Fine-tune each specialist on its 3 classes."""
    finetuned = []
    for classes, weights in specialists:
        w = weights.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([w], lr=lr)
        label_map = {c: i for i, c in enumerate(sorted(classes))}
        for _ in range(steps):
            optimizer.zero_grad()
            imgs_list, labels_list = [], []
            for c in sorted(classes):
                pool = class_images[c]
                idx = torch.randperm(len(pool))[:n_per_class]
                imgs_list.append(pool[idx].to(device))
                labels_list.extend([label_map[c]] * len(idx))
            task_images = torch.cat(imgs_list)
            task_labels = torch.tensor(labels_list, dtype=torch.long, device=device)
            logits = differentiable_forward(w, task_images, target_cfg)
            loss = F.cross_entropy(logits, task_labels)
            loss.backward()
            optimizer.step()
        finetuned.append((classes, w.detach()))
    return finetuned


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def evaluate_individual_specialists(
    specialists: List[Tuple[List[int], torch.Tensor]],
    class_images: Dict[int, torch.Tensor],
    target_cfg: TargetMLPConfig,
    device: str,
    n_per_class: int = 200,
) -> List[float]:
    """Evaluate each specialist on its own 3-class task."""
    accs = []
    for classes, weights in specialists:
        label_map = {c: i for i, c in enumerate(sorted(classes))}
        imgs, labels = [], []
        for c in sorted(classes):
            pool = class_images[c]
            n = min(n_per_class, len(pool))
            idx = torch.randperm(len(pool))[:n]
            imgs.append(pool[idx])
            labels.extend([label_map[c]] * n)
        task_imgs = torch.cat(imgs).to(device)
        task_labels = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = differentiable_forward(weights, task_imgs, target_cfg)
            correct = (logits.argmax(1) == task_labels).sum().item()
        accs.append(correct / len(task_labels))
    return accs


# ---------------------------------------------------------------------------
# Ensemble evaluation — multiple aggregation strategies
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    specialists: List[Tuple[List[int], torch.Tensor]],
    all_images: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int,
    target_cfg: TargetMLPConfig,
    device: str,
    image_embeds: torch.Tensor = None,
    class_embeds: Dict[int, torch.Tensor] = None,
    batch_size: int = 2048,
) -> Dict:
    """Evaluate ensemble with multiple aggregation strategies.

    Key strategy: top-K specialist selection via prototype similarity.
    For each test image, find the K nearest class prototypes, then only
    aggregate logits from specialists that cover those candidate classes.
    """
    N = all_images.size(0)
    has_proto = image_embeds is not None and class_embeds is not None

    # Precompute per-image top-K nearest classes via prototype similarity
    topk_indices = {}
    if has_proto:
        embed_dim = next(iter(class_embeds.values())).shape[0]
        class_embed_mat = torch.zeros(num_classes, embed_dim)
        for c, emb in class_embeds.items():
            class_embed_mat[c] = emb.cpu()
        class_embed_norm = F.normalize(class_embed_mat, dim=1)
        img_embed_norm = F.normalize(image_embeds, dim=1)
        all_sims = img_embed_norm @ class_embed_norm.t()  # (N, C)
        for k in [3, 6, 10]:
            topk_indices[k] = all_sims.topk(k, dim=1).indices  # (N, k)

    # Accumulators — ungated
    g_max = torch.full((N, num_classes), -1e9)
    g_sum = torch.zeros(N, num_classes)
    g_cnt = torch.zeros(N, num_classes)

    # Accumulators — top-K gated (only count when class is in image's top-K)
    tk_max = {k: torch.full((N, num_classes), -1e9) for k in topk_indices}
    tk_sum = {k: torch.zeros(N, num_classes) for k in topk_indices}
    tk_cnt = {k: torch.zeros(N, num_classes) for k in topk_indices}

    for classes, weights in specialists:
        sorted_classes = sorted(classes)

        logits_parts = []
        for start in range(0, N, batch_size):
            batch = all_images[start : start + batch_size].to(device)
            with torch.no_grad():
                logits = differentiable_forward(weights, batch, target_cfg)
            logits_parts.append(logits.cpu())
        logits_all = torch.cat(logits_parts, dim=0)  # (N, 3)

        for local_idx, gc in enumerate(sorted_classes):
            lgt = logits_all[:, local_idx]
            g_max[:, gc] = torch.max(g_max[:, gc], lgt)
            g_sum[:, gc] += lgt
            g_cnt[:, gc] += 1

            for k, tk_idx in topk_indices.items():
                mask = (tk_idx == gc).any(dim=1)  # (N,) bool
                tk_max[k][:, gc] = torch.where(mask, torch.max(tk_max[k][:, gc], lgt), tk_max[k][:, gc])
                tk_sum[k][:, gc] += mask.float() * lgt
                tk_cnt[k][:, gc] += mask.float()

    g_mean = g_sum / g_cnt.clamp(min=1)
    g_mean[g_cnt == 0] = -1e9

    strategies = {"max_logit": g_max, "mean_logit": g_mean}
    for k in topk_indices:
        tm = tk_sum[k] / tk_cnt[k].clamp(min=1)
        tm[tk_cnt[k] == 0] = -1e9
        strategies[f"topk_{k}_mean"] = tm
        strategies[f"topk_{k}_max"] = tk_max[k]

    results = {}
    for name, scores in strategies.items():
        r = _score_predictions(scores, all_labels, num_classes)
        results[name] = r
        print(f"          {name}: {r['overall_accuracy']:.4f}")

    best_name = max(results, key=lambda k: results[k]["overall_accuracy"])
    best = results[best_name]
    best["strategy"] = best_name
    best["all_strategies"] = {k: v["overall_accuracy"] for k, v in results.items()}
    return best


def evaluate_nearest_prototype(
    image_embeds: torch.Tensor,
    class_embeds: Dict[int, torch.Tensor],
    all_labels: torch.Tensor,
    num_classes: int,
) -> Dict:
    """Pure nearest-class-mean classifier in prototype embedding space."""
    embed_dim = next(iter(class_embeds.values())).shape[0]
    class_embed_mat = torch.zeros(num_classes, embed_dim)
    for c, emb in class_embeds.items():
        class_embed_mat[c] = emb.cpu()
    class_embed_norm = F.normalize(class_embed_mat, dim=1)
    img_embed_norm = F.normalize(image_embeds, dim=1)
    sims = img_embed_norm @ class_embed_norm.t()
    return _score_predictions(sims, all_labels, num_classes)


def _score_predictions(scores: torch.Tensor, all_labels: torch.Tensor, num_classes: int) -> Dict:
    N = scores.size(0)
    predictions = scores.argmax(dim=1)
    correct = (predictions == all_labels).sum().item()
    per_class, per_class_n = {}, {}
    for c in range(num_classes):
        mask = all_labels == c
        n = mask.sum().item()
        if n > 0:
            per_class[c] = (predictions[mask] == c).float().mean().item()
            per_class_n[c] = n
    return {"overall_accuracy": correct / N, "per_class_accuracy": per_class, "per_class_n": per_class_n, "n_test": N}


# ---------------------------------------------------------------------------
# Baseline MLP
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_hidden_dim(target_params: int, input_dim: int = 784, num_classes: int = 62) -> int:
    H = (target_params - num_classes) / (input_dim + 1 + num_classes)
    return max(1, round(H))


def train_baseline_mlp(
    hidden_dim: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 10,
) -> Dict:
    model = BaselineMLP(784, hidden_dim, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    MLP arch: 784 -> {hidden_dim} -> {num_classes}  ({total_params:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(batch_imgs), batch_labels).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(test_images), 2048):
                b_img = test_images[start : start + 2048].to(device)
                b_lbl = test_labels[start : start + 2048].to(device)
                correct += (model(b_img).argmax(1) == b_lbl).sum().item()
                total += b_lbl.size(0)
        acc = correct / total

        if acc > best_acc:
            best_acc, no_improve = acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  acc={acc:.4f}  best={best_acc:.4f}")
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device).eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_images), 2048):
            all_preds.append(model(test_images[start:start+2048].to(device)).argmax(1).cpu())
    all_preds = torch.cat(all_preds)

    per_class = {}
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            per_class[c] = (all_preds[mask] == c).float().mean().item()

    return {"overall_accuracy": best_acc, "per_class_accuracy": per_class,
            "total_params": total_params, "hidden_dim": hidden_dim}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def class_name(c: int) -> str:
    return EMNIST_BYCLASS_NAMES.get(c, str(c))


def print_summary_table(results: Dict, coverages: List[int], hypernet_params: int):
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print("\nAggregation strategy comparison:")
    strat_names = set()
    for cov in coverages:
        strat_names.update(results[cov]["ensemble"].get("all_strategies", {}).keys())
    strat_names = sorted(strat_names)
    hdr = f"{'Cov':<5}" + "".join(f" {s:<14}" for s in strat_names) + f" {'Best':<14}"
    print(hdr)
    print("-" * len(hdr))
    for cov in coverages:
        strats = results[cov]["ensemble"].get("all_strategies", {})
        best = results[cov]["ensemble"].get("strategy", "?")
        line = f"{cov}x{'':<3}"
        for s in strat_names:
            line += f" {strats.get(s, 0):<14.4f}"
        line += f" {best}"
        print(line)

    print(f"\n{'Cov':<5} {'#Sub':<6} {'Ens Params':<13} {'Ensemble':<10} {'Baseline':<10} {'Random':<10} {'E-B':<8} {'E-R':<8}")
    print("-" * 80)
    for cov in coverages:
        r = results[cov]
        ens = r["ensemble"]["overall_accuracy"]
        base = r["baseline"]["overall_accuracy"]
        rand = r["random"]["overall_accuracy"]
        print(
            f"{cov}x{'':<3} {r['n_subsets']:<6} {r['total_specialist_params']:<13,} "
            f"{ens:<10.4f} {base:<10.4f} {rand:<10.4f} "
            f"{ens - base:+.4f}   {ens - rand:+.4f}"
        )

    print(f"\nHypernetwork params: {hypernet_params:,}")
    print("SOTA reference: ~88.4% (EMNIST ByClass 62cls, WaveMix, Jeevan et al. 2022)")
    print("                ~88.1% (committee of 7 deep CNNs, Ciresan et al. 2011)")
    print("                ~70%   (MLP 10K hidden, Cohen et al. 2017)")


def print_per_class_table(results: Dict, coverage: int, num_classes: int):
    r = results[coverage]
    ens_pc = r["ensemble"]["per_class_accuracy"]
    base_pc = r["baseline"]["per_class_accuracy"]
    rand_pc = r["random"]["per_class_accuracy"]
    ens_n = r["ensemble"].get("per_class_n", {})

    print(f"\nPer-class accuracy ({coverage}x coverage, {r['n_subsets']} subsets):")
    print(f"{'Class':<8} {'Name':<6} {'N_test':<8} {'Ensemble':<10} {'Baseline':<10} {'Random':<10} {'E-B':<8}")
    print("-" * 68)
    for c in range(num_classes):
        if c not in ens_pc:
            continue
        e, b, rd = ens_pc.get(c, 0), base_pc.get(c, 0), rand_pc.get(c, 0)
        print(f"{c:<8} {class_name(c):<6} {ens_n.get(c, 0):<8} {e:<10.4f} {b:<10.4f} {rd:<10.4f} {e - b:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverages", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--finetune", type=int, default=0)
    parser.add_argument("--finetune-lr", type=float, default=1e-2)
    parser.add_argument("--proto-samples", type=int, default=5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--baseline-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = Config()
    device = args.device

    all_classes = sorted(set(cfg.zoo.seen_classes + cfg.zoo.unseen_classes))
    num_classes = max(all_classes) + 1
    print(f"EMNIST ByClass: {len(all_classes)} classes (0-{max(all_classes)})")
    print(f"  Seen (letters): {len(cfg.zoo.seen_classes)} | Unseen (digits): {len(cfg.zoo.unseen_classes)}")

    # Load hypernetwork
    print("\nLoading hypernetwork...")
    hypernet, w_mean, w_std = load_hypernet("data/v4/checkpoints/hypernet.pt", device)
    hypernet_params = sum(p.numel() for p in hypernet.parameters())
    specialist_param_count = cfg.target_weight_dim()
    print(f"  Hypernetwork params: {hypernet_params:,}")
    print(f"  Specialist params (each): {specialist_param_count:,}")

    # Load EMNIST (with caching to avoid 15-min reload)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "class_images.pt"

    if cache_path.exists():
        print("\nLoading cached class image indices...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        class_images_train = cached["train"]
        class_images_test = cached["test"]
    else:
        print("\nLoading EMNIST data (first run, will cache)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
        emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)

        print("  Building class image indices...")
        class_images_train = build_class_image_index(emnist_train)
        class_images_test = build_class_image_index(emnist_test)
        print("  Caching to disk...")
        torch.save({"train": class_images_train, "test": class_images_test}, cache_path)
        print(f"  Cached to {cache_path}")

    data_classes = sorted(set(class_images_train.keys()) & set(class_images_test.keys()))
    all_classes = [c for c in all_classes if c in data_classes]
    print(f"  Classes with data: {len(all_classes)}")

    # Build full tensors
    print("  Building flat tensors...")
    test_imgs, test_lbls = [], []
    for c in sorted(class_images_test.keys()):
        test_imgs.append(class_images_test[c])
        test_lbls.append(torch.full((len(class_images_test[c]),), c, dtype=torch.long))
    all_test_images = torch.cat(test_imgs)
    all_test_labels = torch.cat(test_lbls)

    train_imgs, train_lbls = [], []
    for c in sorted(class_images_train.keys()):
        train_imgs.append(class_images_train[c])
        train_lbls.append(torch.full((len(class_images_train[c]),), c, dtype=torch.long))
    all_train_images = torch.cat(train_imgs)
    all_train_labels = torch.cat(train_lbls)
    print(f"  Train: {len(all_train_labels):,} images | Test: {len(all_test_labels):,} images")

    target_cfg = TargetMLPConfig(
        input_dim=cfg.target.input_dim,
        hidden_dims=cfg.target.hidden_dims,
        num_classes=cfg.target.num_classes,
    )
    K = cfg.hypernet.prototypes_per_class

    # Build prototype embeddings for gating
    print("\n  Building prototype embeddings for gating...")
    proto_encoder = hypernet.prototype_encoder
    class_embeds = build_class_prototypes(
        proto_encoder, class_images_train, all_classes, K, device, n_samples=args.proto_samples,
    )
    print("  Computing test image embeddings...")
    image_embeds = compute_image_embeddings(proto_encoder, all_test_images, device)
    print(f"  Image embeddings: {image_embeds.shape}")

    # Pure nearest-prototype classifier (no specialists, just embedding space)
    print("\n  Nearest-prototype classifier (no specialists)...")
    ncm_result = evaluate_nearest_prototype(image_embeds, class_embeds, all_test_labels, num_classes)
    print(f"  Nearest-prototype accuracy: {ncm_result['overall_accuracy']:.4f}")

    # -----------------------------------------------------------------------
    # Run for each coverage level
    # -----------------------------------------------------------------------
    results = {}

    for coverage in args.coverages:
        print(f"\n{'='*80}")
        print(f"COVERAGE {coverage}x")
        print(f"{'='*80}")

        subsets = generate_covering_subsets(all_classes, coverage, seed=args.seed)
        n_subsets = len(subsets)
        total_specialist_params = n_subsets * specialist_param_count
        stats = coverage_stats(subsets, all_classes)
        print(f"  Subsets: {n_subsets}")
        print(f"  Total specialist params: {total_specialist_params:,}")
        print(f"  Per-class coverage: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")

        # --- Generate specialists ---
        print("\n  [1/4] Generating hypernetwork specialists...")
        t0 = time.time()
        specialists = generate_specialists(
            hypernet, w_mean, w_std, subsets,
            class_images_train, K, device, n_proto_samples=args.proto_samples,
        )
        print(f"        {n_subsets} specialists in {time.time() - t0:.1f}s")

        if args.finetune > 0:
            print(f"        Fine-tuning ({args.finetune} steps)...")
            specialists = finetune_specialists(
                specialists, class_images_train, target_cfg, device,
                steps=args.finetune, lr=args.finetune_lr,
            )

        # --- Diagnostic: individual specialist accuracy ---
        print("\n  [2/4] Diagnostic: individual specialist accuracy...")
        spec_accs = evaluate_individual_specialists(
            specialists, class_images_test, target_cfg, device,
        )
        mean_acc = sum(spec_accs) / len(spec_accs)
        print(f"        mean={mean_acc:.4f}  min={min(spec_accs):.4f}  max={max(spec_accs):.4f}")
        # Show worst 3
        worst_idx = sorted(range(len(spec_accs)), key=lambda i: spec_accs[i])[:3]
        for i in worst_idx:
            cls_str = ", ".join(f"{c}({class_name(c)})" for c in subsets[i])
            print(f"        worst: [{cls_str}] acc={spec_accs[i]:.4f}")

        # --- Ensemble evaluation ---
        print("\n  [3/4] Evaluating ensemble...")
        ens_result = evaluate_ensemble(
            specialists, all_test_images, all_test_labels,
            num_classes, target_cfg, device,
            image_embeds=image_embeds, class_embeds=class_embeds,
        )
        print(f"        Best: {ens_result['overall_accuracy']:.4f} ({ens_result['strategy']})")

        # --- Random ensemble ---
        print("\n  [4/4] Random-weight ensemble (control)...")
        rand_specialists = generate_random_specialists(subsets, specialist_param_count, device)
        rand_result = evaluate_ensemble(
            rand_specialists, all_test_images, all_test_labels,
            num_classes, target_cfg, device,
            image_embeds=image_embeds, class_embeds=class_embeds,
        )
        print(f"        Best: {rand_result['overall_accuracy']:.4f} ({rand_result['strategy']})")

        # --- Baseline MLP ---
        print(f"\n  [5/4] Training baseline MLP (~{total_specialist_params:,} params)...")
        hidden_dim = compute_hidden_dim(total_specialist_params, num_classes=num_classes)
        baseline_result = train_baseline_mlp(
            hidden_dim, all_train_images, all_train_labels,
            all_test_images, all_test_labels, num_classes, device,
            epochs=args.baseline_epochs,
        )
        print(f"        Baseline accuracy: {baseline_result['overall_accuracy']:.4f}")

        results[coverage] = {
            "n_subsets": n_subsets,
            "total_specialist_params": total_specialist_params,
            "ensemble": ens_result,
            "baseline": baseline_result,
            "random": rand_result,
            "specialist_accs": {"mean": mean_acc, "min": min(spec_accs), "max": max(spec_accs)},
        }

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print_summary_table(results, args.coverages, hypernet_params)
    print_per_class_table(results, max(args.coverages), num_classes)

    # Save
    if args.save:
        save_data = {}
        for cov, r in results.items():
            save_data[str(cov)] = {
                "n_subsets": r["n_subsets"],
                "total_specialist_params": r["total_specialist_params"],
                "ensemble_accuracy": r["ensemble"]["overall_accuracy"],
                "ensemble_strategy": r["ensemble"].get("strategy"),
                "ensemble_all_strategies": r["ensemble"].get("all_strategies"),
                "ensemble_per_class": {str(k): v for k, v in r["ensemble"]["per_class_accuracy"].items()},
                "baseline_accuracy": r["baseline"]["overall_accuracy"],
                "baseline_per_class": {str(k): v for k, v in r["baseline"]["per_class_accuracy"].items()},
                "baseline_params": r["baseline"]["total_params"],
                "random_accuracy": r["random"]["overall_accuracy"],
                "specialist_accs": r["specialist_accs"],
            }
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
