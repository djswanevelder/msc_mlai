"""
V4 Evaluation — Prototype-conditioned hypernetwork on EMNIST.

Measures:
  1. Single-pass accuracy (no fine-tuning)
  2. Post-fine-tuning accuracy (few gradient steps on generated weights)
  3. Comparison: fine-tuning generated vs random init (same budget)

Usage:
  python eval_meta_v4.py --split unseen --threshold 0.45
  python eval_meta_v4.py --split unseen --finetune 10 --threshold 0.70
  python eval_meta_v4.py --split unseen --finetune 10 --compare-random
  python eval_meta_v4.py --split seen_holdout --threshold 0.70
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.prototype.config import Config
from src.prototype.models import HyperNetwork, PrototypeEncoder, differentiable_forward
from src.prototype.zoo import TargetMLP, make_class_subset
from src.prototype.train import build_class_image_index, sample_prototypes, sample_task_data


def load_hypernet(checkpoint_path: str, device: str) -> Tuple:
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


def evaluate_weights(
    flat_weights: torch.Tensor,
    class_images: Dict[int, torch.Tensor],
    classes: List[int],
    cfg: Config,
    device: str,
    n_per_class: int = 100,
) -> float:
    """Evaluate flat weights on test data for a task."""
    task_images, task_labels = sample_task_data(
        class_images, classes, n_per_class, device
    )
    with torch.no_grad():
        logits = differentiable_forward(flat_weights, task_images, cfg.target)
        correct = (logits.argmax(1) == task_labels).sum().item()
    return correct / task_labels.size(0)


def finetune_weights(
    flat_weights: torch.Tensor,
    class_images: Dict[int, torch.Tensor],
    classes: List[int],
    cfg: Config,
    device: str,
    steps: int = 10,
    lr: float = 1e-2,
    n_per_class: int = 50,
) -> torch.Tensor:
    """Fine-tune generated weights with a few gradient steps."""
    w = flat_weights.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([w], lr=lr)

    for step in range(steps):
        task_images, task_labels = sample_task_data(
            class_images, classes, n_per_class, device
        )
        optimizer.zero_grad()
        logits = differentiable_forward(w, task_images, cfg.target)
        loss = F.cross_entropy(logits, task_labels)
        loss.backward()
        optimizer.step()

    return w.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="unseen", choices=["train", "seen_holdout", "unseen"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--finetune", type=int, default=0)
    parser.add_argument("--compare-random", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    device = cfg.hypernet.device

    zoo = torch.load(str(Path(cfg.zoo.zoo_dir) / "zoo.pt"), map_location="cpu", weights_only=False)
    hypernet, w_mean, w_std = load_hypernet("data/v4/checkpoints/hypernet.pt", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
    class_images_test = build_class_image_index(emnist_test)

    # Also use train split for fine-tuning data (separate from eval)
    emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    class_images_train = build_class_image_index(emnist_train)

    classes_list = zoo[args.split]["classes"]
    zoo_accs = zoo[args.split]["accuracies"]
    K = cfg.hypernet.prototypes_per_class

    results = []
    for i, classes in enumerate(classes_list):
        # Generate weights using prototypes from TRAIN split
        prototypes = sample_prototypes(class_images_train, classes, K, device)
        with torch.no_grad():
            gen_w_norm = hypernet(prototypes)
            gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

        # Evaluate single-pass on TEST split
        single_acc = evaluate_weights(gen_w, class_images_test, classes, cfg, device)

        result = {
            "classes": classes,
            "single_pass_accuracy": single_acc,
            "zoo_accuracy": zoo_accs[i],
        }

        if args.finetune > 0:
            # Fine-tune on TRAIN data, evaluate on TEST data
            ft_w = finetune_weights(
                gen_w, class_images_train, classes, cfg, device,
                steps=args.finetune, lr=cfg.hypernet.finetune_lr
            )
            ft_acc = evaluate_weights(ft_w, class_images_test, classes, cfg, device)
            result["finetune_accuracy"] = ft_acc

            if args.compare_random:
                # Random init baseline: same architecture, same fine-tuning budget
                rand_w = torch.randn_like(gen_w) * 0.01
                rand_ft_w = finetune_weights(
                    rand_w, class_images_train, classes, cfg, device,
                    steps=args.finetune, lr=cfg.hypernet.finetune_lr
                )
                rand_ft_acc = evaluate_weights(rand_ft_w, class_images_test, classes, cfg, device)
                result["random_finetune_accuracy"] = rand_ft_acc

        results.append(result)

    # Summary
    single_accs = [r["single_pass_accuracy"] for r in results]
    mean_single = sum(single_accs) / len(single_accs)
    mean_zoo = sum(r["zoo_accuracy"] for r in results) / len(results)

    print(f"\nMeta-Model Evaluation ({args.split} split, {len(results)} subsets)")
    print(f"  Zoo baseline:     mean={mean_zoo:.4f}")
    print(f"  Single-pass:      mean={mean_single:.4f}, min={min(single_accs):.4f}, max={max(single_accs):.4f}")

    if args.finetune > 0:
        ft_accs = [r["finetune_accuracy"] for r in results]
        mean_ft = sum(ft_accs) / len(ft_accs)
        print(f"  After {args.finetune} ft steps: mean={mean_ft:.4f}, min={min(ft_accs):.4f}, max={max(ft_accs):.4f}")

        if args.compare_random:
            rand_accs = [r["random_finetune_accuracy"] for r in results]
            mean_rand = sum(rand_accs) / len(rand_accs)
            print(f"  Random init + ft:  mean={mean_rand:.4f}, min={min(rand_accs):.4f}, max={max(rand_accs):.4f}")
            print(f"\n  Generated ft vs Random ft: {mean_ft:.4f} vs {mean_rand:.4f} (delta={mean_ft - mean_rand:+.4f})")
            passed = mean_ft > mean_rand
            print(f"  Structure test: {'PASS' if passed else 'FAIL'} (generated > random)")

    # Per-subset details (top 10 + bottom 5)
    key = "finetune_accuracy" if args.finetune > 0 else "single_pass_accuracy"
    sorted_results = sorted(results, key=lambda x: x[key])

    header = f"  {'Classes':<20} {'Single':>8} {'Zoo':>8}"
    if args.finetune > 0:
        header += f" {'FT({args.finetune})':>8}"
        if args.compare_random:
            header += f" {'Rand FT':>8}"
    print(f"\n{header}")
    print(f"  {'-'*60}")

    for r in sorted_results[:5]:
        line = f"  {str(r['classes']):<20} {r['single_pass_accuracy']:>8.4f} {r['zoo_accuracy']:>8.4f}"
        if args.finetune > 0:
            line += f" {r['finetune_accuracy']:>8.4f}"
            if args.compare_random:
                line += f" {r['random_finetune_accuracy']:>8.4f}"
        print(line)
    if len(sorted_results) > 10:
        print(f"  {'...':<20}")
    for r in sorted_results[-5:]:
        line = f"  {str(r['classes']):<20} {r['single_pass_accuracy']:>8.4f} {r['zoo_accuracy']:>8.4f}"
        if args.finetune > 0:
            line += f" {r['finetune_accuracy']:>8.4f}"
            if args.compare_random:
                line += f" {r['random_finetune_accuracy']:>8.4f}"
        print(line)

    if args.threshold is not None:
        check_val = mean_ft if args.finetune > 0 else mean_single
        passed = check_val >= args.threshold
        print(f"\n  Exit criterion (>= {args.threshold:.0%}): {'PASS' if passed else 'FAIL'} ({check_val:.4f})")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
