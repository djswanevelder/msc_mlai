"""
Evaluate meta-model (hypernetwork) on held-out class subsets.
Generates weights in a single forward pass and measures accuracy.

Usage:
    python eval_meta.py --split test                    # evaluate on held-out subsets
    python eval_meta.py --split test --threshold 0.90   # check >=90% accuracy
    python eval_meta.py --split train                   # evaluate on training subsets
    python eval_meta.py --check-single-pass             # verify single forward pass
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from src.v3.config import Config
from src.v3.models import HyperNetwork, differentiable_forward
from src.v3.zoo import TargetMLP, get_mnist_data, make_class_subset
from torch.utils.data import DataLoader


def load_hypernet(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load trained hypernetwork and normalization stats."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hypernet = HyperNetwork(
        target_weight_dim=data["config"]["target_weight_dim"],
        num_classes=data["config"]["num_classes"],
        class_embed_dim=data["config"]["class_embed_dim"],
        num_classes_per_task=data["config"]["num_classes_per_task"],
        hidden_dims=data["config"]["hidden_dims"],
    )
    hypernet.load_state_dict(data["hypernet_state"])
    hypernet.to(device)
    hypernet.eval()

    w_mean = data["normalization"]["w_mean"].to(device)
    w_std = data["normalization"]["w_std"].to(device)

    return hypernet, w_mean, w_std


def evaluate_split(
    split: str = "test",
    threshold: float = None,
    device: str = "mps",
) -> dict:
    cfg = Config()
    zoo_path = str(Path(cfg.zoo.zoo_dir) / "zoo.pt")
    checkpoint_path = "data/v3/checkpoints/hypernet.pt"

    # Load
    zoo = torch.load(zoo_path, map_location="cpu", weights_only=False)
    hypernet, w_mean, w_std = load_hypernet(checkpoint_path, device)

    classes_list = zoo[split]["classes"]
    zoo_accs = zoo[split]["accuracies"]

    _, test_data = get_mnist_data()

    results = []
    for i, classes in enumerate(classes_list):
        # Generate weights
        class_tensor = torch.tensor([sorted(classes)], dtype=torch.long, device=device)
        with torch.no_grad():
            gen_w_norm = hypernet(class_tensor)
            gen_w = gen_w_norm * w_std + w_mean

        # Evaluate on test data
        subset, label_map = make_class_subset(test_data, classes)
        loader = DataLoader(subset, batch_size=512, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                labels = torch.tensor([label_map[l.item()] for l in labels])
                images = images.to(device).view(images.size(0), -1)
                logits = differentiable_forward(gen_w[0], images, cfg.target)
                correct += (logits.argmax(1) == labels.to(device)).sum().item()
                total += labels.size(0)

        acc = correct / total
        results.append({
            "classes": classes,
            "generated_accuracy": acc,
            "zoo_accuracy": zoo_accs[i],
        })

    # Summary
    gen_accs = [r["generated_accuracy"] for r in results]
    zoo_accs_list = [r["zoo_accuracy"] for r in results]
    mean_gen = sum(gen_accs) / len(gen_accs)
    mean_zoo = sum(zoo_accs_list) / len(zoo_accs_list)

    print(f"\nMeta-Model Evaluation ({split} split, {len(results)} subsets)")
    print(f"  Zoo baseline:     mean={mean_zoo:.4f}")
    print(f"  Generated models: mean={mean_gen:.4f}, min={min(gen_accs):.4f}, max={max(gen_accs):.4f}")

    # Per-subset details
    print(f"\n  {'Classes':<15} {'Generated':>10} {'Zoo':>10}")
    print(f"  {'-'*35}")
    for r in sorted(results, key=lambda x: x["generated_accuracy"]):
        cls_str = str(r["classes"])
        print(f"  {cls_str:<15} {r['generated_accuracy']:>10.4f} {r['zoo_accuracy']:>10.4f}")

    if threshold is not None:
        passed = mean_gen >= threshold
        print(f"\n  Exit criterion (mean >= {threshold:.0%}): {'PASS' if passed else 'FAIL'} ({mean_gen:.4f})")
        return {"mean_accuracy": mean_gen, "passed": passed, "results": results}

    return {"mean_accuracy": mean_gen, "results": results}


def check_single_pass(device: str = "mps") -> dict:
    """Verify that weight generation is a single forward pass."""
    cfg = Config()
    checkpoint_path = "data/v3/checkpoints/hypernet.pt"
    hypernet, w_mean, w_std = load_hypernet(checkpoint_path, device)

    class_tensor = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

    # Time it
    start = time.time()
    with torch.no_grad():
        gen_w = hypernet(class_tensor) * w_std + w_mean
    elapsed = time.time() - start

    print(f"Single forward pass: {elapsed*1000:.2f}ms")
    print(f"Output shape: {gen_w.shape}")
    passed = elapsed < 1.0
    print(f"Exit criterion (< 1s): {'PASS' if passed else 'FAIL'}")
    return {"single_pass": True, "time_ms": elapsed * 1000, "passed": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--check-single-pass", action="store_true")
    args = parser.parse_args()

    if args.check_single_pass:
        result = check_single_pass()
    else:
        result = evaluate_split(split=args.split, threshold=args.threshold)

    if "passed" in result:
        sys.exit(0 if result["passed"] else 1)
