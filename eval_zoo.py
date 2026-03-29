"""
Evaluate model zoo baseline accuracy.
Exit criterion: mean test accuracy >= 95% across all zoo models.
"""

import sys
from pathlib import Path

import torch

from src.v3.config import Config
from src.v3.zoo import TargetMLP, get_mnist_data, make_class_subset
from torch.utils.data import DataLoader


def evaluate_zoo(zoo_path: str = None) -> dict:
    cfg = Config()
    if zoo_path is None:
        zoo_path = str(Path(cfg.zoo.zoo_dir) / "zoo.pt")

    zoo = torch.load(zoo_path, map_location="cpu", weights_only=False)

    _, test_data = get_mnist_data()
    results = {"train": [], "test": []}

    for split in ["train", "test"]:
        weights = zoo[split]["weights"]
        classes_list = zoo[split]["classes"]

        for i, (w, classes) in enumerate(zip(weights, classes_list)):
            model = TargetMLP(cfg.target)
            model.set_flat_weights(w)
            model.eval()

            subset, label_map = make_class_subset(test_data, classes)
            loader = DataLoader(subset, batch_size=256, shuffle=False)

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    labels = torch.tensor([label_map[l.item()] for l in labels])
                    logits = model(images)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            results[split].append({"classes": classes, "accuracy": acc})

    # Summary
    train_accs = [r["accuracy"] for r in results["train"]]
    test_accs = [r["accuracy"] for r in results["test"]]
    all_accs = train_accs + test_accs

    print(f"Zoo Evaluation")
    print(f"  Train split ({len(train_accs)} models): mean={sum(train_accs)/len(train_accs):.4f}, min={min(train_accs):.4f}")
    print(f"  Test split  ({len(test_accs)} models):  mean={sum(test_accs)/len(test_accs):.4f}, min={min(test_accs):.4f}")
    print(f"  Overall     ({len(all_accs)} models):  mean={sum(all_accs)/len(all_accs):.4f}, min={min(all_accs):.4f}")

    mean_acc = sum(all_accs) / len(all_accs)
    passed = mean_acc >= 0.95
    print(f"\n  Exit criterion (mean >= 95%): {'PASS' if passed else 'FAIL'} ({mean_acc:.4f})")

    return {"mean_accuracy": mean_acc, "passed": passed, "results": results}


if __name__ == "__main__":
    result = evaluate_zoo()
    sys.exit(0 if result["passed"] else 1)
