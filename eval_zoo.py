"""Evaluate V4 zoo baseline accuracy."""
import sys
from pathlib import Path
import torch
from src.prototype.config import Config
from src.prototype.zoo import TargetMLP, get_emnist_data, make_class_subset
from torch.utils.data import DataLoader


def evaluate_zoo():
    cfg = Config()
    zoo = torch.load(str(Path(cfg.zoo.zoo_dir) / "zoo.pt"), map_location="cpu", weights_only=False)
    _, test_data = get_emnist_data(cfg.zoo.emnist_split)

    all_accs = []
    for split in ["train", "seen_holdout", "unseen"]:
        weights = zoo[split]["weights"]
        classes_list = zoo[split]["classes"]
        split_accs = []
        for w, classes in zip(weights, classes_list):
            model = TargetMLP(cfg.target)
            model.set_flat_weights(w)
            model.eval()
            subset, label_map = make_class_subset(test_data, classes)
            loader = DataLoader(subset, batch_size=256, shuffle=False)
            correct = total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images = images.view(images.size(0), -1)
                    labels = torch.tensor([label_map[l.item()] for l in labels])
                    correct += (model(images).argmax(1) == labels).sum().item()
                    total += labels.size(0)
            split_accs.append(correct / total)
        all_accs.extend(split_accs)
        mean = sum(split_accs) / len(split_accs)
        print(f"  {split:15s} ({len(split_accs):3d} models): mean={mean:.4f}, min={min(split_accs):.4f}")

    overall = sum(all_accs) / len(all_accs)
    passed = overall >= 0.90
    print(f"\n  Overall ({len(all_accs)} models): mean={overall:.4f}")
    print(f"  Exit criterion (>= 90%): {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    evaluate_zoo()
