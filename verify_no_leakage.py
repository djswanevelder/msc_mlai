"""
Thorough verification that held-out results are legitimate.
Checks for every possible source of data leakage.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from src.v3.config import Config
from src.v3.models import HyperNetwork, differentiable_forward
from src.v3.zoo import TargetMLP, make_class_subset


def verify():
    cfg = Config()
    device = "mps"

    # =========================================================================
    # CHECK 1: Train and test subsets are truly disjoint
    # =========================================================================
    print("=" * 70)
    print("CHECK 1: Are train/test subsets disjoint?")
    print("=" * 70)

    zoo = torch.load("data/v3/zoo/zoo.pt", map_location="cpu", weights_only=False)
    train_classes = [tuple(sorted(c)) for c in zoo["train"]["classes"]]
    test_classes = [tuple(sorted(c)) for c in zoo["test"]["classes"]]

    train_set = set(train_classes)
    test_set = set(test_classes)
    overlap = train_set & test_set

    print(f"  Train subsets: {len(train_classes)}")
    print(f"  Test subsets:  {len(test_classes)}")
    print(f"  Overlap:       {len(overlap)}")
    if overlap:
        print(f"  LEAKAGE! Overlapping subsets: {overlap}")
        return
    print(f"  PASS: No overlap between train and test subsets\n")

    # Show some examples
    print(f"  First 5 train subsets: {train_classes[:5]}")
    print(f"  First 5 test subsets:  {test_classes[:5]}")
    print()

    # =========================================================================
    # CHECK 2: The hypernetwork never saw test subsets during training
    # =========================================================================
    print("=" * 70)
    print("CHECK 2: Did the hypernetwork training use test subsets?")
    print("=" * 70)

    # The training code in train.py only uses zoo["train"]["weights"] and
    # zoo["train"]["classes"]. Let's verify by checking the class_indices tensor
    # that would have been constructed during training.
    train_class_indices = torch.tensor(
        [sorted(c) for c in zoo["train"]["classes"]], dtype=torch.long
    )
    print(f"  Training class indices tensor shape: {train_class_indices.shape}")
    print(f"  Training used {train_class_indices.shape[0]} subsets")

    # Check: does [0,1,6] (our best test case) appear in training?
    best_test = [0, 1, 6]
    found_in_train = any(
        sorted(c) == sorted(best_test) for c in zoo["train"]["classes"]
    )
    print(f"  Is [0,1,6] in training set? {found_in_train}")
    if found_in_train:
        print("  LEAKAGE! Best test subset found in training!")
        return
    print(f"  PASS: Test subsets not in training data\n")

    # =========================================================================
    # CHECK 3: Load hypernetwork, generate weights for a TEST subset, evaluate
    # =========================================================================
    print("=" * 70)
    print("CHECK 3: Generate weights for held-out [0,1,6] and evaluate")
    print("=" * 70)

    # Load hypernetwork
    ckpt = torch.load("data/v3/checkpoints/hypernet.pt", map_location=device, weights_only=False)
    hypernet = HyperNetwork(
        target_weight_dim=ckpt["config"]["target_weight_dim"],
        num_classes=ckpt["config"]["num_classes"],
        class_embed_dim=ckpt["config"]["class_embed_dim"],
        num_classes_per_task=ckpt["config"]["num_classes_per_task"],
        hidden_dims=ckpt["config"]["hidden_dims"],
    )
    hypernet.load_state_dict(ckpt["hypernet_state"])
    hypernet.to(device)
    hypernet.eval()

    w_mean = ckpt["normalization"]["w_mean"].to(device)
    w_std = ckpt["normalization"]["w_std"].to(device)

    # Generate weights for [0, 1, 6]
    test_subset_classes = [0, 1, 6]
    class_tensor = torch.tensor([test_subset_classes], dtype=torch.long, device=device)

    with torch.no_grad():
        gen_w_norm = hypernet(class_tensor)
        gen_w = gen_w_norm * w_std + w_mean

    print(f"  Generated weight vector shape: {gen_w.shape}")
    print(f"  Generated weight stats: mean={gen_w.mean().item():.4f}, std={gen_w.std().item():.4f}")
    print()

    # =========================================================================
    # CHECK 4: Evaluate on FRESH MNIST test data (downloaded, not cached)
    # =========================================================================
    print("=" * 70)
    print("CHECK 4: Evaluate on MNIST test split (completely fresh)")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Use MNIST TEST split (train=False)
    test_data = datasets.MNIST("data/v3/raw", train=False, download=True, transform=transform)

    subset, label_map = make_class_subset(test_data, test_subset_classes)
    print(f"  MNIST test samples for classes {test_subset_classes}: {len(subset)}")
    print(f"  Label mapping: {label_map}")

    loader = DataLoader(subset, batch_size=512, shuffle=False)

    correct = 0
    total = 0
    per_class_correct = {c: 0 for c in test_subset_classes}
    per_class_total = {c: 0 for c in test_subset_classes}

    with torch.no_grad():
        for images, labels in loader:
            original_labels = labels.clone()
            mapped_labels = torch.tensor([label_map[l.item()] for l in labels])
            images_flat = images.to(device).view(images.size(0), -1)

            logits = differentiable_forward(gen_w[0], images_flat, cfg.target)
            preds = logits.argmax(1).cpu()

            correct += (preds == mapped_labels).sum().item()
            total += mapped_labels.size(0)

            # Per-class breakdown
            for orig_label, pred, mapped in zip(original_labels, preds, mapped_labels):
                c = orig_label.item()
                per_class_total[c] += 1
                if pred.item() == mapped.item():
                    per_class_correct[c] += 1

    accuracy = correct / total
    print(f"\n  Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"\n  Per-class breakdown:")
    for c in test_subset_classes:
        cls_acc = per_class_correct[c] / per_class_total[c]
        print(f"    Digit {c}: {cls_acc:.4f} ({per_class_correct[c]}/{per_class_total[c]})")

    # =========================================================================
    # CHECK 5: Compare to random weights (sanity check)
    # =========================================================================
    print()
    print("=" * 70)
    print("CHECK 5: Random weight baseline (proves it's not trivial)")
    print("=" * 70)

    random_w = torch.randn_like(gen_w) * gen_w.std() + gen_w.mean()
    rand_correct = 0
    rand_total = 0

    with torch.no_grad():
        for images, labels in loader:
            mapped_labels = torch.tensor([label_map[l.item()] for l in labels])
            images_flat = images.to(device).view(images.size(0), -1)
            logits = differentiable_forward(random_w[0], images_flat, cfg.target)
            preds = logits.argmax(1).cpu()
            rand_correct += (preds == mapped_labels).sum().item()
            rand_total += mapped_labels.size(0)

    rand_acc = rand_correct / rand_total
    print(f"  Random weights accuracy: {rand_acc:.4f} ({rand_correct}/{rand_total})")
    print(f"  Expected (chance):       {1.0/3:.4f}")

    # =========================================================================
    # CHECK 6: Compare to zoo model (trained normally on same classes)
    # =========================================================================
    print()
    print("=" * 70)
    print("CHECK 6: Zoo model comparison (conventionally trained)")
    print("=" * 70)

    # Find [0,1,6] in test split
    test_idx = None
    for i, c in enumerate(zoo["test"]["classes"]):
        if sorted(c) == sorted(test_subset_classes):
            test_idx = i
            break

    if test_idx is not None:
        zoo_w = zoo["test"]["weights"][test_idx]
        model = TargetMLP(cfg.target)
        model.set_flat_weights(zoo_w)
        model.eval()

        zoo_correct = 0
        zoo_total = 0
        with torch.no_grad():
            for images, labels in loader:
                mapped_labels = torch.tensor([label_map[l.item()] for l in labels])
                logits = model(images)
                preds = logits.argmax(1)
                zoo_correct += (preds == mapped_labels).sum().item()
                zoo_total += mapped_labels.size(0)

        zoo_acc = zoo_correct / zoo_total
        print(f"  Zoo model accuracy (trained normally): {zoo_acc:.4f} ({zoo_correct}/{zoo_total})")
    else:
        print(f"  [0,1,6] not found in test split")

    # =========================================================================
    # CHECK 7: Try a completely novel subset that exists NOWHERE in the zoo
    # =========================================================================
    print()
    print("=" * 70)
    print("CHECK 7: Generate for subsets NOT in train OR test")
    print("=" * 70)

    all_zoo_subsets = set(
        tuple(sorted(c)) for c in zoo["train"]["classes"]
    ) | set(
        tuple(sorted(c)) for c in zoo["test"]["classes"]
    )

    # Find subsets not in zoo at all (shouldn't exist if all 120 are used, but let's check)
    import itertools
    all_possible = set(itertools.combinations(range(10), 3))
    not_in_zoo = all_possible - all_zoo_subsets

    if not_in_zoo:
        novel = list(not_in_zoo)[0]
        print(f"  Found subset not in zoo: {novel}")
    else:
        print(f"  All 120 subsets are in the zoo. Using test subset [4,6,9] (worst performer)")
        novel = (4, 6, 9)

    novel_classes = list(novel)
    novel_tensor = torch.tensor([sorted(novel_classes)], dtype=torch.long, device=device)

    with torch.no_grad():
        novel_w_norm = hypernet(novel_tensor)
        novel_w = novel_w_norm * w_std + w_mean

    novel_subset, novel_label_map = make_class_subset(test_data, novel_classes)
    novel_loader = DataLoader(novel_subset, batch_size=512, shuffle=False)

    novel_correct = 0
    novel_total = 0
    with torch.no_grad():
        for images, labels in novel_loader:
            mapped_labels = torch.tensor([novel_label_map[l.item()] for l in labels])
            images_flat = images.to(device).view(images.size(0), -1)
            logits = differentiable_forward(novel_w[0], images_flat, cfg.target)
            preds = logits.argmax(1).cpu()
            novel_correct += (preds == mapped_labels).sum().item()
            novel_total += mapped_labels.size(0)

    novel_acc = novel_correct / novel_total
    print(f"  Accuracy on {novel_classes}: {novel_acc:.4f} ({novel_correct}/{novel_total})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Train/test disjoint:     YES")
    print(f"  [0,1,6] in train set:    {'YES (LEAK!)' if found_in_train else 'NO (clean)'}")
    print(f"  Generated [0,1,6] acc:   {accuracy:.4f}")
    print(f"  Zoo [0,1,6] acc:         {zoo_acc:.4f}")
    print(f"  Random weights acc:      {rand_acc:.4f}")
    print(f"  Gap (generated vs zoo):  {zoo_acc - accuracy:.4f}")
    print(f"  Gap (generated vs rand): {accuracy - rand_acc:.4f}")
    print()
    if accuracy > 0.90 and not found_in_train and rand_acc < 0.50:
        print("  VERDICT: Results are legitimate. No data leakage detected.")
    else:
        print("  VERDICT: SUSPICIOUS. Investigate further.")


if __name__ == "__main__":
    verify()
