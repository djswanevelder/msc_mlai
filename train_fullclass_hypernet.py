"""
Train a hypernetwork on ALL 62 EMNIST ByClass classes.

The original hypernetwork only trained on letter classes (10-61).
This retrains with 3-class tasks drawn from ALL classes (0-61),
so the hypernetwork learns to generate specialists for any class combo
including digits.

Steps:
  1. Generate a new model zoo with 3-class tasks from all 62 classes
  2. Train a new hypernetwork on this zoo
  3. Save checkpoint for evaluation

Usage:
  python train_fullclass_hypernet.py
"""

import itertools
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from src.prototype.config import Config, TargetMLPConfig
from src.prototype.models import HyperNetwork, differentiable_forward
from src.prototype.train import build_class_image_index, sample_prototypes, sample_task_data
from src.prototype.zoo import TargetMLP, _train_one_subset

SAVE_DIR = Path("data/v4_fullclass")


def generate_fullclass_zoo(cfg: Config, num_train: int = 300, num_val: int = 50, seed: int = 42):
    """Generate a zoo with 3-class tasks from ALL 62 classes."""
    zoo_dir = SAVE_DIR / "zoo"
    zoo_dir.mkdir(parents=True, exist_ok=True)

    all_classes = list(range(62))
    rng = random.Random(seed)

    # Generate random 3-class subsets from ALL classes
    all_combos = list(itertools.combinations(all_classes, 3))
    rng.shuffle(all_combos)
    all_combos = [list(c) for c in all_combos]

    train_subsets = all_combos[:num_train]
    val_subsets = all_combos[num_train:num_train + num_val]

    ref = TargetMLP(cfg.target)
    total_params = ref.total_params()
    print(f"Target MLP: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Parameters per model: {total_params:,}")
    print(f"Tasks: {num_train} train + {num_val} val = {num_train + num_val} total")

    all_tasks = train_subsets + val_subsets

    worker_fn = partial(
        _train_one_subset,
        input_dim=cfg.target.input_dim,
        hidden_dims=cfg.target.hidden_dims,
        num_classes=cfg.target.num_classes,
        train_epochs=cfg.zoo.train_epochs,
        lr=cfg.zoo.lr,
        batch_size=cfg.zoo.batch_size,
    )

    num_workers = min(mp.cpu_count(), 8)
    print(f"Using {num_workers} parallel workers")

    results = []
    with mp.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(worker_fn, all_tasks), total=len(all_tasks), desc="Training zoo"):
            results.append(result)

    train_results = results[:num_train]
    val_results = results[num_train:]

    for name, res in [("Train", train_results), ("Val", val_results)]:
        accs = [r["test_accuracy"] for r in res]
        print(f"  {name}: mean={sum(accs)/len(accs):.4f}, min={min(accs):.4f}, max={max(accs):.4f}")

    def pack(res_list):
        return {
            "weights": torch.stack([r["flat_weights"] for r in res_list]),
            "classes": [r["classes"] for r in res_list],
            "accuracies": [r["test_accuracy"] for r in res_list],
        }

    save_data = {
        "train": pack(train_results),
        "val": pack(val_results),
        "architecture": {
            "input_dim": cfg.target.input_dim,
            "hidden_dims": cfg.target.hidden_dims,
            "num_classes": cfg.target.num_classes,
            "total_params": total_params,
        },
        "all_classes": all_classes,
    }

    save_path = zoo_dir / "zoo.pt"
    torch.save(save_data, save_path)
    print(f"Zoo saved to {save_path}")
    return str(zoo_dir)


def train_fullclass_hypernetwork(cfg: Config, zoo_path: str, num_epochs: int = 300, patience: int = 60):
    """Train hypernetwork on the full-class zoo."""
    device = cfg.hypernet.device

    print("Loading model zoo...")
    zoo = torch.load(zoo_path, map_location=device, weights_only=False)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    arch = zoo["architecture"]
    weight_dim = arch["total_params"]

    # Normalize weights
    w_mean = train_weights.mean(dim=0, keepdim=True)
    w_std = train_weights.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_weights_norm = (train_weights - w_mean) / w_std

    # Build image index
    print("Building class image index...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
    class_images_train = build_class_image_index(emnist_train)
    class_images_test = build_class_image_index(emnist_test)

    n_train = len(train_classes)
    K = cfg.hypernet.prototypes_per_class
    print(f"Zoo: {n_train} training tasks, weight dim = {weight_dim}")

    # Create hypernetwork (same architecture as original)
    hypernet = HyperNetwork(
        target_weight_dim=weight_dim,
        num_classes_per_task=cfg.hypernet.num_classes_per_task,
        input_dim=cfg.target.input_dim,
        prototype_encoder_hidden=cfg.hypernet.prototype_encoder_hidden,
        prototype_dim=cfg.hypernet.prototype_dim,
        hidden_dims=cfg.hypernet.hidden_dims,
    ).to(device)

    total_params = sum(p.numel() for p in hypernet.parameters())
    print(f"HyperNetwork parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        hypernet.parameters(), lr=cfg.hypernet.lr, weight_decay=cfg.hypernet.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print(f"\nTraining HyperNetwork on ALL 62 classes")
    print(f"  Epochs: {num_epochs} | Patience: {patience} | K: {K}")

    for epoch in range(1, num_epochs + 1):
        hypernet.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        perm = torch.randperm(n_train)
        batch_size = cfg.hypernet.batch_size

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                target_w_norm = train_weights_norm[idx]

                prototypes = sample_prototypes(class_images_train, classes, K, device)
                task_images, task_labels = sample_task_data(
                    class_images_train, classes,
                    cfg.hypernet.num_train_samples // cfg.hypernet.num_classes_per_task, device,
                )

                gen_w_norm = hypernet(prototypes)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                logits = differentiable_forward(gen_w, task_images, cfg.target)
                func_loss = F.cross_entropy(logits, task_labels)
                mse_loss = F.mse_loss(gen_w_norm, target_w_norm)

                loss = cfg.hypernet.functional_loss_weight * func_loss + cfg.hypernet.weight_mse_weight * mse_loss
                batch_loss = batch_loss + loss

                with torch.no_grad():
                    epoch_correct += (logits.argmax(1) == task_labels).sum().item()
                    epoch_total += task_labels.size(0)

            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

        scheduler.step()
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        # Validation: accuracy on validation tasks using TEST images
        hypernet.eval()
        val_correct = 0
        val_total = 0
        val_classes = zoo["val"]["classes"]
        with torch.no_grad():
            for idx in range(len(val_classes)):
                classes = val_classes[idx]
                prototypes = sample_prototypes(class_images_test, classes, K, device)
                task_images, task_labels = sample_task_data(class_images_test, classes, 50, device)
                gen_w_norm = hypernet(prototypes)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                logits = differentiable_forward(gen_w, task_images, cfg.target)
                val_correct += (logits.argmax(1) == task_labels).sum().item()
                val_total += task_labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {epoch_loss/n_batches:.4f} | "
                f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                f"Best: {best_val_acc:.4f}",
                flush=True,
            )

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        hypernet.load_state_dict(best_state)
    hypernet.to(device)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Save
    save_dir = SAVE_DIR / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "hypernet.pt"
    torch.save({
        "hypernet_state": hypernet.state_dict(),
        "config": {
            "target_weight_dim": weight_dim,
            "num_classes_per_task": cfg.hypernet.num_classes_per_task,
            "input_dim": cfg.target.input_dim,
            "prototype_encoder_hidden": cfg.hypernet.prototype_encoder_hidden,
            "prototype_dim": cfg.hypernet.prototype_dim,
            "hidden_dims": cfg.hypernet.hidden_dims,
        },
        "normalization": {"w_mean": w_mean.cpu(), "w_std": w_std.cpu()},
    }, save_path)
    print(f"Saved to {save_path}")
    return hypernet


def main():
    cfg = Config()

    # Step 1: Generate zoo with all 62 classes
    print("=" * 60)
    print("STEP 1: Generate full-class zoo")
    print("=" * 60)
    zoo_dir = generate_fullclass_zoo(cfg, num_train=300, num_val=50)

    # Step 2: Train hypernetwork
    print("\n" + "=" * 60)
    print("STEP 2: Train hypernetwork on full-class zoo")
    print("=" * 60)
    zoo_path = str(Path(zoo_dir) / "zoo.pt")
    train_fullclass_hypernetwork(cfg, zoo_path)


if __name__ == "__main__":
    main()
