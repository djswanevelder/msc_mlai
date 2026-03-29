"""
V4 Training — Prototype-Conditioned Hypernetwork
==================================================
Conditions on actual images, not class IDs.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from src.v4.config import Config
from src.v4.models import HyperNetwork, differentiable_forward


def load_zoo(zoo_path: str, device: str = "cpu") -> Dict:
    return torch.load(zoo_path, map_location=device, weights_only=False)


def build_class_image_index(
    dataset: datasets.EMNIST,
) -> Dict[int, torch.Tensor]:
    """Build index: class -> tensor of all flattened images for that class."""
    class_images = {}
    for i, (img, label) in enumerate(dataset):
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(img.view(-1))

    return {k: torch.stack(v) for k, v in class_images.items()}


def sample_prototypes(
    class_images: Dict[int, torch.Tensor],
    classes: List[int],
    k: int,
    device: str,
) -> List[torch.Tensor]:
    """Sample K prototype images per class."""
    prototypes = []
    for c in sorted(classes):
        imgs = class_images[c]
        indices = torch.randperm(len(imgs))[:k]
        prototypes.append(imgs[indices].to(device))
    return prototypes


def sample_task_data(
    class_images: Dict[int, torch.Tensor],
    classes: List[int],
    n_per_class: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample training/eval data for a task."""
    label_map = {c: i for i, c in enumerate(sorted(classes))}
    all_images = []
    all_labels = []
    for c in sorted(classes):
        imgs = class_images[c]
        indices = torch.randperm(len(imgs))[:n_per_class]
        all_images.append(imgs[indices])
        all_labels.extend([label_map[c]] * len(indices))
    return (
        torch.cat(all_images).to(device),
        torch.tensor(all_labels, dtype=torch.long, device=device),
    )


def train_hypernetwork(cfg: Config, zoo_path: Optional[str] = None) -> HyperNetwork:
    device = cfg.hypernet.device
    if zoo_path is None:
        zoo_path = str(Path(cfg.zoo.zoo_dir) / "zoo.pt")

    print("Loading model zoo...")
    zoo = load_zoo(zoo_path)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    arch = zoo["architecture"]
    weight_dim = arch["total_params"]

    # Normalize weights
    w_mean = train_weights.mean(dim=0, keepdim=True)
    w_std = train_weights.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_weights_norm = (train_weights - w_mean) / w_std

    # Build image index from EMNIST train split
    print("Building class image index...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
    class_images_train = build_class_image_index(emnist_train)

    # Also build from test split for validation
    emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
    class_images_test = build_class_image_index(emnist_test)

    print(f"Zoo: {len(train_classes)} training tasks, weight dim = {weight_dim}")

    # Create hypernetwork
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
        hypernet.parameters(), lr=cfg.hypernet.lr, weight_decay=cfg.hypernet.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.hypernet.num_epochs)

    n_train = len(train_classes)
    K = cfg.hypernet.prototypes_per_class
    history = {"train_func": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training Prototype-Conditioned HyperNetwork")
    print(f"  Epochs: {cfg.hypernet.num_epochs} | Patience: {cfg.hypernet.patience}")
    print(f"  Prototypes per class: {K}")
    print(f"  Batch size: {cfg.hypernet.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.hypernet.num_epochs + 1):
        hypernet.train()
        epoch_func = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        perm = torch.randperm(n_train)

        for start in range(0, n_train, cfg.hypernet.batch_size):
            batch_idx = perm[start:start + cfg.hypernet.batch_size]
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=device)
            batch_correct = 0
            batch_total = 0

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                target_w_norm = train_weights_norm[idx]

                # Sample prototypes and task data (fresh each time)
                prototypes = sample_prototypes(class_images_train, classes, K, device)
                task_images, task_labels = sample_task_data(
                    class_images_train, classes,
                    cfg.hypernet.num_train_samples // cfg.hypernet.num_classes_per_task, device
                )

                # Generate weights
                gen_w_norm = hypernet(prototypes)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                # Functional loss
                logits = differentiable_forward(gen_w, task_images, cfg.target)
                func_loss = F.cross_entropy(logits, task_labels)

                # MSE loss
                mse_loss = F.mse_loss(gen_w_norm, target_w_norm)

                loss = cfg.hypernet.functional_loss_weight * func_loss + cfg.hypernet.weight_mse_weight * mse_loss
                batch_loss = batch_loss + loss

                with torch.no_grad():
                    batch_correct += (logits.argmax(1) == task_labels).sum().item()
                    batch_total += task_labels.size(0)

            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_func += batch_loss.item()
            epoch_correct += batch_correct
            epoch_total += batch_total
            n_batches += 1

        scheduler.step()
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        # Validation: accuracy on train tasks using TEST images
        hypernet.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for idx in range(min(n_train, 50)):  # sample 50 for speed
                classes = train_classes[idx]
                prototypes = sample_prototypes(class_images_test, classes, K, device)
                task_images, task_labels = sample_task_data(
                    class_images_test, classes, 50, device
                )
                gen_w_norm = hypernet(prototypes)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                logits = differentiable_forward(gen_w, task_images, cfg.target)
                val_correct += (logits.argmax(1) == task_labels).sum().item()
                val_total += task_labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        history["train_func"].append(epoch_func / n_batches)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{cfg.hypernet.num_epochs} | "
                f"Func: {epoch_func/n_batches:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Best: {best_val_acc:.4f}",
                flush=True,
            )

        if epochs_no_improve >= cfg.hypernet.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        hypernet.load_state_dict(best_state)
    hypernet.to(device)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    save_dir = Path("data/v4/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "hypernet.pt"
    torch.save({
        "hypernet_state": hypernet.state_dict(),
        "history": history,
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


if __name__ == "__main__":
    cfg = Config()
    train_hypernetwork(cfg)
