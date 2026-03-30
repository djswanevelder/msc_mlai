"""
V3 Training — Hypernetwork with Functional Loss
=================================================
Trains the hypernetwork to generate weights that achieve high accuracy
on their conditioning dataset, using cross-entropy as the primary loss.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.direct.config import Config
from src.direct.models import HyperNetwork, batched_differentiable_forward


def load_zoo(zoo_path: str, device: str = "cpu") -> Dict:
    """Load the model zoo."""
    return torch.load(zoo_path, map_location=device, weights_only=False)


def get_class_data_loaders(
    all_classes: List[List[int]],
    num_samples: int,
    batch_size: int,
    train: bool = True,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-load a fixed batch of images+labels per class subset for functional loss.
    Returns dict mapping 'c0_c1_c2' → (images, labels) tensors.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST("data/v3/raw", train=train, download=False, transform=transform)

    # Build index: digit → list of sample indices
    digit_indices = {d: [] for d in range(10)}
    for i, (_, label) in enumerate(dataset):
        digit_indices[label].append(i)

    result = {}
    for classes in all_classes:
        key = "_".join(str(c) for c in classes)
        label_map = {c: i for i, c in enumerate(sorted(classes))}

        # Gather samples from each class
        images_list = []
        labels_list = []
        per_class = num_samples // len(classes)
        for cls in sorted(classes):
            indices = digit_indices[cls][:per_class]
            for idx in indices:
                img, _ = dataset[idx]
                images_list.append(img.view(-1))  # flatten
                labels_list.append(label_map[cls])

        result[key] = (
            torch.stack(images_list),
            torch.tensor(labels_list, dtype=torch.long),
        )

    return result


def train_hypernetwork(cfg: Config, zoo_path: Optional[str] = None) -> HyperNetwork:
    """Train the hypernetwork with functional + optional MSE loss."""
    device = cfg.hypernet.device
    if zoo_path is None:
        zoo_path = str(Path(cfg.zoo.zoo_dir) / "zoo.pt")

    # Load zoo
    print("Loading model zoo...")
    zoo = load_zoo(zoo_path)
    train_weights = zoo["train"]["weights"].to(device)  # (N_train, weight_dim)
    train_classes = zoo["train"]["classes"]  # list of [c0, c1, c2]
    train_accs = zoo["train"]["accuracies"]
    arch = zoo["architecture"]

    weight_dim = arch["total_params"]
    n_train = train_weights.size(0)

    print(f"Zoo: {n_train} training models, weight dim = {weight_dim}")
    print(f"Zoo accuracy: mean={sum(train_accs)/len(train_accs):.4f}")

    # Pre-load class data for functional loss
    print("Pre-loading class subset data...")
    class_data = get_class_data_loaders(
        train_classes,
        num_samples=cfg.hypernet.num_train_samples,
        batch_size=cfg.hypernet.batch_size,
    )

    # Also pre-load test data for each class subset (for evaluation during training)
    test_class_data = get_class_data_loaders(
        train_classes,
        num_samples=cfg.hypernet.num_train_samples,
        batch_size=cfg.hypernet.batch_size,
        train=False,
    )

    # Build class index tensors: (N_train, 3) — sorted class indices per model
    class_indices = torch.tensor(
        [sorted(c) for c in train_classes], dtype=torch.long, device=device
    )

    # Normalize zoo weights for MSE loss stability
    w_mean = train_weights.mean(dim=0, keepdim=True)
    w_std = train_weights.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_weights_norm = (train_weights - w_mean) / w_std

    # Create hypernetwork
    hypernet = HyperNetwork(
        target_weight_dim=weight_dim,
        num_classes=cfg.hypernet.num_classes,
        class_embed_dim=cfg.hypernet.class_embed_dim,
        num_classes_per_task=cfg.hypernet.num_classes_per_task,
        hidden_dims=cfg.hypernet.hidden_dims,
    ).to(device)

    total_params = sum(p.numel() for p in hypernet.parameters())
    print(f"HyperNetwork parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        hypernet.parameters(),
        lr=cfg.hypernet.lr,
        weight_decay=cfg.hypernet.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.hypernet.num_epochs
    )

    # Training loop
    history = {
        "train_func_loss": [], "train_mse_loss": [], "train_total_loss": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training HyperNetwork")
    print(f"  Epochs: {cfg.hypernet.num_epochs} | Patience: {cfg.hypernet.patience}")
    print(f"  Functional loss weight: {cfg.hypernet.functional_loss_weight}")
    print(f"  MSE loss weight: {cfg.hypernet.weight_mse_weight}")
    print(f"  Batch size: {cfg.hypernet.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.hypernet.num_epochs + 1):
        hypernet.train()
        epoch_func_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_total_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        # Shuffle training order
        perm = torch.randperm(n_train)

        for start in range(0, n_train, cfg.hypernet.batch_size):
            batch_idx = perm[start:start + cfg.hypernet.batch_size]
            batch_classes = class_indices[batch_idx]  # (B, 3)
            batch_target_w = train_weights_norm[batch_idx]  # (B, weight_dim)

            optimizer.zero_grad()

            # Generate weights
            generated_w_norm = hypernet(batch_classes)  # (B, weight_dim)

            # Denormalize for functional evaluation
            generated_w = generated_w_norm * w_std + w_mean

            # Functional loss: cross-entropy of generated models on their class data
            func_loss = torch.tensor(0.0, device=device)
            batch_correct = 0
            batch_total = 0

            for i, idx in enumerate(batch_idx):
                key = "_".join(str(c) for c in sorted(train_classes[idx.item()]))
                images, labels = class_data[key]
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass with generated weights (differentiable)
                from src.direct.models import differentiable_forward
                logits = differentiable_forward(generated_w[i], images, cfg.target)
                func_loss = func_loss + F.cross_entropy(logits, labels)

                # Track accuracy
                with torch.no_grad():
                    batch_correct += (logits.argmax(1) == labels).sum().item()
                    batch_total += labels.size(0)

            func_loss = func_loss / len(batch_idx)

            # MSE loss on normalized weights (auxiliary)
            mse_loss = F.mse_loss(generated_w_norm, batch_target_w)

            # Combined loss
            total_loss = (
                cfg.hypernet.functional_loss_weight * func_loss
                + cfg.hypernet.weight_mse_weight * mse_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_func_loss += func_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_correct += batch_correct
            epoch_total += batch_total
            n_batches += 1

        scheduler.step()

        avg_func = epoch_func_loss / n_batches
        avg_mse = epoch_mse_loss / n_batches
        avg_total = epoch_total_loss / n_batches
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        # Validation: accuracy on test data for training class subsets
        hypernet.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(n_train):
                key = "_".join(str(c) for c in sorted(train_classes[i]))
                if key not in test_class_data:
                    continue
                images, labels = test_class_data[key]
                images = images.to(device)
                labels = labels.to(device)

                gen_w_norm = hypernet(class_indices[i:i+1])
                gen_w = gen_w_norm * w_std + w_mean

                from src.direct.models import differentiable_forward
                logits = differentiable_forward(gen_w[0], images, cfg.target)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_func_loss"].append(avg_func)
        history["train_mse_loss"].append(avg_mse)
        history["train_total_loss"].append(avg_total)
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
                f"Func: {avg_func:.4f} | MSE: {avg_mse:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Best: {best_val_acc:.4f}",
                flush=True,
            )

        if epochs_no_improve >= cfg.hypernet.patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {cfg.hypernet.patience} epochs)")
            break

    # Restore best
    if best_state is not None:
        hypernet.load_state_dict(best_state)
    hypernet.to(device)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Save
    save_dir = Path("data/v3/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "hypernet.pt"
    torch.save({
        "hypernet_state": hypernet.state_dict(),
        "history": history,
        "config": {
            "target_weight_dim": weight_dim,
            "num_classes": cfg.hypernet.num_classes,
            "class_embed_dim": cfg.hypernet.class_embed_dim,
            "num_classes_per_task": cfg.hypernet.num_classes_per_task,
            "hidden_dims": cfg.hypernet.hidden_dims,
        },
        "normalization": {
            "w_mean": w_mean.cpu(),
            "w_std": w_std.cpu(),
        },
    }, save_path)
    print(f"Saved to {save_path}")

    return hypernet


if __name__ == "__main__":
    cfg = Config()
    train_hypernetwork(cfg)
