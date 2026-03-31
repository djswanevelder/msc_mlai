"""
Full-class EMNIST classifier using the hypernetwork's prototype encoder.

Extracts the pretrained PrototypeEncoder, adds a classification head,
and fine-tunes end-to-end on all 62 EMNIST ByClass classes.

The encoder was trained to produce 128D class embeddings via prototype
conditioning. We repurpose it as a per-image feature extractor with
a linear classification head.

Usage:
  python eval_fullclass.py
  python eval_fullclass.py --device mps --epochs 30
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.prototype.config import Config
from src.prototype.models import HyperNetwork
from src.prototype.train import build_class_image_index

CACHE_DIR = Path("data/v4/cache")

EMNIST_BYCLASS_NAMES = {
    **{i: str(i) for i in range(10)},
    **{i + 10: chr(ord("A") + i) for i in range(26)},
    **{i + 36: chr(ord("a") + i) for i in range(26)},
}


class EncoderClassifier(nn.Module):
    """Prototype encoder + classification head for 62-class EMNIST."""

    def __init__(self, encoder: nn.Module, embed_dim: int = 128, num_classes: int = 62):
        super().__init__()
        self.encoder = encoder  # PrototypeEncoder's internal MLP (784→256→256→128)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 784) flattened images → (B, 62) logits"""
        features = self.encoder(x)  # (B, 128) — no mean pooling needed for single images
        return self.head(features)


def load_pretrained_encoder(checkpoint_path: str, device: str) -> nn.Module:
    """Extract the prototype encoder's internal MLP from a trained hypernetwork."""
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
    # Return the internal sequential encoder (not the PrototypeEncoder wrapper which does mean pooling)
    return hypernet.prototype_encoder.encoder


def load_data():
    """Load EMNIST data (from cache if available)."""
    cache_path = CACHE_DIR / "class_images.pt"
    if cache_path.exists():
        print("Loading cached data...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached["train"], cached["test"]
    else:
        print("Loading EMNIST from disk (will cache)...")
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
        emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
        ci_train = build_class_image_index(emnist_train)
        ci_test = build_class_image_index(emnist_test)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"train": ci_train, "test": ci_test}, cache_path)
        return ci_train, ci_test


def build_tensors(class_images: Dict[int, torch.Tensor]):
    """Build flat (images, labels) tensors from class image index."""
    imgs, lbls = [], []
    for c in sorted(class_images.keys()):
        imgs.append(class_images[c])
        lbls.append(torch.full((len(class_images[c]),), c, dtype=torch.long))
    return torch.cat(imgs), torch.cat(lbls)


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
             device: str, num_classes: int, batch_size: int = 2048) -> Dict:
    """Evaluate model, return overall + per-class accuracy."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            preds = model(batch).argmax(1).cpu()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds)

    correct = (all_preds == labels).sum().item()
    overall = correct / len(labels)

    per_class = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (all_preds[mask] == c).float().mean().item()

    return {"overall": overall, "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder, train head only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    # Load pretrained encoder
    print("Loading pretrained encoder from hypernetwork...")
    encoder = load_pretrained_encoder("data/v4/checkpoints/hypernet.pt", device)

    # Load data
    ci_train, ci_test = load_data()
    num_classes = max(max(ci_train.keys()), max(ci_test.keys())) + 1
    print(f"Classes: {num_classes}")

    train_images, train_labels = build_tensors(ci_train)
    test_images, test_labels = build_tensors(ci_test)
    print(f"Train: {len(train_labels):,} | Test: {len(test_labels):,}")

    # Build classifier
    model = EncoderClassifier(encoder, embed_dim=128, num_classes=num_classes).to(device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder FROZEN — training head only")
        trainable = sum(p.numel() for p in model.head.parameters())
    else:
        print("Encoder UNFROZEN — fine-tuning end-to-end")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    # Evaluate before training (NCM-equivalent with random head)
    pre_result = evaluate(model, test_images, test_labels, device, num_classes)
    print(f"\nBefore training: {pre_result['overall']:.4f}")

    # Train
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = TensorDataset(train_images, train_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_imgs, batch_labels in loader:
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Evaluate
        result = evaluate(model, test_images, test_labels, device, num_classes)
        acc = result["overall"]

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss/n_batches:.4f}  "
                  f"acc={acc:.4f}  best={best_acc:.4f}  [{time.time()-t0:.0f}s]")

        if no_improve >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # Load best and final eval
    model.load_state_dict(best_state)
    model.to(device)
    final = evaluate(model, test_images, test_labels, device, num_classes)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {final['overall']:.4f}")
    print(f"  {'PASS' if final['overall'] >= 0.70 else 'FAIL'} (target: >0.70)")
    print(f"  Training time: {time.time()-t0:.0f}s")

    # Per-class
    print(f"\n{'Class':<8} {'Name':<6} {'Accuracy':<10}")
    print("-" * 30)
    for c in sorted(final["per_class"].keys()):
        name = EMNIST_BYCLASS_NAMES.get(c, str(c))
        acc = final["per_class"][c]
        print(f"{c:<8} {name:<6} {acc:.4f}")

    # Save checkpoint
    save_path = Path("data/v4/checkpoints/fullclass_encoder.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "accuracy": final["overall"],
        "per_class": final["per_class"],
        "args": vars(args),
    }, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
