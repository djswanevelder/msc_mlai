"""
ViT Prototype Encoder — Frozen CLIP backbone for prototype encoding
====================================================================
Replaces the learned 3-layer MLP prototype encoder with a frozen
OpenCLIP ViT-B/32. The ViT already understands visual concepts
(letters AND digits) from web-scale pretraining.

Architecture:
  Images (28x28 grayscale) → resize 224x224, repeat 3ch → frozen ViT → 512D
  3 class prototypes (mean-pooled) → concat 1536D → MLP → 50K weights
"""

import itertools
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from torchvision import datasets, transforms
import open_clip

from src.prototype.config import Config, TargetMLPConfig
from src.prototype.models import differentiable_forward
from src.prototype.train import build_class_image_index, sample_task_data


class ViTPrototypeEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 as prototype encoder."""

    def __init__(self, device: str = "mps"):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.visual = model.visual
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad = False
        self.output_dim = 512

        # EMNIST-specific transform: grayscale 28x28 → RGB 224x224
        self.emnist_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (K, 784) flattened grayscale EMNIST images (already normalised)
        Returns:
            (output_dim,) mean-pooled ViT embedding
        """
        # Reshape to image format
        x = images.view(-1, 1, 28, 28)
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Repeat to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # ViT forward (frozen)
        with torch.no_grad():
            features = self.visual(x)  # (K, 512)
        return features.mean(dim=0)  # (512,)


class ViTHyperNetwork(nn.Module):
    """HyperNetwork with frozen ViT prototype encoder."""

    def __init__(
        self,
        target_weight_dim: int,
        num_classes_per_task: int = 3,
        hidden_dims: List[int] = None,
        device: str = "mps",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.vit_encoder = ViTPrototypeEncoder(device)
        vit_dim = self.vit_encoder.output_dim  # 512

        # Projection from ViT space to task embedding
        concat_dim = vit_dim * num_classes_per_task  # 1536
        layers = []
        d = concat_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, target_weight_dim))
        self.weight_generator = nn.Sequential(*layers)

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prototypes: list of 3 tensors, each (K, 784)
        Returns:
            (target_weight_dim,) flat weight vector
        """
        class_embs = [self.vit_encoder(p) for p in prototypes]
        task_emb = torch.cat(class_embs, dim=0)  # (1536,)
        return self.weight_generator(task_emb)


def sample_prototypes(class_images, classes, k, device):
    """Sample K prototype images per class."""
    prototypes = []
    for c in sorted(classes):
        imgs = class_images[c]
        indices = torch.randperm(len(imgs))[:k]
        prototypes.append(imgs[indices].to(device))
    return prototypes


def run():
    cfg = Config()
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Params: {weight_dim:,}")

    # Load zoo (digits-unseen split)
    zoo = torch.load(str(Path(cfg.zoo.zoo_dir) / "zoo.pt"), map_location=device, weights_only=False)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    w_mean = train_weights.mean(0, keepdim=True)
    w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_weights - w_mean) / w_std

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    # Build ViT hypernetwork
    hypernet = ViTHyperNetwork(
        target_weight_dim=weight_dim,
        hidden_dims=[512, 512],
        device=device,
    ).to(device)

    trainable = sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in hypernet.parameters())
    print(f"ViT HyperNet: {total:,} total, {trainable:,} trainable (ViT frozen)")

    # Train with functional loss
    optimizer = torch.optim.Adam(
        [p for p in hypernet.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    n_train = len(train_classes)
    batch_size = 4  # smaller — ViT forward is heavier
    best_val = 0
    best_state = None
    no_improve = 0

    print(f"\n{'='*60}")
    print("Training ViT-HyperNetwork (frozen CLIP ViT-B/32)")
    print(f"{'='*60}\n")

    for epoch in range(1, 201):
        hypernet.weight_generator.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                gen_w_norm = hypernet(protos)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                logits = differentiable_forward(gen_w, imgs, cfg.target)
                func = F.cross_entropy(logits, labs)
                mse = F.mse_loss(gen_w_norm, train_norm[idx])
                bl = bl + func + 0.1 * mse

                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in hypernet.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            el += bl.item()
            nb += 1

        scheduler.step()
        ta = ec / et if et > 0 else 0

        # Val
        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(n_train, 30)):
                classes = train_classes[idx]
                protos = sample_prototypes(ci_test, classes, 20, device)
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                gen_w = hypernet(protos) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()
                         if "vit_encoder.visual" not in k}  # don't save frozen ViT
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 40:
            print(f"  Early stop at {epoch}")
            break

    # Restore best generator weights
    current_state = hypernet.state_dict()
    for k, v in best_state.items():
        current_state[k] = v.to(device)
    hypernet.load_state_dict(current_state)
    hypernet.to(device).eval()
    print(f"\nBest val: {best_val:.4f}")

    # === Evaluate on digits-unseen ===
    print(f"\n{'='*60}")
    print("Evaluating ViT-HyperNet on digits-unseen (3/3)")
    print(f"{'='*60}")

    # Also load the MLP-encoder hypernetwork for comparison
    ckpt = torch.load("data/v4/checkpoints/hypernet.pt", map_location=device, weights_only=False)
    from src.prototype.models import HyperNetwork as MLPHyperNetwork
    hypernet_mlp = MLPHyperNetwork(
        target_weight_dim=ckpt["config"]["target_weight_dim"],
        num_classes_per_task=ckpt["config"]["num_classes_per_task"],
        input_dim=ckpt["config"]["input_dim"],
        prototype_encoder_hidden=ckpt["config"]["prototype_encoder_hidden"],
        prototype_dim=ckpt["config"]["prototype_dim"],
        hidden_dims=ckpt["config"]["hidden_dims"],
    ).to(device)
    hypernet_mlp.load_state_dict(ckpt["hypernet_state"])
    hypernet_mlp.eval()
    w_mean_mlp = ckpt["normalization"]["w_mean"].to(device).squeeze(0)
    w_std_mlp = ckpt["normalization"]["w_std"].to(device).squeeze(0)

    unseen = list(range(10))
    all_tasks = [list(c) for c in itertools.combinations(unseen, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        for i in range(len(dims) - 1):
            fan_in = dims[i]; std = (2.0/fan_in)**0.5
            w_size = dims[i]*dims[i+1]; b_size = dims[i+1]
            w[idx:idx+w_size] = torch.randn(w_size, device=device)*std
            idx += w_size+b_size
        return w

    from src.prototype.train import sample_prototypes as sp_mlp

    zs_vit, zs_mlp, ft_vit, ft_mlp, ft_kai = [], [], [], [], []

    for ti, classes in enumerate(tasks):
        protos_vit = sample_prototypes(ci_train, classes, 20, device)
        protos_mlp = sp_mlp(ci_train, classes, 20, device)

        with torch.no_grad():
            gw_vit = hypernet(protos_vit) * w_std.squeeze(0) + w_mean.squeeze(0)
            gw_mlp = hypernet_mlp(protos_mlp) * w_std_mlp + w_mean_mlp

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs_vit.append((differentiable_forward(gw_vit, imgs, cfg.target).argmax(1)==labs).float().mean().item())
            zs_mlp.append((differentiable_forward(gw_mlp, imgs, cfg.target).argmax(1)==labs).float().mean().item())

        # +3000 FT
        for init_w, store in [(gw_vit.clone(), ft_vit), (gw_mlp.clone(), ft_mlp), (kaiming_init(), ft_kai)]:
            w = init_w.detach().requires_grad_(True)
            opt = torch.optim.SGD([w], lr=0.01)
            for _ in range(3000):
                i2, l2 = sample_task_data(ci_train, classes, 50, device)
                opt.zero_grad()
                F.cross_entropy(differentiable_forward(w, i2, cfg.target), l2).backward()
                opt.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                store.append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1)==labs).float().mean().item())

        if (ti+1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    v = np.array(ft_vit); m = np.array(ft_mlp); k = np.array(ft_kai)

    print(f"\nViT vs MLP PROTOTYPE ENCODER (digits-unseen, n=50)")
    print(f"{'='*60}")
    print(f"{'':>25} {'Zero-shot':>10} {'+3000 FT':>10}")
    print(f"{'ViT-HyperNet (CLIP)':>25} {np.mean(zs_vit):>10.4f} {v.mean():>10.4f}")
    print(f"{'MLP-HyperNet (learned)':>25} {np.mean(zs_mlp):>10.4f} {m.mean():>10.4f}")
    print(f"{'Kaiming':>25} {'0.337':>10} {k.mean():>10.4f}")

    for n1, a, n2, b in [("ViT", v, "MLP", m), ("ViT", v, "Kaiming", k), ("MLP", m, "Kaiming", k)]:
        gap = a - b
        wins = int(np.sum(gap > 0))
        _, p = stats.ttest_rel(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {n1} vs {n2}: {gap.mean()*100:+.2f}pp, wins={wins}/50, p={p:.6f} {sig}")


if __name__ == "__main__":
    run()
