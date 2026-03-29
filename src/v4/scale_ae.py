"""
V4 Autoencoder-Based Scaling — Weight AE + Hypernetwork in Latent Space
========================================================================
1. Train Weight AE: 109K -> 512D latent -> 109K (with functional loss)
2. Train HyperNet: prototypes -> 512D latent code
3. Decode latent -> full weights

This solves the scaling problem: the hypernetwork only needs to output 512D,
and the decoder (trained on zoo weights) handles expansion to 109K.
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

from src.v4.config import Config, TargetMLPConfig
from src.v4.models import PrototypeEncoder, differentiable_forward
from src.v4.train import build_class_image_index, sample_prototypes, sample_task_data


class WeightAutoencoder(nn.Module):
    """Compresses 109K weight vectors to a small latent space."""

    def __init__(self, weight_dim: int, latent_dim: int = 512):
        super().__init__()
        # Moderate AE with dropout — balances capacity vs overfitting on 150 models
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, weight_dim),
        )

    def encode(self, w: torch.Tensor) -> torch.Tensor:
        return self.encoder(w)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(w))


class LatentHyperNetwork(nn.Module):
    """Generates latent codes (512D) from prototype images."""

    def __init__(self, latent_dim: int = 512, prototype_dim: int = 128,
                 num_classes_per_task: int = 3, input_dim: int = 784):
        super().__init__()
        self.prototype_encoder = PrototypeEncoder(
            input_dim=input_dim, hidden_dim=256, output_dim=prototype_dim
        )
        concat_dim = prototype_dim * num_classes_per_task
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)
        return self.net(task_emb)


def run():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()
    latent_dim = 512

    print(f"Target: 784 -> [128, 64] -> 3, params: {weight_dim:,}")
    print(f"Latent dim: {latent_dim}")
    print(f"Compression ratio: {weight_dim / latent_dim:.0f}x")

    zoo = torch.load("data/v4b/zoo/zoo.pt", map_location=device, weights_only=False)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    n_train = len(train_classes)

    # Normalize weights
    w_mean = train_weights.mean(0, keepdim=True)
    w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_weights - w_mean) / w_std

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    # ===================================================================
    # STAGE 1: Train Weight Autoencoder
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Weight Autoencoder (109K -> 512D -> 109K)")
    print(f"{'='*60}")

    ae = WeightAutoencoder(weight_dim, latent_dim).to(device)
    ae_params = sum(p.numel() for p in ae.parameters())
    print(f"AE params: {ae_params:,}")

    # Split into train/val for AE
    n_ae_val = 20
    ae_train = train_norm[:n_train - n_ae_val]
    ae_val = train_norm[n_train - n_ae_val:]

    ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-3)
    ae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ae_opt, T_max=500)
    ae_bs = 16
    best_ae_val = float('inf')
    best_ae_state = None
    no_improve = 0

    for epoch in range(1, 501):
        ae.train()
        perm = torch.randperm(len(ae_train))
        el = nb = 0
        for start in range(0, len(ae_train), ae_bs):
            batch = ae_train[perm[start:start + ae_bs]]
            ae_opt.zero_grad()
            recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            ae_opt.step()
            el += loss.item()
            nb += 1
        ae_sched.step()

        ae.eval()
        with torch.no_grad():
            val_recon = ae(ae_val)
            val_mse = F.mse_loss(val_recon, ae_val).item()

        if val_mse < best_ae_val:
            best_ae_val = val_mse
            best_ae_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            # Check functional accuracy of reconstructed weights
            ae.eval()
            func_accs = []
            with torch.no_grad():
                for idx in range(min(20, n_train)):
                    recon_norm = ae(train_norm[idx:idx+1])
                    recon_w = recon_norm * w_std + w_mean
                    classes = train_classes[idx]
                    imgs, labs = sample_task_data(ci_test, classes, 100, device)
                    logits = differentiable_forward(recon_w.squeeze(0), imgs, cfg.target)
                    func_accs.append((logits.argmax(1) == labs).float().mean().item())
            print(f"  Epoch {epoch:3d} | Train MSE: {el/nb:.6f} | Val MSE: {val_mse:.6f} | Recon Acc: {np.mean(func_accs):.4f}", flush=True)

        if no_improve >= 100:
            print(f"  Early stop at {epoch}")
            break

    ae.load_state_dict(best_ae_state)
    ae.to(device)
    print(f"  Best val MSE: {best_ae_val:.6f}")

    # Final functional eval of AE
    ae.eval()
    func_accs = []
    with torch.no_grad():
        for idx in range(min(40, n_train)):
            recon_norm = ae(train_norm[idx:idx+1])
            recon_w = recon_norm * w_std + w_mean
            classes = train_classes[idx]
            imgs, labs = sample_task_data(ci_test, classes, 100, device)
            logits = differentiable_forward(recon_w.squeeze(0), imgs, cfg.target)
            func_accs.append((logits.argmax(1) == labs).float().mean().item())
    print(f"  AE reconstruction accuracy: {np.mean(func_accs):.4f} (zoo baseline ~98.5%)")

    # ===================================================================
    # STAGE 1b: Fine-tune AE with functional loss
    # ===================================================================
    print(f"\n--- Fine-tuning AE with functional loss ---")
    ae_opt2 = torch.optim.Adam(ae.parameters(), lr=1e-4)

    for epoch in range(1, 101):
        ae.train()
        perm = torch.randperm(n_train - n_ae_val)
        el = nb = ec = et = 0

        for start in range(0, len(perm), 8):
            batch_idx = perm[start:start + 8]
            ae_opt2.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                recon_norm = ae(train_norm[idx:idx+1]).squeeze(0)
                recon_w = recon_norm * w_std.squeeze(0) + w_mean.squeeze(0)
                classes = train_classes[idx]
                imgs, labs = sample_task_data(ci_train, classes, 60, device)
                logits = differentiable_forward(recon_w, imgs, cfg.target)
                func_loss = F.cross_entropy(logits, labs)
                mse_loss = F.mse_loss(recon_norm, train_norm[idx])
                batch_loss = batch_loss + func_loss + 0.5 * mse_loss
                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            ae_opt2.step()
            el += batch_loss.item()
            nb += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train Acc: {ec/et:.4f}", flush=True)

    # Re-check functional accuracy
    ae.eval()
    func_accs = []
    with torch.no_grad():
        for idx in range(min(40, n_train)):
            recon_norm = ae(train_norm[idx:idx+1])
            recon_w = recon_norm * w_std + w_mean
            classes = train_classes[idx]
            imgs, labs = sample_task_data(ci_test, classes, 100, device)
            logits = differentiable_forward(recon_w.squeeze(0), imgs, cfg.target)
            func_accs.append((logits.argmax(1) == labs).float().mean().item())
    print(f"  AE+func reconstruction accuracy: {np.mean(func_accs):.4f}")

    # Save AE
    ae_path = Path("data/v4b/checkpoints/weight_ae.pt")
    ae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu().clone() for k, v in ae.state_dict().items()},
                "w_mean": w_mean.cpu(), "w_std": w_std.cpu()}, ae_path)

    # ===================================================================
    # STAGE 2: Train Latent HyperNetwork
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Latent HyperNetwork (prototypes -> 512D -> decode -> 109K)")
    print(f"{'='*60}")

    # Pre-compute latent codes for all zoo models
    ae.eval()
    with torch.no_grad():
        train_latents = ae.encode(train_norm)  # (N, 512)
    print(f"  Latent codes shape: {train_latents.shape}")

    hypernet = LatentHyperNetwork(latent_dim=latent_dim).to(device)
    hn_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  Latent HyperNet params: {hn_params:,}")

    # Freeze AE decoder during hypernet training
    for p in ae.parameters():
        p.requires_grad = False

    hn_opt = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
    hn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hn_opt, T_max=300)
    best_val = 0
    best_hn_state = None
    no_improve = 0

    for epoch in range(1, 301):
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
            hn_opt.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                # Generate latent code, decode to weights
                z = hypernet(protos)
                gen_w_norm = ae.decode(z)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                # Functional loss
                logits = differentiable_forward(gen_w, imgs, cfg.target)
                func_loss = F.cross_entropy(logits, labs)
                # Latent MSE (match zoo's latent code)
                latent_mse = F.mse_loss(z, train_latents[idx])

                batch_loss = batch_loss + func_loss + 0.1 * latent_mse
                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            hn_opt.step()
            el += batch_loss.item()
            nb += 1

        hn_sched.step()
        ta = ec / et if et > 0 else 0

        # Val
        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(40, n_train)):
                classes = train_classes[idx]
                protos = sample_prototypes(ci_test, classes, 20, device)
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                z = hypernet(protos)
                gen_w = ae.decode(z) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val:
            best_val = va
            best_hn_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    hypernet.load_state_dict(best_hn_state)
    hypernet.to(device)
    print(f"  Best val: {best_val:.4f}")

    # Save
    torch.save({"state": best_hn_state}, Path("data/v4b/checkpoints/latent_hypernet.pt"))

    # ===================================================================
    # STAGE 3: All-unseen evaluation
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: All-unseen eval (109K params via AE)")
    print(f"{'='*60}")

    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = (2.0 / fan_in) ** 0.5
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    zs, ft_gen, ft_kai, ft_rand = [], [], [], []
    STEPS = 3000

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            z = hypernet(protos)
            gen_w = ae.decode(z) * w_std.squeeze(0) + w_mean.squeeze(0)

        # Zero-shot
        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        # Fine-tune from generated, kaiming, random
        for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_init(), ft_kai),
                              (torch.randn(weight_dim, device=device) * 0.01, ft_rand)]:
            w = init_w.detach().requires_grad_(True)
            opt = torch.optim.SGD([w], lr=0.01)
            for _ in range(STEPS):
                ti2, tl2 = sample_task_data(ci_train, classes, 50, device)
                opt.zero_grad()
                F.cross_entropy(differentiable_forward(w, ti2, cfg.target), tl2).backward()
                opt.step()
            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                store.append((differentiable_forward(w.detach(), imgs, cfg.target).argmax(1) == labs).float().mean().item())

        if (ti + 1) % 10 == 0:
            print(f"  {ti+1}/{len(tasks)} done", flush=True)

    g = np.array(ft_gen)
    k = np.array(ft_kai)
    r = np.array(ft_rand)

    print(f"\n109K PARAMS VIA WEIGHT AE — ALL-UNSEEN (3/3), n={len(tasks)}")
    print(f"{'='*60}")
    print(f"Zero-shot:           {np.mean(zs):.4f} +- {np.std(zs):.4f}")
    print(f"Generated +{STEPS}FT:  {g.mean():.4f} +- {g.std():.4f}")
    print(f"Kaiming +{STEPS}FT:    {k.mean():.4f} +- {k.std():.4f}")
    print(f"Random +{STEPS}FT:     {r.mean():.4f} +- {r.std():.4f}")

    for name, b in [("Kaiming", k), ("Random", r)]:
        gap = g - b
        wins = np.sum(gap > 0)
        t_stat, t_p = stats.ttest_rel(g, b)
        nz = gap[gap != 0]
        w_stat, w_p = stats.wilcoxon(nz) if len(nz) > 0 else (0, 1)
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"\n  vs {name}: gap={gap.mean()*100:+.2f}pp, wins={wins}/{len(gap)}, p={t_p:.8f} {sig}")


if __name__ == "__main__":
    run()
