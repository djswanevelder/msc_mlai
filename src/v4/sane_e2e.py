"""
SANE End-to-End — Joint fine-tuning of decoder + hypernetwork
==============================================================
Takes the pre-trained SANE AE + latent hypernetwork from sane.py,
unfreezes the decoder, and fine-tunes everything with functional loss.

This should fix the better-minimum problem: the decoder learns to
produce weights that are not just close to the zoo in MSE, but
functionally correct.
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
from src.v4.sane import (
    SANEAutoencoder, LatentHyperNetwork, PositionEmbedding,
    tokenize_mlp_weights, detokenize_to_flat,
)


def run():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()

    dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
    n_tokens = sum(dims[i + 1] for i in range(len(dims) - 1))
    token_size = max(dims[i] + 1 for i in range(len(dims) - 1))
    d_z = 32

    print(f"Target: 784 -> [128, 64] -> 3, params: {weight_dim:,}")
    print(f"Tokens: {n_tokens}, Token size: {token_size}, Latent: {n_tokens}x{d_z}")

    zoo = torch.load("data/v4b/zoo/zoo.pt", map_location=device, weights_only=False)
    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    n_train = len(train_classes)

    w_mean = train_weights.mean(0, keepdim=True)
    w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_weights - w_mean) / w_std

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))

    # Tokenize zoo
    all_tokens, all_masks = [], []
    ref_positions = None
    for i in range(n_train):
        tokens, mask, positions = tokenize_mlp_weights(train_norm[i], cfg.target, token_size)
        all_tokens.append(tokens)
        all_masks.append(mask)
        if ref_positions is None:
            ref_positions = positions
    all_tokens = torch.stack(all_tokens).to(device)
    all_masks = torch.stack(all_masks).to(device)

    # Load pre-trained SANE + hypernetwork
    sane_ckpt = Path("data/v4b/checkpoints/sane_ae.pt")
    if sane_ckpt.exists():
        print("Loading pre-trained SANE AE...")
        sane_data = torch.load(sane_ckpt, map_location=device, weights_only=False)
        sane = SANEAutoencoder(token_dim=token_size, d_model=256, d_z=d_z,
                               n_heads=8, n_layers=4, dropout=0.1).to(device)
        sane.load_state_dict(sane_data["state"])
    else:
        print("No pre-trained SANE found. Training from scratch...")
        sane = SANEAutoencoder(token_dim=token_size, d_model=256, d_z=d_z,
                               n_heads=8, n_layers=4, dropout=0.1).to(device)
        # Quick MSE pre-train
        opt = torch.optim.Adam(sane.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_batch = ref_positions.unsqueeze(0).expand(min(16, n_train), -1, -1)
        for ep in range(200):
            sane.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, 16):
                batch_idx = perm[start:start + 16]
                bt = all_tokens[batch_idx]
                bp = ref_positions.unsqueeze(0).expand(len(batch_idx), -1, -1)
                bm = all_masks[batch_idx]
                opt.zero_grad()
                _, recon = sane(bt, bp)
                loss = ((recon - bt) ** 2 * bm.float()).sum() / bm.float().sum()
                loss.backward()
                opt.step()
            if (ep + 1) % 50 == 0:
                print(f"  MSE pre-train epoch {ep+1}: {loss.item():.4f}")

    hypernet = LatentHyperNetwork(n_tokens=n_tokens, d_z=d_z).to(device)

    # Try loading pre-trained hypernetwork
    hn_ckpt = Path("data/v4b/checkpoints/latent_hypernet.pt")
    if hn_ckpt.exists():
        print("Loading pre-trained latent hypernetwork...")
        hn_data = torch.load(hn_ckpt, map_location=device, weights_only=False)
        hypernet.load_state_dict(hn_data["state"])

    total_params = sum(p.numel() for p in sane.parameters()) + sum(p.numel() for p in hypernet.parameters())
    print(f"Total params (SANE + HyperNet): {total_params:,}")

    # ===================================================================
    # END-TO-END FINE-TUNING — everything unfrozen, functional loss
    # ===================================================================
    print(f"\n{'='*60}")
    print("END-TO-END FINE-TUNING (SANE decoder + HyperNet, functional loss)")
    print(f"{'='*60}")

    # All parameters trainable — decoder at lower LR
    sane_params = list(sane.parameters())
    hn_params = list(hypernet.parameters())
    optimizer = torch.optim.Adam([
        {"params": sane_params, "lr": 1e-4},  # decoder at lower LR
        {"params": hn_params, "lr": 5e-4},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    best_val = 0
    best_sane_state = None
    best_hn_state = None
    no_improve = 0

    for epoch in range(1, 301):
        sane.train()
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
            optimizer.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                # Forward: prototypes -> latent -> decode -> weights
                z_pred = hypernet(protos)
                recon_tokens = sane.decode(
                    z_pred.unsqueeze(0), ref_positions.unsqueeze(0)
                ).squeeze(0)
                gen_w_norm = detokenize_to_flat(recon_tokens, all_masks[0], cfg.target)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                # Functional loss (primary)
                logits = differentiable_forward(gen_w, imgs, cfg.target)
                func_loss = F.cross_entropy(logits, labs)

                # Token reconstruction regularization (keeps decoder grounded)
                target_z = sane.encode(
                    all_tokens[idx:idx+1], ref_positions.unsqueeze(0)
                ).squeeze(0).detach()
                latent_reg = F.mse_loss(z_pred, target_z)

                bl = bl + func_loss + 0.05 * latent_reg

                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(list(sane.parameters()) + list(hypernet.parameters()), 5.0)
            optimizer.step()
            el += bl.item()
            nb += 1

        scheduler.step()
        ta = ec / et if et > 0 else 0

        # Val
        sane.eval()
        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(40, n_train)):
                classes = train_classes[idx]
                protos = sample_prototypes(ci_test, classes, 20, device)
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                z = hypernet(protos)
                recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w = detokenize_to_flat(recon, all_masks[0], cfg.target) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val:
            best_val = va
            best_sane_state = {k: v.cpu().clone() for k, v in sane.state_dict().items()}
            best_hn_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    sane.load_state_dict(best_sane_state)
    hypernet.load_state_dict(best_hn_state)
    sane.to(device)
    hypernet.to(device)
    print(f"  Best val: {best_val:.4f}")

    # Save
    torch.save({"sane": best_sane_state, "hypernet": best_hn_state,
                "w_mean": w_mean.cpu(), "w_std": w_std.cpu()},
               Path("data/v4b/checkpoints/sane_e2e.pt"))

    # ===================================================================
    # ALL-UNSEEN EVAL
    # ===================================================================
    print(f"\n{'='*60}")
    print("ALL-UNSEEN EVAL (SANE E2E, 109K params)")
    print(f"{'='*60}")

    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std = (2.0 / fan_in) ** 0.5
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    zs, ft_gen, ft_kai = [], [], []
    STEPS = 3000

    sane.eval()
    hypernet.eval()

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            z = hypernet(protos)
            recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
            gen_w = detokenize_to_flat(recon, all_masks[0], cfg.target) * w_std.squeeze(0) + w_mean.squeeze(0)

        # Zero-shot
        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

        # Fine-tune
        for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_init(), ft_kai)]:
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
    gap = g - k
    wins = np.sum(gap > 0)
    t_stat, t_p = stats.ttest_rel(g, k)
    nz = gap[gap != 0]
    w_stat, w_p = stats.wilcoxon(nz) if len(nz) > 0 else (0, 1)

    print(f"\nSANE E2E — 109K PARAMS, ALL-UNSEEN (3/3), n={len(tasks)}, {STEPS} steps")
    print(f"{'='*60}")
    print(f"Zero-shot:            {np.mean(zs):.4f} +- {np.std(zs):.4f}")
    print(f"Generated +{STEPS}FT:   {g.mean():.4f} +- {g.std():.4f}")
    print(f"Kaiming +{STEPS}FT:     {k.mean():.4f} +- {k.std():.4f}")
    print(f"Gap: {gap.mean()*100:+.2f}pp, Wins: {wins}/{len(tasks)}")
    print(f"t-test: t={t_stat:.3f}, p={t_p:.8f}")
    print(f"Wilcoxon: W={w_stat:.0f}, p={w_p:.8f}")

    # Also test at fewer FT steps (the practical regime)
    print(f"\n--- Quick eval at 10 and 100 FT steps ---")
    for ft_steps in [10, 100]:
        gen_acc, kai_acc = [], []
        for ti, classes in enumerate(tasks[:20]):  # 20 tasks for speed
            protos = sample_prototypes(ci_train, classes, 20, device)
            with torch.no_grad():
                z = hypernet(protos)
                recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gw = detokenize_to_flat(recon, all_masks[0], cfg.target) * w_std.squeeze(0) + w_mean.squeeze(0)

            for init_w, store in [(gw.clone(), gen_acc), (kaiming_init(), kai_acc)]:
                w = init_w.detach().requires_grad_(True)
                o = torch.optim.SGD([w], lr=0.01)
                for _ in range(ft_steps):
                    i2, l2 = sample_task_data(ci_train, classes, 50, device)
                    o.zero_grad()
                    F.cross_entropy(differentiable_forward(w, i2, cfg.target), l2).backward()
                    o.step()
                i2, l2 = sample_task_data(ci_test, classes, 200, device)
                with torch.no_grad():
                    store.append((differentiable_forward(w.detach(), i2, cfg.target).argmax(1) == l2).float().mean().item())

        g_short = np.mean(gen_acc)
        k_short = np.mean(kai_acc)
        print(f"  {ft_steps} steps: Generated={g_short:.4f}, Kaiming={k_short:.4f}, Gap={g_short-k_short:+.4f}")


if __name__ == "__main__":
    run()
