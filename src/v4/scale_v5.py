"""
Scale V5 -- Larger Target MLP with SANE Functional Pipeline
==============================================================
Target MLP: 784 -> 256 -> 128 -> 64 -> 3  (~236K params)

Stages:
  1. Generate zoo (200 train from seen 0-49, 50 unseen from 50-61)
  2. Tokenize weights per-neuron (SANE approach)
  3. Train SANE AE with functional loss warmup (MSE first 50 epochs, then ramp)
  4. Train latent hypernetwork with functional loss
  5. End-to-end fine-tuning
  6. Eval at 0/1/2/3 unseen classes (50 tasks each, 3000 FT steps, vs Kaiming)
"""

import itertools
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

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
    SANEAutoencoder, LatentHyperNetwork,
    tokenize_mlp_weights, detokenize_to_flat,
)
from src.v4.zoo import _train_one_subset


# =====================================================================
# Zoo generation
# =====================================================================

def generate_v5_zoo(target_cfg: TargetMLPConfig, zoo_dir: str, num_workers: int = 8):
    """Generate a model zoo for the larger target MLP."""
    zoo_path = Path(zoo_dir)
    zoo_path.mkdir(parents=True, exist_ok=True)

    save_file = zoo_path / "zoo.pt"
    if save_file.exists():
        print(f"Zoo already exists at {save_file}, loading...")
        return torch.load(save_file, map_location="cpu", weights_only=False)

    print("Downloading/verifying EMNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=True, transform=transform)

    seen = list(range(50))
    unseen = list(range(50, 62))
    rng = random.Random(42)

    # Seen-only subsets for training
    all_seen_subsets = list(itertools.combinations(seen, 3))
    rng.shuffle(all_seen_subsets)
    all_seen_subsets = [list(s) for s in all_seen_subsets]
    train_subsets = all_seen_subsets[:200]

    # Unseen subsets: at least one class from unseen set
    unseen_combos = []
    for u in unseen:
        for pair in itertools.combinations(seen, 2):
            unseen_combos.append(sorted(list(pair) + [u]))
    rng.shuffle(unseen_combos)
    seen_set = set()
    unseen_subsets = []
    for combo in unseen_combos:
        key = tuple(combo)
        if key not in seen_set:
            seen_set.add(key)
            unseen_subsets.append(list(combo))
        if len(unseen_subsets) >= 50:
            break

    all_tasks = train_subsets + unseen_subsets
    total = len(all_tasks)

    # Compute param count
    dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
    total_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    print(f"Target MLP: {target_cfg.input_dim} -> {target_cfg.hidden_dims} -> {target_cfg.num_classes}")
    print(f"Parameters per model: {total_params:,}")
    print(f"Tasks: {len(train_subsets)} train + {len(unseen_subsets)} unseen = {total} total")

    worker_fn = partial(
        _train_one_subset,
        input_dim=target_cfg.input_dim,
        hidden_dims=target_cfg.hidden_dims,
        num_classes=target_cfg.num_classes,
        train_epochs=30,
        lr=1e-3,
        batch_size=128,
    )

    num_workers = min(mp.cpu_count(), num_workers)
    print(f"Using {num_workers} parallel workers")

    results = []
    with mp.Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap(worker_fn, all_tasks)):
            results.append(result)
            done = i + 1
            if done % 25 == 0 or done == total:
                print(f"  Zoo progress: {done}/{total} models trained", flush=True)

    train_results = results[:200]
    unseen_results = results[200:]

    for name, res in [("Train", train_results), ("Unseen", unseen_results)]:
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
        "unseen": pack(unseen_results),
        "architecture": {
            "input_dim": target_cfg.input_dim,
            "hidden_dims": target_cfg.hidden_dims,
            "num_classes": target_cfg.num_classes,
            "total_params": total_params,
        },
        "seen_classes": list(range(50)),
        "unseen_classes": list(range(50, 62)),
    }

    torch.save(save_data, save_file)
    print(f"Zoo saved to {save_file}")
    return save_data


# =====================================================================
# Main pipeline
# =====================================================================

def run():
    device = "mps"
    target_cfg = TargetMLPConfig(input_dim=784, hidden_dims=[256, 128, 64], num_classes=3)
    dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
    weight_dim = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
    n_tokens = sum(dims[i + 1] for i in range(len(dims) - 1))  # 256+128+64+3 = 451
    token_size = max(dims[i] + 1 for i in range(len(dims) - 1))  # 785 (784+1)
    d_z = 32

    print("=" * 70)
    print("SCALE V5: Larger Target MLP with SANE Functional Pipeline")
    print("=" * 70)
    print(f"Target: 784 -> [256, 128, 64] -> 3")
    print(f"Weight dim: {weight_dim:,}")
    print(f"Tokens: {n_tokens}, Token size: {token_size}")
    print(f"Latent: {n_tokens} x {d_z} = {n_tokens * d_z}D")
    print()

    # =================================================================
    # STAGE 0: Generate zoo
    # =================================================================
    print(f"{'='*60}")
    print("STAGE 0: Generate model zoo")
    print(f"{'='*60}")

    zoo_dir = "data/v4c/zoo"
    zoo = generate_v5_zoo(target_cfg, zoo_dir, num_workers=8)

    train_weights = zoo["train"]["weights"].to(device)
    train_classes = zoo["train"]["classes"]
    n_train = len(train_classes)

    w_mean = train_weights.mean(0, keepdim=True)
    w_std = train_weights.std(0, keepdim=True).clamp(min=1e-6)
    train_norm = (train_weights - w_mean) / w_std

    print(f"\nLoaded {n_train} training models, weight dim={weight_dim:,}")
    print(f"Zoo accuracies: mean={np.mean(zoo['train']['accuracies']):.4f}")

    # Build EMNIST image indices
    print("Building EMNIST class image indices...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)),
    ])
    ci_train = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform))
    ci_test = build_class_image_index(
        datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform))
    print("  Done.")

    # =================================================================
    # Tokenize zoo weights
    # =================================================================
    print("\nTokenizing zoo weights...")
    all_tokens, all_masks = [], []
    ref_positions = None
    for i in range(n_train):
        tokens, mask, positions = tokenize_mlp_weights(train_norm[i], target_cfg, token_size)
        all_tokens.append(tokens)
        all_masks.append(mask)
        if ref_positions is None:
            ref_positions = positions
    all_tokens = torch.stack(all_tokens).to(device)
    all_masks = torch.stack(all_masks).to(device)
    print(f"  Tokenized: {all_tokens.shape} (n_models, n_tokens, token_size)")

    # Verify roundtrip
    recon_flat = detokenize_to_flat(all_tokens[0], all_masks[0], target_cfg)
    assert torch.allclose(recon_flat, train_norm[0].to(device), atol=1e-5), "Roundtrip failed!"
    print("  Tokenization roundtrip: OK")

    # =================================================================
    # STAGE 1: SANE AE with MSE + functional loss warmup
    # =================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: SANE AE (MSE warmup 50 epochs, then ramp functional loss)")
    print(f"{'='*60}")

    sane = SANEAutoencoder(
        token_dim=token_size, d_model=256, d_z=d_z,
        n_heads=8, n_layers=4, dropout=0.1,
    ).to(device)
    sane_params = sum(p.numel() for p in sane.parameters())
    print(f"  SANE AE params: {sane_params:,}")

    sane_opt = torch.optim.Adam(sane.parameters(), lr=1e-3, weight_decay=1e-4)
    sane_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sane_opt, T_max=500)
    BS = 8  # smaller batch for functional loss (each needs forward pass)
    best_val_acc = 0
    best_sane_state = None
    no_improve = 0
    func_weight = 0.0

    for epoch in range(1, 501):
        sane.train()
        perm = torch.randperm(n_train)
        el = ec = et = nb = 0

        # Warm up functional loss
        if epoch <= 50:
            func_weight = 0.0
        elif epoch <= 100:
            func_weight = (epoch - 50) / 50.0  # linear ramp 0->1
        else:
            func_weight = 1.0

        for start in range(0, n_train, BS):
            batch_idx = perm[start:start + BS]
            sane_opt.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                bt = all_tokens[idx:idx + 1]
                bp = ref_positions.unsqueeze(0)
                bm = all_masks[idx:idx + 1]

                _, recon = sane(bt, bp)

                # MSE loss (masked)
                diff = (recon - bt) ** 2
                mse = (diff * bm.float()).sum() / bm.float().sum()

                # Functional loss
                if func_weight > 0:
                    recon_flat = detokenize_to_flat(recon.squeeze(0), bm.squeeze(0), target_cfg)
                    recon_w = recon_flat * w_std.squeeze(0) + w_mean.squeeze(0)
                    classes = train_classes[idx]
                    imgs, labs = sample_task_data(ci_train, classes, 60, device)
                    logits = differentiable_forward(recon_w, imgs, target_cfg)
                    func_loss = F.cross_entropy(logits, labs).clamp(max=5.0)
                    bl = bl + mse + func_weight * func_loss

                    with torch.no_grad():
                        ec += (logits.argmax(1) == labs).sum().item()
                        et += labs.size(0)
                else:
                    bl = bl + mse

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(sane.parameters(), 5.0)
            sane_opt.step()
            el += bl.item()
            nb += 1

        sane_sched.step()

        # Validation: functional accuracy of reconstructed weights
        sane.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(30, n_train)):
                _, recon = sane(all_tokens[idx:idx + 1], ref_positions.unsqueeze(0))
                recon_flat = detokenize_to_flat(recon.squeeze(0), all_masks[idx], target_cfg)
                recon_w = recon_flat * w_std.squeeze(0) + w_mean.squeeze(0)
                classes = train_classes[idx]
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                logits = differentiable_forward(recon_w, imgs, target_cfg)
                vc += (logits.argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val_acc:
            best_val_acc = va
            best_sane_state = {k: v.cpu().clone() for k, v in sane.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            ta = ec / et if et > 0 else 0
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val Acc: {va:.4f} | fw: {func_weight:.2f} | Best: {best_val_acc:.4f}", flush=True)

        if no_improve >= 80 and epoch > 150:
            print(f"  Early stop at epoch {epoch}")
            break

    sane.load_state_dict(best_sane_state)
    sane.to(device)
    print(f"  Best AE reconstruction accuracy: {best_val_acc:.4f}")

    # Save AE checkpoint
    ckpt_dir = Path("data/v4c/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_sane_state, "w_mean": w_mean.cpu(), "w_std": w_std.cpu()},
               ckpt_dir / "sane_ae_v5.pt")

    # =================================================================
    # STAGE 2: Latent HyperNetwork (functional loss through frozen decoder)
    # =================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Latent HyperNetwork")
    print(f"{'='*60}")

    sane.eval()
    with torch.no_grad():
        all_latents = sane.encode(all_tokens, ref_positions.unsqueeze(0).expand(n_train, -1, -1))
    print(f"  Latent shape: {all_latents.shape}")

    hypernet = LatentHyperNetwork(n_tokens=n_tokens, d_z=d_z).to(device)
    hn_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  HyperNet params: {hn_params:,}")

    for p in sane.parameters():
        p.requires_grad = False

    hn_opt = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
    hn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hn_opt, T_max=200)
    best_val = 0
    best_hn_state = None
    no_improve = 0

    for epoch in range(1, 201):
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
            hn_opt.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                z_pred = hypernet(protos)
                recon_tokens = sane.decode(z_pred.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w_norm = detokenize_to_flat(recon_tokens, all_masks[0], target_cfg)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                logits = differentiable_forward(gen_w, imgs, target_cfg)
                func_loss = F.cross_entropy(logits, labs).clamp(max=5.0)
                latent_mse = F.mse_loss(z_pred, all_latents[idx])
                bl = bl + func_loss + 0.1 * latent_mse

                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 5.0)
            hn_opt.step()
            el += bl.item()
            nb += 1

        hn_sched.step()

        hypernet.eval()
        vc = vt = 0
        with torch.no_grad():
            for idx in range(min(40, n_train)):
                classes = train_classes[idx]
                protos = sample_prototypes(ci_test, classes, 20, device)
                imgs, labs = sample_task_data(ci_test, classes, 50, device)
                z = hypernet(protos)
                recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w = detokenize_to_flat(recon, all_masks[0], target_cfg) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, target_cfg).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val:
            best_val = va
            best_hn_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ec/et:.4f} | Val: {va:.4f} | Best: {best_val:.4f}", flush=True)
        if no_improve >= 50:
            print(f"  Early stop at epoch {epoch}")
            break

    hypernet.load_state_dict(best_hn_state)
    hypernet.to(device)

    torch.save({"state": best_hn_state}, ckpt_dir / "hypernet_v5.pt")

    # =================================================================
    # STAGE 3: End-to-end fine-tuning (everything unfrozen)
    # =================================================================
    print(f"\n{'='*60}")
    print("STAGE 3: End-to-end fine-tuning")
    print(f"{'='*60}")

    for p in sane.parameters():
        p.requires_grad = True

    e2e_opt = torch.optim.Adam([
        {"params": sane.parameters(), "lr": 5e-5},
        {"params": hypernet.parameters(), "lr": 2e-4},
    ], weight_decay=1e-5)
    e2e_sched = torch.optim.lr_scheduler.CosineAnnealingLR(e2e_opt, T_max=150)
    best_e2e_val = 0
    no_improve = 0

    for epoch in range(1, 151):
        sane.train()
        hypernet.train()
        perm = torch.randperm(n_train)
        ec = et = nb = 0
        el = 0.0

        for start in range(0, n_train, 8):
            batch_idx = perm[start:start + 8]
            e2e_opt.zero_grad()
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                z_pred = hypernet(protos)
                recon_tokens = sane.decode(z_pred.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w = detokenize_to_flat(recon_tokens, all_masks[0], target_cfg) * w_std.squeeze(0) + w_mean.squeeze(0)

                logits = differentiable_forward(gen_w, imgs, target_cfg)
                bl = bl + F.cross_entropy(logits, labs).clamp(max=5.0)

                with torch.no_grad():
                    ec += (logits.argmax(1) == labs).sum().item()
                    et += labs.size(0)

            bl = bl / len(batch_idx)
            bl.backward()
            torch.nn.utils.clip_grad_norm_(list(sane.parameters()) + list(hypernet.parameters()), 5.0)
            e2e_opt.step()
            el += bl.item()
            nb += 1

        e2e_sched.step()

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
                gen_w = detokenize_to_flat(recon, all_masks[0], target_cfg) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, target_cfg).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_e2e_val:
            best_e2e_val = va
            best_sane_state = {k: v.cpu().clone() for k, v in sane.state_dict().items()}
            best_hn_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ec/et:.4f} | Val: {va:.4f} | Best: {best_e2e_val:.4f}", flush=True)
        if no_improve >= 50:
            print(f"  Early stop at epoch {epoch}")
            break

    sane.load_state_dict(best_sane_state)
    hypernet.load_state_dict(best_hn_state)
    sane.to(device)
    hypernet.to(device)

    # Save final checkpoint
    torch.save({
        "sane": best_sane_state,
        "hypernet": best_hn_state,
        "w_mean": w_mean.cpu(),
        "w_std": w_std.cpu(),
    }, ckpt_dir / "sane_func_v5.pt")
    print(f"  Saved final checkpoint to {ckpt_dir / 'sane_func_v5.pt'}")

    # =================================================================
    # STAGE 4: Eval across 0/1/2/3 unseen classes
    # =================================================================
    print(f"\n{'='*60}")
    print("STAGE 4: Eval across novelty levels (0/1/2/3 unseen)")
    print(f"{'='*60}")

    seen = set(range(50))
    unseen = set(range(50, 62))
    rng = random.Random(42)

    def make_tasks(n_unseen_in_task, count=50):
        tasks = []
        attempts = 0
        while len(tasks) < count and attempts < 10000:
            attempts += 1
            u = rng.sample(list(unseen), n_unseen_in_task)
            s = rng.sample(list(seen), 3 - n_unseen_in_task)
            t = sorted(u + s)
            if tuple(t) not in {tuple(x) for x in tasks}:
                tasks.append(t)
        return tasks

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            std_val = (2.0 / fan_in) ** 0.5
            w_size = dims[i] * dims[i + 1]
            b_size = dims[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std_val
            idx += w_size + b_size
        return w

    sane.eval()
    hypernet.eval()
    STEPS = 3000

    for n_unseen in [0, 1, 2, 3]:
        label = 'seen-only (novel combos)' if n_unseen == 0 else f'{n_unseen}/3 unseen'
        tasks = make_tasks(n_unseen, 50)

        zs, ft_gen, ft_kai = [], [], []
        for ti, classes in enumerate(tasks):
            protos = sample_prototypes(ci_train, classes, 20, device)
            with torch.no_grad():
                z = hypernet(protos)
                recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w = detokenize_to_flat(recon, all_masks[0], target_cfg) * w_std.squeeze(0) + w_mean.squeeze(0)

            imgs, labs = sample_task_data(ci_test, classes, 200, device)
            with torch.no_grad():
                zs.append((differentiable_forward(gen_w, imgs, target_cfg).argmax(1) == labs).float().mean().item())

            for init_w, store in [(gen_w.clone(), ft_gen), (kaiming_init(), ft_kai)]:
                w = init_w.detach().requires_grad_(True)
                opt = torch.optim.SGD([w], lr=0.01)
                for step in range(STEPS):
                    i2, l2 = sample_task_data(ci_train, classes, 50, device)
                    opt.zero_grad()
                    F.cross_entropy(differentiable_forward(w, i2, target_cfg), l2).backward()
                    opt.step()
                imgs, labs = sample_task_data(ci_test, classes, 200, device)
                with torch.no_grad():
                    store.append((differentiable_forward(w.detach(), imgs, target_cfg).argmax(1) == labs).float().mean().item())

            if (ti + 1) % 10 == 0:
                print(f"    {label}: {ti+1}/{len(tasks)} tasks done", flush=True)

        g = np.array(ft_gen)
        k = np.array(ft_kai)
        gap = g - k
        wins = int(np.sum(gap > 0))
        t_stat, t_p = stats.ttest_rel(g, k)

        print(f"\n  {label}:")
        print(f"    Zero-shot:      {np.mean(zs):.4f} +- {np.std(zs):.4f}")
        print(f"    Gen +{STEPS}FT:  {g.mean():.4f} +- {g.std():.4f}")
        print(f"    Kai +{STEPS}FT:  {k.mean():.4f} +- {k.std():.4f}")
        print(f"    Gap: {gap.mean()*100:+.2f}pp, Wins: {wins}/{len(tasks)}, p={t_p:.6f}")

    print(f"\n{'='*60}")
    print("SCALE V5 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
