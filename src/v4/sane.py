"""
SANE-Inspired Weight Tokenization + Transformer AE
====================================================
Based on "Towards Scalable and Versatile Weight Space Learning" (Schurholt et al., ICML 2024)

Key idea: instead of flattening all weights into one vector, tokenize per output-neuron
and process the sequence with a Transformer autoencoder.

For our 784->128->64->3 MLP:
  Layer 1: 128 neurons, each with 785 params (784 weights + 1 bias) -> 128 tokens of 785D
  Layer 2: 64 neurons, each with 129 params (128 weights + 1 bias)  -> 64 tokens of 129D
  Layer 3: 3 neurons, each with 65 params (64 weights + 1 bias)     -> 3 tokens of 65D
  Total: 195 tokens. Pad all to token_size=785. Process with Transformer.

This is ~200x fewer dimensions per token than the full 109K vector.
"""

import math
import itertools
import random
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


def tokenize_mlp_weights(
    flat_weights: torch.Tensor,
    target_cfg: TargetMLPConfig,
    token_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize a flat MLP weight vector into per-neuron tokens.

    Returns:
        tokens: (N_tokens, token_size) padded weight tokens
        mask: (N_tokens, token_size) bool mask (True = real weight)
        positions: (N_tokens, 2) integer positions [layer_idx, neuron_idx]
    """
    dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
    tokens, masks, positions = [], [], []
    idx = 0

    for layer_i in range(len(dims) - 1):
        fan_in, fan_out = dims[layer_i], dims[layer_i + 1]
        w_size = fan_in * fan_out
        b_size = fan_out

        W = flat_weights[idx:idx + w_size].view(fan_out, fan_in)  # (out, in)
        b = flat_weights[idx + w_size:idx + w_size + b_size]  # (out,)
        idx += w_size + b_size

        for neuron_i in range(fan_out):
            # Token = [weights_to_this_neuron, bias]
            neuron_params = torch.cat([W[neuron_i], b[neuron_i:neuron_i + 1]])  # (fan_in + 1,)
            param_len = neuron_params.shape[0]

            # Pad to token_size
            if param_len < token_size:
                padded = torch.zeros(token_size, device=flat_weights.device)
                padded[:param_len] = neuron_params
                mask_vec = torch.zeros(token_size, dtype=torch.bool, device=flat_weights.device)
                mask_vec[:param_len] = True
            else:
                padded = neuron_params[:token_size]
                mask_vec = torch.ones(token_size, dtype=torch.bool, device=flat_weights.device)

            tokens.append(padded)
            masks.append(mask_vec)
            positions.append([layer_i, neuron_i])

    return torch.stack(tokens), torch.stack(masks), torch.tensor(positions, device=flat_weights.device)


def detokenize_to_flat(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    target_cfg: TargetMLPConfig,
) -> torch.Tensor:
    """Reconstruct flat weight vector from tokens."""
    dims = [target_cfg.input_dim] + target_cfg.hidden_dims + [target_cfg.num_classes]
    parts = []
    token_idx = 0

    for layer_i in range(len(dims) - 1):
        fan_in, fan_out = dims[layer_i], dims[layer_i + 1]
        W_rows = []
        biases = []
        for neuron_i in range(fan_out):
            tok = tokens[token_idx]
            W_rows.append(tok[:fan_in])
            biases.append(tok[fan_in:fan_in + 1])
            token_idx += 1
        parts.append(torch.stack(W_rows).flatten())  # W matrix flattened
        parts.append(torch.cat(biases))  # bias vector

    return torch.cat(parts)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_layers: int = 16, max_neurons: int = 256):
        super().__init__()
        d_half = d_model // 2
        self.emb_layer = nn.Embedding(max_layers, d_half)
        self.emb_neuron = nn.Embedding(max_neurons, d_model - d_half)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, N, 2) or (N, 2)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        e_l = self.emb_layer(positions[:, :, 0])
        e_n = self.emb_neuron(positions[:, :, 1])
        return torch.cat([e_l, e_n], dim=-1)


class SANEAutoencoder(nn.Module):
    """Transformer autoencoder for weight tokens."""

    def __init__(self, token_dim: int = 785, d_model: int = 256,
                 d_z: int = 32, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_z = d_z

        self.tokenizer = nn.Linear(token_dim, d_model)
        self.pos_emb = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Linear(d_model, d_z)

        self.from_latent = nn.Linear(d_z, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.detokenizer = nn.Linear(d_model, token_dim)

    def encode(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(tokens) + self.pos_emb(positions)
        x = self.dropout(x)
        x = self.encoder(x)
        return self.to_latent(x)  # (B, N, d_z)

    def decode(self, z: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z) + self.pos_emb(positions)
        x = self.dropout(x)
        x = self.decoder(x)
        return self.detokenizer(x)  # (B, N, token_dim)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor):
        z = self.encode(tokens, positions)
        recon = self.decode(z, positions)
        return z, recon


class LatentHyperNetwork(nn.Module):
    """Generates per-token latent codes from prototypes."""

    def __init__(self, n_tokens: int, d_z: int = 32,
                 prototype_dim: int = 128, num_classes_per_task: int = 3,
                 input_dim: int = 784):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_z = d_z
        self.prototype_encoder = PrototypeEncoder(
            input_dim=input_dim, hidden_dim=256, output_dim=prototype_dim
        )
        concat_dim = prototype_dim * num_classes_per_task
        # Generate all token latents at once: concat_dim -> n_tokens * d_z
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, n_tokens * d_z),
        )

    def forward(self, prototypes: List[torch.Tensor]) -> torch.Tensor:
        embs = [self.prototype_encoder(p) for p in prototypes]
        task_emb = torch.cat(embs, dim=0)
        flat = self.net(task_emb)
        return flat.view(self.n_tokens, self.d_z)  # (N_tokens, d_z)


def run():
    cfg = Config()
    cfg.target = TargetMLPConfig(input_dim=784, hidden_dims=[128, 64], num_classes=3)
    device = cfg.hypernet.device
    weight_dim = cfg.target_weight_dim()

    # Compute token structure
    dims = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
    n_tokens = sum(dims[i + 1] for i in range(len(dims) - 1))  # 128 + 64 + 3 = 195
    max_neuron_params = max(dims[i] + 1 for i in range(len(dims) - 1))  # 785 (784+1)
    token_size = max_neuron_params

    print(f"Target: {cfg.target.input_dim} -> {cfg.target.hidden_dims} -> {cfg.target.num_classes}")
    print(f"Target params: {weight_dim:,}")
    print(f"Tokens: {n_tokens}, Token size: {token_size}")
    print(f"Total token dims: {n_tokens * token_size:,} (vs flat {weight_dim:,})")

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

    # Tokenize all zoo models
    print("Tokenizing zoo weights...")
    all_tokens = []
    all_masks = []
    ref_positions = None
    for i in range(n_train):
        tokens, mask, positions = tokenize_mlp_weights(train_norm[i], cfg.target, token_size)
        all_tokens.append(tokens)
        all_masks.append(mask)
        if ref_positions is None:
            ref_positions = positions
    all_tokens = torch.stack(all_tokens)  # (N_train, N_tokens, token_size)
    all_masks = torch.stack(all_masks)
    positions = ref_positions.unsqueeze(0).expand(n_train, -1, -1)  # (N_train, N_tokens, 2)
    print(f"Tokenized: {all_tokens.shape}")

    # Verify detokenization roundtrip
    recon_flat = detokenize_to_flat(all_tokens[0], all_masks[0], cfg.target)
    assert torch.allclose(recon_flat, train_norm[0], atol=1e-5), "Tokenization roundtrip failed!"
    print("Tokenization roundtrip: OK")

    # ===================================================================
    # STAGE 1: Train SANE Autoencoder
    # ===================================================================
    d_z = 32
    sane = SANEAutoencoder(token_dim=token_size, d_model=256, d_z=d_z,
                           n_heads=8, n_layers=4, dropout=0.1).to(device)
    sane_params = sum(p.numel() for p in sane.parameters())
    print(f"\nSANE AE params: {sane_params:,}")
    print(f"Latent: {n_tokens} tokens x {d_z}D = {n_tokens * d_z}D total")

    n_ae_val = 20
    ae_train_tokens = all_tokens[:n_train - n_ae_val]
    ae_val_tokens = all_tokens[n_train - n_ae_val:]
    ae_train_pos = positions[:n_train - n_ae_val]
    ae_val_pos = positions[n_train - n_ae_val:]
    ae_train_masks = all_masks[:n_train - n_ae_val]
    ae_val_masks = all_masks[n_train - n_ae_val:]

    sane_opt = torch.optim.Adam(sane.parameters(), lr=1e-3, weight_decay=1e-4)
    sane_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sane_opt, T_max=500)
    BS = 16
    best_val = float('inf')
    best_sane_state = None
    no_improve = 0

    print(f"\n--- Training SANE AE ---")
    for epoch in range(1, 501):
        sane.train()
        perm = torch.randperm(len(ae_train_tokens))
        el = nb = 0
        for start in range(0, len(perm), BS):
            batch_idx = perm[start:start + BS]
            batch_t = ae_train_tokens[batch_idx]
            batch_p = ae_train_pos[batch_idx]
            batch_m = ae_train_masks[batch_idx]

            sane_opt.zero_grad()
            z, recon = sane(batch_t, batch_p)
            # Masked MSE
            diff = (recon - batch_t) ** 2
            loss = (diff * batch_m.float()).sum() / batch_m.float().sum()
            loss.backward()
            sane_opt.step()
            el += loss.item()
            nb += 1
        sane_sched.step()

        # Val
        sane.eval()
        with torch.no_grad():
            z_v, recon_v = sane(ae_val_tokens, ae_val_pos)
            diff_v = (recon_v - ae_val_tokens) ** 2
            val_loss = (diff_v * ae_val_masks.float()).sum() / ae_val_masks.float().sum()
            val_loss = val_loss.item()

        if val_loss < best_val:
            best_val = val_loss
            best_sane_state = {k: v.cpu().clone() for k, v in sane.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            # Functional eval
            sane.eval()
            func_accs = []
            with torch.no_grad():
                for idx in range(min(20, n_train)):
                    z_i, recon_i = sane(all_tokens[idx:idx+1], positions[idx:idx+1])
                    recon_flat = detokenize_to_flat(recon_i.squeeze(0), all_masks[idx], cfg.target)
                    recon_w = recon_flat * w_std.squeeze(0) + w_mean.squeeze(0)
                    classes = train_classes[idx]
                    imgs, labs = sample_task_data(ci_test, classes, 100, device)
                    logits = differentiable_forward(recon_w, imgs, cfg.target)
                    func_accs.append((logits.argmax(1) == labs).float().mean().item())
            print(f"  Epoch {epoch:3d} | Train: {el/nb:.6f} | Val: {val_loss:.6f} | Recon Acc: {np.mean(func_accs):.4f}", flush=True)

        if no_improve >= 100:
            print(f"  Early stop at {epoch}")
            break

    sane.load_state_dict(best_sane_state)
    sane.to(device)
    print(f"  Best val loss: {best_val:.6f}")

    # Save
    sane_path = Path("data/v4b/checkpoints/sane_ae.pt")
    sane_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_sane_state, "w_mean": w_mean.cpu(), "w_std": w_std.cpu()}, sane_path)

    # ===================================================================
    # STAGE 2: Train Latent HyperNetwork
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Latent HyperNetwork (prototypes -> {n_tokens}x{d_z}D)")
    print(f"{'='*60}")

    # Pre-compute latent codes
    sane.eval()
    with torch.no_grad():
        all_latents = sane.encode(all_tokens, positions)  # (N, N_tokens, d_z)
    print(f"  Latent shape: {all_latents.shape}")

    hypernet = LatentHyperNetwork(
        n_tokens=n_tokens, d_z=d_z
    ).to(device)
    hn_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  HyperNet params: {hn_params:,}")
    print(f"  HyperNet output: {n_tokens * d_z} = {n_tokens}x{d_z}")

    # Freeze SANE decoder
    for p in sane.parameters():
        p.requires_grad = False

    hn_opt = torch.optim.Adam(hypernet.parameters(), lr=5e-4, weight_decay=1e-5)
    hn_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hn_opt, T_max=300)
    best_val_acc = 0
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
            bl = torch.tensor(0.0, device=device)

            for i in batch_idx:
                idx = i.item()
                classes = train_classes[idx]
                protos = sample_prototypes(ci_train, classes, 20, device)
                imgs, labs = sample_task_data(ci_train, classes, 60, device)

                # Generate latent tokens
                z_pred = hypernet(protos)  # (N_tokens, d_z)

                # Decode to weight tokens
                recon_tokens = sane.decode(z_pred.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w_norm = detokenize_to_flat(recon_tokens, all_masks[0], cfg.target)
                gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)

                # Functional loss
                logits = differentiable_forward(gen_w, imgs, cfg.target)
                func_loss = F.cross_entropy(logits, labs)
                # Latent MSE
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
                recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
                gen_w = detokenize_to_flat(recon, all_masks[0], cfg.target) * w_std.squeeze(0) + w_mean.squeeze(0)
                vc += (differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).sum().item()
                vt += labs.size(0)
        va = vc / vt if vt > 0 else 0

        if va > best_val_acc:
            best_val_acc = va
            best_hn_state = {k: v.cpu().clone() for k, v in hypernet.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {el/nb:.4f} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val_acc:.4f}", flush=True)
        if no_improve >= 60:
            print(f"  Early stop at {epoch}")
            break

    hypernet.load_state_dict(best_hn_state)
    hypernet.to(device)

    # ===================================================================
    # STAGE 3: All-unseen eval
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 3: All-unseen eval (SANE, 109K params)")
    print(f"{'='*60}")

    unseen_classes = list(range(50, 62))
    all_tasks = [list(c) for c in itertools.combinations(unseen_classes, 3)]
    rng = random.Random(99)
    tasks = rng.sample(all_tasks, 50)

    def kaiming_init():
        w = torch.zeros(weight_dim, device=device)
        idx = 0
        dims_t = [cfg.target.input_dim] + cfg.target.hidden_dims + [cfg.target.num_classes]
        for i in range(len(dims_t) - 1):
            fan_in = dims_t[i]
            std = (2.0 / fan_in) ** 0.5
            w_size = dims_t[i] * dims_t[i + 1]
            b_size = dims_t[i + 1]
            w[idx:idx + w_size] = torch.randn(w_size, device=device) * std
            idx += w_size + b_size
        return w

    zs, ft_gen, ft_kai = [], [], []
    STEPS = 3000

    for ti, classes in enumerate(tasks):
        protos = sample_prototypes(ci_train, classes, 20, device)
        with torch.no_grad():
            z = hypernet(protos)
            recon = sane.decode(z.unsqueeze(0), ref_positions.unsqueeze(0)).squeeze(0)
            gen_w = detokenize_to_flat(recon, all_masks[0], cfg.target) * w_std.squeeze(0) + w_mean.squeeze(0)

        imgs, labs = sample_task_data(ci_test, classes, 200, device)
        with torch.no_grad():
            zs.append((differentiable_forward(gen_w, imgs, cfg.target).argmax(1) == labs).float().mean().item())

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

    print(f"\nSANE-BASED 109K PARAMS — ALL-UNSEEN (3/3), n={len(tasks)}")
    print(f"{'='*60}")
    print(f"Zero-shot:            {np.mean(zs):.4f} +- {np.std(zs):.4f}")
    print(f"Generated +{STEPS}FT:   {g.mean():.4f} +- {g.std():.4f}")
    print(f"Kaiming +{STEPS}FT:     {k.mean():.4f} +- {k.std():.4f}")
    print(f"Gap: {gap.mean()*100:+.2f}pp, Wins: {wins}/{len(tasks)}, p={t_p:.8f}")


if __name__ == "__main__":
    run()
