"""
Tournament Classifier — Adaptive specialist generation for full EMNIST.

Uses the hypernetwork to generate specialists ON-THE-FLY for each test image,
organized as a 2-stage tournament bracket:

  Stage 1: Prototype encoder picks top-9 candidate classes.
           Generate 3 specialists for groups {1,2,3}, {4,5,6}, {7,8,9}.
           Each specialist picks its winner.

  Stage 2: 3 winners compete in a final specialist.
           Generate 1 specialist for {winner_A, winner_B, winner_C}.
           Output = final classification.

This avoids the OOD problem of static ensembles: every specialist operates
within its designed 3-class scope on relevant candidates.

Usage:
  python eval_tournament.py
  python eval_tournament.py --top-k 9 --device mps
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.prototype.config import Config, TargetMLPConfig
from src.prototype.models import HyperNetwork, differentiable_forward
from src.prototype.train import build_class_image_index, sample_prototypes

CACHE_DIR = Path("data/v4/cache")

EMNIST_BYCLASS_NAMES = {
    **{i: str(i) for i in range(10)},
    **{i + 10: chr(ord("A") + i) for i in range(26)},
    **{i + 36: chr(ord("a") + i) for i in range(26)},
}


def load_hypernet(checkpoint_path: str, device: str):
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
    hypernet.to(device)
    hypernet.eval()
    w_mean = data["normalization"]["w_mean"].to(device)
    w_std = data["normalization"]["w_std"].to(device)
    return hypernet, w_mean, w_std


def build_class_prototypes(
    proto_encoder, class_images, all_classes, K, device, n_samples=5,
) -> Dict[int, torch.Tensor]:
    proto_encoder.eval()
    embeds = {}
    for c in all_classes:
        parts = []
        for _ in range(n_samples):
            imgs = class_images[c]
            idx = torch.randperm(len(imgs))[:K]
            with torch.no_grad():
                emb = proto_encoder(imgs[idx].to(device))
            parts.append(emb)
        embeds[c] = torch.stack(parts).mean(dim=0)
    return embeds


def generate_weights(hypernet, w_mean, w_std, classes, class_images, K, device, n_samples=3):
    """Generate specialist weights for a 3-class task."""
    weights_accum = []
    for _ in range(n_samples):
        prototypes = sample_prototypes(class_images, classes, K, device)
        with torch.no_grad():
            gen_w_norm = hypernet(prototypes)
            gen_w = gen_w_norm * w_std.squeeze(0) + w_mean.squeeze(0)
        weights_accum.append(gen_w)
    return torch.stack(weights_accum).mean(dim=0)


def tournament_classify_batch(
    images: torch.Tensor,
    candidate_classes: torch.Tensor,
    hypernet: HyperNetwork,
    w_mean: torch.Tensor,
    w_std: torch.Tensor,
    class_images: Dict[int, torch.Tensor],
    K: int,
    target_cfg: TargetMLPConfig,
    device: str,
    n_proto_samples: int = 3,
) -> torch.Tensor:
    """Tournament classification for a batch of images.

    Args:
        images: (B, 784) batch of images
        candidate_classes: (B, 9) top-9 class indices per image
        ...
    Returns:
        predictions: (B,) global class predictions
    """
    B = images.size(0)
    candidates = candidate_classes  # (B, 9)

    # --- Stage 1: 3 groups of 3 → 3 winners ---
    # Seeded by prototype distance: top-3 candidates go to different groups.
    # Rank 1,4,7 → group 0 | Rank 2,5,8 → group 1 | Rank 3,6,9 → group 2
    # This ensures the strongest candidates only meet in the final.
    seed_order = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]  # indices into candidates (already sorted by similarity)
    round1_winners = torch.zeros(B, 3, dtype=torch.long)

    for group_idx, seed_indices in enumerate(seed_order):
        group_classes = candidates[:, seed_indices]  # (B, 3)

        # Find unique class triplets to avoid redundant specialist generation
        unique_triplets = {}
        for i in range(B):
            triplet = tuple(sorted(group_classes[i].tolist()))
            if triplet not in unique_triplets:
                unique_triplets[triplet] = []
            unique_triplets[triplet].append(i)

        for triplet, indices in unique_triplets.items():
            classes = list(triplet)
            # Generate specialist for this triplet
            weights = generate_weights(
                hypernet, w_mean, w_std, classes, class_images, K, device, n_proto_samples,
            )
            # Run images through specialist
            batch_imgs = images[indices].to(device)
            with torch.no_grad():
                logits = differentiable_forward(weights, batch_imgs, target_cfg)
                local_winners = logits.argmax(dim=1)  # (len(indices),) indices 0-2
            # Map local winners back to global class IDs
            sorted_classes = sorted(classes)
            for j, img_idx in enumerate(indices):
                round1_winners[img_idx, group_idx] = sorted_classes[local_winners[j].item()]

    # --- Stage 2: 3 winners compete → final prediction ---
    predictions = torch.zeros(B, dtype=torch.long)

    unique_finals = {}
    for i in range(B):
        triplet = tuple(sorted(round1_winners[i].tolist()))
        if triplet not in unique_finals:
            unique_finals[triplet] = []
        unique_finals[triplet].append(i)

    for triplet, indices in unique_finals.items():
        classes = list(triplet)
        if len(set(classes)) < 3:
            # Degenerate case: duplicate winners. Fill with unique classes.
            unique = list(set(classes))
            if len(unique) == 1:
                # All same class — just predict it
                for idx in indices:
                    predictions[idx] = unique[0]
                continue
            elif len(unique) == 2:
                # Two unique — add a dummy (won't matter, the duplicate already won twice)
                # Just pick the duplicate as winner
                from collections import Counter
                c = Counter(classes)
                winner = c.most_common(1)[0][0]
                for idx in indices:
                    predictions[idx] = winner
                continue

        weights = generate_weights(
            hypernet, w_mean, w_std, classes, class_images, K, device, n_proto_samples,
        )
        batch_imgs = images[indices].to(device)
        with torch.no_grad():
            logits = differentiable_forward(weights, batch_imgs, target_cfg)
            local_winners = logits.argmax(dim=1)
        sorted_classes = sorted(classes)
        for j, img_idx in enumerate(indices):
            predictions[img_idx] = sorted_classes[local_winners[j].item()]

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Tournament classifier using adaptive hypernetwork specialists")
    parser.add_argument("--top-k", type=int, default=9, help="Top-K candidates from prototype similarity")
    parser.add_argument("--proto-samples", type=int, default=5, help="Prototype draws for class embeddings")
    parser.add_argument("--gen-samples", type=int, default=3, help="Prototype draws for weight generation")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512, help="Tournament batch size")
    parser.add_argument("--checkpoint", default=None, help="Hypernetwork checkpoint path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = Config()
    device = args.device
    K = cfg.hypernet.prototypes_per_class

    all_classes = sorted(set(cfg.zoo.seen_classes + cfg.zoo.unseen_classes))
    num_classes = max(all_classes) + 1
    print(f"EMNIST ByClass: {len(all_classes)} classes (0-{max(all_classes)})")

    # Load hypernetwork
    print("Loading hypernetwork...")
    default_ckpt = "data/v4_fullclass/checkpoints/hypernet.pt"
    ckpt_path = args.checkpoint if hasattr(args, 'checkpoint') and args.checkpoint else default_ckpt
    hypernet, w_mean, w_std = load_hypernet(ckpt_path, device)
    hypernet_params = sum(p.numel() for p in hypernet.parameters())
    specialist_params = cfg.target_weight_dim()
    print(f"  Hypernetwork params: {hypernet_params:,}")
    print(f"  Specialist params: {specialist_params:,}")

    # Load data (cached)
    cache_path = CACHE_DIR / "class_images.pt"
    if cache_path.exists():
        print("Loading cached class image indices...")
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        class_images_train = cached["train"]
        class_images_test = cached["test"]
    else:
        print("Loading EMNIST data (will cache)...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        emnist_train = datasets.EMNIST("data/v4/raw", split="byclass", train=True, download=False, transform=transform)
        emnist_test = datasets.EMNIST("data/v4/raw", split="byclass", train=False, download=False, transform=transform)
        class_images_train = build_class_image_index(emnist_train)
        class_images_test = build_class_image_index(emnist_test)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"train": class_images_train, "test": class_images_test}, cache_path)

    # Build test tensors
    test_imgs, test_lbls = [], []
    for c in sorted(class_images_test.keys()):
        test_imgs.append(class_images_test[c])
        test_lbls.append(torch.full((len(class_images_test[c]),), c, dtype=torch.long))
    all_test_images = torch.cat(test_imgs)
    all_test_labels = torch.cat(test_lbls)
    N = len(all_test_labels)
    print(f"Test set: {N:,} images")

    target_cfg = TargetMLPConfig(
        input_dim=cfg.target.input_dim,
        hidden_dims=cfg.target.hidden_dims,
        num_classes=cfg.target.num_classes,
    )

    # Build class prototype embeddings
    print("Building class prototype embeddings...")
    proto_encoder = hypernet.prototype_encoder
    class_embeds = build_class_prototypes(
        proto_encoder, class_images_train, all_classes, K, device, n_samples=args.proto_samples,
    )

    # Compute test image embeddings
    print("Computing test image embeddings...")
    proto_encoder.eval()
    img_embed_parts = []
    for start in range(0, N, 2048):
        batch = all_test_images[start : start + 2048].to(device)
        with torch.no_grad():
            emb = proto_encoder.encoder(batch)
        img_embed_parts.append(emb.cpu())
    image_embeds = torch.cat(img_embed_parts, dim=0)

    # Compute prototype similarities → top-K candidates
    embed_dim = next(iter(class_embeds.values())).shape[0]
    class_embed_mat = torch.zeros(num_classes, embed_dim)
    for c, emb in class_embeds.items():
        class_embed_mat[c] = emb.cpu()
    class_embed_norm = F.normalize(class_embed_mat, dim=1)
    img_embed_norm = F.normalize(image_embeds, dim=1)
    all_sims = img_embed_norm @ class_embed_norm.t()  # (N, C)

    topk_result = all_sims.topk(args.top_k, dim=1)
    topk_classes = topk_result.indices  # (N, 9)

    # Nearest-prototype baseline (top-1)
    ncm_preds = all_sims.argmax(dim=1)
    ncm_correct = (ncm_preds == all_test_labels).sum().item()
    ncm_acc = ncm_correct / N
    print(f"\nNearest-prototype (top-1): {ncm_acc:.4f}")

    # Top-K recall (is true class in the top-K?)
    for k in [3, 6, 9]:
        topk_k = all_sims.topk(k, dim=1).indices
        recall = sum(
            all_test_labels[i].item() in topk_k[i].tolist() for i in range(N)
        ) / N
        print(f"Top-{k} recall: {recall:.4f}")

    # --- Tournament classification ---
    print(f"\nRunning tournament (top-{args.top_k}, 3-3-3 → 1-1-1)...")
    t0 = time.time()

    all_preds = []
    BS = args.batch_size
    n_batches = (N + BS - 1) // BS

    for batch_idx in range(n_batches):
        start = batch_idx * BS
        end = min(start + BS, N)
        batch_imgs = all_test_images[start:end]
        batch_candidates = topk_classes[start:end]

        preds = tournament_classify_batch(
            batch_imgs, batch_candidates,
            hypernet, w_mean, w_std, class_images_train, K,
            target_cfg, device, n_proto_samples=args.gen_samples,
        )
        all_preds.append(preds)

        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            done = end
            eta = elapsed / done * (N - done) if done > 0 else 0
            running_preds = torch.cat(all_preds)
            running_acc = (running_preds == all_test_labels[:len(running_preds)]).float().mean().item()
            print(f"  [{done:,}/{N:,}] acc={running_acc:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    all_preds = torch.cat(all_preds)
    tournament_time = time.time() - t0

    # Overall accuracy
    correct = (all_preds == all_test_labels).sum().item()
    overall_acc = correct / N
    print(f"\nTournament accuracy: {overall_acc:.4f} ({tournament_time:.1f}s)")

    # Per-class accuracy
    print(f"\n{'Class':<8} {'Name':<6} {'N_test':<8} {'Tourn':<10} {'NCM':<10} {'Delta':<8}")
    print("-" * 56)

    per_class_tourn = {}
    per_class_ncm = {}
    for c in range(num_classes):
        mask = all_test_labels == c
        n = mask.sum().item()
        if n == 0:
            continue
        t_acc = (all_preds[mask] == c).float().mean().item()
        n_acc = (ncm_preds[mask] == c).float().mean().item()
        per_class_tourn[c] = t_acc
        per_class_ncm[c] = n_acc
        name = EMNIST_BYCLASS_NAMES.get(c, str(c))
        print(f"{c:<8} {name:<6} {n:<8} {t_acc:<10.4f} {n_acc:<10.4f} {t_acc - n_acc:+.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Nearest-prototype (top-1):  {ncm_acc:.4f}")
    print(f"  Tournament (top-{args.top_k}, 3+1):   {overall_acc:.4f}")
    print(f"  Delta:                      {overall_acc - ncm_acc:+.4f}")
    print(f"\n  Hypernetwork params:        {hypernet_params:,}")
    print(f"  Specialists per image:      4 (3 round-1 + 1 final)")
    print(f"  Specialist params:          {specialist_params:,}")
    print(f"  Inference time:             {tournament_time:.1f}s for {N:,} images")
    print(f"\n  SOTA (EMNIST ByClass 62cls):")
    print(f"    WaveMix:                  ~88.4%")
    print(f"    MLP (10K hidden):         ~70%")
    print(f"    Baseline MLP (1M):        ~84.6% (from ensemble experiment)")


if __name__ == "__main__":
    main()
