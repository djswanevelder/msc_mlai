#!/usr/bin/env python3
"""
Standalone script to encode ResNet18 models from a folder into latent space representations.

This script takes a trained weight space autoencoder and a folder of ResNet18 model files,
and encodes each model into its latent space representation, saving the results as a dictionary.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from src.data_prep.resNet18 import ResNetClassifier

class ResNet18WeightUtils:
    """Utilities for working with ResNet18 named parameters & flat vectors."""

    @staticmethod
    def create_resnet18_3class() -> nn.Module:
        """Create a randomly initialized ResNet18 with 3 output classes."""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model

    @staticmethod
    def param_specs(model: nn.Module) -> List[Dict[str, Any]]:
        """List of {"name":str, "shape":tuple, "numel":int} in model param order."""
        specs = []
        for name, p in model.named_parameters():
            specs.append({"name": name, "shape": tuple(p.shape), "numel": p.numel()})
        return specs

    @staticmethod
    def get_weight_vector(model: nn.Module) -> torch.Tensor:
        """Flatten all parameters to a single 1D tensor (no grad)."""
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    @staticmethod
    def set_weights_from_vector(model: nn.Module, weight_vector: torch.Tensor) -> None:
        """Write a flat vector back into model parameters (no grad path needed)."""
        idx = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(weight_vector[idx : idx + n].view_as(p))
                idx += n

    @staticmethod
    def total_params() -> int:
        """Get total number of parameters in a ResNet18 3-class model."""
        model = ResNet18WeightUtils.create_resnet18_3class()
        return sum(p.numel() for p in model.parameters())

class WeightSpaceAE(nn.Module):
    """
    Standalone autoencoder model for weight-space compression. We train the autoencoder
    to reconstruct PCA coefficients, which in turn reconstruct the full weight vector.

    So after training, we have:
        z  --(AE decode)-->  z_rec  --(PCA inverse)-->  w_rec
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = (
                [1024, 512] if input_dim >= 1024 else [max(256, input_dim // 2)]
            )

        # Encoder
        enc = []
        d = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ELU(), nn.Dropout(0.1)]
            d = h
        enc += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        # Decoder
        dec = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), nn.ELU(), nn.Dropout(0.1)]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

        # normalization stats (will be set by training module)
        self.register_buffer("z_mean", torch.zeros(input_dim))
        self.register_buffer("z_std", torch.ones(input_dim))

    def set_z_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics for PCA coefficients."""
        self.z_mean.copy_(mean)
        self.z_std.copy_(std.clamp_min(1e-4))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode PCA coefficients to latent representation."""
        return self.encoder(z)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to PCA coefficients."""
        return self.decoder(h)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns reconstructed coefficients and latent representation."""
        h = self.encode(z)
        z_rec = self.decode(h)
        return z_rec, h

class PerParamPCAMapper:
    """
    Adaptive per-named-parameter mapper:
      - mode "pca":      PCA with k_i = min(k_default, d_i-1)
      - mode "identity": no PCA; z = (x - mean)  (dims = d_i) for tiny tensors
      - mode "constant": near-constant param; no coeffs; reconstruct mean
    """

    def __init__(
        self,
        model_ref: nn.Module,
        dataset_dir: str,
        n_components: int = 8,
        min_dim_identity: int = 4,
        const_var_eps: float = 1e-8,
    ):
        from sklearn.decomposition import IncrementalPCA

        self.IncrementalPCA = IncrementalPCA
        self.dataset_dir = Path(dataset_dir)
        self.pca_dir = self.dataset_dir / "pca"
        self.pca_dir.mkdir(parents=True, exist_ok=True)

        self.specs = ResNet18WeightUtils.param_specs(model_ref)
        self.slices = []

        # creating the slices for each named parameter, so we can take slices of the flat vector easily (transforming and inverse transforming)
        start = 0
        for s in self.specs:
            end = start + s["numel"]
            self.slices.append(slice(start, end))
            start = end
        self.total_dims = start

        self.k_default = int(n_components)
        self.min_dim_identity = int(min_dim_identity)
        self.const_var_eps = float(const_var_eps)

        self.index_file = self.pca_dir / "index.json"
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _path(self, idx: int) -> Path:
        return self.pca_dir / f"param_{idx:04d}.npz"

    def is_fit(self) -> bool:
        return self.index_file.exists() and all(
            self._path(i).exists() for i in range(len(self.specs))
        )

    def load(self) -> None:
        """Load fitted PCA mappings from disk."""
        with open(self.index_file) as f:
            idx = json.load(f)
        self.modes = idx["modes"]
        self.k_list = idx["k_list"]
        self.specs = idx["specs"]
        self.total_dims = idx["total_dims"]
        self.coeff_dim = idx["coeff_dim"]

        # rebuild slices
        self.slices = []
        start = 0
        for s in self.specs:
            end = start + s["numel"]
            self.slices.append(slice(start, end))
            start = end

        # warm cache
        self._cache = {}
        for p_idx in range(len(self.specs)):
            npz = np.load(self._path(p_idx))
            mode = str(npz["mode"])
            entry = {"mode": mode, "mean": torch.from_numpy(npz["mean"]).float()}
            if mode == "pca":
                entry["components"] = torch.from_numpy(
                    npz["components"]
                ).float()  # (k_i, d_i)
            self._cache[p_idx] = entry

    def transform(self, flat_weights: torch.Tensor) -> torch.Tensor:
        """Transform flat weight vector to PCA coefficients."""
        if flat_weights.ndim != 1:
            flat_weights = flat_weights.flatten()
        zs = []
        fw = flat_weights.detach().cpu()

        for p_idx, sl in enumerate(self.slices):
            c = self._cache[p_idx]
            mode = c["mode"]
            x = fw[sl]
            if mode == "pca":
                z = torch.mv(c["components"], (x - c["mean"]))
                zs.append(z)
            elif mode == "identity":
                zs.append(x - c["mean"])
            else:
                # constant → no coeffs
                pass

        return torch.cat(zs) if zs else torch.zeros(0)

    def inverse_transform(self, coeffs_concat: torch.Tensor) -> torch.Tensor:
        """Transform PCA coefficients back to flat weight vector."""
        if coeffs_concat.ndim != 1:
            coeffs_concat = coeffs_concat.flatten()

        out = torch.empty(self.total_dims, dtype=torch.float32)
        pos_coeff = 0
        pos_out = 0

        for p_idx, sl in enumerate(self.slices):
            c = self._cache[p_idx]
            mode = c["mode"]
            d_i = sl.stop - sl.start  # get the original dimension size

            if mode == "pca":
                k_i = c["components"].shape[0]
                z = coeffs_concat[pos_coeff : pos_coeff + k_i]
                x = torch.mv(c["components"].t(), z) + c["mean"]
                out[pos_out : pos_out + d_i] = x
                pos_coeff += k_i
            elif mode == "identity":
                z = coeffs_concat[pos_coeff : pos_coeff + d_i]
                x = z + c["mean"]
                out[pos_out : pos_out + d_i] = x
                pos_coeff += d_i
            else:
                out[pos_out : pos_out + d_i] = c["mean"]
            pos_out += d_i

        return out

def load_autoencoder(
    checkpoint_path: str, dataset_dir: str, hidden_dims: Optional[list] = None
) -> tuple[WeightSpaceAE, PerParamPCAMapper]:
    """Load trained autoencoder and PCA mapper from disk."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Create autoencoder
    weight_ae = WeightSpaceAE(
        input_dim=675,
        latent_dim=512,
        hidden_dims=[],
    )
    weight_ae.load_state_dict(ckpt["model_state_dict"])
    weight_ae.set_z_stats(ckpt["z_mean"], ckpt["z_std"])

    # Create and load PCA mapper
    model_ref = ResNet18WeightUtils.create_resnet18_3class()
    mapper = PerParamPCAMapper(model_ref, dataset_dir)
    mapper.load()

    print(f"Loaded autoencoder from {checkpoint_path}")
    return weight_ae, mapper

def decode_latent_to_resnet_model(
    weight_ae, mapper,
    latent_vector: torch.Tensor
) -> torch.nn.Module:

    latent_vector = latent_vector.to('cpu').unsqueeze(0)

    with torch.no_grad():
        pca_coeffs_norm = weight_ae.decode(latent_vector).squeeze(0) 

    pca_coeffs = (pca_coeffs_norm * weight_ae.z_std) + weight_ae.z_mean
    flat_weights = mapper.inverse_transform(pca_coeffs.cpu())

    decoded_model = ResNet18WeightUtils.create_resnet18_3class()
    ResNet18WeightUtils.set_weights_from_vector(decoded_model, flat_weights)
    
    return decoded_model

def encode_resnet_model(
    model: torch.nn.Module, weight_ae: WeightSpaceAE, mapper: PerParamPCAMapper
) -> torch.Tensor:
    """Encode a ResNet model to latent space representation."""
    # Get flat weight vector
    flat_weights = ResNet18WeightUtils.get_weight_vector(model)

    # Transform to PCA coefficients
    pca_coeffs = mapper.transform(flat_weights)

    # Normalize
    device = next(weight_ae.parameters()).device
    pca_coeffs = pca_coeffs.to(device)
    pca_coeffs_norm = (pca_coeffs - weight_ae.z_mean) / weight_ae.z_std

    # Encode to latent space
    with torch.no_grad():
        weight_ae.eval()
        latent = weight_ae.encode(pca_coeffs_norm.unsqueeze(0))

    return latent.squeeze(0)


def encode_models_folder(
    model_dir: str,
    checkpoint_path: str,
    dataset_dir: str,
    output_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Encode all ResNet18 models in a folder to latent space representations.

    Args:
        model_dir: Directory containing ResNet18 model files (.pth/.pt)
        checkpoint_path: Path to trained autoencoder checkpoint
        dataset_dir: Dataset directory containing PCA mapper data
        output_path: Where to save the encoded latents dictionary
        device: Device to use for computation

    Returns:
        Dictionary mapping model filenames to latent vectors
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)

    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    # Find model files
    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    if not model_files:
        raise ValueError(f"No .pth or .pt files found in {model_dir}")

    print(f"Found {len(model_files)} model files to encode")

    # Load autoencoder and PCA mapper
    print("Loading autoencoder and PCA mapper...")
    weight_ae, mapper = load_autoencoder(checkpoint_path, dataset_dir)
    weight_ae = weight_ae.to(device)
    weight_ae.eval()

    # Encode models
    encoded_models = {}
    failed_models = []
    exp_params = ResNet18WeightUtils.total_params()

    print("Encoding models...")
    for model_file in tqdm(model_files, desc="Encoding models"):
        try:
            # Load model
            model = ResNet18WeightUtils.create_resnet18_3class()
            ckpt = torch.load(model_file, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(ckpt, dict) and (
                "state_dict" in ckpt or "model_state_dict" in ckpt
            ):
                state = ckpt.get("state_dict", ckpt.get("model_state_dict"))
            elif isinstance(ckpt, dict):
                state = ckpt
            elif hasattr(ckpt, "state_dict"):
                state = ckpt.state_dict()
            else:
                failed_models.append((model_file.name, "Invalid checkpoint format"))
                continue

            # Load weights
            model.load_state_dict(state, strict=False)

            # Verify parameter count
            flat_weights = ResNet18WeightUtils.get_weight_vector(model)
            if flat_weights.numel() != exp_params:
                failed_models.append((model_file.name, "Parameter count mismatch"))
                continue

            # Encode to latent space
            latent = encode_resnet_model(model, weight_ae, mapper)
            encoded_models[model_file.name] = latent.cpu()

        except Exception as e:
            failed_models.append((model_file.name, str(e)))

    # Save results
    print(f"Successfully encoded {len(encoded_models)} models")
    if failed_models:
        print(f"Failed to encode {len(failed_models)} models:")
        for name, error in failed_models[:5]:  # Show first 5 failures
            print(f"  {name}: {error}")
        if len(failed_models) > 5:
            print(f"  ... and {len(failed_models) - 5} more")

    # Save encoded models
    save_data = {
        "encoded_models": encoded_models,
        "metadata": {
            "total_models": len(model_files),
            "successful_encodings": len(encoded_models),
            "failed_encodings": len(failed_models),
            "latent_dim": weight_ae.latent_dim,
            "checkpoint_path": str(checkpoint_path),
            "dataset_dir": str(dataset_dir),
            "failed_models": failed_models,
        },
    }

    torch.save(save_data, output_path)
    print(f"Saved encoded models to {output_path}")

    return encoded_models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Encode ResNet18 models from a folder into latent space representations"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing ResNet18 model files (.pth/.pt)",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained autoencoder checkpoint",
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset directory containing PCA mapper data",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="encoded_latents.pth",
        help="Path to save encoded latents dictionary (default: encoded_latents.pth)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (default: cpu)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    try:
        encoded_models = encode_models_folder(
            model_dir=args.model_dir,
            checkpoint_path=args.checkpoint_path,
            dataset_dir=args.dataset_dir,
            output_path=args.output_path,
            device=args.device,
        )

        print(f"\nEncoding complete!")
        print(f"Encoded {len(encoded_models)} models to latent space")
        print(f"Results saved to: {args.output_path}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
