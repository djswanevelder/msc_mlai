import json
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models

from tqdm import tqdm
from loguru import logger
from PIL import Image
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Callback
from lightning.pytorch.loggers import WandbLogger
import wandb


class ResNet18WeightUtils:
    """Utilities for working with ResNet18 named parameters & flat vectors."""

    @staticmethod
    def create_resnet18_3class() -> nn.Module:
        # randomly initialized ResNet18 with 3 output classes
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
        model = ResNet18WeightUtils.create_resnet18_3class()
        return sum(p.numel() for p in model.parameters())


class WeightVectorDatasetCreator:
    """
    Creates and saves ResNet18 weight vectors from .pth files. We save a flattened version of it
    to make manipulation easier.

    This generate_dataset_from_models function creates a dataset of weight vectors from a directory of trained models,
    saving a new pth file for each model in the specified dataset directory, containing the flattened weight vector,
    the source file name, and the model index. It also saves metadata about the dataset in a JSON file.

    We feed into this the trained resnet18s from resnet18.py.
    """

    @staticmethod
    def generate_dataset_from_models(
        dataset_dir: str, models_dir: str, max_models: Optional[int] = None
    ) -> None:
        dpath = Path(dataset_dir)
        (dpath / "weights").mkdir(parents=True, exist_ok=True)
        mpath = Path(models_dir)
        if not mpath.exists():
            raise ValueError(f"{models_dir} does not exist")

        # check for pth or pt files
        pth_files = list(mpath.glob("*.pth")) + list(mpath.glob("*.pt"))
        if not pth_files:
            raise ValueError(f"No .pth or .pt files in {models_dir}")

        if max_models is not None:
            pth_files = pth_files[:max_models]

        exp_params = ResNet18WeightUtils.total_params()
        loaded = 0
        failed = []

        logger.info(f"Loading up to {len(pth_files)} trained models...")
        for pth in tqdm(pth_files):
            try:
                model = ResNet18WeightUtils.create_resnet18_3class()
                ckpt = torch.load(pth, map_location="cpu")

                # load trained weights into the resnet18
                if isinstance(ckpt, dict) and (
                    "state_dict" in ckpt or "model_state_dict" in ckpt
                ):
                    state = ckpt.get("state_dict", ckpt.get("model_state_dict"))
                elif isinstance(ckpt, dict):
                    state = ckpt
                elif hasattr(ckpt, "state_dict"):
                    state = ckpt.state_dict()
                else:
                    failed.append((pth.name, "bad format"))
                    continue

                model.load_state_dict(state, strict=False)
                flat = ResNet18WeightUtils.get_weight_vector(model).cpu()
                if flat.numel() != exp_params:
                    failed.append((pth.name, "param count mismatch"))
                    continue
                out = dpath / "weights" / f"model_{loaded:06d}.pt"
                torch.save(
                    {
                        "weight_vector": flat,
                        "source_file": pth.name,
                        "model_idx": loaded,
                    },
                    out,
                )
                loaded += 1
            except Exception as e:
                failed.append((pth.name, str(e)))

        meta = {
            "num_models": loaded,
            "input_dim": exp_params,
            "model_type": "resnet18_3class",
            "source": "trained_models",
            "failed_count": len(failed),
            "failed_examples": failed[:5],
        }
        with open(dpath / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Loaded {loaded} models; {len(failed)} failed.")


class WeightVectorDataset(Dataset):
    """Loads flat weight vectors from disk (used for PCA fitting and transforms)."""

    def __init__(self, dataset_dir: str):
        self.dpath = Path(dataset_dir)
        self.files = sorted((self.dpath / "weights").glob("model_*.pt"))
        if not self.files:
            raise ValueError(f"No weights in {self.dpath/'weights'}")

        with open(self.dpath / "metadata.json") as f:
            self.meta = json.load(f)
        self.input_dim = self.meta["input_dim"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> torch.Tensor:
        obj = torch.load(self.files[idx], map_location="cpu")
        flat = (
            obj["weight_vector"].float().clamp_(-10.0, 10.0)
        )  # clamping massive weights to help with smoothness during training
        return flat


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

    def _precompute_stats(
        self, dataset: WeightVectorDataset, max_models: int = 512
    ) -> Tuple[List[float], List[float]]:
        """Streaming mean/var per named param to detect constants & set modes."""
        n = min(max_models, len(dataset))
        sums = [0.0] * len(self.specs)
        sums2 = [0.0] * len(self.specs)
        counts = [0] * len(self.specs)

        # loop over our models, get the different statistics at various slices of the flat vector
        for i in range(n):
            flat = dataset[i].numpy()
            for p_idx, sl in enumerate(self.slices):
                x = flat[sl]
                sums[p_idx] += float(x.mean())
                sums2[p_idx] += float((x**2).mean())
                counts[p_idx] += 1

        means = [s / c for s, c in zip(sums, counts)]
        vars_ = [max(s2 / c - m * m, 0.0) for s2, c, m in zip(sums2, counts, means)]
        return means, vars_

    def fit(
        self,
        dataset: WeightVectorDataset,
        max_models_for_pca: int = 256,
        load_batch: int = 64,
        train_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Fast adaptive fit:
          - use only training data if train_indices provided
          - random-subselect up to max_models_for_pca models from training set
          - load flat vectors in batches (load_batch)
          - for each batch, slice once per param and partial_fit
        """
        # Use only training indices if provided
        if train_indices is not None:
            available_idxs = train_indices
            logger.info(
                f"Fitting per-named-parameter PCA on TRAINING SET ONLY: {len(train_indices)} models "
                f"(max_models_for_pca={max_models_for_pca}, load_batch={load_batch})..."
            )
        else:
            available_idxs = list(range(len(dataset)))
            logger.info(
                f"Fitting per-named-parameter PCA on ALL models: {len(dataset)} models "
                f"(max_models_for_pca={max_models_for_pca}, load_batch={load_batch})..."
            )

        # Randomly subsample from available indices
        idxs = list(available_idxs)
        random.shuffle(idxs)
        idxs = idxs[: min(max_models_for_pca, len(idxs))]
        idxs.sort()  # improves sequential disk reads

        # stats to decide modes
        _, vars_s = self._precompute_stats(dataset, max_models=min(512, len(idxs)))

        modes: List[str] = []
        k_list: List[int] = []
        ipcas: List[Optional[Any]] = []

        # setup all the ways we will reduce the dimensions (if we even downsample)
        for p_idx, s in enumerate(self.specs):
            d_i = s["numel"]  # original dimension size
            if vars_s[p_idx] < self.const_var_eps:
                modes.append("constant")
                k_list.append(0)
                ipcas.append(None)
            elif d_i <= self.min_dim_identity:
                modes.append("identity")
                k_list.append(d_i)
                ipcas.append(None)
            else:
                k_i = min(
                    self.k_default, d_i - 1
                )  # number of projected down dimensions
                modes.append("pca")
                k_list.append(k_i)
                ipcas.append(self.IncrementalPCA(n_components=k_i))

        # partial_fit in model-batches
        for bstart in tqdm(
            range(0, len(idxs), load_batch), desc="PCA batched partial_fit"
        ):
            bidx = idxs[bstart : bstart + load_batch]
            flats = [dataset[i].numpy() for i in bidx]
            X = np.stack(flats, axis=0)  # (B, D)
            for p_idx, sl in enumerate(self.slices):
                if modes[p_idx] != "pca":
                    continue
                Xi = X[:, sl]  # (B, d_i)
                if Xi.shape[0] >= max(k_list[p_idx], 2):
                    ipcas[p_idx].partial_fit(
                        Xi
                    )  # fit PCA on the batch of sliced weight vectors

        # save per-named-param artifacts, creating a unique .npz per named parameter
        for p_idx, s in enumerate(self.specs):
            sl = self.slices[p_idx]
            d_i = s["numel"]
            if modes[p_idx] == "pca":
                comp = ipcas[p_idx].components_.astype(np.float32)
                mean = ipcas[p_idx].mean_.astype(np.float32)
                np.savez_compressed(
                    self._path(p_idx),
                    mode="pca",
                    k=k_list[p_idx],
                    d=d_i,
                    components=comp,
                    mean=mean,
                )
            elif modes[p_idx] == "identity":
                # robust mean over the subset
                acc = np.zeros(d_i, dtype=np.float64)
                c = 0
                for bstart in range(0, len(idxs), load_batch):
                    bidx = idxs[bstart : bstart + load_batch]
                    flats = [dataset[i].numpy() for i in bidx]
                    X = np.stack(flats, axis=0)
                    acc += X[:, sl].mean(axis=0)
                    c += 1
                mean = (acc / max(c, 1)).astype(
                    np.float32
                )  # creating the mean of the identity mode
                np.savez_compressed(
                    self._path(p_idx), mode="identity", k=d_i, d=d_i, mean=mean
                )
            else:
                # constant → mean from subset (often zeros in synthetic inits)
                acc = np.zeros(d_i, dtype=np.float64)
                c = 0
                for bstart in range(0, len(idxs), load_batch):
                    bidx = idxs[bstart : bstart + load_batch]
                    flats = [dataset[i].numpy() for i in bidx]
                    X = np.stack(flats, axis=0)
                    acc += X[:, sl].mean(axis=0)
                    c += 1
                mean = (acc / max(c, 1)).astype(np.float32)
                np.savez_compressed(
                    self._path(p_idx), mode="constant", k=0, d=d_i, mean=mean
                )

        coeff_dim = int(
            sum(
                k_list_i if m == "pca" else (s["numel"] if m == "identity" else 0)
                for k_list_i, m, s in zip(k_list, modes, self.specs)
            )
        )

        # creating meta data needed to use the PCA mapper later on (saving it as a json file)
        index = {
            "k_default": self.k_default,
            "modes": modes,
            "k_list": k_list,
            "specs": self.specs,
            "total_dims": self.total_dims,
            "coeff_dim": coeff_dim,
        }
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Saved PCA/identity/constant maps. Total coeff_dim={coeff_dim}")

    def load(self) -> None:
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


class PCACoeffDataset(Dataset):
    """Streams PCA coefficients from flat weight vectors (uses PerParamPCAMapper)."""

    def __init__(self, dataset_dir: str, mapper: PerParamPCAMapper):
        self.raw = WeightVectorDataset(dataset_dir)
        self.mapper = mapper
        self.z_dim = mapper.coeff_dim

        # compute mean/std over a sample for normalization in AE space
        logger.info(
            "Computing PCA-coeff normalization stats (sample up to 512 models)..."
        )
        sample = min(512, len(self.raw))
        zs = []
        for i in range(sample):
            flat = (
                torch.load(self.raw.files[i], map_location="cpu")["weight_vector"]
                .float()
                .clamp_(-10.0, 10.0)
            )
            zs.append(self.mapper.transform(flat))
        Z = torch.stack(zs, dim=0) if zs else torch.zeros(1, self.z_dim)
        self.mean = Z.mean(dim=0)
        self.std = Z.std(dim=0).clamp_min(1e-4)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx) -> torch.Tensor:
        flat = (
            torch.load(self.raw.files[idx], map_location="cpu")["weight_vector"]
            .float()
            .clamp_(-10.0, 10.0)
        )
        z = self.mapper.transform(
            flat
        )  # create PCA coefficients from the flat weight vector
        z = (
            z - self.mean
        ) / self.std  # normalize the PCA coefficients for training stability
        return z


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
        # Buffer is something that is not a parameter (trainable), but part of the module state
        self.register_buffer("z_mean", torch.zeros(input_dim))
        self.register_buffer("z_std", torch.ones(input_dim))

    def set_z_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.z_mean.copy_(mean)
        self.z_std.copy_(std.clamp_min(1e-4))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(z)
        z_rec = self.decode(h)
        return z_rec, h


class WeightSpaceAELightning(L.LightningModule):
    """Lightning wrapper for WeightSpaceAE with training logic."""

    def __init__(
        self,
        weight_ae: WeightSpaceAE,
        lr: float = 1e-3,
        wd: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weight_ae"])

        self.weight_ae = weight_ae
        self.lr = lr
        self.wd = wd
        self.crit = nn.MSELoss()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weight_ae(z)

    def training_step(self, batch, _):
        z_rec, _ = self(batch)
        loss = self.crit(z_rec, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        z_rec, _ = self(batch)
        loss = self.crit(z_rec, batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


class WeightSpaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        models_dir: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        max_models: Optional[int] = None,
        pca_components_per_param: int = 8,
        pca_min_dim_identity: int = 4,
        pca_const_var_eps: float = 1e-8,
        pca_max_models_for_pca: int = 256,
        pca_load_batch: int = 64,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.max_models = max_models

        # Ensure splits sum to 1.0
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")

        self.model_ref = ResNet18WeightUtils.create_resnet18_3class()
        self.mapper = PerParamPCAMapper(
            self.model_ref,
            dataset_dir,
            n_components=pca_components_per_param,
            min_dim_identity=pca_min_dim_identity,
            const_var_eps=pca_const_var_eps,
        )
        self.pca_max_models_for_pca = pca_max_models_for_pca
        self.pca_load_batch = pca_load_batch

    def prepare_data(self):
        dpath = Path(self.dataset_dir)

        # Only generate dataset if it doesn't exist
        if not (dpath / "metadata.json").exists():
            if self.models_dir is None:
                raise ValueError(
                    "Dataset does not exist and --models-dir was not provided. "
                    "Please specify --models-dir to generate the dataset."
                )
            WeightVectorDatasetCreator.generate_dataset_from_models(
                self.dataset_dir, self.models_dir, self.max_models
            )
        else:
            logger.info(f"Found existing dataset at {self.dataset_dir}")

        # Fit PCA if missing
        if not self.mapper.is_fit():
            raw = WeightVectorDataset(self.dataset_dir)

            # If we have split info, use only training indices for PCA fitting
            split_file = Path(self.dataset_dir) / "data_splits.json"
            train_indices = None
            if split_file.exists():
                with open(split_file) as f:
                    split_data = json.load(f)
                    train_indices = split_data["train_indices"]
                    logger.info(
                        f"Using {len(train_indices)} training samples for PCA fitting"
                    )

            self.mapper.fit(
                raw,
                max_models_for_pca=self.pca_max_models_for_pca,
                load_batch=self.pca_load_batch,
                train_indices=train_indices,
            )
        else:
            logger.info("Found existing per-param PCA; will reuse.")

    def setup(self, stage: Optional[str] = None):
        self.mapper.load()
        full = PCACoeffDataset(self.dataset_dir, self.mapper)
        self.input_dim = self.mapper.coeff_dim
        self.z_mean = full.mean
        self.z_std = full.std

        # Create reproducible train/val/test split
        n = len(full)
        n_train = int(self.train_split * n)
        n_val = int(self.val_split * n)
        n_test = n - n_train - n_val

        # Use fixed seed for reproducible splits
        generator = torch.Generator().manual_seed(self.random_seed)
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            full, [n_train, n_val, n_test], generator=generator
        )

        # Save split indices for reference
        self.train_indices = self.train_ds.indices
        self.val_indices = self.val_ds.indices
        self.test_indices = self.test_ds.indices

        # Save split metadata
        split_metadata = {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "train_size": len(self.train_ds),
            "val_size": len(self.val_ds),
            "test_size": len(self.test_ds),
            "random_seed": self.random_seed,
        }

        split_file = Path(self.dataset_dir) / "data_splits.json"
        with open(split_file, "w") as f:
            json.dump(split_metadata, f, indent=2)

        logger.info(
            f"PCA-coeff dataset: train={len(self.train_ds)} val={len(self.val_ds)} test={len(self.test_ds)} dims={self.input_dim}"
        )
        logger.info(f"Saved data splits to {split_file}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_test_indices(self) -> List[int]:
        """Return the test set indices for external evaluation."""
        if hasattr(self, "test_indices"):
            return self.test_indices
        else:
            logger.warning("Test indices not available, returning empty list")
            return []


@torch.no_grad()
def compare_model_outputs(
    original_model: nn.Module,
    reconstructed_model: nn.Module,
    num_test_samples: int = 100,
    image_folder_path: Optional[str] = "images/labrador_retriever/",
) -> Dict[str, Any]:
    """
    Compares the outputs of the original and reconstructed models, giving us some metrics
    and plots to visualize how well the reconstruction worked.

    Args:
        original_model: The original model
        reconstructed_model: The reconstructed model
        num_test_samples: Number of test samples to use
        image_folder_path: Optional path to folder containing real images to test on
    """

    # Detect device and move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_model = original_model.to(device)
    reconstructed_model = reconstructed_model.to(device)

    original_model.eval()
    reconstructed_model.eval()

    test_inputs = []

    # Try to load real images if folder path is provided
    if image_folder_path is not None:
        from pathlib import Path

        image_folder = Path(image_folder_path)

        if image_folder.exists() and image_folder.is_dir():
            # Define image preprocessing transform
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Find image files
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(image_folder.glob(ext)))
                image_files.extend(list(image_folder.glob(ext.upper())))

            # Load images up to num_test_samples
            loaded_count = 0
            for img_path in image_files:
                if loaded_count >= num_test_samples:
                    break
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    test_inputs.append(img_tensor)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")

            if loaded_count > 0:
                logger.info(
                    f"Loaded {loaded_count} real images from {image_folder_path}"
                )

    # Fill remaining samples with synthetic patterns if needed
    remaining_samples = (
        num_test_samples - len(test_inputs) if len(test_inputs) == 0 else 0
    )
    if remaining_samples > 0:
        # Generate synthetic patterns on the target device
        for _ in range(remaining_samples // 4):  # noise
            test_inputs.append(torch.randn(1, 3, 224, 224, device=device))
        for i in range(remaining_samples // 4):  # structured
            pattern = torch.zeros(1, 3, 224, 224, device=device)
            if i % 3 == 0:
                pattern[:, :, ::4, :] = 1.0
            elif i % 3 == 1:
                pattern[:, :, ::8, ::8] = 1.0
                pattern[:, :, 4::8, 4::8] = 1.0
            else:
                for j in range(224):
                    pattern[:, :, j, :] = j / 224.0
            test_inputs.append(pattern)
        for i in range(remaining_samples // 4):  # edge cases
            if i % 3 == 0:
                test_inputs.append(torch.zeros(1, 3, 224, 224, device=device))
            elif i % 3 == 1:
                test_inputs.append(torch.ones(1, 3, 224, 224, device=device))
            else:
                test_inputs.append(torch.randn(1, 3, 224, 224, device=device) * 5)
        while len(test_inputs) < num_test_samples:
            test_inputs.append(
                torch.randn(1, 3, 224, 224, device=device) * np.random.uniform(0.1, 2.0)
            )

    orig_outs, recon_outs = [], []
    for x in test_inputs:
        orig_outs.append(original_model(x).squeeze())
        recon_outs.append(reconstructed_model(x).squeeze())
    original_outputs = torch.stack(orig_outs)
    reconstructed_outputs = torch.stack(recon_outs)

    # ------------ metrics ------------
    output_mse = torch.mean((original_outputs - reconstructed_outputs) ** 2).item()
    output_mae = torch.mean(torch.abs(original_outputs - reconstructed_outputs)).item()
    cos_sim = (
        torch.nn.functional.cosine_similarity(
            original_outputs.flatten(1), reconstructed_outputs.flatten(1), dim=1
        )
        .mean()
        .item()
    )
    orig_flat = original_outputs.flatten().cpu().numpy()
    recon_flat = reconstructed_outputs.flatten().cpu().numpy()
    correlation = float(np.corrcoef(orig_flat, recon_flat)[0, 1])
    pred_agreement = (
        (original_outputs.argmax(1) == reconstructed_outputs.argmax(1))
        .float()
        .mean()
        .item()
    )

    result: Dict[str, Any] = {
        "output_mse": output_mse,
        "output_mae": output_mae,
        "cosine_similarity": cos_sim,
        "correlation": correlation,
        "prediction_agreement": pred_agreement,
    }

    return result


@torch.no_grad()
def test_reconstruction_quality(
    weight_ae: WeightSpaceAE,
    mapper: PerParamPCAMapper,
    pca_coeff_vector: torch.Tensor,
    split="test",
) -> Dict[str, float]:

    weight_ae.eval()
    device = next(weight_ae.parameters()).device
    z = pca_coeff_vector.to(device)

    # Measure reconstruction in PCA space
    z_norm = (z - weight_ae.z_mean.to(device)) / weight_ae.z_std.to(device)
    h = weight_ae.encode(z_norm.unsqueeze(0))
    z_rec = weight_ae.decode(h).squeeze(0)

    mse = torch.mean((z_norm - z_rec) ** 2).item()
    mae = torch.mean(torch.abs(z_norm - z_rec)).item()
    compression_ratio = z_norm.numel() / h.numel()

    # Reconstruct resnets and compare outputs
    z_rec_denorm = z_rec * weight_ae.z_std.to(device) + weight_ae.z_mean.to(device)
    z_rec_denorm = z_rec_denorm.detach().cpu()
    flat_rec = mapper.inverse_transform(z_rec_denorm)
    flat_orig = mapper.inverse_transform(z.detach().cpu())

    # apply same clamping as during dataset creation
    flat_rec.clamp_(-5.0, 5.0)
    flat_orig.clamp_(-5.0, 5.0)

    # models
    m_orig = ResNet18WeightUtils.create_resnet18_3class()
    m_recn = ResNet18WeightUtils.create_resnet18_3class()
    ResNet18WeightUtils.set_weights_from_vector(m_orig, flat_orig)
    ResNet18WeightUtils.set_weights_from_vector(m_recn, flat_rec)

    # metrics + image paths
    out = compare_model_outputs(m_orig, m_recn)

    all_metrics = {
        f"{split}_pca_mse": mse,
        f"{split}_pca_mae": mae,
        f"{split}_compression_ratio": compression_ratio,
        f"{split}_output_mse": out["output_mse"],
        f"{split}_output_mae": out["output_mae"],
        f"{split}_cosine_similarity": out["cosine_similarity"],
        f"{split}_correlation": out["correlation"],
        f"{split}_prediction_agreement": out["prediction_agreement"],
    }

    logger.info(
        f"Reconstruction quality: PCA MSE={mse:.6f} MAE={mae:.6f} CompRatio={compression_ratio:.1f} | "
        f"Output MSE={out['output_mse']:.6f} MAE={out['output_mae']:.6f} Corr={out['correlation']:.4f} PredAgree={out['prediction_agreement']:.4f}"
    )

    return all_metrics


class FunctionalMetricsCallback(Callback):
    """
    Periodically measures functional reconstruction quality.
    Uses test set models for evaluation, ensuring no data leakage.
    """

    def __init__(
        self,
        dataset_dir: str,
        mapper: PerParamPCAMapper,
        log_every_n_epochs: int = 2,
        num_test_samples: int = 5,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mapper = mapper
        self.log_every_n_epochs = log_every_n_epochs
        self.num_test_samples = num_test_samples

        # Load test indices for evaluation
        split_file = Path(dataset_dir) / "data_splits.json"
        if split_file.exists():
            with open(split_file) as f:
                split_data = json.load(f)
                self.test_indices = split_data["test_indices"]
        else:
            # Fallback: use first few models as "test"
            logger.warning("No data splits found, using first 10 models as test set")
            self.test_indices = list(range(10))

        # Prepare test PCA coeff vectors
        raw = WeightVectorDataset(dataset_dir)
        self.test_vectors = []

        # Use a subset of test indices for evaluation
        eval_indices = self.test_indices[
            : min(self.num_test_samples, len(self.test_indices))
        ]

        for idx in eval_indices:
            if idx < len(raw.files):
                flat = (
                    torch.load(raw.files[idx], map_location="cpu")["weight_vector"]
                    .float()
                    .clamp_(-10.0, 10.0)
                )
                z = self.mapper.transform(flat)
                self.test_vectors.append(z)

        logger.info(
            f"Prepared {len(self.test_vectors)} test vectors for functional evaluation"
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: WeightSpaceAELightning
    ) -> None:
        if trainer.current_epoch == 0:
            return
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        try:
            # Average metrics across multiple test vectors
            all_metrics = []
            for i, test_z in enumerate(self.test_vectors):
                metrics = test_reconstruction_quality(
                    weight_ae=pl_module.weight_ae,
                    mapper=self.mapper,
                    pca_coeff_vector=test_z,
                    split="test",  # Use consistent split name
                )
                all_metrics.append(metrics)

            # Compute average metrics with better names
            if all_metrics:
                avg_metrics = {}
                name_mapping = {
                    "test_pca_mse": "test_pca_reconstruction_mse",
                    "test_pca_mae": "test_pca_reconstruction_mae", 
                    "test_output_mse": "test_model_output_mse",
                    "test_output_mae": "test_model_output_mae",
                    "test_cosine_similarity": "test_output_cosine_similarity",
                    "test_correlation": "test_output_correlation",
                    "test_prediction_agreement": "test_prediction_agreement",
                    "test_compression_ratio": "test_compression_ratio"
                }
                
                for key in all_metrics[0].keys():
                    if key.startswith("test_"):  # Only average test metrics
                        values = [m[key] for m in all_metrics]
                        new_key = name_mapping.get(key, key)  # Use mapping or keep original
                        avg_metrics[new_key] = sum(values) / len(values)

                # Log averaged metrics to Lightning and Wandb
                for k, v in avg_metrics.items():
                    pl_module.log(f"func_{k}", v, on_epoch=True, prog_bar=False)

                wandb.log(avg_metrics)
                logger.info(f"Logged functional metrics: {list(avg_metrics.keys())}")

        except Exception as e:
            logger.warning(f"FunctionalMetricsCallback failed: {e}")
            import traceback

            logger.warning(f"Traceback: {traceback.format_exc()}")


def main():
    # Config
    BATCH_SIZE = 128
    LATENT_DIM = 512
    NUM_EPOCHS = 100
    LR = 1e-3
    WD = 1e-5
    DATASET_DIR = "../data/dataset/"
    PROJECT_NAME = "weight-space-ae-pca"
    PCA_K_PER_PARAM = 8
    PCA_MIN_DIM_ID = 4
    PCA_CONST_VAR_EPS = 1e-8
    PCA_MAX_MODELS_FOR_PCA = 256
    PCA_LOAD_BATCH = 64
    NUM_WORKERS = 4
    MODELS_DIR = "../data/weights/downloaded_artifacts/"
    MAX_MODELS = None
    HIDDEN_DIMS = [512]

    # Logger
    wandb_logger = WandbLogger(
        project=PROJECT_NAME, name=f"k{PCA_K_PER_PARAM}-lat{LATENT_DIM}-lr{LR}"
    )

    # Data
    dm = WeightSpaceDataModule(
        dataset_dir=DATASET_DIR,
        models_dir=MODELS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_models=MAX_MODELS,
        pca_components_per_param=PCA_K_PER_PARAM,
        pca_min_dim_identity=PCA_MIN_DIM_ID,
        pca_const_var_eps=PCA_CONST_VAR_EPS,
        pca_max_models_for_pca=PCA_MAX_MODELS_FOR_PCA,
        pca_load_batch=PCA_LOAD_BATCH,
        random_seed=42,  # Fixed seed for reproducible splits
    )
    dm.prepare_data()
    dm.setup()

    # Create standalone autoencoder
    weight_ae = WeightSpaceAE(
        input_dim=dm.input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )
    weight_ae.set_z_stats(dm.z_mean, dm.z_std)

    # Wrap in Lightning module
    pl_model = WeightSpaceAELightning(weight_ae=weight_ae, lr=LR, wd=WD)

    # Trainer
    use_cuda = torch.cuda.is_available()
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=15, verbose=True),
        FunctionalMetricsCallback(
            dataset_dir=DATASET_DIR,
            mapper=dm.mapper,
            log_every_n_epochs=2,
            num_test_samples=3,
        ),
    ]
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="gpu" if use_cuda else "cpu",
        devices=[3] if use_cuda else "auto",
        precision="16-mixed" if use_cuda else "32",
        gradient_clip_val=1.0,
    )

    # Log basics
    wandb_logger.log_hyperparams(
        {
            "pca_k_per_param": PCA_K_PER_PARAM,
            "pca_input_dim": dm.input_dim,
            "latent_dim": LATENT_DIM,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "wd": WD,
            "pca_max_models_for_pca": PCA_MAX_MODELS_FOR_PCA,
            "pca_load_batch": PCA_LOAD_BATCH,
            "hidden_dims": HIDDEN_DIMS,
        }
    )

    logger.info("Starting training...")
    trainer.fit(pl_model, dm)

    # Final functional evaluation on test set
    logger.info("Final functional evaluation on test set...")

    # Load test indices
    split_file = Path(DATASET_DIR) / "data_splits.json"
    if split_file.exists():
        with open(split_file) as f:
            split_data = json.load(f)
            test_indices = split_data["test_indices"]
    else:
        test_indices = [0]  # fallback

    raw = WeightVectorDataset(DATASET_DIR)
    dm.mapper.load()

    # Evaluate on multiple test samples
    test_metrics = []
    eval_indices = test_indices[
        : min(10, len(test_indices))
    ]  # Use up to 10 test samples

    for i, idx in enumerate(eval_indices):
        if idx < len(raw.files):
            flat = (
                torch.load(raw.files[idx], map_location="cpu")["weight_vector"]
                .float()
                .clamp_(-10.0, 10.0)
            )
            z_eval = dm.mapper.transform(flat)
            metrics = test_reconstruction_quality(
                pl_model.weight_ae, dm.mapper, z_eval, split="final_test"
            )
            test_metrics.append(metrics)

    # Log average final test metrics
    if test_metrics:
        avg_final_metrics = {}
        for key in test_metrics[0].keys():
            if key.startswith("final_test_"):
                values = [m[key] for m in test_metrics]
                avg_final_metrics[key] = sum(values) / len(values)

        wandb.log(avg_final_metrics)
        logger.info(f"Final test reconstruction metrics: {avg_final_metrics}")

    # Save ONLY the weight_ae state dict
    ae_checkpoint_path = Path(DATASET_DIR) / "weight_ae_final.pth"
    torch.save(
        {
            "model_state_dict": pl_model.weight_ae.state_dict(),
            "input_dim": dm.input_dim,
            "latent_dim": LATENT_DIM,
            "z_mean": dm.z_mean,
            "z_std": dm.z_std,
        },
        ae_checkpoint_path,
    )
    logger.info(f"Saved weight_ae checkpoint to {ae_checkpoint_path}")

    wandb.finish()
    logger.info("Done.")
    return trainer, pl_model


if __name__ == "__main__":
    main()
    
