"""
pyunwrap.training.trainer
============================

Training loop for `AmbiguityNet`, implementing:

1. **Curriculum learning**: dynamic per-epoch filtering of the training tile
   set, starting from "easy" (high-coherence, low-gradient) tiles and
   progressively including harder ones (moderate coherence, atmospheric/
   orbital artifacts, then full difficulty including sub-Nyquist-violating
   deformation gradients).
2. **SNAPHU pseudo-ground-truth fine-tuning**: a final training phase that
   switches to real Sentinel-1 tiles whose "ground truth" is SNAPHU's own
   unwrapped output, to bridge the synthetic-to-real domain gap.
3. Standard training infrastructure: AdamW + warmup/cosine LR schedule,
   gradient clipping, periodic validation with unwrapping-specific metrics,
   TensorBoard logging, and a CLI entry point.

Curriculum tile metadata
--------------------------
Curriculum filtering needs, per tile, (a) a coherence summary and (b) a
deformation-gradient summary. Rather than requiring `pyunwrap.data.preprocessing`
to precompute and store these (which would bloat every HDF5 tile file whether
or not curriculum learning is used), `CurriculumIndex` computes them once,
lazily, by scanning the dataset's coherence/true_unwrapped arrays on first
use, and caches the result in memory for the life of the `Trainer`.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    _HAS_TENSORBOARD = False

from pyunwrap.data.dataloader import InSARTileDataset
from pyunwrap.models.ambiguity_net import AmbiguityNet, AmbiguityNetOutput
from pyunwrap.models.losses import PhysicsInformedUnwrapLoss, PhysicsLossOutput

# --------------------------------------------------------------------------- #
# Curriculum learning
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class TileDifficulty:
    """Per-tile difficulty summary used to drive curriculum filtering.

    Attributes:
        index: Index into the underlying `InSARTileDataset`.
        mean_coherence: Mean coherence over the tile, [0, 1].
        p99_gradient: 99th-percentile absolute spatial gradient of the true
            unwrapped phase within the tile, radians/pixel (used to detect
            sub-Nyquist-violating deformation, i.e. gradient > pi). A
            percentile rather than a hard max is used deliberately: the
            ground-truth unwrapped phase (per the generator's design -- see
            `pyunwrap.synthetic.generator`) legitimately includes
            spatially-*uncorrelated* decorrelation noise, so even physically
            "easy" tiles routinely contain a handful of isolated single-pixel
            noise spikes whose raw gradient exceeds pi. A hard max would
            therefore misclassify nearly every tile as "hard"; the 99th
            percentile is robust to that noise while still reliably flagging
            tiles with genuinely widespread steep deformation gradients.
    """

    index: int
    mean_coherence: float
    p99_gradient: float


class CurriculumIndex:
    """Computes and caches per-tile difficulty stats, and exposes epoch-aware
    index subsets implementing the 3-stage curriculum described in Prompt 4.

    Stage boundaries (by 1-indexed epoch number):
        - Epochs 1-20:  mean_coherence > 0.7 AND p99_gradient <= pi (easy)
        - Epochs 21-50: mean_coherence > 0.4 (moderate; includes atmosphere/
          orbital-ramp-heavy tiles, which are present throughout the
          synthetic dataset regardless of coherence)
        - Epochs 51+:   full dataset, no filtering (hard; includes
          low-coherence tiles and gradients exceeding the Nyquist limit)
    """

    def __init__(self, dataset: InSARTileDataset) -> None:
        """
        Args:
            dataset: The full training `InSARTileDataset` to index. Must have
                `require_ground_truth=True` (curriculum stats depend on
                `true_unwrapped`).
        """
        self.dataset = dataset
        self._stats: list[TileDifficulty] = self._compute_stats()

    def _compute_stats(self) -> list[TileDifficulty]:
        """Scan every tile once (without augmentation) to compute difficulty stats."""
        stats = []
        was_augmenting = self.dataset.augment
        self.dataset.augment = False  # stats must reflect the canonical, unaugmented tile
        try:
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                coherence = sample["coherence"].numpy()
                mean_coh = float(coherence.mean())

                if "true_unwrapped" in sample:
                    unwrapped = sample["true_unwrapped"].numpy()
                    grad_y = np.abs(np.diff(unwrapped, axis=-2))
                    grad_x = np.abs(np.diff(unwrapped, axis=-1))
                    combined = np.concatenate([grad_y.ravel(), grad_x.ravel()])
                    p99_grad = float(np.percentile(combined, 99)) if combined.size > 0 else 0.0
                else:
                    p99_grad = 0.0

                stats.append(
                    TileDifficulty(index=i, mean_coherence=mean_coh, p99_gradient=p99_grad)
                )
        finally:
            self.dataset.augment = was_augmenting
        return stats

    def indices_for_epoch(self, epoch: int) -> list[int]:
        """Return the list of dataset indices eligible for training at `epoch` (1-indexed).

        Args:
            epoch: Current 1-indexed epoch number.

        Returns:
            List of dataset indices satisfying the curriculum stage's
            difficulty criteria. Falls back to the full dataset if a stage's
            filter is too strict and would otherwise yield an empty set
            (logged via a warning-equivalent print, since an empty epoch
            would silently stall training).
        """
        if epoch <= 20:
            eligible = [
                s.index for s in self._stats if s.mean_coherence > 0.7 and s.p99_gradient <= math.pi
            ]
        elif epoch <= 50:
            eligible = [s.index for s in self._stats if s.mean_coherence > 0.4]
        else:
            eligible = [s.index for s in self._stats]

        if len(eligible) == 0:
            print(
                f"[CurriculumIndex] WARNING: epoch {epoch}'s curriculum filter matched 0 "
                "tiles; falling back to the full dataset for this epoch to avoid stalling training."
            )
            eligible = [s.index for s in self._stats]
        return eligible


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ValidationMetrics:
    """Aggregate validation metrics computed every `validate_every` epochs.

    Attributes:
        rmse_rad: RMSE of the reconstructed unwrapped phase, radians.
        pct_pixels_under_0p1_rad: Percentage of pixels with absolute phase
            error < 0.1 radians.
        residue_count: Total count of simple ambiguity-map discontinuities
            (see `_count_simple_residues`) summed over the validation set --
            a lightweight proxy for unwrapping-induced artifacts, not the
            full Goldstein residue analysis provided by
            `pyunwrap.analytics.phase_stats` (Prompt 6).
        n_samples: Number of validation tiles the metrics were computed over.
    """

    rmse_rad: float
    pct_pixels_under_0p1_rad: float
    residue_count: int
    n_samples: int


def _count_simple_residues(k_hat: torch.Tensor, threshold: float = 1.5) -> int:
    """Lightweight proxy residue count: number of pixels whose ambiguity value
    differs from the mean of its 4 neighbors by more than `threshold`.

    This is intentionally simple (an O(1) diagnostic for training-time
    logging); `pyunwrap.analytics.phase_stats` (Prompt 6) provides the
    rigorous Goldstein-style residue analysis for scientific reporting.

    Args:
        k_hat: Predicted ambiguity map, [B, 1, H, W].
        threshold: Minimum deviation from the local neighbor mean to count as
            a flagged discontinuity.

    Returns:
        Total flagged pixel count across the batch.
    """
    neighbor_mean = (
        torch.roll(k_hat, 1, dims=-1)
        + torch.roll(k_hat, -1, dims=-1)
        + torch.roll(k_hat, 1, dims=-2)
        + torch.roll(k_hat, -1, dims=-2)
    ) / 4.0
    deviation = (k_hat - neighbor_mean).abs()
    # Exclude the wrap-around border pixels torch.roll introduces artifacts at.
    interior = deviation[:, :, 1:-1, 1:-1]
    return int((interior > threshold).sum().item())


@torch.no_grad()
def evaluate(
    model: AmbiguityNet,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> ValidationMetrics:
    """Run validation over (optionally a subset of) `dataloader` and compute metrics.

    Args:
        model: The `AmbiguityNet` to evaluate (switched to eval mode
            internally, restored to its prior mode on return).
        dataloader: Validation `DataLoader` (should be built with
            `augment=False`).
        device: Device to run evaluation on.
        max_batches: Optional cap on the number of batches evaluated, for
            fast periodic validation on large validation sets.

    Returns:
        Aggregated `ValidationMetrics` over all evaluated samples.
    """
    was_training = model.training
    model.eval()

    total_sq_error = 0.0
    total_pixels = 0
    total_under_threshold = 0
    total_residues = 0
    n_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = torch.cat([batch["wrapped_phase"], batch["coherence"], batch["amplitude"]], dim=1).to(
            device
        )
        true_unwrapped = batch["true_unwrapped"].to(device)

        out = model(x)
        error = (out.phi_hat - true_unwrapped).abs()

        total_sq_error += float((error**2).sum().item())
        total_pixels += error.numel()
        total_under_threshold += int((error < 0.1).sum().item())
        total_residues += _count_simple_residues(out.k_hat)
        n_samples += x.shape[0]

    if was_training:
        model.train()

    rmse = math.sqrt(total_sq_error / max(total_pixels, 1))
    pct_under = 100.0 * total_under_threshold / max(total_pixels, 1)

    return ValidationMetrics(
        rmse_rad=rmse,
        pct_pixels_under_0p1_rad=pct_under,
        residue_count=total_residues,
        n_samples=n_samples,
    )


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine annealing
# --------------------------------------------------------------------------- #


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a `LambdaLR` implementing linear warmup followed by cosine annealing.

    LR multiplier schedule (relative to the optimizer's base LR):
        - epoch < warmup_epochs: linear ramp from 0 -> 1
        - epoch >= warmup_epochs: cosine decay from 1 -> 0 over the
          remaining `total_epochs - warmup_epochs` epochs

    Args:
        optimizer: The optimizer to schedule (e.g. AdamW).
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of linear warmup epochs.

    Returns:
        A `LambdaLR` scheduler; call `.step()` once per epoch.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# --------------------------------------------------------------------------- #
# Visualizer
# --------------------------------------------------------------------------- #


class Visualizer:
    """Plots training curves and side-by-side predicted vs. ground-truth phase.

    All plotting is done with matplotlib and figures are saved to disk (no
    interactive display assumed, since training typically runs headless).
    """

    def __init__(self, out_dir: str | Path) -> None:
        """
        Args:
            out_dir: Directory to save plots into (created if missing).
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Each metric key maps to its own list of (epoch, value) pairs, since
        # not every metric is logged every epoch (e.g. validation metrics are
        # only computed every `validate_every` epochs) -- a single shared
        # "epoch" column would silently misalign against sparser metrics.
        self.history: dict[str, list[tuple[int, float]]] = {}

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record a dict of scalar metrics for a given epoch.

        Args:
            epoch: The epoch these metrics correspond to.
            metrics: Mapping of metric name -> value. Keys need not be the
                same across calls (e.g. validation-only keys may only appear
                every `validate_every` epochs); each key's history simply
                grows its own (epoch, value) list.
        """
        for key, value in metrics.items():
            self.history.setdefault(key, []).append((epoch, value))

    def plot_training_curves(self, filename: str = "training_curves.png") -> Path:
        """Plot every logged scalar, one subplot each, against its own epoch axis.

        Returns:
            Path to the saved figure.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = list(self.history.keys())
        if not keys:
            raise ValueError("No metrics logged yet; call `.log()` before plotting.")

        fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, keys):
            epochs, values = zip(*self.history[key])
            ax.plot(epochs, values, marker="o", markersize=3)
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("epoch")
        fig.tight_layout()

        path = self.out_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_phase_comparison(
        self,
        wrapped_phase_rad: np.ndarray,
        predicted_unwrapped: np.ndarray,
        true_unwrapped: np.ndarray,
        filename: str = "phase_comparison.png",
    ) -> Path:
        """Plot wrapped input, predicted unwrapped, ground-truth unwrapped, and
        the error map, side by side.

        Args:
            wrapped_phase_rad: 2D wrapped phase, radians.
            predicted_unwrapped: 2D predicted unwrapped phase, radians.
            true_unwrapped: 2D ground-truth unwrapped phase, radians.
            filename: Output filename within `self.out_dir`.

        Returns:
            Path to the saved figure.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        error = predicted_unwrapped - true_unwrapped
        vmax = max(np.abs(true_unwrapped).max(), np.abs(predicted_unwrapped).max())

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        im0 = axes[0].imshow(wrapped_phase_rad, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[0].set_title("Wrapped phase (input)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(predicted_unwrapped, cmap="jet", vmin=-vmax, vmax=vmax)
        axes[1].set_title("Predicted unwrapped")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(true_unwrapped, cmap="jet", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Ground truth unwrapped")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        err_vmax = max(np.abs(error).max(), 1e-6)
        im3 = axes[3].imshow(error, cmap="RdBu_r", vmin=-err_vmax, vmax=err_vmax)
        axes[3].set_title("Error (pred - true)")
        fig.colorbar(im3, ax=axes[3], fraction=0.046)

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

        path = self.out_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class Trainer:
    """Orchestrates curriculum training, optional SNAPHU fine-tuning,
    validation, checkpointing, and logging for `AmbiguityNet`.
    """

    def __init__(
        self,
        model: AmbiguityNet,
        train_dataset: InSARTileDataset,
        val_dataset: InSARTileDataset,
        out_dir: str | Path,
        total_epochs: int = 60,
        warmup_epochs: int = 5,
        base_lr: float = 1e-4,
        batch_size: int = 16,
        num_workers: int = 4,
        grad_clip_norm: float = 1.0,
        validate_every: int = 5,
        device: str | None = None,
        use_curriculum: bool = True,
        finetune_dataset: InSARTileDataset | None = None,
        finetune_start_epoch: int = 55,
    ) -> None:
        """
        Args:
            model: The `AmbiguityNet` instance to train.
            train_dataset: Synthetic training `InSARTileDataset`
                (`require_ground_truth=True`, `augment=True` recommended).
            val_dataset: Held-out validation `InSARTileDataset`
                (`augment=False`).
            out_dir: Directory for checkpoints, TensorBoard logs, and plots.
            total_epochs: Total number of epochs to train (curriculum stages
                and the cosine schedule are both defined relative to this).
            warmup_epochs: Linear LR warmup duration, epochs.
            base_lr: Peak learning rate for AdamW.
            batch_size: Training/validation batch size.
            num_workers: DataLoader worker count.
            grad_clip_norm: Max gradient norm for `clip_grad_norm_`.
            validate_every: Run `evaluate()` every N epochs.
            device: `"cuda"`, `"cpu"`, or None to auto-detect.
            use_curriculum: Whether to apply the 3-stage curriculum filter to
                `train_dataset`. If False, the full dataset is used every
                epoch (still followed by SNAPHU fine-tuning if configured).
            finetune_dataset: Optional real-data `InSARTileDataset` whose
                `true_unwrapped` is SNAPHU's pseudo-ground-truth output, used
                for the final fine-tuning phase.
            finetune_start_epoch: 1-indexed epoch at which to switch from the
                (synthetic, curriculum) `train_dataset` to `finetune_dataset`.
        """
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.finetune_dataset = finetune_dataset
        self.finetune_start_epoch = finetune_start_epoch

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grad_clip_norm = grad_clip_norm
        self.validate_every = validate_every
        self.use_curriculum = use_curriculum

        self.criterion = PhysicsInformedUnwrapLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr)
        self.scheduler = build_warmup_cosine_scheduler(self.optimizer, total_epochs, warmup_epochs)

        self.curriculum_index = CurriculumIndex(train_dataset) if use_curriculum else None

        self.writer: SummaryWriter | None = None
        if _HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.out_dir / "tensorboard"))
        else:
            print("[Trainer] tensorboard not installed; skipping TensorBoard logging.")

        self.visualizer = Visualizer(self.out_dir / "plots")

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # ----------------------------------------------------------------- #
    # Per-epoch data selection
    # ----------------------------------------------------------------- #

    def _build_epoch_loader(self, epoch: int) -> DataLoader:
        """Build the training DataLoader for `epoch`, applying curriculum
        filtering or switching to the SNAPHU fine-tuning dataset as configured.
        """
        if self.finetune_dataset is not None and epoch >= self.finetune_start_epoch:
            print(
                f"[Trainer] Epoch {epoch}: switching to SNAPHU pseudo-ground-truth fine-tuning dataset."
            )
            return DataLoader(
                self.finetune_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

        if self.use_curriculum and self.curriculum_index is not None:
            indices = self.curriculum_index.indices_for_epoch(epoch)
            subset = Subset(self.train_dataset, indices)
            print(
                f"[Trainer] Epoch {epoch}: curriculum subset size = {len(indices)} / {len(self.train_dataset)}"
            )
            return DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    # ----------------------------------------------------------------- #
    # Training
    # ----------------------------------------------------------------- #

    def _train_one_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            x = torch.cat(
                [batch["wrapped_phase"], batch["coherence"], batch["amplitude"]], dim=1
            ).to(self.device)
            k_true = batch["true_ambiguity"].to(self.device)
            coherence = batch["coherence"].to(self.device)
            wrapped_phase_norm = batch["wrapped_phase"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            out: AmbiguityNetOutput = self.model(x)
            loss_out: PhysicsLossOutput = self.criterion(
                out,
                k_true=k_true,
                wrapped_phase_norm=wrapped_phase_norm,
                coherence=coherence,
            )
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            for key, value in loss_out.as_dict().items():
                running[key] = running.get(key, 0.0) + value
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def fit(self) -> None:
        """Run the full training loop for `self.total_epochs` epochs, with
        periodic validation, TensorBoard logging, and checkpointing.
        """
        for epoch in range(1, self.total_epochs + 1):
            start = time.time()
            loader = self._build_epoch_loader(epoch)
            train_metrics = self._train_one_epoch(epoch, loader)
            self.scheduler.step()
            elapsed = time.time() - start

            lr = self.optimizer.param_groups[0]["lr"]
            log_line = (
                f"[Trainer] Epoch {epoch}/{self.total_epochs} "
                f"loss={train_metrics.get('loss/total', float('nan')):.4f} "
                f"lr={lr:.2e} ({elapsed:.1f}s)"
            )

            epoch_scalars = dict(train_metrics)
            epoch_scalars["lr"] = lr

            if epoch % self.validate_every == 0 or epoch == self.total_epochs:
                val_metrics = evaluate(self.model, self.val_loader, self.device)
                log_line += (
                    f" | val_rmse={val_metrics.rmse_rad:.4f} rad "
                    f"val_pct<0.1rad={val_metrics.pct_pixels_under_0p1_rad:.2f}% "
                    f"val_residues={val_metrics.residue_count}"
                )
                epoch_scalars["val/rmse_rad"] = val_metrics.rmse_rad
                epoch_scalars["val/pct_pixels_under_0p1_rad"] = val_metrics.pct_pixels_under_0p1_rad
                epoch_scalars["val/residue_count"] = float(val_metrics.residue_count)
                self._save_checkpoint(epoch, val_metrics)

            print(log_line)
            self.visualizer.log(epoch, epoch_scalars)
            if self.writer is not None:
                for key, value in epoch_scalars.items():
                    self.writer.add_scalar(key, value, epoch)

        self.visualizer.plot_training_curves()
        if self.writer is not None:
            self.writer.close()

    def _save_checkpoint(self, epoch: int, val_metrics: ValidationMetrics) -> None:
        """Save a checkpoint with model/optimizer state and validation metrics."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_rmse_rad": val_metrics.rmse_rad,
        }
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `pyunwrap-train`."""
    parser = argparse.ArgumentParser(description="Train AmbiguityNet for InSAR phase unwrapping.")
    parser.add_argument(
        "--train-hdf5", type=str, required=True, help="Path to training tiles HDF5 file."
    )
    parser.add_argument(
        "--val-hdf5", type=str, required=True, help="Path to validation tiles HDF5 file."
    )
    parser.add_argument(
        "--finetune-hdf5",
        type=str,
        default=None,
        help="Optional path to SNAPHU pseudo-ground-truth fine-tuning tiles HDF5 file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./runs/pyunwrap",
        help="Output directory for logs/checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-start-epoch", type=int, default=55)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Disable ImageNet-pretrained encoder init."
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning; use full dataset every epoch.",
    )
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    return parser


def main() -> None:
    """CLI entry point (`pyunwrap-train`): parse args, build datasets/model, and train."""
    parser = build_argparser()
    args = parser.parse_args()

    train_dataset = InSARTileDataset(args.train_hdf5, augment=True, require_ground_truth=True)
    val_dataset = InSARTileDataset(args.val_hdf5, augment=False, require_ground_truth=True)
    finetune_dataset = (
        InSARTileDataset(args.finetune_hdf5, augment=True, require_ground_truth=True)
        if args.finetune_hdf5
        else None
    )

    model = AmbiguityNet(pretrained=not args.no_pretrained, k_max=args.k_max)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_dir=args.out_dir,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
        validate_every=args.validate_every,
        device=args.device,
        use_curriculum=not args.no_curriculum,
        finetune_dataset=finetune_dataset,
        finetune_start_epoch=args.finetune_start_epoch,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
