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
from pyunwrap.models.losses import PhysicsInformedUnwrapLoss, PhysicsLossOutput, SmoothnessConfig
from pyunwrap.training.curriculum import (
    CurriculumIndex,
    CurriculumReplayConfig,
    CurriculumReplayIndex,
    TileDifficulty,
    classify_difficulty_tier,
    compute_tile_difficulty_stats,
)

__all__ = [
    # Re-exported from pyunwrap.training.curriculum for backward
    # compatibility -- these classes originated in this module before being
    # extracted; existing `from pyunwrap.training.trainer import
    # CurriculumIndex`-style imports continue to work unchanged.
    "CurriculumIndex",
    "CurriculumReplayConfig",
    "CurriculumReplayIndex",
    "TileDifficulty",
    "Trainer",
    "ValidationMetrics",
    "build_curriculum_aware_scheduler",
    "build_warmup_cosine_scheduler",
    "classify_difficulty_tier",
    "compute_tile_difficulty_stats",
    "evaluate",
    "evaluate_stratified",
]

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


def evaluate_stratified(
    model: AmbiguityNet,
    dataset: InSARTileDataset,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
) -> dict[str, ValidationMetrics]:
    """Run validation broken out by difficulty tier ("easy"/"moderate"/"hard",
    via `classify_difficulty_tier`) in addition to the overall aggregate.

    Why this exists: `Trainer` originally only tracked one aggregate
    `val_rmse` across the whole validation set. On a real 60-epoch run, a
    curriculum-stage LR restart fixed accuracy on hard scenes while
    silently regressing accuracy on easy ones (a real, measured 74% RMSE
    regression on synthetic-benchmark scenes resembling the "easy" tier) --
    and the aggregate metric never revealed it, because improvement on the
    (numerically larger) hard-tile errors outweighed the regression on
    easy tiles in the overall average. Stratified tracking makes this kind
    of regime-specific regression visible during training itself, not just
    discoverable after the fact via a separate downstream benchmark. See
    `docs/experiments.md`'s Experiment 3 for the full story.

    Args:
        model: The `AmbiguityNet` to evaluate.
        dataset: Validation `InSARTileDataset` (should have `augment=False`
            and `require_ground_truth=True`).
        device: Device to run evaluation on.
        batch_size: Batch size for each tier's temporary `DataLoader`.
        num_workers: Worker count for each tier's temporary `DataLoader`.

    Returns:
        A dict with keys `"overall"`, `"easy"`, `"moderate"`, `"hard"`,
        each mapping to a `ValidationMetrics`. A tier with zero matching
        tiles in `dataset` is omitted (rather than reported as a
        misleading zero-sample metric) -- check for a tier's presence
        before reading it.
    """
    results: dict[str, ValidationMetrics] = {}

    overall_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    results["overall"] = evaluate(model, overall_loader, device)

    stats = compute_tile_difficulty_stats(dataset)
    tier_indices: dict[str, list[int]] = {"easy": [], "moderate": [], "hard": []}
    for stat in stats:
        tier_indices[classify_difficulty_tier(stat)].append(stat.index)

    for tier, indices in tier_indices.items():
        if len(indices) == 0:
            continue
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        results[tier] = evaluate(model, loader, device)

    return results


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine annealing
# --------------------------------------------------------------------------- #


def _safe_drop_last(dataset_size: int, batch_size: int) -> bool:
    """Whether it's safe to pass `drop_last=True` to a `DataLoader` without
    risking zero batches per epoch.

    Only drops the final ragged batch when there's at least one full batch
    of data besides it (`dataset_size > batch_size`). If the whole dataset
    is smaller than or equal to one batch, dropping it would silently
    produce zero batches for the entire epoch -- see `_build_epoch_loader`'s
    docstring for the real training run this was caught on.

    Args:
        dataset_size: Number of samples in the dataset/subset being loaded.
        batch_size: The `DataLoader`'s configured batch size.

    Returns:
        True if dropping the ragged final batch is safe, False otherwise.
    """
    return dataset_size > batch_size


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


def build_curriculum_aware_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    restart_epochs: list[int],
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a `LambdaLR` that runs an independent warmup+cosine cycle within
    each curriculum/fine-tuning segment, restarting the LR at each segment
    boundary instead of letting one long cosine decay run across all of them.

    Motivation (found on a real 60-epoch training run, not a hypothetical):
    `build_warmup_cosine_scheduler` decays smoothly across the *entire* run,
    with no awareness of `CurriculumIndex`'s stage transitions (by default
    at epochs 21 and 51) or `Trainer`'s SNAPHU fine-tuning switch (at
    `finetune_start_epoch`). On a real run, curriculum stage 3 -- the
    hardest data, unlocked at epoch 51 -- arrived when the cosine schedule
    had already decayed the LR to roughly 1/15,000th of its peak value. The
    model's training loss visibly jumped at that transition (harder data)
    and never recovered before training ended, because there was no
    meaningful gradient step size left to adapt with. This scheduler fixes
    that by giving each segment (stage 1, stage 2, stage 3, and the
    fine-tuning phase if configured) its own short warmup back up to the
    original peak LR followed by its own cosine decay over just that
    segment's epoch span.

    Degenerates to `build_warmup_cosine_scheduler`'s exact schedule when
    `restart_epochs` contains no epoch other than `1` (i.e. curriculum
    learning and fine-tuning are both disabled, so there is only one
    segment spanning the whole run) -- verified by direct comparison in
    `tests/test_trainer.py`, not just argued for here.

    Args:
        optimizer: The optimizer to schedule (e.g. AdamW).
        total_epochs: Total number of training epochs.
        restart_epochs: 1-indexed epoch numbers at which a new segment (and
            therefore a fresh warmup+cosine cycle) begins. `1` is always
            treated as an implicit segment start regardless of whether it's
            included; values outside `[1, total_epochs]` are ignored.
        warmup_epochs: Warmup length for each segment, capped to at most
            `segment_length - 1` for any segment shorter than this (so a
            short final curriculum stage doesn't spend its entire budget on
            warmup with no room left to actually decay).

    Returns:
        A `LambdaLR` scheduler; call `.step()` once per epoch, exactly like
        `build_warmup_cosine_scheduler`.
    """
    starts = sorted({1, *(e for e in restart_epochs if 1 < e <= total_epochs)})
    segments: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else total_epochs
        segments.append((start, end))

    def lr_lambda(step: int) -> float:
        epoch = (
            step + 1
        )  # PyTorch's LambdaLR step counter is 0-indexed; Trainer's epochs are 1-indexed.
        seg_start, seg_end = segments[-1]
        for start, end in segments:
            if start <= epoch <= end:
                seg_start, seg_end = start, end
                break

        seg_length = seg_end - seg_start + 1
        local_epoch = epoch - seg_start
        seg_warmup = min(warmup_epochs, max(seg_length - 1, 0))

        if local_epoch < seg_warmup:
            return (local_epoch + 1) / max(seg_warmup, 1)
        progress = (local_epoch - seg_warmup) / max(seg_length - seg_warmup, 1)
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
        stratified_validation: bool = True,
        curriculum_replay_config: CurriculumReplayConfig | None = None,
        forgetting_lr_reduction_factor: float | None = 0.5,
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
            use_curriculum: Whether to apply the original 3-stage sequential
                curriculum filter to `train_dataset`. Ignored (has no
                effect) when `curriculum_replay_config` is also given and
                enabled -- see that parameter's docstring for precedence.
                If both are `False`/`None`, the full dataset is used every
                epoch (still followed by SNAPHU fine-tuning if configured).
            finetune_dataset: Optional real-data `InSARTileDataset` whose
                `true_unwrapped` is SNAPHU's pseudo-ground-truth output, used
                for the final fine-tuning phase. Build this dataset with
                `pyunwrap.utils.snaphu_integration.generate_snaphu_finetune_dataset`,
                which runs real SNAPHU unwrapping and filters out any tile
                where SNAPHU wasn't confident enough to trust as ground truth.
            finetune_start_epoch: 1-indexed epoch at which to switch from the
                (synthetic, curriculum) `train_dataset` to `finetune_dataset`.
            stratified_validation: Whether periodic validation also reports
                RMSE broken out by difficulty tier ("easy"/"moderate"/"hard",
                via `evaluate_stratified`) in addition to the overall
                aggregate. On by default -- it reuses the same per-tile
                stats the curriculum already computes, so the extra cost is
                a handful of additional forward passes over `val_dataset`,
                not a second data-processing pipeline.
            curriculum_replay_config: Optional `CurriculumReplayConfig`. If
                given and `config.use_curriculum_replay` is `True`, replaces
                the original sequential `CurriculumIndex` staging with
                `CurriculumReplayIndex`'s fixed-proportion easy/moderate/hard
                mixture, sampled fresh every epoch, with automatic
                forgetting-triggered rebalancing (see
                `pyunwrap.training.curriculum` for why this exists: the
                original sequential curriculum was found to cause
                catastrophic forgetting of easy-tier performance once the
                hardest stage took over every epoch -- see
                `docs/experiments.md`, Experiment 3). `None` (the default)
                preserves the original `use_curriculum` behavior exactly,
                with zero change to existing training runs.

                Precedence when both `use_curriculum=True` and a
                `curriculum_replay_config` are given: replay wins, and a
                one-time note is printed, since sequential staging and
                fixed-proportion replay are alternative mechanisms for the
                same underlying goal, not composable ones.

                Curriculum replay also changes the learning-rate schedule's
                restart points: because replay never introduces an abrupt
                "suddenly 100% hard data" transition in the first place (the
                mixture is constant throughout, by design), the
                sequential-curriculum-stage LR restarts are not added when
                replay is active. The SNAPHU fine-tuning restart (if
                configured) still applies either way, since that is a real
                distribution change regardless of which curriculum mechanism
                preceded it.
            forgetting_lr_reduction_factor: When curriculum replay detects
                forgetting (see `CurriculumReplayIndex.record_easy_validation_metric`),
                the current learning rate is also multiplied by this factor
                (in addition to rebalancing the replay mixture), as a
                persistent scale applied on top of the underlying schedule.
                `None` disables the LR-reduction side effect; rebalancing
                still occurs. Has no effect when curriculum replay is not
                active.
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
        self.stratified_validation = stratified_validation
        self.forgetting_lr_reduction_factor = forgetting_lr_reduction_factor
        self._lr_scale_factor = 1.0  # persistent multiplier applied on top of the schedule

        replay_active = (
            curriculum_replay_config is not None and curriculum_replay_config.use_curriculum_replay
        )
        if replay_active and use_curriculum:
            print(
                "[Trainer] Both use_curriculum=True and an enabled curriculum_replay_config "
                "were given; curriculum replay takes precedence and the original sequential "
                "curriculum staging will not be used this run."
            )
        self.curriculum_replay_config = curriculum_replay_config
        self.curriculum_replay_index: CurriculumReplayIndex | None = None
        if replay_active:
            assert (
                curriculum_replay_config is not None
            )  # implied by replay_active, for mypy's narrowing
            self.curriculum_replay_index = CurriculumReplayIndex(
                train_dataset, curriculum_replay_config
            )

        self.criterion = PhysicsInformedUnwrapLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr)

        # Restart the LR schedule at every point the training data
        # distribution changes abruptly: each curriculum stage transition,
        # and the SNAPHU fine-tuning switch if configured. See
        # build_curriculum_aware_scheduler's docstring for the real run
        # that motivated this (a plain single-run cosine decay left the LR
        # near zero exactly when curriculum stage 3's harder data arrived).
        # Not added for curriculum replay -- see curriculum_replay_config's
        # docstring above for why replay doesn't need this.
        restart_epochs = [1]
        if use_curriculum and not replay_active:
            restart_epochs += [
                CurriculumIndex.STAGE_1_END_EPOCH + 1,
                CurriculumIndex.STAGE_2_END_EPOCH + 1,
            ]
        if finetune_dataset is not None:
            restart_epochs.append(finetune_start_epoch)
        self.scheduler = build_curriculum_aware_scheduler(
            self.optimizer, total_epochs, restart_epochs, warmup_epochs
        )

        self.curriculum_index = (
            CurriculumIndex(train_dataset) if (use_curriculum and not replay_active) else None
        )

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

        Note on `drop_last`: PyTorch's `DataLoader(drop_last=True)` discards
        the final ragged batch -- but if the *entire* dataset is smaller than
        `batch_size` (e.g. an early, strict curriculum stage matches only a
        handful of tiles), there is no batch other than that ragged one, so
        `drop_last=True` silently yields ZERO batches for the whole epoch.
        `_train_one_epoch` would then run its `for batch in loader` loop zero
        times, leave `running` empty, and report `loss=nan` -- an epoch that
        looks like it "completed" while doing no actual training at all. This
        was caught during a real training run (a curriculum subset of 4 tiles
        against `batch_size=8` silently produced 20 consecutive no-op
        epochs). `_safe_drop_last` below prevents it by only dropping the
        ragged batch when there's at least one full batch of data besides it.
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
                drop_last=_safe_drop_last(len(self.finetune_dataset), self.batch_size),
            )

        if self.curriculum_replay_index is not None:
            indices = self.curriculum_replay_index.indices_for_epoch(epoch)
            subset = Subset(self.train_dataset, indices)
            print(
                f"[Trainer] Epoch {epoch}: curriculum REPLAY mixture size = {len(indices)} "
                f"(easy={self.curriculum_replay_index.easy_fraction:.2f}, "
                f"moderate={self.curriculum_replay_index.medium_fraction:.2f}, "
                f"hard={self.curriculum_replay_index.hard_fraction:.2f})"
            )
            return DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=_safe_drop_last(len(subset), self.batch_size),
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
                drop_last=_safe_drop_last(len(subset), self.batch_size),
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=_safe_drop_last(len(self.train_dataset), self.batch_size),
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
            if not train_metrics:
                # Defense in depth: _safe_drop_last prevents the known cause
                # (a curriculum/fine-tune subset smaller than one batch), but
                # if any other future edge case ever produces zero batches
                # for an epoch, fail loudly rather than silently logging
                # loss=nan and proceeding as though training happened.
                raise RuntimeError(
                    f"Epoch {epoch} produced zero training batches (dataset/subset "
                    f"size may be smaller than batch_size={self.batch_size}). No "
                    "gradient updates occurred this epoch -- refusing to silently "
                    "continue. Reduce batch_size or increase the dataset/subset size."
                )
            self.scheduler.step()
            # Re-apply any persistent forgetting-triggered LR reduction:
            # LambdaLR recomputes lr = base_lr * lambda(step) from the
            # *original* base_lr on every .step() call, so a one-time
            # manual reduction would otherwise be silently overwritten by
            # the very next .step() -- this must be reapplied every epoch
            # to actually persist. See CurriculumReplayIndex's forgetting
            # detection and forgetting_lr_reduction_factor's docstring.
            if self._lr_scale_factor != 1.0:
                for group in self.optimizer.param_groups:
                    group["lr"] *= self._lr_scale_factor
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

                needs_stratified = (
                    self.stratified_validation or self.curriculum_replay_index is not None
                )
                if needs_stratified:
                    tiered = evaluate_stratified(
                        self.model, self.val_dataset, self.device, self.batch_size, self.num_workers
                    )
                    if self.stratified_validation:
                        tier_summary_parts = []
                        for tier in ("easy", "moderate", "hard"):
                            if tier not in tiered:
                                continue
                            tier_metrics = tiered[tier]
                            epoch_scalars[f"val/rmse_rad_{tier}"] = tier_metrics.rmse_rad
                            tier_summary_parts.append(
                                f"{tier}={tier_metrics.rmse_rad:.3f}({tier_metrics.n_samples})"
                            )
                        if tier_summary_parts:
                            log_line += " | val_rmse_by_tier: " + " ".join(tier_summary_parts)

                    if self.curriculum_replay_index is not None and "easy" in tiered:
                        adjusted = self.curriculum_replay_index.record_easy_validation_metric(
                            epoch, tiered["easy"].rmse_rad
                        )
                        epoch_scalars["curriculum_replay/easy_fraction"] = (
                            self.curriculum_replay_index.easy_fraction
                        )
                        epoch_scalars["curriculum_replay/medium_fraction"] = (
                            self.curriculum_replay_index.medium_fraction
                        )
                        epoch_scalars["curriculum_replay/hard_fraction"] = (
                            self.curriculum_replay_index.hard_fraction
                        )
                        if adjusted and self.forgetting_lr_reduction_factor is not None:
                            self._lr_scale_factor *= self.forgetting_lr_reduction_factor
                            for group in self.optimizer.param_groups:
                                group["lr"] *= self.forgetting_lr_reduction_factor
                            log_line += (
                                f" | FORGETTING DETECTED: replay rebalanced, "
                                f"lr scaled by {self.forgetting_lr_reduction_factor:g} "
                                f"(cumulative scale={self._lr_scale_factor:.3g})"
                            )

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
        "--train-hdf5", type=str, default=None, help="Path to training tiles HDF5 file."
    )
    parser.add_argument(
        "--val-hdf5", type=str, default=None, help="Path to validation tiles HDF5 file."
    )
    parser.add_argument(
        "--finetune-hdf5",
        type=str,
        default=None,
        help=(
            "Optional path to a SNAPHU pseudo-ground-truth fine-tuning tiles "
            "HDF5 file, built via "
            "pyunwrap.utils.snaphu_integration.generate_snaphu_finetune_dataset."
        ),
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
    parser.add_argument(
        "--no-stratified-validation",
        action="store_true",
        help=(
            "Disable per-difficulty-tier validation breakdown (easy/moderate/hard); "
            "report only the overall aggregate val_rmse."
        ),
    )
    parser.add_argument(
        "--use-curriculum-replay",
        action="store_true",
        help=(
            "Use fixed-proportion easy/moderate/hard curriculum replay (with automatic "
            "forgetting detection) instead of the original sequential 3-stage curriculum. "
            "See pyunwrap.training.curriculum.CurriculumReplayConfig for the underlying "
            "defaults and docs/experiments.md for why this exists."
        ),
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=None,
        help=(
            "If set, enables the edge-preserving CoherenceWeightedPhaseSmoothnessLoss "
            "(Component 5) at this weight, in addition to the existing (always-on, "
            "simpler) smoothness term. Unset (default) leaves Component 5 fully "
            "disabled, for exact backward compatibility."
        ),
    )
    parser.add_argument(
        "--real-data-path",
        type=str,
        default=None,
        help=(
            "Optional path to a directory containing a real (or real-like) data stack "
            "(amplitude.npy, wrapped_phase.npy, coherence.npy -- see "
            "pyunwrap.data.real_injection.load_real_stack). If given, training and "
            "validation datasets are built via the real-data + synthetic-injection "
            "pipeline (pyunwrap.data.real_injection.build_real_injection_datasets) "
            "instead of --train-hdf5/--val-hdf5, which become optional in that case."
        ),
    )
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    return parser


def main() -> None:
    """CLI entry point (`pyunwrap-train`): parse args, build datasets/model, and train."""
    parser = build_argparser()
    args = parser.parse_args()

    if args.real_data_path:
        from pyunwrap.data.real_injection import (
            RealSyntheticInjectionConfig,
            build_real_injection_datasets,
            load_real_stack,
        )

        real_stack = load_real_stack(
            amplitude_path=f"{args.real_data_path}/amplitude.npy",
            wrapped_phase_path=f"{args.real_data_path}/wrapped_phase.npy",
            coherence_path=f"{args.real_data_path}/coherence.npy",
        )
        injection_config = RealSyntheticInjectionConfig()
        dataset_paths = build_real_injection_datasets(
            real_stack,
            injection_config,
            out_dir=f"{args.out_dir}/real_injection_cache",
        )
        train_dataset = InSARTileDataset(
            dataset_paths["train"], augment=True, require_ground_truth=True
        )
        val_dataset = InSARTileDataset(
            dataset_paths["val"], augment=False, require_ground_truth=True
        )
    elif args.train_hdf5 and args.val_hdf5:
        train_dataset = InSARTileDataset(args.train_hdf5, augment=True, require_ground_truth=True)
        val_dataset = InSARTileDataset(args.val_hdf5, augment=False, require_ground_truth=True)
    else:
        parser.error("Either --real-data-path, or both --train-hdf5 and --val-hdf5, is required.")

    finetune_dataset = (
        InSARTileDataset(args.finetune_hdf5, augment=True, require_ground_truth=True)
        if args.finetune_hdf5
        else None
    )

    model = AmbiguityNet(pretrained=not args.no_pretrained, k_max=args.k_max)

    curriculum_replay_config = None
    if args.use_curriculum_replay:
        curriculum_replay_config = CurriculumReplayConfig()

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
        stratified_validation=not args.no_stratified_validation,
        curriculum_replay_config=curriculum_replay_config,
    )

    if args.smoothness_weight is not None:
        trainer.criterion = PhysicsInformedUnwrapLoss(
            smoothness_config=SmoothnessConfig(smoothness_weight=args.smoothness_weight),
        )

    trainer.fit()


if __name__ == "__main__":
    main()
