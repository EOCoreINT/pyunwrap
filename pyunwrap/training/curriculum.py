"""
pyunwrap.training.curriculum
================================

Curriculum learning and curriculum replay for
`pyunwrap.training.trainer.Trainer`.

Two distinct mechanisms live here:

1. **Sequential curriculum** (`CurriculumIndex`, the original mechanism,
   moved here unchanged from `pyunwrap.training.trainer` -- still
   importable from there too, for backward compatibility): training
   progresses through 3 fixed stages (easy -> moderate -> hard), and each
   later stage's eligible-tile set is a strict expansion of the previous
   one. Simple, and mirrors common curriculum-learning practice, but a
   real training run found it can cause the network to catastrophically
   forget the easy regime once the hardest stage dominates every
   subsequent epoch (see `docs/experiments.md`, Experiment 3) -- easy
   tiles are entirely absent from every stage-3 epoch's training data.

2. **Curriculum replay** (`CurriculumReplayIndex` / `CurriculumReplayConfig`,
   new): every epoch instead sees a fixed-proportion *mixture* of
   easy/medium/hard tiles (e.g. 25% / 25% / 50%), so easy and medium data
   are never fully dropped from training regardless of how far along the
   schedule is. Also includes automatic forgetting detection: if
   validation performance on the easy tier stops improving for
   `forgetting_patience` epochs, `easy_replay_fraction` is increased
   automatically (taken from `hard_fraction` first, then `medium_replay_fraction`).

`Trainer`'s `curriculum_replay_config` parameter selects mechanism (2) when
given; `None` (the default) preserves mechanism (1) exactly, for full
backward compatibility with every existing training run and test.

A note on terminology: this module's difficulty tiers are named
`"easy"`, `"moderate"`, `"hard"` throughout (matching the pre-existing
`classify_difficulty_tier`/`evaluate_stratified` API, which already shipped
before curriculum replay was added). `CurriculumReplayConfig`'s field is
named `medium_replay_fraction` to match the terminology used when curriculum
replay was specified; it controls the same `"moderate"` tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyunwrap.data.dataloader import InSARTileDataset

# --------------------------------------------------------------------------- #
# Shared difficulty classification (used by both curriculum mechanisms and
# by pyunwrap.training.trainer.evaluate_stratified)
# --------------------------------------------------------------------------- #


@dataclass
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


def compute_tile_difficulty_stats(dataset: InSARTileDataset) -> list[TileDifficulty]:
    """Scan every tile in `dataset` once (without augmentation) and compute
    its difficulty stats.

    Standalone (not private to any one curriculum class) specifically so
    every consumer -- `CurriculumIndex`, `CurriculumReplayIndex`, and
    `pyunwrap.training.trainer.evaluate_stratified` -- uses the exact same
    coherence/gradient computation, rather than three easily-drifting
    reimplementations of "how hard is this tile."

    Args:
        dataset: An `InSARTileDataset` to scan. Must have
            `require_ground_truth=True` for `p99_gradient` to be meaningful
            (silently reports `0.0` otherwise).

    Returns:
        One `TileDifficulty` per tile, in dataset order.
    """
    stats = []
    was_augmenting = dataset.augment
    dataset.augment = False  # stats must reflect the canonical, unaugmented tile
    try:
        for i in range(len(dataset)):
            sample = dataset[i]
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

            stats.append(TileDifficulty(index=i, mean_coherence=mean_coh, p99_gradient=p99_grad))
    finally:
        dataset.augment = was_augmenting
    return stats


def classify_difficulty_tier(stat: TileDifficulty) -> str:
    """Classify a single tile's difficulty into exactly one of three tiers,
    as a strict partition (every tile gets exactly one label).

    Args:
        stat: A `TileDifficulty` (e.g. from `compute_tile_difficulty_stats`).

    Returns:
        `"easy"`, `"moderate"`, or `"hard"`.
    """
    if stat.mean_coherence > 0.7 and stat.p99_gradient <= math.pi:
        return "easy"
    if stat.mean_coherence > 0.4:
        return "moderate"
    return "hard"


def _bucket_indices_by_tier(stats: list[TileDifficulty]) -> dict[str, list[int]]:
    """Partition `stats` into `{"easy": [...], "moderate": [...], "hard": [...]}`
    dataset-index lists via `classify_difficulty_tier`."""
    buckets: dict[str, list[int]] = {"easy": [], "moderate": [], "hard": []}
    for stat in stats:
        buckets[classify_difficulty_tier(stat)].append(stat.index)
    return buckets


# --------------------------------------------------------------------------- #
# Mechanism 1: sequential curriculum (original)
# --------------------------------------------------------------------------- #


class CurriculumIndex:
    """Computes and caches per-tile difficulty stats, and exposes epoch-aware
    index subsets implementing the original 3-stage sequential curriculum.

    Stage boundaries (by 1-indexed epoch number):
        - Epochs 1-20:  mean_coherence > 0.7 AND p99_gradient <= pi (easy)
        - Epochs 21-50: mean_coherence > 0.4 (moderate; includes atmosphere/
          orbital-ramp-heavy tiles, which are present throughout the
          synthetic dataset regardless of coherence)
        - Epochs 51+:   full dataset, no filtering (hard; includes
          low-coherence tiles and gradients exceeding the Nyquist limit)

    `STAGE_1_END_EPOCH`/`STAGE_2_END_EPOCH` are exposed as class attributes
    (not just inlined in `indices_for_epoch`) specifically so
    `Trainer`'s learning-rate schedule can align its own restart points to
    these same boundaries -- see
    `pyunwrap.training.trainer.build_curriculum_aware_scheduler`'s
    docstring for why that alignment matters.

    Known limitation this class has (fixed by `CurriculumReplayIndex`, not
    by this class): once training reaches stage 3, easy tiles are entirely
    absent from every subsequent epoch's training data, which a real
    training run showed can cause the network to forget how to perform
    well on them (see `docs/experiments.md`, Experiment 3). Use
    `CurriculumReplayIndex` (via `Trainer`'s `curriculum_replay_config`) if
    that matters for your use case.
    """

    #: Last epoch of curriculum stage 1 (easy). Stage 2 begins the next epoch.
    STAGE_1_END_EPOCH = 20
    #: Last epoch of curriculum stage 2 (moderate). Stage 3 (full difficulty)
    #: begins the next epoch and continues to the end of training.
    STAGE_2_END_EPOCH = 50

    def __init__(self, dataset: InSARTileDataset) -> None:
        """
        Args:
            dataset: The full training `InSARTileDataset` to index. Must have
                `require_ground_truth=True` (curriculum stats depend on
                `true_unwrapped`).
        """
        self.dataset = dataset
        self._stats: list[TileDifficulty] = compute_tile_difficulty_stats(dataset)

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
        if epoch <= self.STAGE_1_END_EPOCH:
            eligible = [
                s.index for s in self._stats if s.mean_coherence > 0.7 and s.p99_gradient <= math.pi
            ]
        elif epoch <= self.STAGE_2_END_EPOCH:
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
# Mechanism 2: curriculum replay (new)
# --------------------------------------------------------------------------- #


@dataclass
class CurriculumReplayConfig:
    """Configuration for `CurriculumReplayIndex`.

    Attributes:
        use_curriculum_replay: Master on/off switch, read by `Trainer` to
            decide whether to use replay mixing (this mechanism) or the
            original sequential `CurriculumIndex` staging.
        easy_replay_fraction: Target fraction of each epoch's tiles drawn
            from the "easy" tier.
        medium_replay_fraction: Target fraction drawn from the "moderate"
            tier (named `medium_replay_fraction` to match this feature's
            specification; see the module docstring's terminology note).
        hard_fraction: Target fraction drawn from the "hard" tier. The
            three fractions must sum to `1.0`.
        min_easy_validation_f1_or_rmse_threshold: Optional absolute
            threshold on easy-tier validation RMSE. If set and easy-tier
            RMSE ever exceeds it, forgetting is flagged immediately
            (independent of the patience-based trend detection below).
            `None` disables this absolute check, relying only on the
            trend-based detection.
        forgetting_patience: Number of consecutive validation checks
            without an easy-tier RMSE improvement before automatically
            increasing `easy_replay_fraction`.
        log_per_difficulty_metrics: Whether `Trainer` logs a per-tier
            metrics breakdown (via `evaluate_stratified`) at every
            validation step when replay is active. Independent of whether
            `Trainer.stratified_validation` is also set -- replay's own
            forgetting-detection loop needs the easy-tier metric regardless
            of whether it's also being logged.
        rebalance_every_n_epochs: How often (in epochs) forgetting
            detection is checked and fractions are potentially adjusted.
            Checking every epoch (`1`) is the most responsive; larger
            values reduce sensitivity to single noisy validation passes.

    Raises:
        ValueError: If the three fractions don't sum to `1.0` (within
            floating-point tolerance), if any is outside `[0, 1]`, or if
            `forgetting_patience`/`rebalance_every_n_epochs` is < 1.
    """

    use_curriculum_replay: bool = True
    easy_replay_fraction: float = 0.25
    medium_replay_fraction: float = 0.25
    hard_fraction: float = 0.50
    min_easy_validation_f1_or_rmse_threshold: float | None = None
    forgetting_patience: int = 5
    log_per_difficulty_metrics: bool = True
    rebalance_every_n_epochs: int = 1

    def __post_init__(self) -> None:
        for name, value in [
            ("easy_replay_fraction", self.easy_replay_fraction),
            ("medium_replay_fraction", self.medium_replay_fraction),
            ("hard_fraction", self.hard_fraction),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        total = self.easy_replay_fraction + self.medium_replay_fraction + self.hard_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"easy_replay_fraction + medium_replay_fraction + hard_fraction must sum to "
                f"1.0, got {total}"
            )
        if self.forgetting_patience < 1:
            raise ValueError(f"forgetting_patience must be >= 1, got {self.forgetting_patience}")
        if self.rebalance_every_n_epochs < 1:
            raise ValueError(
                f"rebalance_every_n_epochs must be >= 1, got {self.rebalance_every_n_epochs}"
            )


class CurriculumReplayIndex:
    """Builds per-epoch training index samples as a fixed-proportion mixture
    of easy/moderate/hard tiles, with automatic forgetting-triggered
    rebalancing.

    Unlike `CurriculumIndex` (which changes *which* tiles are eligible as
    training progresses through discrete stages), this class samples from
    all three difficulty tiers *every* epoch, so easy and moderate tiles
    are never entirely absent from training data regardless of how long
    training runs.

    Example:
        >>> config = CurriculumReplayConfig(easy_replay_fraction=0.3, medium_replay_fraction=0.3, hard_fraction=0.4)
        >>> replay = CurriculumReplayIndex(train_dataset, config, seed=0)
        >>> indices = replay.indices_for_epoch(epoch=1)  # a mix of all 3 tiers, every epoch
        >>> # after validating on the easy tier:
        >>> adjusted = replay.record_easy_validation_metric(epoch=5, rmse=12.3)
        >>> if adjusted:
        ...     print(f"Forgetting detected; easy_fraction is now {replay.easy_fraction}")
    """

    #: Amount easy_replay_fraction is increased by on each detected forgetting event.
    FORGETTING_EASY_FRACTION_INCREMENT = 0.10

    def __init__(
        self,
        dataset: InSARTileDataset,
        config: CurriculumReplayConfig,
        epoch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            dataset: The full training `InSARTileDataset` to sample from.
                Must have `require_ground_truth=True`.
            config: Replay behavior configuration.
            epoch_size: Number of tiles to sample per epoch. Defaults to
                `len(dataset)` (matches the original dataset's size, so a
                fresh mix is drawn each epoch rather than accumulating).
            seed: Seed for the internal sampling RNG, for deterministic,
                reproducible epoch composition across runs.
        """
        self.dataset = dataset
        self.config = config
        self._stats = compute_tile_difficulty_stats(dataset)
        self._tier_indices = _bucket_indices_by_tier(self._stats)
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self._rng = np.random.default_rng(seed)

        # Mutable current fractions -- start at the configured values, may
        # be adjusted upward (for "easy") by forgetting detection.
        self.easy_fraction = config.easy_replay_fraction
        self.medium_fraction = config.medium_replay_fraction
        self.hard_fraction = config.hard_fraction

        self._easy_val_history: list[tuple[int, float]] = []
        self._epochs_since_improvement = 0
        self.n_forgetting_adjustments = 0

        for tier in ("easy", "moderate", "hard"):
            if len(self._tier_indices[tier]) == 0:
                print(
                    f"[CurriculumReplayIndex] WARNING: tier '{tier}' has 0 tiles in this "
                    "dataset; epochs will draw 0 samples from it regardless of its configured "
                    "fraction until the dataset changes."
                )

    def indices_for_epoch(self, epoch: int) -> list[int]:
        """Return a fresh, randomly-sampled mixture of tile indices for
        `epoch`, drawn from all three tiers according to the current
        (possibly forgetting-adjusted) fractions.

        Args:
            epoch: Current 1-indexed epoch number (accepted for interface
                symmetry with `CurriculumIndex.indices_for_epoch`; sampling
                itself does not depend on the epoch number beyond advancing
                the internal RNG state).

        Returns:
            A shuffled list of dataset indices, length approximately
            `epoch_size` (exact count can differ slightly due to rounding
            across three fractions, and a tier with fewer available tiles
            than its target count is sampled with replacement rather than
            silently under-filling the epoch).
        """
        del epoch  # not used for sampling itself; see docstring
        targets = {
            "easy": round(self.epoch_size * self.easy_fraction),
            "moderate": round(self.epoch_size * self.medium_fraction),
            "hard": round(self.epoch_size * self.hard_fraction),
        }

        indices: list[int] = []
        for tier, n_target in targets.items():
            pool = self._tier_indices[tier]
            if n_target <= 0 or len(pool) == 0:
                continue
            replace = n_target > len(pool)
            chosen = self._rng.choice(pool, size=n_target, replace=replace)
            indices.extend(int(i) for i in chosen)

        self._rng.shuffle(indices)
        return indices

    def record_easy_validation_metric(self, epoch: int, rmse: float) -> bool:
        """Feed in the latest easy-tier validation RMSE and apply automatic
        forgetting-triggered rebalancing if warranted.

        Two independent triggers can fire an adjustment: an **absolute
        threshold** (`rmse` exceeds `config.min_easy_validation_f1_or_rmse_threshold`,
        if set) or a **trend-based** check (`rmse` has not improved on its
        best-seen value for `config.forgetting_patience` consecutive calls).

        On trigger, `easy_fraction` increases by
        `FORGETTING_EASY_FRACTION_INCREMENT` (capped at `1.0`), with the
        difference taken from `hard_fraction` first and then
        `medium_fraction` if `hard_fraction` alone isn't enough -- hard
        data is reduced before moderate data, since hard-tile
        overrepresentation is the mechanism actually implicated in the
        forgetting failure mode this class exists to prevent (see
        `docs/experiments.md`, Experiment 3).

        Args:
            epoch: Current 1-indexed epoch number, recorded for the
                internal history (not used in the adjustment logic itself).
            rmse: Easy-tier validation RMSE for this epoch (e.g. from
                `evaluate_stratified(...)["easy"].rmse_rad`).

        Returns:
            `True` if fractions were adjusted this call, `False` otherwise.
        """
        self._easy_val_history.append((epoch, rmse))

        best_so_far = min((r for _, r in self._easy_val_history[:-1]), default=math.inf)
        improved = rmse < best_so_far
        if improved:
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        threshold = self.config.min_easy_validation_f1_or_rmse_threshold
        absolute_trigger = threshold is not None and rmse > threshold
        trend_trigger = self._epochs_since_improvement >= self.config.forgetting_patience

        if not (absolute_trigger or trend_trigger):
            return False

        old_easy = self.easy_fraction
        self.easy_fraction = min(1.0, self.easy_fraction + self.FORGETTING_EASY_FRACTION_INCREMENT)
        delta = self.easy_fraction - old_easy

        take_from_hard = min(delta, self.hard_fraction)
        self.hard_fraction -= take_from_hard
        remaining = delta - take_from_hard
        if remaining > 0:
            take_from_medium = min(remaining, self.medium_fraction)
            self.medium_fraction -= take_from_medium

        self._epochs_since_improvement = 0
        self.n_forgetting_adjustments += 1
        print(
            f"[CurriculumReplayIndex] Forgetting detected at epoch {epoch} "
            f"(easy RMSE={rmse:.4f}, {'above absolute threshold' if absolute_trigger else 'no improvement for ' + str(self.config.forgetting_patience) + ' checks'}); "
            f"easy_fraction {old_easy:.3f} -> {self.easy_fraction:.3f}"
        )
        return True
