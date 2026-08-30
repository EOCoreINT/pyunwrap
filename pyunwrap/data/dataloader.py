"""
pyunwrap.data.dataloader
==========================

PyTorch `Dataset` for loading pre-tiled InSAR patches produced by
`pyunwrap.data.preprocessing`, with on-the-fly data augmentation.

Each sample is returned as a dict of tensors:

    {
        "wrapped_phase":  FloatTensor [1, H, W]  (normalized to [-1, 1])
        "coherence":      FloatTensor [1, H, W]  ([0, 1])
        "amplitude":      FloatTensor [1, H, W]  (~[0, 1])
        "true_unwrapped": FloatTensor [1, H, W]  (radians; training targets only)
        "true_ambiguity": FloatTensor [1, H, W]  (integer-valued; training targets only)
    }

Augmentation note on phase correctness
---------------------------------------
Random 90-degree rotations and horizontal/vertical flips are applied
*identically* across every channel (wrapped phase, coherence, amplitude, true
unwrapped phase). Because interferometric phase is a per-pixel scalar
quantity (not a vector field with a spatial direction baked into its sign),
spatially permuting the array does not require negating the phase values
themselves — what matters is that every channel undergoes the *exact same*
spatial transform so their per-pixel correspondence (and therefore the
`wrapped + 2*pi*k = unwrapped` identity) is preserved after augmentation. This
class enforces that by applying one sampled transform to the whole tile dict
at once, and by recomputing (rather than transforming) the integer ambiguity
implicitly through consistent joint transformation of wrapped/unwrapped
phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py

    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False

from pyunwrap.data.preprocessing import denormalize_phase
from pyunwrap.synthetic.generator import compute_ambiguity


class InSARTileDataset(Dataset):
    """PyTorch Dataset over pre-tiled InSAR patches stored in an HDF5 file.

    The HDF5 file is expected to follow the layout produced by
    `pyunwrap.data.preprocessing.save_tiles_hdf5`: one group per tile
    (`tile_00000000`, `tile_00000001`, ...), each containing datasets
    `wrapped_phase`, `coherence`, `amplitude`, and optionally
    `true_unwrapped`.

    Example:
        >>> ds = InSARTileDataset("tiles.h5", augment=True)
        >>> sample = ds[0]
        >>> sample["wrapped_phase"].shape
        torch.Size([1, 256, 256])
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        augment: bool = True,
        require_ground_truth: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            hdf5_path: Path to the HDF5 tile file.
            augment: If True, apply random 90-degree rotations and
                horizontal/vertical flips (8-fold dihedral augmentation).
            require_ground_truth: If True, raise at access time if a tile is
                missing `true_unwrapped` (i.e. this is a training dataset, not
                a raw-inference dataset). Set False for unlabeled/real data.
            seed: Optional seed for the augmentation RNG (kept separate from
                any global RNG for reproducibility across DataLoader workers).
        """
        if not _HAS_H5PY:
            raise ImportError("h5py is required for InSARTileDataset.")
        self.hdf5_path = Path(hdf5_path)
        self.augment = augment
        self.require_ground_truth = require_ground_truth
        self._rng = np.random.default_rng(seed)

        # Discover tile group names once; the actual file handle is opened
        # lazily per-worker in __getitem__ to remain safe under
        # multiprocessing DataLoader workers (h5py file handles do not
        # reliably survive process forking).
        with h5py.File(self.hdf5_path, "r") as f:
            self._tile_names = sorted(f.keys())
            if len(self._tile_names) == 0:
                raise ValueError(f"No tiles found in {self.hdf5_path}")
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return len(self._tile_names)

    def _get_file(self) -> h5py.File:
        """Lazily open (and cache, per-process) the HDF5 file handle."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        f = self._get_file()
        grp = f[self._tile_names[index]]

        wrapped_phase = np.asarray(grp["wrapped_phase"], dtype=np.float32)
        coherence = np.asarray(grp["coherence"], dtype=np.float32)
        amplitude = np.asarray(grp["amplitude"], dtype=np.float32)

        has_gt = "true_unwrapped" in grp
        if self.require_ground_truth and not has_gt:
            raise KeyError(
                f"Tile {self._tile_names[index]} in {self.hdf5_path} has no "
                "'true_unwrapped' dataset, but require_ground_truth=True."
            )
        true_unwrapped = np.asarray(grp["true_unwrapped"], dtype=np.float32) if has_gt else None

        if self.augment:
            wrapped_phase, coherence, amplitude, true_unwrapped = self._augment(
                wrapped_phase, coherence, amplitude, true_unwrapped
            )

        sample: dict[str, torch.Tensor] = {
            "wrapped_phase": torch.from_numpy(wrapped_phase).unsqueeze(0).float(),
            "coherence": torch.from_numpy(coherence).unsqueeze(0).float(),
            "amplitude": torch.from_numpy(amplitude).unsqueeze(0).float(),
        }

        if true_unwrapped is not None:
            sample["true_unwrapped"] = torch.from_numpy(true_unwrapped).unsqueeze(0).float()

            # Recompute the ground-truth integer ambiguity map *after*
            # augmentation, directly from the (jointly transformed) wrapped
            # and unwrapped phase, rather than transforming a precomputed k.
            # This guarantees k stays exactly consistent with the augmented
            # wrapped/unwrapped pair even at edge cases near +-pi.
            wrapped_rad = denormalize_phase(wrapped_phase.astype(np.float64))
            unwrapped_rad = true_unwrapped.astype(np.float64)
            true_ambiguity = compute_ambiguity(unwrapped_rad, wrapped_rad).astype(np.float32)
            sample["true_ambiguity"] = torch.from_numpy(true_ambiguity).unsqueeze(0).float()

        return sample

    # ----------------------------------------------------------------- #
    # Augmentation
    # ----------------------------------------------------------------- #

    def _augment(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        amplitude: np.ndarray,
        true_unwrapped: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply one randomly-sampled dihedral (rotation + flip) transform
        identically to every channel in the tile.

        Samples uniformly from the 8-element dihedral group D4: 4 rotations
        (0/90/180/270 degrees) x 2 (flip / no flip), matching the standard
        "8-fold" augmentation used for imagery with no canonical orientation.

        Args:
            wrapped_phase, coherence, amplitude: 2D arrays, same shape.
            true_unwrapped: Optional 2D array, same shape, or None.

        Returns:
            The four arrays after applying the same transform to each
            (`true_unwrapped` stays None if it was None on input).
        """
        k_rot = int(self._rng.integers(0, 4))  # number of 90-degree rotations
        do_flip = bool(self._rng.integers(0, 2))  # whether to mirror horizontally

        def _transform(arr: np.ndarray) -> np.ndarray:
            out = np.rot90(arr, k=k_rot)
            if do_flip:
                out = np.fliplr(out)
            return np.ascontiguousarray(out)

        wrapped_phase = _transform(wrapped_phase)
        coherence = _transform(coherence)
        amplitude = _transform(amplitude)
        if true_unwrapped is not None:
            true_unwrapped = _transform(true_unwrapped)

        return wrapped_phase, coherence, amplitude, true_unwrapped

    def __del__(self) -> None:
        # Best-effort cleanup of the lazily-opened file handle.
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass


def build_dataloader(
    hdf5_path: str | Path,
    batch_size: int = 16,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    require_ground_truth: bool = True,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    """Convenience factory for a `torch.utils.data.DataLoader` over `InSARTileDataset`.

    Args:
        hdf5_path: Path to the HDF5 tile file.
        batch_size: Batch size.
        augment: Whether to apply dihedral augmentation.
        shuffle: Whether to shuffle tile order each epoch.
        num_workers: Number of DataLoader worker processes.
        require_ground_truth: See `InSARTileDataset`.
        seed: Optional augmentation RNG seed.

    Returns:
        A configured `DataLoader`.
    """
    dataset = InSARTileDataset(
        hdf5_path,
        augment=augment,
        require_ground_truth=require_ground_truth,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop a ragged final batch only during training/shuffled use
    )
