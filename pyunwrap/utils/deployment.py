"""
pyunwrap.utils.deployment
============================

Production deployment utilities for `AmbiguityNet`:

1. **ONNX export**, with optional FP16 quantization, for fast CPU/GPU
   inference without a PyTorch runtime dependency.
2. **Weight caching**: auto-download pretrained weights from a Zenodo DOI on
   first use, cached under `~/.pyunwrap/models/`.
3. **Inference engine selection with fallback**: prefer ONNX Runtime on GPU,
   fall back to OpenVINO (or ONNX Runtime CPU) if GPU execution is
   unavailable or fails.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort

    _HAS_ONNXRUNTIME = True
except ImportError:  # pragma: no cover
    _HAS_ONNXRUNTIME = False

try:
    import openvino as ov

    _HAS_OPENVINO = True
except ImportError:  # pragma: no cover
    _HAS_OPENVINO = False


DEFAULT_CACHE_DIR = Path.home() / ".pyunwrap" / "models"


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    fp16: bool = True,
    opset_version: int = 18,
    dynamic_spatial_size: bool = True,
) -> Path:
    """Export a trained `AmbiguityNet` to ONNX format, optionally FP16-quantized.

    The exported graph's outputs match `AmbiguityNetOutput`'s field order:
    `(k_hat, k_continuous, residue_prob, phi_hat)`.

    Args:
        model: Trained `AmbiguityNet` (or any module with a compatible
            forward signature returning an object with `.k_hat`,
            `.k_continuous`, `.residue_prob`, `.phi_hat` attributes).
        output_path: Destination `.onnx` file path.
        input_shape: Shape used to trace the export (batch, channels, H, W).
            Only used as a tracing template; see `dynamic_spatial_size`.
        fp16: If True, convert the exported graph's weights and activations
            to FP16 for faster inference and half the disk/memory footprint.
            Requires the `onnx` and `onnxconverter-common` packages.
        opset_version: ONNX opset version to target.
        dynamic_spatial_size: If True, mark the H and W dimensions as dynamic
            in the exported graph, so the ONNX model can run on tiles of any
            size (not just `input_shape`'s H/W) -- important since
            `PhaseUnwrapper` may use a different tile size than was used at
            export time.

    Returns:
        Path to the written `.onnx` file.

    Raises:
        ImportError: If the `onnx` package (required for FP16 conversion) is
            not installed and `fp16=True`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().cpu()
    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_spatial_size:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "k_hat": {0: "batch", 2: "height", 3: "width"},
            "k_continuous": {0: "batch", 2: "height", 3: "width"},
            "residue_prob": {0: "batch", 2: "height", 3: "width"},
            "phi_hat": {0: "batch", 2: "height", 3: "width"},
        }

    # Wrap the model so the traced graph returns a flat tuple of tensors
    # (ONNX export cannot trace through the `AmbiguityNetOutput` dataclass).
    class _ONNXWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor):
            out = self.inner(x)
            return out.k_hat, out.k_continuous, out.residue_prob, out.phi_hat

    wrapped = _ONNXWrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["k_hat", "k_continuous", "residue_prob", "phi_hat"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        external_data=False,  # keep everything in one .onnx file: ModelCache and
        # the Zenodo-based auto-download flow both assume a single downloadable
        # artifact per model name, not an .onnx + .onnx.data pair.
    )

    if fp16:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as exc:
            raise ImportError(
                "FP16 export requires the 'onnx' and 'onnxconverter-common' "
                "packages (`pip install onnx onnxconverter-common`)."
            ) from exc
        onnx_model = onnx.load(str(output_path))
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, str(output_path))

    return output_path


# --------------------------------------------------------------------------- #
# Weight caching / auto-download
# --------------------------------------------------------------------------- #


def zenodo_download_url(doi_record_id: str, filename: str) -> str:
    """Build a direct-download URL for a file within a Zenodo record.

    Args:
        doi_record_id: The numeric Zenodo record id (the digits after the
            last `.` in a DOI like `10.5281/zenodo.1234567` -> `1234567`).
        filename: The filename to fetch from that record.

    Returns:
        A direct-download URL string.
    """
    return f"https://zenodo.org/records/{doi_record_id}/files/{filename}?download=1"


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelCache:
    """Manages a local cache of downloaded pretrained model weights under
    `~/.pyunwrap/models/` (or a custom `cache_dir`).

    Example:
        >>> cache = ModelCache()
        >>> path = cache.get_or_download(
        ...     name="pyunwrap-v1",
        ...     url="https://zenodo.org/records/1234567/files/pyunwrap_v1.onnx?download=1",
        ...     expected_sha256=None,
        ... )
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Args:
            cache_dir: Directory to store cached weight files in. Defaults to
                `~/.pyunwrap/models/`.
        """
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def local_path(self, name: str) -> Path:
        """Return the local cache path a given model `name` would be stored at."""
        return self.cache_dir / name

    def is_cached(self, name: str) -> bool:
        """Whether `name` already exists in the local cache."""
        return self.local_path(name).exists()

    def get_or_download(
        self,
        name: str,
        url: str,
        expected_sha256: str | None = None,
        force: bool = False,
    ) -> Path:
        """Return the local path to `name`, downloading it from `url` first if needed.

        Args:
            name: Cache filename (e.g. `"pyunwrap-v1.onnx"`).
            url: Direct-download URL to fetch from if not already cached
                (typically built with `zenodo_download_url`).
            expected_sha256: Optional SHA-256 hex digest to verify the
                download's integrity against; raises `ValueError` on mismatch.
            force: If True, re-download even if already cached.

        Returns:
            Path to the (now-guaranteed-present) local weight file.

        Raises:
            RuntimeError: If the download fails (network error, non-2xx
                response, etc.) -- with a clear, actionable message rather
                than letting a raw connection exception propagate.
            ValueError: If `expected_sha256` is provided and does not match
                the downloaded file's checksum.
        """
        dest = self.local_path(name)
        if dest.exists() and not force:
            return dest

        import urllib.error
        import urllib.request

        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        try:
            print(f"[ModelCache] Downloading {name} from {url} ...")
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                open(tmp_dest, "wb") as out_file,
            ):
                shutil.copyfileobj(response, out_file)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if tmp_dest.exists():
                tmp_dest.unlink()
            raise RuntimeError(
                f"Failed to download model weights '{name}' from {url}: {exc}. "
                "Check network connectivity, or manually place the weight file "
                f"at {dest}."
            ) from exc

        if expected_sha256 is not None:
            actual = _sha256(tmp_dest)
            if actual.lower() != expected_sha256.lower():
                tmp_dest.unlink()
                raise ValueError(
                    f"Checksum mismatch for '{name}': expected {expected_sha256}, got {actual}. "
                    "The download may be corrupted or the URL may point to an unexpected file."
                )

        tmp_dest.rename(dest)
        print(f"[ModelCache] Cached at {dest}")
        return dest


# --------------------------------------------------------------------------- #
# Inference engine selection with fallback
# --------------------------------------------------------------------------- #


class InferenceEngine:
    """Wraps an ONNX model for inference, preferring GPU execution via ONNX
    Runtime and falling back to OpenVINO (or ONNX Runtime CPU) if GPU
    execution is unavailable or raises at session-creation or run time.

    Example:
        >>> engine = InferenceEngine("model.onnx")
        >>> outputs = engine.run(input_array)  # list of 4 numpy arrays
    """

    def __init__(self, onnx_path: str | Path, prefer_gpu: bool = True) -> None:
        """
        Args:
            onnx_path: Path to the exported `.onnx` model.
            prefer_gpu: Whether to attempt GPU execution first.

        Raises:
            ImportError: If neither `onnxruntime` nor `openvino` is installed.
            RuntimeError: If session creation fails on every available backend.
        """
        if not _HAS_ONNXRUNTIME and not _HAS_OPENVINO:
            raise ImportError(
                "Neither onnxruntime nor openvino is installed; install at "
                "least one (`pip install onnxruntime` or `pip install openvino`)."
            )
        self.onnx_path = Path(onnx_path)
        self.backend: str
        self._session = None
        self._ov_compiled = None
        self._init_session(prefer_gpu)

    def _init_session(self, prefer_gpu: bool) -> None:
        """Try backends in priority order, falling back on failure."""
        errors: list[str] = []

        if _HAS_ONNXRUNTIME and prefer_gpu:
            try:
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    self._session = ort.InferenceSession(
                        str(self.onnx_path),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self.backend = "onnxruntime-cuda"
                    return
            except Exception as exc:  # pragma: no cover - depends on GPU availability
                errors.append(f"onnxruntime-cuda failed: {exc}")

        if _HAS_ONNXRUNTIME:
            try:
                self._session = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self.backend = "onnxruntime-cpu"
                return
            except Exception as exc:
                errors.append(f"onnxruntime-cpu failed: {exc}")

        if _HAS_OPENVINO:
            try:
                core = ov.Core()
                model = core.read_model(str(self.onnx_path))
                self._ov_compiled = core.compile_model(model, "CPU")
                self.backend = "openvino-cpu"
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"openvino-cpu failed: {exc}")

        raise RuntimeError(
            "Failed to initialize any inference backend for "
            f"{self.onnx_path}. Errors: " + "; ".join(errors)
        )

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        """Run inference on a single input batch.

        Args:
            x: Input array, shape [B, 3, H, W], float32.

        Returns:
            List of 4 output arrays: `[k_hat, k_continuous, residue_prob, phi_hat]`.
        """
        x = x.astype(np.float32)
        if self.backend.startswith("onnxruntime"):
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: x})
            return outputs
        elif self.backend == "openvino-cpu":
            result = self._ov_compiled(x)
            # OpenVINO returns an ordered mapping keyed by output tensors;
            # rely on declaration order, matching the ONNX export's
            # output_names=["k_hat","k_continuous","residue_prob","phi_hat"].
            return [np.asarray(result[i]) for i in range(4)]
        else:  # pragma: no cover - unreachable given _init_session's contract
            raise RuntimeError(f"Unknown backend: {self.backend}")
