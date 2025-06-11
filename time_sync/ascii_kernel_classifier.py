#!/usr/bin/env python3
"""ASCII classifier using tensor backends for parallel evaluation."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from typing import Any
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tensors import (
        AbstractTensorOperations,
        get_tensor_operations,
        PyTorchTensorOperations,
        NumPyTensorOperations,
        Faculty,
    )
except Exception as e:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from fontmapper.FM16.modules.charset_ops import obtain_charset


def _backend_numpy(ops: AbstractTensorOperations) -> bool:
    """Return True if ``ops`` uses a NumPy-based backend."""
    return isinstance(ops, NumPyTensorOperations)


def _backend_torch(ops: AbstractTensorOperations) -> bool:
    """Return True if ``ops`` uses a PyTorch backend."""
    return isinstance(ops, PyTorchTensorOperations)

class AsciiKernelClassifier:
    def __init__(
        self,
        ramp: str,
        font_path: str = "fontmapper/FM16/consola.ttf",
        font_size: int = 16,
        char_size: tuple[int, int] = (16, 16),
        loss_mode: str = "sad",
        tensor_ops: AbstractTensorOperations | None = None,
    ) -> None:
        self.ramp = ramp
        self.vocab_size = len(ramp)
        self.font_path = font_path
        self.font_size = font_size
        self.char_size = char_size
        self.loss_mode = loss_mode  # "sad" or "ssim"
        self.tensor_ops = tensor_ops or get_tensor_operations()
        self.charset: list[str] | None = None
        self.charBitmasks: list[Any] | None = None
        self._prepare_reference_bitmasks()

    def set_font(self, font_path=None, font_size=None, char_size=None):
        """Set font parameters and regenerate reference bitmasks."""
        if font_path is not None:
            self.font_path = font_path
        if font_size is not None:
            self.font_size = font_size
        if char_size is not None:
            self.char_size = char_size
        self._prepare_reference_bitmasks()

    def _prepare_reference_bitmasks(self) -> None:
        """Generate reference glyph tensors for the current ASCII ramp."""
        fonts, charset, charBitmasks, _max_w, _max_h = obtain_charset(
            font_files=[self.font_path], font_size=self.font_size, complexity_level=0
        )
        filtered = [(c, bm) for c, bm in zip(charset, charBitmasks) if c in self.ramp and bm is not None]
        self.charset = [c for c, _ in filtered]
        processed = [self._resize_to_char_size(bm) for _, bm in filtered]
        self.charBitmasks = processed

    def _resize_to_char_size(self, arr: np.ndarray):
        """Resize ``arr`` to ``self.char_size`` using the tensor abstraction."""
        np_ops = get_tensor_operations(Faculty.NUMPY)
        tensor_np = np_ops.__class__()
        tensor_np.data = np_ops.tensor_from_list(arr.tolist(), dtype=np_ops.float_dtype, device=None)
        tensor_ops = tensor_np.to_backend(self.tensor_ops)
        tensor_ops_data = tensor_ops.data / 255.0
        resized = self.tensor_ops.interpolate(tensor_ops_data, self.char_size)
        out_np = self.tensor_ops.__class__()
        out_np.data = resized
        out_np_converted = out_np.to_backend(np_ops)
        if out_np_converted.shape() != self.char_size:
            raise ValueError(f"Resized shape {out_np_converted.shape()} does not match char_size {self.char_size}")
        if getattr(out_np_converted, "data", None) is None:
            raise ValueError("Bitmask conversion failed: .data is None after resizing/conversion.")
        return out_np_converted  # Store backend-native data, not a NumPy array

    def _resize_tensor_to_char(self, tensor: Any) -> Any:
        """Resize ``tensor`` to ``self.char_size`` using the tensor abstraction."""
        return self.tensor_ops.interpolate(tensor, self.char_size)

    def sad_loss(self, candidate: Any, reference: Any) -> float:
        """Sum of absolute differences between ``candidate`` and ``reference``."""
        ops = self.tensor_ops
        diff = candidate - reference
        abs_diff = ops.sqrt(ops.pow(diff, 2))
        total = ops.mean(abs_diff) * ops.numel(abs_diff)
        return float(ops.item(total))

    def ssim_loss(self, candidate: Any, reference: Any) -> float:
        if not SSIM_AVAILABLE:
            raise RuntimeError("SSIM loss requires scikit-image")
        np_ops = get_tensor_operations(Faculty.NUMPY)
        arr1 = self.tensor_ops.to_backend(np_ops)
        arr2 = self.tensor_ops.to_backend(np_ops)
        return 1.0 - ssim(arr1.data.astype(np.float32), arr2.data.astype(np.float32), data_range=1.0)

    def classify_batch(self, subunit_batch: np.ndarray) -> dict:
        """Classify a batch of subunit images in parallel using tensor ops."""
        dtype = self.tensor_ops.float_dtype
        device = getattr(self.tensor_ops, "default_device", None)

        # Wrap the numpy array in a NumPyTensorOperations instance and convert to the target backend
        np_ops = get_tensor_operations(Faculty.NUMPY)
        batch_tensor = np_ops.__class__()
        luminance_tensor = self.tensor_ops.__class__()
        batch_tensor.data = subunit_batch
        batch = batch_tensor.to_backend(self.tensor_ops)
        batch.to_dtype("float")
        batch_shape = batch.shape()
        N = batch_shape[0]

        # Compute luminance using the tensor's own methods
        if len(batch_shape) == 4 and batch_shape[3] == 3:
            luminance_tensor.from_numpy(batch.mean(dim=3) / 255.0)
        elif len(batch_shape) == 3:
            luminance_tensor.from_numpy(batch / 255.0)
        else:
            luminance_tensor.zeros((N, *self.char_size), dtype=dtype, device=device)

        if luminance_tensor.shape()[1:] != tuple(self.char_size):
            resized = [self._resize_tensor_to_char(luminance[i]) for i in range(N)]
            luminance_tensor.stack(resized, dim=0)

        refs_data = [bm.data for bm in self.charBitmasks]
        refs = self.tensor_ops.stack(refs_data, dim=0)
        diff = luminance_tensor[:, None, :, :] - refs[None, :, :, :]
        abs_diff = self.tensor_ops.sqrt(self.tensor_ops.pow(diff, 2))

        losses = self.tensor_ops.mean(abs_diff, dim=(2, 3))
        idxs = self.tensor_ops.argmin(losses, dim=1)
        row_indices = self.tensor_ops.arange(N, device=device, dtype=self.tensor_ops.long_dtype)
        selected_losses = self.tensor_ops.select_by_indices(losses, row_indices, idxs)

        chars = [self.charset[int(i)] for i in idxs.tolist()]
        return {
            "indices": idxs,
            "chars": chars,
            "losses": selected_losses,
            "logits": None,
        }
