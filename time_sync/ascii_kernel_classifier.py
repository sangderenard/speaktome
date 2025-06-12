#!/usr/bin/env python3
"""ASCII classifier using tensor backends for parallel evaluation."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from typing import Any
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tensors import (
        AbstractTensor,
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


def _backend_numpy(ops: AbstractTensor) -> bool:
    """Return True if ``ops`` uses a NumPy-based backend."""
    return isinstance(ops, NumPyTensorOperations)


def _backend_torch(ops: AbstractTensor) -> bool:
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
        tensor_ops: AbstractTensor | None = None,
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
        processed = [self._resize_tensor_to_char(bm) for _, bm in filtered]
        self.charBitmasks = processed


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
        cand_tensor = self.tensor_ops.ensure_tensor(candidate)
        ref_tensor = self.tensor_ops.ensure_tensor(reference)
        arr1 = cand_tensor.to_backend(np_ops)
        arr2 = ref_tensor.to_backend(np_ops)
        return 1.0 - ssim(
            arr1.data.astype(np.float32),
            arr2.data.astype(np.float32),
            data_range=1.0,
        )

    def classify_batch(self, subunit_batch: np.ndarray) -> dict:
        """Classify a batch of subunit images in parallel using tensor ops."""
        dtype = self.tensor_ops.float_dtype
        device = getattr(self.tensor_ops, "default_device", None)

        # Wrap the numpy array in a NumPyTensorOperations instance and convert to the target backend
        np_ops = get_tensor_operations(Faculty.NUMPY)
        batch = self.tensor_ops.ensure_tensor(subunit_batch).to_dtype("float")
        batch_shape = batch.shape()
        N = batch_shape[0]

        # Compute luminance using the tensor's own methods
        if len(batch_shape) == 4 and batch_shape[3] == 3:
            luminance_tensor = batch.mean(dim=3) / 255.0
        elif len(batch_shape) == 3:
            luminance_tensor = batch / 255.0
        else:
            luminance_tensor = self.tensor_ops.zeros((N, *self.char_size), dtype=dtype, device=device)

        if luminance_tensor.shape()[1:] != tuple(self.char_size):
            resized = [self._resize_tensor_to_char(luminance_tensor[i]) for i in range(N)]
            luminance_tensor = self.tensor_ops.stack(resized, dim=0)

        refs = self.tensor_ops.stack(self.charBitmasks, dim=0)
        expanded_inputs = self.tensor_ops.repeat_interleave(
            luminance_tensor[:, None, :, :], refs.shape[0], dim=1
        )
        expanded_refs = self.tensor_ops.repeat_interleave(
            refs[None, :, :, :], N, dim=0
        )
        diff = expanded_inputs - expanded_refs
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
