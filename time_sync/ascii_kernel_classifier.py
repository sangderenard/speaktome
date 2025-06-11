import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from fontmapper.FM16.modules.charset_ops import obtain_charset

class AsciiKernelClassifier:
    def __init__(self, ramp: str, font_path="fontmapper/FM16/consola.ttf", font_size=16, char_size=(16, 16), loss_mode="sad"):
        self.ramp = ramp
        self.vocab_size = len(ramp)
        self.font_path = font_path
        self.font_size = font_size
        self.char_size = char_size
        self.loss_mode = loss_mode  # "sad" or "ssim"
        self.charset = None
        self.charBitmasks = None
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

    def _prepare_reference_bitmasks(self):
        # Use obtain_charset from charset_ops to get charset and bitmasks
        fonts, charset, charBitmasks, max_width, max_height = obtain_charset(
            font_files=[self.font_path], font_size=self.font_size, complexity_level=0
        )
        # Only keep chars in our ramp, preserving charset order
        filtered = [(c, bm) for c, bm in zip(charset, charBitmasks) if c in self.ramp]
        self.charset = [c for c, _ in filtered]
        self.charBitmasks = [self._resize_to_char_size(bm) for _, bm in filtered]

    def _resize_to_char_size(self, arr):
        img = Image.fromarray(arr)
        img = img.resize(self.char_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def sad_loss(self, candidate: np.ndarray, reference: np.ndarray) -> float:
        return np.sum(np.abs(candidate.astype(np.float32) - reference.astype(np.float32)))

    def ssim_loss(self, candidate: np.ndarray, reference: np.ndarray) -> float:
        if not SSIM_AVAILABLE:
            raise RuntimeError("SSIM loss requires scikit-image")
        # ssim returns similarity, so loss is 1 - ssim
        return 1.0 - ssim(candidate, reference, data_range=1.0)

    def classify_batch(self, subunit_batch: np.ndarray) -> dict:
        N = subunit_batch.shape[0]
        indices = np.zeros(N, dtype=int)
        chars = []
        losses = np.zeros(N)
        for i in range(N):
            subunit = subunit_batch[i]
            if subunit.ndim == 3 and subunit.shape[2] == 3:
                luminance_map = np.mean(subunit, axis=2) / 255.0
            elif subunit.ndim == 2:
                luminance_map = subunit / 255.0
            else:
                indices[i] = 0
                chars.append(self.ramp[0])
                losses[i] = 0
                continue
            # Resize input to match char image size if needed
            if luminance_map.shape != self.char_size:
                luminance_map = np.array(
                    Image.fromarray((luminance_map * 255).astype(np.uint8)).resize(self.char_size, Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            # Compute loss to all reference glyphs
            if self.loss_mode == "ssim" and SSIM_AVAILABLE:
                losses_all = [self.ssim_loss(luminance_map, ref) for ref in self.charBitmasks]
            else:
                losses_all = [self.sad_loss(luminance_map, ref) for ref in self.charBitmasks]
            idx = int(np.argmin(losses_all))
            indices[i] = idx
            chars.append(self.charset[idx])
            losses[i] = losses_all[idx]
        return {
            "indices": indices,
            "chars": chars,
            "losses": losses,
            "logits": None,
        }