#!/usr/bin/env python3
"""Optional dependency loader for FontMapper."""
from __future__ import annotations

import importlib

# Basic function for optional imports

def optional_import(module_name: str, attr: str | None = None):
    """Attempt to import a module or attribute.

    Returns the imported module or attribute if available, otherwise ``None``.
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr) if attr else module
    except Exception:
        return None

# Optional groups

pika = optional_import("pika")
PIKA_AVAILABLE = pika is not None

qtwidgets = optional_import("PyQt5.QtWidgets")
qtgui = optional_import("PyQt5.QtGui")
qtcore = optional_import("PyQt5.QtCore")
PYQT_AVAILABLE = all([qtwidgets, qtgui, qtcore])

colorsys = optional_import("colorsys")
ssim = optional_import("skimage.metrics", "structural_similarity")
COLOR_MIXING_AVAILABLE = colorsys is not None and ssim is not None

pynvml = optional_import("pynvml")
if pynvml is not None:
    try:
        pynvml.nvmlInit()
    except Exception:
        pynvml = None
NVML_AVAILABLE = pynvml is not None
