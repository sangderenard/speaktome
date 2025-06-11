#!/usr/bin/env python3
"""Helper classes for FM16."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from .charset_ops import (
    list_printable_characters,
    generate_checkerboard_pattern,
    generate_variants,
    bytemaps_as_ascii,
    obtain_charset,
)

try:  # optional ML stack
    import torch  # pragma: no cover - optional backend

    from .model_configs import ModelCompatConfig, ModelConfig
    from .char_sorter import CharSorter
    from .transforms import (
        AddRandomNoise,
        DistortionChain,
        RandomGaussianBlur,
        ToTensorAndToDevice,
    )
    from .datasets import CustomDataset, CustomInputDataset
    from .model_ops import load_char_sorter, evaluate_batch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - torch not installed
    TORCH_AVAILABLE = False

__all__ = [
    "list_printable_characters",
    "generate_checkerboard_pattern",
    "generate_variants",
    "bytemaps_as_ascii",
    "obtain_charset",
]

if TORCH_AVAILABLE:
    __all__ += [
        "ModelCompatConfig",
        "ModelConfig",
        "CharSorter",
        "AddRandomNoise",
        "DistortionChain",
        "RandomGaussianBlur",
        "ToTensorAndToDevice",
        "CustomDataset",
        "CustomInputDataset",
        "load_char_sorter",
        "evaluate_batch",
    ]
