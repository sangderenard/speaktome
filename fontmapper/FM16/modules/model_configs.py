#!/usr/bin/env python3
"""Model configuration classes used by FM38 and FMS6."""
from __future__ import annotations

# --- END HEADER ---

from dataclasses import dataclass
from typing import Any

@dataclass
class ModelCompatConfig:
    """Compatibility options for loading models."""
    charset: str
    font_files: Any
    font_size: int
    conv1_out: int
    conv2_out: int
    linear_out: int
    width: int
    height: int

@dataclass
class ModelConfig:
    """Core model parameters."""
    charset: str
    refresh_rate: Any
    demo_image: Any
    demo_images: Any
    training_categories: Any
    training_gradients: Any
    training_images: Any
    human_in_the_loop: Any
    image_batch_limit: Any
    epochs_per_preview: int
    font_files: Any
    font_size: int
    conv1_out: int
    conv2_out: int
    linear_out: int
    width: int
    height: int
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    gradient_loss_function: Any
    training_loss_function: Any
    demo_loss_function: Any
    model_path: str
    version: str
    model: Any = None
    dataloader: Any = None
