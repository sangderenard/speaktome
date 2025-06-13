#!/usr/bin/env python3
"""Model loading and batch evaluation helpers."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from .char_sorter import CharSorter
from .datasets import CustomInputDataset
from .model_configs import ModelConfig
from .transforms import ToTensorAndToDevice
# --- END HEADER ---

from pathlib import Path
from typing import Iterable, Tuple


def load_char_sorter(config: ModelConfig, model_path: str, device: torch.device = torch.device("cpu")) -> CharSorter:
    """Load a ``CharSorter`` model from a file."""
    model = CharSorter(config)
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    return model


def evaluate_batch(
    model: CharSorter,
    images: Iterable[torch.Tensor],
    batch_size: int = 8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evaluate a sequence of tensors and return concatenated logits."""
    dataset = CustomInputDataset(list(images), (model.demo_width, model.demo_height), transform=ToTensorAndToDevice(device or model.fc1.weight.device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    model_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["sub_image"].to(model_device)
            logits = model(img)
            outputs.append(logits.cpu())
    return torch.cat(outputs, dim=0)


__all__ = ["load_char_sorter", "evaluate_batch"]
