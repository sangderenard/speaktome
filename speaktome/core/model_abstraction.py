#!/usr/bin/env python3
"""Abstractions over model interfaces."""
from __future__ import annotations

try:
    from abc import ABC, abstractmethod
    from typing import Any, Dict
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
except Exception:
    import sys
    print("Model abstraction failed to import")
    sys.exit(1)
# --- END HEADER ---

class AbstractModelWrapper(ABC):
    @abstractmethod
    def forward(self, input_ids: Any, attention_mask: Any, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_device(self) -> Any:
        pass

class PyTorchModelWrapper(AbstractModelWrapper):
    def __init__(self, model: torch.nn.Module):
        if torch is None:
            raise RuntimeError("PyTorch is required for this wrapper")
        self.model = model
        # Inference remains the conservative default. FluxGraph can opt into
        # a gradient-enabled forward when it is using model scores as a
        # differentiable reward signal for continuous physiology. Discrete
        # token selection (top-k/top-p) is still intentionally outside that
        # graph; enabling this flag never pretends an argmax is differentiable.
        self.track_gradients = False

    def set_gradient_tracking(self, enabled: bool) -> None:
        self.track_gradients = bool(enabled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        context = torch.enable_grad if self.track_gradients else torch.no_grad
        with context():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        return {'logits': logits}

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device
