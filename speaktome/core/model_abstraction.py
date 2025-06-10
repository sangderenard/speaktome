#!/usr/bin/env python3
"""Abstractions over model interfaces."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from abc import ABC, abstractmethod
    from typing import Any, Dict
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
except Exception:
    import sys
    print(ENV_SETUP_BOX)
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
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
