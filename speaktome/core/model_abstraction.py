from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
# --- END HEADER ---
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

class AbstractModelWrapper(ABC):
    HEADER = "Unified interface for varying model types"

    @staticmethod
    def test() -> None:
        pass
    @abstractmethod
    def forward(self, input_ids: Any, attention_mask: Any, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_device(self) -> Any:
        pass

class PyTorchModelWrapper(AbstractModelWrapper):
    HEADER = "Wrapper exposing PyTorch models via common API"

    @staticmethod
    def test() -> None:
        pass
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
