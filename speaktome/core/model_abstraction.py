try:
    from __future__ import annotations

    from abc import ABC, abstractmethod
    from typing import Any, Dict
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
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
