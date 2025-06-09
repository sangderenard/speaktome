"""Prototype for backend-agnostic linear networks."""

from __future__ import annotations

from typing import Any, Iterable, Dict

from .model_abstraction import AbstractModelWrapper
from ..tensors.abstraction import AbstractTensorOperations

# --- END HEADER ---


class AbstractLinearLayer:
    """Single linear transformation using :class:`AbstractTensorOperations`."""

    def __init__(
        self,
        weight: Any,
        bias: Any,
        tensor_ops: AbstractTensorOperations,
    ) -> None:
        self.weight = weight
        self.bias = bias
        self.ops = tensor_ops

    def forward(self, inputs: Any) -> Any:
        # matrix multiply then add bias using backend operators
        out = self.ops._AbstractTensorOperations__apply_operator(
            "matmul", inputs, self.weight
        )
        return self.ops._AbstractTensorOperations__apply_operator(
            "add", out, self.bias
        )


class SequentialLinearModel(AbstractModelWrapper):
    """Compose multiple :class:`AbstractLinearLayer` objects."""

    def __init__(
        self,
        layers: Iterable[AbstractLinearLayer],
        tensor_ops: AbstractTensorOperations,
    ) -> None:
        self.layers = list(layers)
        self.ops = tensor_ops

    def forward(
        self, input_ids: Any, attention_mask: Any | None = None, **kwargs
    ) -> Dict[str, Any]:
        out = input_ids
        for layer in self.layers:
            out = layer.forward(out)
        return {"logits": out}

    def get_device(self) -> Any:
        if not self.layers:
            return "unknown"
        return self.ops.get_device(self.layers[0].weight)
