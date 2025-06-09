try:
    """Backend-agnostic linear layers for small network experiments."""

    from __future__ import annotations

    from typing import Any, Iterable, Dict

    from .model_abstraction import AbstractModelWrapper
    from ..tensors.abstraction import AbstractTensorOperations
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


class AbstractLinearLayer:
    """Single linear transformation with optional activation."""

    def __init__(
        self,
        weight: Any,
        bias: Any,
        tensor_ops: AbstractTensorOperations,
        activation: str | None = None,
    ) -> None:
        self.weight = weight
        self.bias = bias
        self.ops = tensor_ops
        self.activation = activation

    def _apply_activation(self, tensor: Any) -> Any:
        """Apply ``self.activation`` if specified."""
        if self.activation is None:
            return tensor
        if self.activation == "relu":
            return self.ops.clamp(tensor, min_val=0.0)
        raise ValueError(f"Unsupported activation {self.activation!r}")

    def forward(self, inputs: Any) -> Any:
        # matrix multiply then add bias using backend operators
        out = self.ops._AbstractTensorOperations__apply_operator(
            "matmul", inputs, self.weight
        )
        out = self.ops._AbstractTensorOperations__apply_operator(
            "add", out, self.bias
        )
        return self._apply_activation(out)


class SequentialLinearModel(AbstractModelWrapper):
    """Compose multiple :class:`AbstractLinearLayer` objects."""

    def __init__(
        self,
        layers: Iterable[AbstractLinearLayer],
        tensor_ops: AbstractTensorOperations,
    ) -> None:
        self.layers = list(layers)
        self.ops = tensor_ops

    @classmethod
    def from_weights(
        cls,
        weights: Iterable[tuple[Any, Any]],
        tensor_ops: AbstractTensorOperations,
        activation: str | None = None,
    ) -> "SequentialLinearModel":
        """Construct layers from ``(weight, bias)`` pairs."""
        built_layers = [
            AbstractLinearLayer(w, b, tensor_ops, activation=activation)
            for w, b in weights
        ]
        return cls(built_layers, tensor_ops)

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
