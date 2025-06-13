#!/usr/bin/env python3
"""AcceleratorCoordinator double buffering behaviour."""
from __future__ import annotations

try:
    import pytest

    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from tensors.accelerator_backends.coordinator import AcceleratorCoordinator
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def test_double_buffer_synchronization():
    coord = AcceleratorCoordinator("c")
    coord.create_tensor("a", [1.0, 4.0])
    coord.enqueue("a", "sqrt_")
    coord.synchronize("a")
    tensor = coord.get_tensor("a")
    assert tensor.tolist() == [1.0, 2.0]
    coord.terminate("a")


def test_future_completion():
    """Operations may return a ``Future`` for async usage."""
    coord = AcceleratorCoordinator("c")
    coord.create_tensor("b", [4.0, 9.0])
    fut = coord.enqueue("b", "sqrt_", return_future=True)
    coord.synchronize("b")
    result = fut.result(timeout=1)
    assert result.tolist() == [2.0, 3.0]
    coord.terminate("b")

