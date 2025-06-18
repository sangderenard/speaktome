"""Rust backend stub test."""
from __future__ import annotations

try:
    import os
    import pytest
    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    from tensors.accelerator_backends import RustTensorOperations
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

@pytest.mark.stub
def test_rust_backend_placeholder():
    backend = RustTensorOperations()
    with pytest.raises(NotImplementedError):
        backend.full_((2,), 0.0, None, None)
