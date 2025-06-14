#!/usr/bin/env python3
"""Tests for building CTensor with Zig."""
from __future__ import annotations

try:
    import os
    import importlib.util
    from pathlib import Path
    import tempfile

    import pytest

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    from tensors.accelerator_backends import c_backend
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

zig_available = importlib.util.find_spec("ziglang") is not None

@pytest.mark.skipif(not zig_available, reason="ziglang not installed")
def test_build_ctensor_with_zig():
    src = c_backend.SOURCE_PATH
    out_dir = tempfile.mkdtemp()
    lib_path = c_backend.build_ctensor_with_zig(str(src), out_dir)
    assert Path(lib_path).exists()


@pytest.mark.skipif(not zig_available, reason="ziglang not installed")
def test_ctensor_ops_from_zig(monkeypatch):
    src = c_backend.SOURCE_PATH
    out_dir = tempfile.mkdtemp()
    lib_path = c_backend.build_ctensor_with_zig(str(src), out_dir)
    assert Path(lib_path).exists()

    monkeypatch.setenv("SPEAKTOME_CTENSOR_LIB", lib_path)
    import importlib
    import tensors.accelerator_backends.c_backend as cb_reload
    cb_reload = importlib.reload(cb_reload)
    ops = cb_reload.CTensorOperations()

    tensor = ops.tensor_from_list([1.0, 2.0, 3.0], dtype=ops.float_dtype, device=None)
    result = ops.log_softmax(tensor, dim=0)
    values = ops.tolist(result)

    import math
    max_v = 3.0
    exps = [math.exp(1.0 - max_v), math.exp(2.0 - max_v), math.exp(3.0 - max_v)]
    total = sum(exps)
    expected = [math.log(v / total) for v in exps]

    assert values == pytest.approx(expected)

