import importlib.util
from pathlib import Path
import tempfile

import pytest

from speaktome.tensors import c_backend

zig_available = importlib.util.find_spec("ziglang") is not None

@pytest.mark.skipif(not zig_available, reason="ziglang not installed")
def test_build_ctensor_with_zig():
    src = c_backend.SOURCE_PATH
    out_dir = tempfile.mkdtemp()
    lib_path = c_backend.build_ctensor_with_zig(str(src), out_dir)
    assert Path(lib_path).exists()

