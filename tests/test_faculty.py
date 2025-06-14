"""Tests for the :mod:`tensors.faculty` module.

These exercises confirm that the faculty-detection logic honours installed
libraries and environment overrides.  The language used here aims for clarity
and a touch of grace so that future readers grasp the intent without wading
through terse jargon.
"""

import pytest
import sys
from unittest import mock

from tensors.faculty import detect_faculty, Faculty, FORCE_ENV, DEFAULT_FACULTY
# --- END HEADER ---


def test_detect_faculty_enum() -> None:
    """Ensure :func:`detect_faculty` returns a proper :class:`Faculty` value."""
    fac = detect_faculty()
    assert isinstance(fac, Faculty)


def test_detect_faculty_pure_python(monkeypatch) -> None:
    """Verify the fallback when no numerical libraries are present."""
    import builtins

    with mock.patch.dict(sys.modules):
        # Remove any already imported numerical libraries
        sys.modules.pop("torch_geometric", None)
        sys.modules.pop("torch", None)
        sys.modules.pop("numpy", None)

        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"torch", "torch_geometric", "numpy"}:
                raise ModuleNotFoundError
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert detect_faculty() == Faculty.PURE_PYTHON


def test_detect_faculty_numpy() -> None:
    """Confirm that NumPy alone elevates the faculty to :attr:`Faculty.NUMPY`."""
    with mock.patch.dict(sys.modules):
        sys.modules.pop('torch_geometric', None)
        sys.modules.pop('torch', None)
        sys.modules['numpy'] = mock.MagicMock()  # Simulate numpy is importable
        assert detect_faculty() == Faculty.NUMPY


def test_detect_faculty_torch() -> None:
    """Check that PyTorch takes precedence when PyG is missing."""
    with mock.patch.dict(sys.modules):
        sys.modules.pop('torch_geometric', None)
        sys.modules['torch'] = mock.MagicMock()  # Simulate torch is importable
        # Numpy might also be present, torch should take precedence
        sys.modules['numpy'] = mock.MagicMock()
        assert detect_faculty() == Faculty.TORCH


def test_detect_faculty_pygeo() -> None:
    """Ensure the presence of PyG triggers the highest faculty tier."""
    with mock.patch.dict(sys.modules):
        sys.modules['torch_geometric'] = mock.MagicMock()  # Simulate PyG is importable
        # Torch and Numpy would typically also be present for PyG
        sys.modules['torch'] = mock.MagicMock()
        sys.modules['numpy'] = mock.MagicMock()
        assert detect_faculty() == Faculty.PYGEO


@pytest.mark.parametrize("faculty_name, expected_faculty", [
    ("PURE_PYTHON", Faculty.PURE_PYTHON),
    ("NUMPY", Faculty.NUMPY),
    ("TORCH", Faculty.TORCH),
    ("PYGEO", Faculty.PYGEO),
    ("CTENSOR", Faculty.CTENSOR),
])
def test_detect_faculty_env_override_valid(monkeypatch, faculty_name, expected_faculty) -> None:
    """Respect valid environment overrides regardless of installed packages."""
    monkeypatch.setenv(FORCE_ENV, faculty_name)
    # Simulate highest level installed to ensure override takes precedence
    with mock.patch.dict(sys.modules, {'torch_geometric': mock.MagicMock()}):
        assert detect_faculty() == expected_faculty


def test_detect_faculty_env_override_invalid(monkeypatch) -> None:
    """Reject unknown values supplied via ``SPEAKTOME_FACULTY``."""
    monkeypatch.setenv(FORCE_ENV, "INVALID_FACULTY_NAME")
    with pytest.raises(ValueError, match="Unknown faculty override: INVALID_FACULTY_NAME"):
        detect_faculty()


def test_detect_faculty_env_override_case_insensitive(monkeypatch) -> None:
    """Overrides should ignore letter casing."""
    monkeypatch.setenv(FORCE_ENV, "numpy")
    with mock.patch.dict(sys.modules, {"torch_geometric": mock.MagicMock()}):
        assert detect_faculty() == Faculty.NUMPY


def test_default_faculty_respects_env(monkeypatch) -> None:
    """Verify ``DEFAULT_FACULTY`` honours the override environment variable."""
    monkeypatch.setenv(FORCE_ENV, "pure_python")
    with mock.patch.dict(sys.modules, {"torch_geometric": mock.MagicMock()}):
        from importlib import reload
        mod = reload(sys.modules[detect_faculty.__module__])
        assert mod.DEFAULT_FACULTY is mod.Faculty.PURE_PYTHON
