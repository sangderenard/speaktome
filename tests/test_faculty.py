import pytest
import sys
from unittest import mock

from speaktome.faculty import detect_faculty, Faculty, FORCE_ENV


def test_detect_faculty_enum():
    """Checks if detect_faculty returns a Faculty enum instance (basic check)."""
    fac = detect_faculty()
    assert isinstance(fac, Faculty)


def test_detect_faculty_pure_python():
    """Test PURE_PYTHON detection when no optional libraries are 'found'."""
    with mock.patch.dict(sys.modules):
        # Ensure these modules are treated as not imported for this test
        sys.modules.pop('torch_geometric', None)
        sys.modules.pop('torch', None)
        sys.modules.pop('numpy', None)
        assert detect_faculty() == Faculty.PURE_PYTHON


def test_detect_faculty_numpy():
    """Test NUMPY detection when numpy is available, but torch/pyg are not."""
    with mock.patch.dict(sys.modules):
        sys.modules.pop('torch_geometric', None)
        sys.modules.pop('torch', None)
        sys.modules['numpy'] = mock.MagicMock()  # Simulate numpy is importable
        assert detect_faculty() == Faculty.NUMPY


def test_detect_faculty_torch():
    """Test TORCH detection when torch is available, but pyg is not."""
    with mock.patch.dict(sys.modules):
        sys.modules.pop('torch_geometric', None)
        sys.modules['torch'] = mock.MagicMock()  # Simulate torch is importable
        # Numpy might also be present, torch should take precedence
        sys.modules['numpy'] = mock.MagicMock()
        assert detect_faculty() == Faculty.TORCH


def test_detect_faculty_pygeo():
    """Test PYGEO detection when torch_geometric is available."""
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
])
def test_detect_faculty_env_override_valid(monkeypatch, faculty_name, expected_faculty):
    """Test valid SPEAKTOME_FACULTY environment variable overrides."""
    monkeypatch.setenv(FORCE_ENV, faculty_name)
    # Simulate highest level installed to ensure override takes precedence
    with mock.patch.dict(sys.modules, {'torch_geometric': mock.MagicMock()}):
        assert detect_faculty() == expected_faculty


def test_detect_faculty_env_override_invalid(monkeypatch):
    """Test invalid SPEAKTOME_FACULTY environment variable override."""
    monkeypatch.setenv(FORCE_ENV, "INVALID_FACULTY_NAME")
    with pytest.raises(ValueError, match="Unknown faculty override: INVALID_FACULTY_NAME"):
        detect_faculty()
