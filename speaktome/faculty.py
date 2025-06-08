from __future__ import annotations

"""Faculty levels for runtime resources."""

import os
from enum import Enum, auto
# --- END HEADER ---


class Faculty(Enum):
    """Available compute/resource tiers."""

    PURE_PYTHON = auto()  # No third-party numerical libs
    NUMPY = auto()  # Research demo of algorithm
    TORCH = auto()  # Performant production faculty
    PYGEO = auto()  # NN programmable smart search


FORCE_ENV = "SPEAKTOME_FACULTY"


def detect_faculty() -> Faculty:
    """Return the highest available Faculty tier based on installed packages.

    The environment variable ``SPEAKTOME_FACULTY`` may be set to force a
    specific tier regardless of installed libraries.
    """
    forced = os.environ.get(FORCE_ENV)
    if forced:
        try:
            return Faculty[forced.upper()]
        except KeyError as exc:  # pragma: no cover - env misuse
            raise ValueError(f"Unknown faculty override: {forced}") from exc

    try:  # check for PyGeoMind requirements
        import torch_geometric  # type: ignore
        _ = torch_geometric  # silence lint
        return Faculty.PYGEO
    except ModuleNotFoundError:
        pass
    try:
        import torch  # type: ignore
        _ = torch
        return Faculty.TORCH
    except ModuleNotFoundError:
        try:
            import numpy  # type: ignore
            _ = numpy
            return Faculty.NUMPY
        except ModuleNotFoundError:
            return Faculty.PURE_PYTHON


DEFAULT_FACULTY = detect_faculty()
