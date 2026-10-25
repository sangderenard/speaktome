"""Canonical radial and spectral field algebra for OpenRGB effects.

The spectral-analyzer ray tracer transports wavelength bands and postpones its
display reduction until after band accumulation.  This module keeps that
contract for the LED field: ``K @ S`` first, then ``spectrum @ RGB_weights``.
The matrices are authored with SymPy for the compiled path and NumPy for
inspection/tests; both use the ray tracer's eight-band display basis.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import nnls


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SPECTRAL_ANALYZER_ROOT = WORKSPACE / "spectral-analyzer"
TURING_ROOT = WORKSPACE / "turing"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))

from src.compiler.spectral_colorimetry import (
    SRGB_GAMMA,
    SRGB_LINEAR_CUTOFF,
    SRGB_LINEAR_SLOPE,
    SRGB_OFFSET,
    SRGB_SCALE,
    XYZ_TO_LINEAR_SRGB,
)

SPECTRAL_WAVELENGTHS_NM = np.linspace(700.0, 400.0, 8, dtype=np.float64)
RADIAL_ANGLES = np.asarray(
    (-3.0 * math.pi / 4.0, math.pi, 3.0 * math.pi / 4.0, math.pi / 2.0,
     math.pi / 4.0, 0.0, -math.pi / 4.0, -math.pi / 2.0),
    dtype=np.float64,
)
RADIAL_DIRECTIONS = np.column_stack((np.cos(RADIAL_ANGLES), np.sin(RADIAL_ANGLES)))

COLOR_SYSTEMS = ("hsl", "blackbody", "white")
WHITE_POINTS_XY = {
    "E": (1.0 / 3.0, 1.0 / 3.0),
    "D50": (0.34567, 0.35850),
    "D65": (0.31271, 0.32902),
}


@dataclass(frozen=True)
class HelmholtzColor:
    """One colour expressed on both sides of the linear spectral boundary."""

    system: str
    linear_rgb: np.ndarray
    spectrum: np.ndarray


def _nonnegative_finite(values, *, shape, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite, nonnegative, and have shape {shape}")
    return array


def _spectrum_for_linear_rgb(linear_rgb: np.ndarray) -> np.ndarray:
    spectrum, _residual = nnls(raytracer_display_matrix().T, linear_rgb)
    return spectrum


def _normalise_peak(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values)) if values.size else 0.0
    return values / peak if peak > 0.0 else np.zeros_like(values)


def _linear_saturation(values: np.ndarray, saturation: float) -> np.ndarray:
    amount = min(1.0, max(0.0, float(saturation)))
    luminance = float(np.dot(values, np.asarray((0.2126, 0.7152, 0.0722))))
    return luminance + amount * (values - luminance)


def _blackbody_spectrum(temperature_k: float) -> np.ndarray:
    """Use spectral-analyzer's public Planck sampler on the transport grid."""

    root = str(SPECTRAL_ANALYZER_ROOT)
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        from camera_software.camera_manifest import sample_spectral_model

        return np.asarray(sample_spectral_model(
            {"kind": "blackbody", "temperature_k": float(temperature_k)},
            SPECTRAL_WAVELENGTHS_NM,
        ), dtype=np.float64)
    finally:
        if inserted:
            sys.path.remove(root)


def helmholtz_color_law(
    system: str,
    parameters: Mapping[str, object] | None = None,
) -> HelmholtzColor:
    """Resolve a colour description to linear RGB and the shared 8-band SPD.

    The public systems used by the studio are ``hsl``, ``blackbody``, and
    ``white``.  ``hsv``, ``srgb``, ``linear_rgb``, ``xyz``, and ``spectrum`` are also
    accepted so measured and compiler-produced values enter through the same
    law.  Nothing in this function applies the final sRGB transfer twice:
    every result is linear light and encoding remains an output operation.
    """

    kind = str(system).strip().lower()
    values = dict(parameters or {})
    level = max(0.0, float(values.get("level", 1.0)))
    if kind == "hsl":
        encoded = np.asarray(colorsys.hls_to_rgb(
            (float(values.get("hue_degrees", 0.0)) % 360.0) / 360.0,
            min(1.0, max(0.0, float(values.get("lightness", 0.5)))),
            min(1.0, max(0.0, float(values.get("saturation", 1.0)))),
        ), dtype=np.float64)
        linear = srgb_to_linear_rgb(encoded) * level
        spectrum = _spectrum_for_linear_rgb(linear)
    elif kind == "hsv":
        encoded = np.asarray(colorsys.hsv_to_rgb(
            (float(values.get("hue_degrees", 0.0)) % 360.0) / 360.0,
            min(1.0, max(0.0, float(values.get("saturation", 1.0)))),
            min(1.0, max(0.0, float(values.get("value", values.get("lightness", 1.0))))),
        ), dtype=np.float64)
        linear = srgb_to_linear_rgb(encoded) * level
        spectrum = _spectrum_for_linear_rgb(linear)
    elif kind in {"blackbody", "planck", "kelvin"}:
        temperature_k = max(1.0, float(values.get("temperature_k", 5600.0)))
        spectrum = _blackbody_spectrum(temperature_k)
        projected = spectrum @ raytracer_display_matrix()
        peak = float(np.max(projected))
        if peak > 0.0:
            spectrum = spectrum / peak
            projected = projected / peak
        saturation = min(1.0, max(0.0, float(values.get("saturation", 1.0))))
        luminance = float(np.dot(projected, np.asarray((0.2126, 0.7152, 0.0722))))
        spectrum = (
            saturation * spectrum
            + (1.0 - saturation) * luminance * WHITE_SPECTRUM
        ) * level
        linear = spectrum @ raytracer_display_matrix()
        kind = "blackbody"
    elif kind in {"white", "whitepoint", "white_point"}:
        white_name = str(values.get("white_point", "D65")).upper()
        if white_name not in WHITE_POINTS_XY:
            raise ValueError(
                f"unknown white point {white_name!r}; expected one of {tuple(WHITE_POINTS_XY)}"
            )
        x_white, y_white = WHITE_POINTS_XY[white_name]
        xyz = np.asarray(
            (x_white / y_white, 1.0, (1.0 - x_white - y_white) / y_white),
            dtype=np.float64,
        )
        linear = np.maximum(0.0, np.asarray(XYZ_TO_LINEAR_SRGB) @ xyz)
        linear = np.maximum(0.0, _linear_saturation(
            _normalise_peak(linear), float(values.get("saturation", 1.0)),
        )) * level
        spectrum = _spectrum_for_linear_rgb(linear)
        kind = "white"
    elif kind == "srgb":
        encoded = _nonnegative_finite(values.get("rgb"), shape=(3,), name="sRGB")
        linear = srgb_to_linear_rgb(encoded) * level
        spectrum = _spectrum_for_linear_rgb(linear)
    elif kind == "linear_rgb":
        linear = _nonnegative_finite(
            values.get("rgb"), shape=(3,), name="linear RGB",
        ) * level
        spectrum = _spectrum_for_linear_rgb(linear)
    elif kind == "xyz":
        xyz = _nonnegative_finite(values.get("xyz"), shape=(3,), name="XYZ")
        linear = np.maximum(0.0, np.asarray(XYZ_TO_LINEAR_SRGB) @ xyz) * level
        spectrum = _spectrum_for_linear_rgb(linear)
    elif kind in {"spectrum", "spd"}:
        spectrum = _nonnegative_finite(
            values.get("spectrum"), shape=(8,), name="spectrum",
        ) * level
        linear = spectrum @ raytracer_display_matrix()
        kind = "spectrum"
    else:
        raise ValueError(
            f"unknown colour system {system!r}; expected hsl, hsv, blackbody, white, "
            "srgb, linear_rgb, xyz, or spectrum"
        )
    linear = np.asarray(linear, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    linear.setflags(write=False)
    spectrum.setflags(write=False)
    return HelmholtzColor(kind, linear, spectrum)


def srgb_to_linear_rgb(rgb: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
    """Decode IEC 61966-2-1 sRGB values to linear-light RGB."""

    encoded = np.asarray(rgb, dtype=np.float64)
    if not np.isfinite(encoded).all():
        raise ValueError("sRGB values must be finite")
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(
        encoded <= 0.04045,
        encoded / SRGB_LINEAR_SLOPE,
        ((encoded + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA,
    )


def linear_rgb_to_srgb(rgb: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
    """Encode nonnegative linear-light RGB with the IEC sRGB perceptual curve."""

    linear = np.asarray(rgb, dtype=np.float64)
    if not np.isfinite(linear).all():
        raise ValueError("linear RGB values must be finite")
    linear = np.maximum(0.0, linear)
    return np.clip(np.where(
        linear <= SRGB_LINEAR_CUTOFF,
        SRGB_LINEAR_SLOPE * linear,
        SRGB_SCALE * linear ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET,
    ), 0.0, 1.0)


def symbolic_linear_rgb_to_srgb(channel):
    """SymPy form of the compiler-owned IEC sRGB transfer function."""

    positive = sp.Max(sp.Integer(0), channel)
    return sp.Piecewise(
        (sp.Float(SRGB_LINEAR_SLOPE) * positive,
         positive <= sp.Float(SRGB_LINEAR_CUTOFF)),
        (sp.Float(SRGB_SCALE) * positive ** sp.Float(1.0 / SRGB_GAMMA)
         - sp.Float(SRGB_OFFSET), True),
    )


@lru_cache(maxsize=1)
def raytracer_display_matrix() -> np.ndarray:
    """Return the ray tracer's eight-band to linear-display-RGB matrix."""

    if not SPECTRAL_ANALYZER_ROOT.is_dir():
        raise RuntimeError(
            f"spectral-analyzer is required at {SPECTRAL_ANALYZER_ROOT}"
        )
    inserted = False
    root = str(SPECTRAL_ANALYZER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        # scene_orders states and tests that this is byte-for-byte the native
        # ray tracer's band_to_display_rgb transfer.
        from scene_orders import _display_rgb_weights

        matrix = np.asarray(
            _display_rgb_weights(SPECTRAL_WAVELENGTHS_NM), dtype=np.float64
        )
    finally:
        if inserted:
            sys.path.remove(root)
    if matrix.shape != (8, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("spectral-analyzer returned an invalid 8x3 display matrix")
    matrix.setflags(write=False)
    return matrix


def default_cmy_spectra() -> np.ndarray:
    """Three nonnegative ray-tracer-band spectra for cyan, magenta, yellow.

    Cyan and yellow are single-band spectral colours.  Magenta is explicitly
    composite: red plus violet, demonstrating that non-spectral colours remain
    a distribution until the final display projection.
    """

    display = raytracer_display_matrix()
    spectra = np.zeros((3, 8), dtype=np.float64)

    cyan_band = 5
    spectra[0, cyan_band] = 1.0 / float(np.max(display[cyan_band]))

    red_band, violet_band = 1, 7
    spectra[1, violet_band] = 1.0 / float(display[violet_band, 2])
    red_remainder = 1.0 - spectra[1, violet_band] * display[violet_band, 0]
    spectra[1, red_band] = red_remainder / float(display[red_band, 0])

    yellow_band = 3
    spectra[2, yellow_band] = 1.0 / float(np.max(display[yellow_band]))
    return spectra


def equal_display_white_spectrum() -> np.ndarray:
    """Nonnegative three-line spectrum projecting to linear RGB (1, 1, 1)."""

    display = raytracer_display_matrix()
    active = np.asarray((1, 4, 6), dtype=np.int64)
    weights = np.linalg.solve(display[active].T, np.ones(3, dtype=np.float64))
    if np.any(weights < 0.0):
        raise RuntimeError("ray-tracer display basis no longer admits the white spectrum")
    spectrum = np.zeros(8, dtype=np.float64)
    spectrum[active] = weights
    return spectrum


CMY_SPECTRA = default_cmy_spectra()
WHITE_SPECTRUM = equal_display_white_spectrum()


def radial_contributions(
    x: float,
    y: float,
    *,
    center_contribution: float = 0.25,
    falloff_power: float = 2.0,
    directions: Sequence[Sequence[float]] = RADIAL_DIRECTIONS,
) -> np.ndarray:
    """Generate one contribution per supplied real-world light direction."""

    if not np.isfinite((x, y, center_contribution, falloff_power)).all():
        raise ValueError("point and center contribution must be finite")
    if not 0.0 <= center_contribution <= 1.0:
        raise ValueError("center_contribution must be in the inclusive range 0..1")
    if falloff_power <= 0.0:
        raise ValueError("falloff_power must be finite and positive")
    direction_rows = np.asarray(directions, dtype=np.float64)
    if (
        direction_rows.ndim != 2
        or direction_rows.shape[1] != 2
        or not np.isfinite(direction_rows).all()
    ):
        raise ValueError("directions must be a finite matrix with shape (lights,2)")
    distance = math.hypot(float(x), float(y))
    if distance == 0.0:
        return np.full(len(direction_rows), float(center_contribution), dtype=np.float64)
    radial = min(1.0, distance)
    point = radial * np.asarray((x / distance, y / distance), dtype=np.float64)
    squared_distance = np.sum((direction_rows - point) ** 2, axis=1)
    distance_lobe = np.maximum(0.0, 1.0 - squared_distance / 2.0) ** falloff_power
    center_reference = 0.5**falloff_power
    # Preserve the public center value for every power. The correction fades
    # away at the perimeter, where the distance lobe is the complete law.
    # center control with a correction that vanishes quadratically at the
    # perimeter; the former broad linear background is gone.
    center_correction = (center_contribution - center_reference) * (1.0 - radial) ** 2
    return np.clip(distance_lobe + center_correction, 0.0, 1.0)


def symbolic_radial_kernel(
    x, y, direction_x, direction_y, center_contribution, falloff_power=sp.Integer(2)
):
    """SymPy authority for the radial driver field used by every effect."""

    distance = sp.sqrt(x**2 + y**2)
    radial = sp.Min(sp.Integer(1), distance)
    safe_distance = sp.Max(distance, sp.Float(1.0e-12))
    point_x = radial * x / safe_distance
    point_y = radial * y / safe_distance
    squared_distance = (point_x - direction_x) ** 2 + (point_y - direction_y) ** 2
    distance_base = sp.Max(
        sp.Integer(0), sp.Integer(1) - squared_distance / 2
    )
    distance_lobe = distance_base**falloff_power
    center_reference = sp.Pow(sp.Rational(1, 2), falloff_power)
    return sp.Min(
        sp.Integer(1),
        sp.Max(
            sp.Integer(0),
            distance_lobe
            + (center_contribution - center_reference) * (1 - radial) ** 2,
        ),
    )


def accumulate_spectral_field(
    contributions: Sequence[Sequence[float]],
    source_spectra: Sequence[Sequence[float]],
) -> np.ndarray:
    """Matrix field accumulation: ``(lights,sources) @ (sources,bands)``."""

    kernel = np.asarray(contributions, dtype=np.float64)
    spectra = np.asarray(source_spectra, dtype=np.float64)
    if kernel.ndim != 2 or spectra.ndim != 2 or kernel.shape[1] != spectra.shape[0]:
        raise ValueError("contributions and spectra must have shapes (L,S) and (S,B)")
    if spectra.shape[1] != 8:
        raise ValueError("spectra must use the ray tracer's eight-band basis")
    if not np.isfinite(kernel).all() or not np.isfinite(spectra).all():
        raise ValueError("field matrices must be finite")
    if np.any(kernel < 0.0) or np.any(spectra < 0.0):
        raise ValueError("spectral field coefficients must be nonnegative")
    return kernel @ spectra


def spectral_field_to_linear_rgb(field_spectrum: Sequence[Sequence[float]]) -> np.ndarray:
    """Project an accumulated ``(lights,8)`` spectrum to HDR linear RGB."""

    spectrum = np.asarray(field_spectrum, dtype=np.float64)
    if spectrum.ndim != 2 or spectrum.shape[1] != 8:
        raise ValueError("field spectrum must have shape (lights,8)")
    if not np.isfinite(spectrum).all() or np.any(spectrum < 0.0):
        raise ValueError("field spectrum must be finite and nonnegative")
    return spectrum @ raytracer_display_matrix()


def symbolic_spectral_display(
    source_weights: sp.MatrixBase,
    source_spectra: sp.MatrixBase,
    *,
    exposure,
    output_level,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Accumulate spectral matrices and map HDR linear RGB to LED drive.

    The exponential exposure is applied only after spectral addition.  It is
    monotone and asymptotic, so overlap remains distinguishable instead of
    becoming a hard-clipped white plateau.
    """

    weights = sp.Matrix(source_weights)
    spectra = sp.Matrix(source_spectra)
    if weights.cols != 1 or spectra.rows != weights.rows or spectra.cols != 8:
        raise ValueError("symbolic shapes must be weights=(sources,1), spectra=(sources,8)")
    display = sp.ImmutableDenseMatrix(raytracer_display_matrix().tolist())
    field_spectrum = weights.T * spectra
    linear_rgb = field_spectrum * display
    channels = []
    for channel in range(3):
        positive = sp.Max(sp.Integer(0), linear_rgb[0, channel])
        mapped = sp.Integer(1) - sp.exp(-exposure * positive)
        channels.append(output_level * symbolic_linear_rgb_to_srgb(mapped))
    return tuple(channels)


__all__ = [
    "CMY_SPECTRA", "COLOR_SYSTEMS", "HelmholtzColor",
    "RADIAL_ANGLES", "RADIAL_DIRECTIONS",
    "SPECTRAL_WAVELENGTHS_NM", "WHITE_SPECTRUM",
    "WHITE_POINTS_XY",
    "accumulate_spectral_field", "default_cmy_spectra",
    "equal_display_white_spectrum", "radial_contributions",
    "helmholtz_color_law",
    "linear_rgb_to_srgb", "raytracer_display_matrix", "spectral_field_to_linear_rgb",
    "srgb_to_linear_rgb", "symbolic_linear_rgb_to_srgb",
    "symbolic_radial_kernel", "symbolic_spectral_display",
]
