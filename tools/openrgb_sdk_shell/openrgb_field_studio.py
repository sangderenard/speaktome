"""Interactive Pygame studio for the native-HID managed light engines."""

from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from openrgb_cpu_temperature import (
    CpuTemperatureReader,
    DEFAULT_ASSEMBLY,
    DEFAULT_SENSOR_IDENTIFIER,
)
from openrgb_fan_rpm import FanRpmReader
from openrgb_audio_vectorscope import NativeAudioVectorscope
from openrgb_field_config import load_or_create_config, save_config
from openrgb_dt_spirograph import (
    ManagedRadialSpirograph,
    compile_spirograph_law,
    hypotrochoid_repeat_period,
)
from openrgb_dt_three_body import (
    BODY_COUNT,
    ManagedThreeBodyField,
    PHYSICS_GEOMETRIES,
    _EagerEquationPiece,
    _body_coordinates,
    _grb_wire_frame,
    _logical_domain_frame,
    eager_three_body_pieces,
)
from openrgb_dt_walk import address_domain_for
from openrgb_shell import OpenRGBShellError, connect
from asrock_hid_abi import ASRockEffectMode
from asrock_hid_transport import NativeUniformColor, connect_direct
from openrgb_spectral_field import (
    COLOR_SYSTEMS,
    RADIAL_DIRECTIONS,
    WHITE_POINTS_XY,
    accumulate_spectral_field,
    helmholtz_color_law,
    linear_rgb_to_srgb,
    radial_contributions,
    spectral_field_to_linear_rgb,
)


SPIROGRAPH_LAW_NAME = "openrgb_spectral_spirograph"
STROBE_WAVEFORMS = ("off", "sine", "square", "triangle", "saw", "beat", "audio", "timed")
HARDWARE_MODES = (
    ("off", ASRockEffectMode.OFF),
    ("static", ASRockEffectMode.STATIC),
    ("breathing", ASRockEffectMode.BREATHING),
    ("strobe", ASRockEffectMode.STROBE),
    ("spectrum", ASRockEffectMode.SPECTRUM_CYCLE),
    ("random", ASRockEffectMode.RANDOM),
    ("audio", ASRockEffectMode.MUSIC),
    ("wave", ASRockEffectMode.WAVE),
    ("spring", ASRockEffectMode.SPRING),
    ("stack", ASRockEffectMode.STACK),
    ("cram", ASRockEffectMode.CRAM),
    ("scan", ASRockEffectMode.SCAN),
    ("neon", ASRockEffectMode.NEON),
    ("water", ASRockEffectMode.WATER),
    ("rainbow", ASRockEffectMode.RAINBOW),
)
HARDWARE_SPEED_MODES = {
    int(ASRockEffectMode.BREATHING), int(ASRockEffectMode.STROBE),
    int(ASRockEffectMode.SPECTRUM_CYCLE), int(ASRockEffectMode.RANDOM),
    int(ASRockEffectMode.WAVE), int(ASRockEffectMode.SPRING),
    int(ASRockEffectMode.STACK), int(ASRockEffectMode.CRAM),
    int(ASRockEffectMode.SCAN), int(ASRockEffectMode.WATER),
    int(ASRockEffectMode.RAINBOW),
}
HARDWARE_COLOR_MODES = {
    int(ASRockEffectMode.STATIC), int(ASRockEffectMode.BREATHING),
    int(ASRockEffectMode.STROBE),
}
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "openrgb_field_studio_config.json"
ANGLE_RATE_MULTIPLIER_MAX = 100.0
GUARANTEED_HID_RATE_HZ = 9.9
GUARANTEED_HID_RESET_NS = math.ceil(1_000_000_000.0 / GUARANTEED_HID_RATE_HZ)
STRIP_RECEIVER_POINTS = np.asarray(
    ((0.0, 0.9), (0.0, 0.3), (0.0, -0.3), (0.0, -0.9)),
    dtype=np.float64,
)


def _rate_multiplier(value) -> float:
    return min(ANGLE_RATE_MULTIPLIER_MAX, max(1.0, float(value)))


def color_spectrum(
    hue_degrees: float, saturation: float = 1.0, lightness: float = 0.5,
    *, color_system: str = "hsl", temperature_k: float = 5600.0,
    white_point: str = "D65",
) -> np.ndarray:
    """Resolve one studio source through the shared Helmholtz colour law."""

    parameters = {
        "hue_degrees": hue_degrees,
        "saturation": saturation,
        "lightness": lightness,
        "temperature_k": temperature_k,
        "white_point": white_point,
    }
    if color_system != "hsl":
        parameters["level"] = lightness
    return helmholtz_color_law(color_system, parameters).spectrum


def encoded_source_color(
    hue_degrees: float, saturation: float, lightness: float,
    color_system: str = "hsl", temperature_k: float = 5600.0,
    white_point: str = "D65",
) -> np.ndarray:
    parameters = {
        "hue_degrees": hue_degrees,
        "saturation": saturation,
        "lightness": lightness,
        "temperature_k": temperature_k,
        "white_point": white_point,
    }
    if color_system != "hsl":
        parameters["level"] = lightness
    return linear_rgb_to_srgb(
        helmholtz_color_law(color_system, parameters).linear_rgb
    )


def hue_spectrum(hue_degrees: float) -> np.ndarray:
    """Compatibility name for the original fully saturated hue spectrum."""

    return color_spectrum(hue_degrees)


def remap_ring_colors(
    logical: Sequence[tuple[int, int, int]], offset: int, direction: int,
) -> list[tuple[int, int, int]]:
    """Place logical index i at physical site direction*i+offset."""

    count = len(logical)
    mapped = [(0, 0, 0)] * count
    for index, color in enumerate(logical):
        mapped[(direction * index + offset) % count] = color
    return mapped


def layout_site_indices(count: int, offset: int, direction: int) -> np.ndarray:
    """Return the real-world ring site occupied by each controller address."""

    return np.asarray(
        [(direction * address_index + offset) % count for address_index in range(count)],
        dtype=np.int64,
    )


def address_directions(array) -> np.ndarray:
    """Current real-world direction of every address, including live dragging."""

    directions = RADIAL_DIRECTIONS[
        layout_site_indices(len(RADIAL_DIRECTIONS), array.identity_offset, array.identity_direction)
    ]
    angle = array.ring_drag_delta if array.dragging_ring else 0.0
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return apply_array_placement(directions @ rotation.T, array)


def remap_full_frame(domain, full_frame, mapped):
    result = list(full_frame)
    for address, color in zip(domain.addresses, mapped):
        result[address.frame_index] = color
    return result


@dataclass
class StudioState:
    engine: str = "physics"
    falloff_power: float = 2.0
    center_contribution: float = 0.125
    source_rotation_degrees: float = 0.0
    source_rotation_rate_degrees: float = 0.0
    source_rotation_rate_multiplier: float = 1.0
    timing_rate_hz: float = 1.0
    timing_phase_degrees: float = 0.0
    timing_predelay_ns: int = 0
    timing_pulse_width_ns: int = 100_000_000
    hues: list[float] = field(default_factory=lambda: [180.0, 300.0, 60.0])
    color_systems: list[str] = field(default_factory=lambda: ["hsl"] * 3)
    color_temperatures_k: list[float] = field(default_factory=lambda: [5600.0] * 3)
    color_white_points: list[str] = field(default_factory=lambda: ["D65"] * 3)
    saturations: list[float] = field(default_factory=lambda: [1.0] * 3)
    lightnesses: list[float] = field(default_factory=lambda: [0.5] * 3)
    hue_rates: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hue_rate_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    strobe_waveforms: list[str] = field(default_factory=lambda: ["off"] * 3)
    strobe_rates_hz: list[float] = field(default_factory=lambda: [0.0] * 3)
    strobe_phases: list[float] = field(default_factory=lambda: [0.0] * 3)
    beat_polarities: list[float] = field(default_factory=lambda: [1.0] * 3)
    audio_bands: list[int] = field(default_factory=lambda: [0] * 3)
    audio_windows_ms: list[float] = field(default_factory=lambda: [120.0] * 3)
    masses: list[float] = field(default_factory=lambda: [3.0, 4.0, 5.0])
    physics_geometry: str = "torus"
    physics_normalize_source_count: bool = False
    projection_inner_radius: float = 0.41
    projection_outer_radius: float = 0.82
    light_ring_radius: float = 1.0
    temperature_c: float = 40.0
    thermal_coupling: float = 1.0
    spiro_hues: list[float] = field(default_factory=lambda: [180.0, 300.0, 60.0])
    spiro_color_systems: list[str] = field(default_factory=lambda: ["hsl"] * 3)
    spiro_color_temperatures_k: list[float] = field(default_factory=lambda: [5600.0] * 3)
    spiro_color_white_points: list[str] = field(default_factory=lambda: ["D65"] * 3)
    spiro_saturations: list[float] = field(default_factory=lambda: [1.0] * 3)
    spiro_lightnesses: list[float] = field(default_factory=lambda: [0.5] * 3)
    spiro_hue_rates: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spiro_hue_rate_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    spiro_strobe_waveforms: list[str] = field(default_factory=lambda: ["off"] * 3)
    spiro_strobe_rates_hz: list[float] = field(default_factory=lambda: [0.0] * 3)
    spiro_strobe_phases: list[float] = field(default_factory=lambda: [0.0] * 3)
    spiro_beat_polarities: list[float] = field(default_factory=lambda: [1.0] * 3)
    spiro_audio_bands: list[int] = field(default_factory=lambda: [0] * 3)
    spiro_audio_windows_ms: list[float] = field(default_factory=lambda: [120.0] * 3)
    body_count: int = 3
    spiro_count: int = 3
    selected_spiro: int = 0
    spiro_speed: float = 1.0
    spiro_speed_multiplier: float = 1.0
    spiro_normalize_source_count: bool = False
    spiro_ratio_numerators: list[int] = field(default_factory=lambda: [2, 3, 4])
    spiro_ratio_denominators: list[int] = field(default_factory=lambda: [3, 4, 5])
    spiro_angular_offsets: list[float] = field(default_factory=lambda: [0.0, 120.0, 240.0])
    spiro_angular_rates: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spiro_angular_rate_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    manual_positions: list[list[float]] = field(
        default_factory=lambda: [[0.55, 0.0], [-0.3, 0.52], [-0.3, -0.52]]
    )
    manual_hues: list[float] = field(default_factory=lambda: [0.0, 120.0, 240.0])
    manual_color_systems: list[str] = field(default_factory=lambda: ["hsl"] * 3)
    manual_color_temperatures_k: list[float] = field(default_factory=lambda: [5600.0] * 3)
    manual_color_white_points: list[str] = field(default_factory=lambda: ["D65"] * 3)
    manual_saturations: list[float] = field(default_factory=lambda: [1.0] * 3)
    manual_lightnesses: list[float] = field(default_factory=lambda: [0.5] * 3)
    manual_hue_rates: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    manual_hue_rate_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    manual_strobe_waveforms: list[str] = field(default_factory=lambda: ["off"] * 3)
    manual_strobe_rates_hz: list[float] = field(default_factory=lambda: [0.0] * 3)
    manual_strobe_phases: list[float] = field(default_factory=lambda: [0.0] * 3)
    manual_beat_polarities: list[float] = field(default_factory=lambda: [1.0] * 3)
    manual_audio_bands: list[int] = field(default_factory=lambda: [0] * 3)
    manual_audio_windows_ms: list[float] = field(default_factory=lambda: [120.0] * 3)
    selected_manual: int = 0
    manual_deployment_count: int = 3
    manual_normalize_source_count: bool = False
    vectorscope_hue: float = 180.0
    vectorscope_color_system: str = "hsl"
    vectorscope_color_temperature_k: float = 5600.0
    vectorscope_color_white_point: str = "D65"
    vectorscope_saturation: float = 1.0
    vectorscope_lightness: float = 0.5
    vectorscope_hue_rate: float = 0.0
    vectorscope_hue_rate_multiplier: float = 1.0
    vectorscope_strobe_waveform: str = "off"
    vectorscope_strobe_rate_hz: float = 0.0
    vectorscope_strobe_phase: float = 0.0
    vectorscope_beat_polarity: float = 1.0
    vectorscope_audio_band: int = 0
    vectorscope_audio_window_ms: float = 120.0
    vectorscope_gain: float = 6.0
    vectorscope_level: float = 4.0
    vectorscope_normalize_band_count: bool = True
    vectorscope_tail_milliseconds: float = 600.0
    vectorscope_trace_points: int = 128
    vectorscope_fft_size: int = 4096
    vectorscope_oversample: int = 8
    vectorscope_min_frequency_hz: float = 27.5
    vectorscope_max_frequency_hz: float = 18000.0
    uniform_hue: float = 0.0
    uniform_color_system: str = "hsl"
    uniform_color_temperature_k: float = 5600.0
    uniform_color_white_point: str = "D65"
    uniform_hardware_mode: int = int(ASRockEffectMode.STATIC)
    uniform_hardware_speed: int = 0xE0
    uniform_saturation: float = 1.0
    uniform_lightness: float = 0.5
    uniform_hue_rate: float = 0.0
    uniform_hue_rate_multiplier: float = 1.0
    uniform_strobe_waveform: str = "off"
    uniform_strobe_rate_hz: float = 0.0
    uniform_strobe_phase: float = 0.0
    uniform_beat_polarity: float = 1.0
    uniform_audio_band: int = 0
    uniform_audio_window_ms: float = 120.0
    site_sources: list[dict] = field(default_factory=list)
    selected_site_source: int = 0
    refresh_rate: float = 60.0
    paused: bool = False
    active_slider: str | None = None


@dataclass
class ArrayRuntime:
    array_id: str
    label: str
    domain: object
    geometry_kind: str
    identity_offset: int
    identity_direction: int
    channel_order: str
    field_id: str
    transport_zone_type: int
    transport_rg_swap: bool
    placement_x: float
    placement_y: float
    placement_rotation_degrees: float
    placement_rotation_phase_degrees: float
    placement_rotation_rate_degrees: float
    placement_rotation_rate_multiplier: float
    placement_scale: float
    site_rotation_steps: int = 0
    site_rotation_phase_degrees: float = 0.0
    dragging_ring: bool = False
    ring_drag_start: float = 0.0
    ring_drag_delta: float = 0.0
    ring_drag_base_offset: int = 0
    light_tail_rgb: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )


@dataclass
class BeatOnsetTracker:
    previous_amplitude: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    flux_floor: float = 0.0
    flux_peak: float = 0.0
    envelope: float = 0.0

    def update(self, band_power, frame_seconds: float) -> float:
        """Convert positive spectral flux into an adaptive unit onset envelope."""

        power = np.asarray(band_power, dtype=np.float64)
        amplitude = np.power(10.0, 4.0 * np.clip(power, 0.0, 1.0) - 4.0)
        if self.previous_amplitude.shape != amplitude.shape:
            self.previous_amplitude = amplitude.copy()
            self.flux_floor = 0.0
            self.flux_peak = 0.0
            self.envelope = 0.0
            return 0.0
        positive_delta = np.maximum(0.0, amplitude - self.previous_amplitude)
        self.previous_amplitude = amplitude.copy()
        flux = float(np.sqrt(np.mean(positive_delta * positive_delta))) if len(power) else 0.0
        floor_alpha = 1.0 - math.exp(-frame_seconds / 1.5)
        self.flux_floor += floor_alpha * (flux - self.flux_floor)
        onset = max(0.0, flux - self.flux_floor)
        self.flux_peak = max(onset, self.flux_peak * math.exp(-frame_seconds / 0.75))
        instantaneous = onset / self.flux_peak if self.flux_peak > 1.0e-12 else 0.0
        self.envelope = max(
            min(1.0, instantaneous), self.envelope * math.exp(-frame_seconds / 0.08),
        )
        return self.envelope


@dataclass
class FieldRuntime:
    field_id: str
    studio: StudioState
    physics: object
    physics_systems: dict
    spirographs: list
    brightness: int
    exposure: float
    latest_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )
    latest_unrotated_points: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )
    latest_strengths: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    latest_hues: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    latest_frequencies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    latest_band_power: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    latest_source_levels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    beat_tracker: BeatOnsetTracker = field(default_factory=BeatOnsetTracker)
    latest_beat_strength: float = 0.0
    latest_audio_levels: dict = field(default_factory=dict)
    audio_level_windows: dict = field(default_factory=dict)
    vectorscope_history: deque = field(default_factory=deque)
    elapsed_seconds: float = 0.0
    latest_audio_rms: float = 0.0
    audio: object | None = None
    dragging_manual_index: int | None = None


def apply_array_placement(points, array: ArrayRuntime) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    angle = math.radians(array.placement_rotation_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    translated = array.placement_scale * (values @ rotation.T)
    return translated + np.asarray((array.placement_x, array.placement_y))


def rotate_source_points(points, angle_degrees: float) -> np.ndarray:
    """Rotate any engine's emitters in the common light-field coordinates."""

    values = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    return values @ rotation.T


def strobe_level(
    waveform: str, phase: float, beat_strength: float = 0.0,
    beat_polarity: float = 1.0, audio_strength: float = 0.0,
) -> float:
    """Return a unit-range source envelope for one periodic waveform."""

    fraction = (phase / (2.0 * math.pi)) % 1.0
    if waveform == "off":
        return 1.0
    if waveform == "beat":
        beat = min(1.0, max(0.0, beat_strength))
        polarity = min(1.0, max(-1.0, beat_polarity))
        return 1.0 - polarity * (1.0 - beat) if polarity >= 0.0 else 1.0 + polarity * beat
    if waveform == "audio":
        return min(1.0, max(0.0, audio_strength))
    if waveform == "timed":
        return 1.0
    if waveform == "sine":
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * fraction)
    if waveform == "square":
        return 1.0 if fraction < 0.5 else 0.0
    if waveform == "triangle":
        return 1.0 - abs(2.0 * fraction - 1.0)
    if waveform == "saw":
        return fraction
    raise ValueError(f"Unknown strobe waveform {waveform!r}")


def active_mode_uses_beat(studio: StudioState) -> bool:
    if studio.engine == "physics":
        return "beat" in studio.strobe_waveforms[:studio.body_count]
    if studio.engine == "spirograph":
        return "beat" in studio.spiro_strobe_waveforms[:studio.spiro_count]
    if studio.engine == "manual":
        return "beat" in studio.manual_strobe_waveforms
    if studio.engine == "uniform":
        return studio.uniform_strobe_waveform == "beat"
    if studio.engine == "sites":
        return any(source["strobe_waveform"] == "beat" for source in studio.site_sources)
    return studio.vectorscope_strobe_waveform == "beat"


def active_mode_uses_audio(studio: StudioState) -> bool:
    if studio.engine == "physics":
        return "audio" in studio.strobe_waveforms[:studio.body_count]
    if studio.engine == "spirograph":
        return "audio" in studio.spiro_strobe_waveforms[:studio.spiro_count]
    if studio.engine == "manual":
        return "audio" in studio.manual_strobe_waveforms
    if studio.engine == "uniform":
        return studio.uniform_strobe_waveform == "audio"
    if studio.engine == "sites":
        return any(source["strobe_waveform"] == "audio" for source in studio.site_sources)
    return studio.vectorscope_strobe_waveform == "audio"


def active_mode_uses_audio_capture(studio: StudioState) -> bool:
    return studio.engine == "vectorscope" or active_mode_uses_beat(studio) or active_mode_uses_audio(studio)


def active_mode_uses_timed(studio: StudioState) -> bool:
    if studio.engine == "physics":
        return "timed" in studio.strobe_waveforms[:studio.body_count]
    if studio.engine == "spirograph":
        return "timed" in studio.spiro_strobe_waveforms[:studio.spiro_count]
    if studio.engine == "manual":
        return "timed" in studio.manual_strobe_waveforms
    if studio.engine == "uniform":
        return studio.uniform_strobe_waveform == "timed"
    if studio.engine == "sites":
        return any(source["strobe_waveform"] == "timed" for source in studio.site_sources)
    return studio.vectorscope_strobe_waveform == "timed"


def active_mode_normalizes_source_count(studio: StudioState) -> bool:
    """Return the selected accumulated-field engine's source-count policy."""

    if studio.engine == "physics":
        return studio.physics_normalize_source_count
    if studio.engine == "spirograph":
        return studio.spiro_normalize_source_count
    if studio.engine == "manual":
        return studio.manual_normalize_source_count
    if studio.engine == "vectorscope":
        return studio.vectorscope_normalize_band_count
    return False


def toggle_active_mode_source_normalization(studio: StudioState) -> None:
    value = not active_mode_normalizes_source_count(studio)
    if studio.engine == "physics":
        studio.physics_normalize_source_count = value
    elif studio.engine == "spirograph":
        studio.spiro_normalize_source_count = value
    elif studio.engine == "manual":
        studio.manual_normalize_source_count = value
    elif studio.engine == "vectorscope":
        studio.vectorscope_normalize_band_count = value


def advance_strobe_phases(studio: StudioState, frame_seconds: float) -> None:
    for index in range(BODY_COUNT):
        studio.strobe_phases[index] = (
            studio.strobe_phases[index]
            + 2.0 * math.pi * studio.strobe_rates_hz[index] * frame_seconds
        ) % (2.0 * math.pi)
    for index in range(studio.spiro_count):
        studio.spiro_strobe_phases[index] = (
            studio.spiro_strobe_phases[index]
            + 2.0 * math.pi * studio.spiro_strobe_rates_hz[index] * frame_seconds
        ) % (2.0 * math.pi)
    for index in range(len(studio.manual_strobe_phases)):
        studio.manual_strobe_phases[index] = (
            studio.manual_strobe_phases[index]
            + 2.0 * math.pi * studio.manual_strobe_rates_hz[index] * frame_seconds
        ) % (2.0 * math.pi)
    studio.vectorscope_strobe_phase = (
        studio.vectorscope_strobe_phase
        + 2.0 * math.pi * studio.vectorscope_strobe_rate_hz * frame_seconds
    ) % (2.0 * math.pi)
    studio.uniform_strobe_phase = (
        studio.uniform_strobe_phase
        + 2.0 * math.pi * studio.uniform_strobe_rate_hz * frame_seconds
    ) % (2.0 * math.pi)
    for source in studio.site_sources:
        source["strobe_phase"] = (
            source["strobe_phase"]
            + 2.0 * math.pi * source["strobe_rate_hz"] * frame_seconds
        ) % (2.0 * math.pi)


def windowed_audio_level(
    runtime: FieldRuntime, key: str, band: int, window_ms: float,
) -> float:
    """Return a time-window mean of one fftfree logarithmic band."""

    power = runtime.latest_band_power
    value = float(power[min(max(0, int(band)), len(power) - 1)]) if len(power) else 0.0
    history = runtime.audio_level_windows.setdefault(key, deque())
    history.append((runtime.elapsed_seconds, value))
    cutoff = runtime.elapsed_seconds - max(0.0, float(window_ms)) / 1000.0
    while len(history) > 1 and history[0][0] < cutoff:
        history.popleft()
    level = sum(sample for _timestamp, sample in history) / len(history)
    runtime.latest_audio_levels[key] = level
    return level


def receiver_points(array: ArrayRuntime) -> np.ndarray:
    if array.geometry_kind == "ring":
        return address_directions(array)
    if array.geometry_kind == "vertical_strip":
        return apply_array_placement(STRIP_RECEIVER_POINTS, array)
    raise ValueError(f"Unknown receiver geometry {array.geometry_kind!r}")


def studio_state_from_config(field_config: dict, refresh_rate: float) -> StudioState:
    physics = field_config["physics"]
    projection = field_config.get("projection", physics)
    spirograph = field_config["spirograph"]
    manual = field_config.get("manual", {})
    vectorscope = field_config.get("vectorscope", {})
    uniform = field_config.get("uniform", {})
    sites = field_config.get("sites", {})
    timing = field_config.get("timing", {})
    sources = spirograph["sources"]
    manual_positions = [
        [float(position[0]), float(position[1])]
        for position in manual.get(
            "positions", [[0.55, 0.0], [-0.3, 0.52], [-0.3, -0.52]],
        )
    ]
    manual_default_hues = (
        physics["hues"][:len(manual_positions)]
        if len(physics["hues"]) >= len(manual_positions)
        else [360.0 * index / len(manual_positions) for index in range(len(manual_positions))]
    )
    return StudioState(
        engine=str(field_config["engine"]),
        falloff_power=float(field_config["falloff_power"]),
        center_contribution=float(field_config["center_contribution"]),
        source_rotation_degrees=float(field_config.get("source_rotation_degrees", 0.0)),
        source_rotation_rate_degrees=float(
            field_config.get("source_rotation_rate_degrees", 0.0)
        ),
        source_rotation_rate_multiplier=_rate_multiplier(
            field_config.get("source_rotation_rate_multiplier", 1.0)
        ),
        timing_rate_hz=float(timing.get("rate_hz", 1.0)),
        timing_phase_degrees=float(timing.get("phase_degrees", 0.0)),
        timing_predelay_ns=int(timing.get("predelay_ns", 0)),
        timing_pulse_width_ns=int(timing.get("pulse_width_ns", 100_000_000)),
        hues=[float(value) for value in physics["hues"]],
        color_systems=list(physics.get("color_systems", ["hsl"] * BODY_COUNT)),
        color_temperatures_k=[float(value) for value in physics.get(
            "color_temperatures_k", [5600.0] * BODY_COUNT,
        )],
        color_white_points=list(physics.get("color_white_points", ["D65"] * BODY_COUNT)),
        saturations=[float(value) for value in physics.get("saturations", [1.0] * BODY_COUNT)],
        lightnesses=[float(value) for value in physics.get("lightnesses", [0.5] * BODY_COUNT)],
        hue_rates=[float(value) for value in physics["hue_rates"]],
        hue_rate_multipliers=[
            _rate_multiplier(value) for value in physics.get(
                "hue_rate_multipliers", [1.0] * BODY_COUNT,
            )
        ],
        strobe_waveforms=list(physics.get("strobe_waveforms", ["off"] * BODY_COUNT)),
        strobe_rates_hz=[
            float(value) for value in physics.get("strobe_rates_hz", [0.0] * BODY_COUNT)
        ],
        beat_polarities=[
            float(value) for value in physics.get("beat_polarities", [1.0] * BODY_COUNT)
        ],
        audio_bands=[int(value) for value in physics.get("audio_bands", [0] * BODY_COUNT)],
        audio_windows_ms=[float(value) for value in physics.get("audio_windows_ms", [120.0] * BODY_COUNT)],
        masses=[float(value) for value in physics["masses"]],
        physics_geometry=str(physics.get("geometry", "torus")),
        physics_normalize_source_count=bool(physics.get("normalize_source_count", False)),
        projection_inner_radius=float(projection.get("inner_radius", physics.get("projection_inner_radius", 0.41))),
        projection_outer_radius=float(projection.get("outer_radius", physics.get("projection_outer_radius", 0.82))),
        light_ring_radius=float(projection.get("light_ring_radius", physics.get("light_ring_radius", 1.0))),
        temperature_c=float(physics.get("temperature_c", 40.0)),
        thermal_coupling=float(physics.get("thermal_coupling", 1.0)),
        spiro_hues=[float(source["hue"]) for source in sources],
        spiro_color_systems=[str(source.get("color_system", "hsl")) for source in sources],
        spiro_color_temperatures_k=[float(source.get("temperature_k", 5600.0)) for source in sources],
        spiro_color_white_points=[str(source.get("white_point", "D65")) for source in sources],
        spiro_saturations=[float(source.get("saturation", 1.0)) for source in sources],
        spiro_lightnesses=[float(source.get("lightness", 0.5)) for source in sources],
        spiro_hue_rates=[float(source["hue_rate"]) for source in sources],
        spiro_hue_rate_multipliers=[
            _rate_multiplier(source.get("hue_rate_multiplier", 1.0)) for source in sources
        ],
        spiro_strobe_waveforms=[source.get("strobe_waveform", "off") for source in sources],
        spiro_strobe_rates_hz=[float(source.get("strobe_rate_hz", 0.0)) for source in sources],
        spiro_strobe_phases=[0.0] * len(sources),
        spiro_beat_polarities=[float(source.get("beat_polarity", 1.0)) for source in sources],
        spiro_audio_bands=[int(source.get("audio_band", 0)) for source in sources],
        spiro_audio_windows_ms=[float(source.get("audio_window_ms", 120.0)) for source in sources],
        body_count=int(physics["body_count"]),
        spiro_count=len(sources),
        selected_spiro=min(int(spirograph["selected_source"]), len(sources) - 1),
        spiro_speed=float(spirograph["speed"]),
        spiro_speed_multiplier=_rate_multiplier(spirograph.get("speed_multiplier", 1.0)),
        spiro_normalize_source_count=bool(spirograph.get("normalize_source_count", False)),
        spiro_ratio_numerators=[int(source["ratio_numerator"]) for source in sources],
        spiro_ratio_denominators=[int(source["ratio_denominator"]) for source in sources],
        spiro_angular_offsets=[float(source.get("angular_offset_degrees", 0.0)) for source in sources],
        spiro_angular_rates=[float(source.get("angular_rate_degrees", 0.0)) for source in sources],
        spiro_angular_rate_multipliers=[
            _rate_multiplier(source.get("angular_rate_multiplier", 1.0)) for source in sources
        ],
        manual_positions=manual_positions,
        manual_hues=[
            float(value) for value in manual.get("hues", manual_default_hues)
        ],
        manual_color_systems=list(manual.get(
            "color_systems", ["hsl"] * len(manual_positions),
        )),
        manual_color_temperatures_k=[float(value) for value in manual.get(
            "color_temperatures_k", [5600.0] * len(manual_positions),
        )],
        manual_color_white_points=list(manual.get(
            "color_white_points", ["D65"] * len(manual_positions),
        )),
        manual_saturations=[
            float(value) for value in manual.get("saturations", [1.0] * len(manual_positions))
        ],
        manual_lightnesses=[
            float(value) for value in manual.get("lightnesses", [0.5] * len(manual_positions))
        ],
        manual_hue_rates=[
            float(value) for value in manual.get(
                "hue_rates", [0.0] * len(manual_positions)
            )
        ],
        manual_hue_rate_multipliers=[
            float(value) for value in manual.get(
                "hue_rate_multipliers", [1.0] * len(manual_positions)
            )
        ],
        manual_strobe_waveforms=list(
            manual.get("strobe_waveforms", ["off"] * len(manual_positions))
        ),
        manual_strobe_rates_hz=[
            float(value) for value in manual.get(
                "strobe_rates_hz", [0.0] * len(manual_positions)
            )
        ],
        manual_strobe_phases=[0.0] * len(manual_positions),
        manual_beat_polarities=[
            float(value) for value in manual.get(
                "beat_polarities", [1.0] * len(manual_positions)
            )
        ],
        manual_audio_bands=[
            int(value) for value in manual.get("audio_bands", [0] * len(manual_positions))
        ],
        manual_audio_windows_ms=[
            float(value) for value in manual.get("audio_windows_ms", [120.0] * len(manual_positions))
        ],
        selected_manual=min(
            int(manual.get("selected_source", 0)),
            len(manual_positions) - 1,
        ),
        manual_deployment_count=int(
            manual.get("deployment_count", len(manual_positions))
        ),
        manual_normalize_source_count=bool(manual.get("normalize_source_count", False)),
        vectorscope_hue=float(vectorscope.get("hue", 180.0)),
        vectorscope_color_system=str(vectorscope.get("color_system", "hsl")),
        vectorscope_color_temperature_k=float(vectorscope.get("temperature_k", 5600.0)),
        vectorscope_color_white_point=str(vectorscope.get("white_point", "D65")),
        vectorscope_saturation=float(vectorscope.get("saturation", 1.0)),
        vectorscope_lightness=float(vectorscope.get("lightness", 0.5)),
        vectorscope_hue_rate=float(vectorscope.get("hue_rate", 0.0)),
        vectorscope_hue_rate_multiplier=_rate_multiplier(
            vectorscope.get("hue_rate_multiplier", 1.0)
        ),
        vectorscope_strobe_waveform=str(vectorscope.get("strobe_waveform", "off")),
        vectorscope_strobe_rate_hz=float(vectorscope.get("strobe_rate_hz", 0.0)),
        vectorscope_beat_polarity=float(vectorscope.get("beat_polarity", 1.0)),
        vectorscope_audio_band=int(vectorscope.get("audio_band", 0)),
        vectorscope_audio_window_ms=float(vectorscope.get("audio_window_ms", 120.0)),
        vectorscope_gain=float(vectorscope.get("gain", 6.0)),
        vectorscope_level=float(vectorscope.get("level", 4.0)),
        vectorscope_normalize_band_count=bool(vectorscope.get("normalize_band_count", True)),
        vectorscope_tail_milliseconds=float(vectorscope.get("tail_milliseconds", 600.0)),
        vectorscope_trace_points=int(vectorscope.get("trace_points", 128)),
        vectorscope_fft_size=int(vectorscope.get("fft_size", 4096)),
        vectorscope_oversample=int(vectorscope.get("oversample", 8)),
        vectorscope_min_frequency_hz=float(vectorscope.get("min_frequency_hz", 27.5)),
        vectorscope_max_frequency_hz=float(vectorscope.get("max_frequency_hz", 18000.0)),
        uniform_hue=float(uniform.get("hue", 0.0)),
        uniform_color_system=str(uniform.get("color_system", "hsl")),
        uniform_color_temperature_k=float(uniform.get("temperature_k", 5600.0)),
        uniform_color_white_point=str(uniform.get("white_point", "D65")),
        uniform_hardware_mode=int(uniform.get("hardware_mode", ASRockEffectMode.STATIC)),
        uniform_hardware_speed=int(uniform.get("hardware_speed", 0xE0)),
        uniform_saturation=float(uniform.get("saturation", 1.0)),
        uniform_lightness=float(uniform.get("lightness", 0.5)),
        uniform_hue_rate=float(uniform.get("hue_rate", 0.0)),
        uniform_hue_rate_multiplier=_rate_multiplier(uniform.get("hue_rate_multiplier", 1.0)),
        uniform_strobe_waveform=str(uniform.get("strobe_waveform", "off")),
        uniform_strobe_rate_hz=float(uniform.get("strobe_rate_hz", 0.0)),
        uniform_beat_polarity=float(uniform.get("beat_polarity", 1.0)),
        uniform_audio_band=int(uniform.get("audio_band", 0)),
        uniform_audio_window_ms=float(uniform.get("audio_window_ms", 120.0)),
        site_sources=[dict(source, strobe_phase=0.0) for source in sites.get("sources", [])],
        selected_site_source=min(
            int(sites.get("selected_source", 0)), max(0, len(sites.get("sources", [])) - 1),
        ),
        refresh_rate=refresh_rate,
    )


def update_config_from_runtimes(config, arrays, fields, active_array_id, refresh_rate) -> None:
    config["refresh_rate"] = float(refresh_rate)
    config.setdefault("ui", {})["active_array"] = active_array_id
    for array_id, runtime in arrays.items():
        target = config["arrays"][array_id]
        target["channel_order"] = runtime.channel_order
        target["field_id"] = runtime.field_id
        geometry = target["geometry"]
        if runtime.geometry_kind == "ring":
            geometry["identity_offset"] = runtime.identity_offset
            geometry["identity_direction"] = runtime.identity_direction
            geometry["site_rotation_steps"] = runtime.site_rotation_steps
        target["placement"] = {
            "x": runtime.placement_x,
            "y": runtime.placement_y,
            "rotation_degrees": runtime.placement_rotation_degrees,
            "rotation_rate_degrees": runtime.placement_rotation_rate_degrees,
            "rotation_rate_multiplier": runtime.placement_rotation_rate_multiplier,
            "scale": runtime.placement_scale,
        }
    for field_id, runtime in fields.items():
        studio = runtime.studio
        target = config["fields"][field_id]
        target.update({
            "engine": studio.engine,
            "falloff_power": studio.falloff_power,
            "center_contribution": studio.center_contribution,
            "source_rotation_degrees": studio.source_rotation_degrees,
            "source_rotation_rate_degrees": studio.source_rotation_rate_degrees,
            "source_rotation_rate_multiplier": studio.source_rotation_rate_multiplier,
            "brightness": runtime.brightness,
            "exposure": runtime.exposure,
        })
        target["projection"] = {
            "inner_radius": studio.projection_inner_radius,
            "outer_radius": studio.projection_outer_radius,
            "light_ring_radius": studio.light_ring_radius,
        }
        target["timing"] = {
            "rate_hz": studio.timing_rate_hz,
            "phase_degrees": studio.timing_phase_degrees,
            "predelay_ns": studio.timing_predelay_ns,
            "pulse_width_ns": studio.timing_pulse_width_ns,
        }
        target["physics"] = {
            "geometry": studio.physics_geometry,
            "normalize_source_count": studio.physics_normalize_source_count,
            "body_count": studio.body_count,
            "hues": list(studio.hues),
            "color_systems": list(studio.color_systems),
            "color_temperatures_k": list(studio.color_temperatures_k),
            "color_white_points": list(studio.color_white_points),
            "saturations": list(studio.saturations),
            "lightnesses": list(studio.lightnesses),
            "hue_rates": list(studio.hue_rates),
            "hue_rate_multipliers": list(studio.hue_rate_multipliers),
            "strobe_waveforms": list(studio.strobe_waveforms),
            "strobe_rates_hz": list(studio.strobe_rates_hz),
            "beat_polarities": list(studio.beat_polarities),
            "audio_bands": list(studio.audio_bands),
            "audio_windows_ms": list(studio.audio_windows_ms),
            "masses": list(studio.masses),
            "temperature_c": studio.temperature_c,
            "thermal_coupling": studio.thermal_coupling,
        }
        target["spirograph"] = {
            "speed": studio.spiro_speed,
            "speed_multiplier": studio.spiro_speed_multiplier,
            "normalize_source_count": studio.spiro_normalize_source_count,
            "selected_source": studio.selected_spiro,
            "sources": [
                {
                    "hue": studio.spiro_hues[index],
                    "color_system": studio.spiro_color_systems[index],
                    "temperature_k": studio.spiro_color_temperatures_k[index],
                    "white_point": studio.spiro_color_white_points[index],
                    "saturation": studio.spiro_saturations[index],
                    "lightness": studio.spiro_lightnesses[index],
                    "hue_rate": studio.spiro_hue_rates[index],
                    "hue_rate_multiplier": studio.spiro_hue_rate_multipliers[index],
                    "angular_offset_degrees": studio.spiro_angular_offsets[index],
                    "angular_rate_degrees": studio.spiro_angular_rates[index],
                    "angular_rate_multiplier": studio.spiro_angular_rate_multipliers[index],
                    "strobe_waveform": studio.spiro_strobe_waveforms[index],
                    "strobe_rate_hz": studio.spiro_strobe_rates_hz[index],
                    "beat_polarity": studio.spiro_beat_polarities[index],
                    "audio_band": studio.spiro_audio_bands[index],
                    "audio_window_ms": studio.spiro_audio_windows_ms[index],
                    "ratio_numerator": studio.spiro_ratio_numerators[index],
                    "ratio_denominator": studio.spiro_ratio_denominators[index],
                }
                for index in range(studio.spiro_count)
            ],
        }
        target["manual"] = {
            "normalize_source_count": studio.manual_normalize_source_count,
            "positions": [list(position) for position in studio.manual_positions],
            "hues": list(studio.manual_hues),
            "color_systems": list(studio.manual_color_systems),
            "color_temperatures_k": list(studio.manual_color_temperatures_k),
            "color_white_points": list(studio.manual_color_white_points),
            "saturations": list(studio.manual_saturations),
            "lightnesses": list(studio.manual_lightnesses),
            "hue_rates": list(studio.manual_hue_rates),
            "hue_rate_multipliers": list(studio.manual_hue_rate_multipliers),
            "strobe_waveforms": list(studio.manual_strobe_waveforms),
            "strobe_rates_hz": list(studio.manual_strobe_rates_hz),
            "beat_polarities": list(studio.manual_beat_polarities),
            "audio_bands": list(studio.manual_audio_bands),
            "audio_windows_ms": list(studio.manual_audio_windows_ms),
            "selected_source": studio.selected_manual,
            "deployment_count": studio.manual_deployment_count,
        }
        target["vectorscope"] = {
            "hue": studio.vectorscope_hue,
            "color_system": studio.vectorscope_color_system,
            "temperature_k": studio.vectorscope_color_temperature_k,
            "white_point": studio.vectorscope_color_white_point,
            "saturation": studio.vectorscope_saturation,
            "lightness": studio.vectorscope_lightness,
            "hue_rate": studio.vectorscope_hue_rate,
            "hue_rate_multiplier": studio.vectorscope_hue_rate_multiplier,
            "strobe_waveform": studio.vectorscope_strobe_waveform,
            "strobe_rate_hz": studio.vectorscope_strobe_rate_hz,
            "beat_polarity": studio.vectorscope_beat_polarity,
            "audio_band": studio.vectorscope_audio_band,
            "audio_window_ms": studio.vectorscope_audio_window_ms,
            "gain": studio.vectorscope_gain,
            "level": studio.vectorscope_level,
            "normalize_band_count": studio.vectorscope_normalize_band_count,
            "tail_milliseconds": studio.vectorscope_tail_milliseconds,
            "trace_points": studio.vectorscope_trace_points,
            "fft_size": studio.vectorscope_fft_size,
            "oversample": studio.vectorscope_oversample,
            "min_frequency_hz": studio.vectorscope_min_frequency_hz,
            "max_frequency_hz": studio.vectorscope_max_frequency_hz,
        }
        target["uniform"] = {
            "hardware_mode": studio.uniform_hardware_mode,
            "hardware_speed": studio.uniform_hardware_speed,
            "hue": studio.uniform_hue,
            "color_system": studio.uniform_color_system,
            "temperature_k": studio.uniform_color_temperature_k,
            "white_point": studio.uniform_color_white_point,
            "saturation": studio.uniform_saturation,
            "lightness": studio.uniform_lightness,
            "hue_rate": studio.uniform_hue_rate,
            "hue_rate_multiplier": studio.uniform_hue_rate_multiplier,
            "strobe_waveform": studio.uniform_strobe_waveform,
            "strobe_rate_hz": studio.uniform_strobe_rate_hz,
            "beat_polarity": studio.uniform_beat_polarity,
            "audio_band": studio.uniform_audio_band,
            "audio_window_ms": studio.uniform_audio_window_ms,
        }
        target["sites"] = {
            "selected_source": studio.selected_site_source,
            "sources": [
                {key: value for key, value in source.items() if key != "strobe_phase"}
                for source in studio.site_sources
            ],
        }


def _set_column(state, name: str, value: float) -> None:
    getattr(state, name)[:] = float(value)


def spirograph_geometry_from_ratio(numerator: int, denominator: int) -> tuple[float, float]:
    """Map the displayed ``(R-r):r`` integer ratio to ``(R, r)``."""

    if isinstance(numerator, bool) or not isinstance(numerator, int) or numerator <= 0:
        raise ValueError("spirograph ratio numerator must be a positive integer")
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("spirograph ratio denominator must be a positive integer")
    return float(numerator + denominator), float(denominator)


def spirograph_speed_limit(studio: StudioState) -> float:
    """Temporal Nyquist bound for the eight-site ring's highest spatial mode."""

    harmonic = max(
        max(1.0, studio.spiro_ratio_numerators[index] / studio.spiro_ratio_denominators[index])
        for index in range(studio.spiro_count)
    )
    return math.pi * studio.refresh_rate / (4.0 * harmonic)


def apply_spirograph_ratio(studio: StudioState, spirograph, source: int) -> None:
    """Update geometry and its empirically verified closure as one state change."""

    fixed_radius, rolling_radius = spirograph_geometry_from_ratio(
        studio.spiro_ratio_numerators[source],
        studio.spiro_ratio_denominators[source],
    )
    state = spirograph.state
    if (
        float(state.fixed_radius[0]) == fixed_radius
        and float(state.rolling_radius[0]) == rolling_radius
    ):
        return
    repeat_period = hypotrochoid_repeat_period(
        fixed_radius=fixed_radius,
        rolling_radius=rolling_radius,
        pen_offset=float(state.pen_offset[0]),
        path_radius=float(state.path_radius[0]),
    )
    _set_column(state, "fixed_radius", fixed_radius)
    _set_column(state, "rolling_radius", rolling_radius)
    _set_column(state, "repeat_period", repeat_period)
    state.phase[:] = np.mod(state.phase, repeat_period)


def apply_studio_parameters(studio: StudioState, physics, spirographs) -> None:
    _set_column(physics.state, "center_contribution", studio.center_contribution)
    _set_column(physics.state, "falloff_power", studio.falloff_power)
    physics.set_system_temperature(studio.temperature_c)
    physics.set_thermal_coupling(studio.thermal_coupling)
    speed_limit = spirograph_speed_limit(studio)
    studio.spiro_speed = float(np.clip(studio.spiro_speed, -speed_limit, speed_limit))

    for index in range(BODY_COUNT):
        active = index < studio.body_count
        _set_column(physics.state, f"mass{index}", studio.masses[index] if active else 0.0)
        spectrum = (
            color_spectrum(
                studio.hues[index], studio.saturations[index], studio.lightnesses[index],
                color_system=studio.color_systems[index],
                temperature_k=studio.color_temperatures_k[index],
                white_point=studio.color_white_points[index],
            )
            if active else np.zeros(8)
        )
        for band, power in enumerate(spectrum):
            _set_column(physics.state, f"spectrum{index}_b{band}", float(power))
    for index, spirograph in enumerate(spirographs):
        _set_column(spirograph.state, "center_contribution", studio.center_contribution)
        _set_column(spirograph.state, "falloff_power", studio.falloff_power)
        _set_column(
            spirograph.state, "angular_speed",
            studio.spiro_speed * studio.spiro_speed_multiplier,
        )
        apply_spirograph_ratio(studio, spirograph, index)


def deploy_manual_emitters(studio: StudioState) -> None:
    """Replace the manual field with a spatial and spectral colour wheel."""

    count = max(1, min(128, int(studio.manual_deployment_count)))
    studio.manual_deployment_count = count
    if count == 1:
        studio.manual_positions = [[0.0, 0.0]]
    else:
        radius = 0.65
        studio.manual_positions = [
            [
                radius * math.cos(2.0 * math.pi * index / count),
                radius * math.sin(2.0 * math.pi * index / count),
            ]
            for index in range(count)
        ]
    studio.manual_hues = [360.0 * index / count for index in range(count)]
    studio.manual_color_systems = ["hsl"] * count
    studio.manual_color_temperatures_k = [5600.0] * count
    studio.manual_color_white_points = ["D65"] * count
    studio.manual_saturations = [1.0] * count
    studio.manual_lightnesses = [0.5] * count
    studio.manual_hue_rates = [0.0] * count
    studio.manual_hue_rate_multipliers = [1.0] * count
    studio.manual_strobe_waveforms = ["off"] * count
    studio.manual_strobe_rates_hz = [0.0] * count
    studio.manual_strobe_phases = [0.0] * count
    studio.manual_beat_polarities = [1.0] * count
    studio.manual_audio_bands = [0] * count
    studio.manual_audio_windows_ms = [120.0] * count
    studio.selected_manual = 0


def configure_direct_sites(studio: StudioState, count: int) -> None:
    """Match direct-site objects to the receiver's reported address count."""

    count = max(1, int(count))
    while len(studio.site_sources) < count:
        index = len(studio.site_sources)
        studio.site_sources.append({
            "hue": 360.0 * index / count,
            "color_system": "hsl",
            "temperature_k": 5600.0,
            "white_point": "D65",
            "saturation": 1.0,
            "lightness": 0.5,
            "hue_rate": 0.0,
            "hue_rate_multiplier": 1.0,
            "strobe_waveform": "off",
            "strobe_rate_hz": 0.0,
            "strobe_phase": 0.0,
            "beat_polarity": 1.0,
            "audio_band": 0,
            "audio_window_ms": 120.0,
        })
    del studio.site_sources[count:]
    studio.selected_site_source = min(studio.selected_site_source, count - 1)


def _projection_scale(studio: StudioState) -> float:
    return studio.projection_outer_radius / studio.light_ring_radius


def _project_planar_points(points, studio: StudioState) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * _projection_scale(studio)


def spirograph_points(spirographs, studio: StudioState) -> np.ndarray:
    points = []
    for index, spirograph in enumerate(spirographs):
        x = float(spirograph.state.x[0])
        y = float(spirograph.state.y[0])
        angle = math.radians(studio.spiro_angular_offsets[index])
        cosine, sine = math.cos(angle), math.sin(angle)
        points.append((cosine * x - sine * y, sine * x + cosine * y))
    return _project_planar_points(points, studio)


def projected_physics_points(physics, studio: StudioState) -> list[tuple[float, float]]:
    """Project intrinsic torus coordinates using the configured three-radius ratio."""

    inner = studio.projection_inner_radius
    outer = studio.projection_outer_radius
    light_ring = studio.light_ring_radius
    middle = 0.5 * (outer + inner)
    tube = 0.5 * (outer - inner)
    points = []
    for u, v, _old_x, _old_y in _body_coordinates(physics)[:studio.body_count]:
        if physics.geometry == "torus":
            radius = (middle + tube * math.cos(v)) / light_ring
            points.append((radius * math.cos(u), radius * math.sin(u)))
        elif physics.geometry == "sphere":
            scale = outer / light_ring
            points.append((scale * math.cos(v) * math.cos(u), scale * math.sin(v)))
        else:
            scale = outer / light_ring
            points.append((scale * u, scale * v))
    return points


def spectral_source_frame(
    domain,
    points,
    hues,
    receiver_points,
    studio: StudioState,
    brightness: int,
    exposure: float,
    source_strengths=None,
    saturations=None,
    lightnesses=None,
    color_systems=None,
    color_temperatures_k=None,
    color_white_points=None,
):
    """Sample one source field at an arbitrary configured receiver geometry."""

    if len(points) == 0:
        zeros = np.zeros(domain.count, dtype=np.float64)
        return domain.frame_from_channels(zeros, zeros, zeros)
    weights = np.column_stack([
        radial_contributions(
            x, y,
            center_contribution=studio.center_contribution,
            falloff_power=studio.falloff_power,
            directions=receiver_points,
        )
        for x, y in points
    ])
    if saturations is None:
        saturations = 1.0
    if lightnesses is None:
        lightnesses = 0.5
    saturations = np.broadcast_to(np.asarray(saturations, dtype=np.float64), (len(hues),))
    lightnesses = np.broadcast_to(np.asarray(lightnesses, dtype=np.float64), (len(hues),))
    if color_systems is None:
        color_systems = ["hsl"] * len(hues)
    if color_temperatures_k is None:
        color_temperatures_k = [5600.0] * len(hues)
    if color_white_points is None:
        color_white_points = ["D65"] * len(hues)
    if (
        len(hues) > 1
        and all(hue == hues[0] for hue in hues[1:])
        and np.all(saturations == saturations[0])
        and np.all(lightnesses == lightnesses[0])
        and all(system == color_systems[0] for system in color_systems[1:])
        and all(value == color_temperatures_k[0] for value in color_temperatures_k[1:])
        and all(value == color_white_points[0] for value in color_white_points[1:])
    ):
        spectra = np.repeat(
            color_spectrum(
                hues[0], saturations[0], lightnesses[0],
                color_system=color_systems[0],
                temperature_k=color_temperatures_k[0],
                white_point=color_white_points[0],
            )[np.newaxis, :],
            len(hues), axis=0,
        )
    else:
        spectra = np.vstack([
            color_spectrum(
                hue, saturation, lightness,
                color_system=system, temperature_k=temperature_k,
                white_point=white_point,
            )
            for hue, saturation, lightness, system, temperature_k, white_point in zip(
                hues, saturations, lightnesses, color_systems,
                color_temperatures_k, color_white_points,
            )
        ])
    if source_strengths is not None:
        strengths = np.asarray(source_strengths, dtype=np.float64)
        if strengths.shape != (len(spectra),):
            raise ValueError("source strengths must have one value per source")
        spectra = spectra * strengths[:, np.newaxis]
    field = accumulate_spectral_field(weights, spectra)
    linear_rgb = spectral_field_to_linear_rgb(field)
    exposed = 1.0 - np.exp(-exposure * np.maximum(0.0, linear_rgb))
    drive = brightness * linear_rgb_to_srgb(exposed)
    return domain.frame_from_channels(drive[:, 0], drive[:, 1], drive[:, 2])


def merge_domain_frames(frame_count: int, domain_frames) -> list[tuple[int, int, int]]:
    """Merge independently sampled receiver domains into one controller frame."""

    merged = [(0, 0, 0)] * frame_count
    for domain, frame in domain_frames:
        for address in domain.addresses:
            merged[address.frame_index] = frame[address.frame_index]
    return merged


def direct_color(
    hue, saturation, lightness, level, brightness,
    color_system="hsl", temperature_k=5600.0, white_point="D65",
) -> tuple[int, int, int]:
    rgb = encoded_source_color(
        hue, saturation, lightness, color_system, temperature_k, white_point,
    )
    scale = min(1.0, max(0.0, float(level))) * int(brightness)
    return tuple(min(255, max(0, round(scale * channel))) for channel in rgb)


def direct_site_frame(
    domain, hues, saturations, lightnesses, strengths, brightness,
    rotation_steps: int = 0, color_systems=None,
    color_temperatures_k=None, color_white_points=None,
):
    frame = [(0, 0, 0)] * (max(address.frame_index for address in domain.addresses) + 1)
    color_systems = color_systems or ["hsl"] * len(hues)
    color_temperatures_k = color_temperatures_k or [5600.0] * len(hues)
    color_white_points = color_white_points or ["D65"] * len(hues)
    colors = [
        direct_color(
            hue, saturation, lightness, strength, brightness,
            system, temperature_k, white_point,
        )
        for hue, saturation, lightness, strength, system, temperature_k, white_point in zip(
            hues, saturations, lightnesses, strengths, color_systems,
            color_temperatures_k, color_white_points,
        )
    ]
    count = len(domain.addresses)
    for source_index, color in enumerate(colors):
        address = domain.addresses[(source_index + rotation_steps) % count]
        frame[address.frame_index] = color
    return frame


def apply_receiver_tail(
    array: ArrayRuntime, frame, frame_seconds: float, tail_milliseconds: float,
):
    """Peak-hold and exponentially fade solved receiver RGB without renormalizing it."""

    current = np.asarray(
        [frame[address.frame_index] for address in array.domain.addresses],
        dtype=np.float64,
    )
    if array.light_tail_rgb.shape != current.shape or tail_milliseconds <= 0.0:
        array.light_tail_rgb = current.copy()
    else:
        decay = math.exp(-frame_seconds / (tail_milliseconds / 1000.0))
        array.light_tail_rgb = np.maximum(current, array.light_tail_rgb * decay)
        array.light_tail_rgb[array.light_tail_rgb < 0.5] = 0.0
    result = list(frame)
    for address, color in zip(array.domain.addresses, array.light_tail_rgb):
        result[address.frame_index] = tuple(
            min(255, max(0, round(float(channel)))) for channel in color
        )
    return result


def _slider_value(
    name: str, studio: StudioState, array: ArrayRuntime | None = None,
) -> float:
    if name == "falloff":
        return studio.falloff_power
    if name == "center":
        return studio.center_contribution
    if name == "spiro_speed":
        return studio.spiro_speed
    if name == "spiro_speed_multiplier":
        return studio.spiro_speed_multiplier
    if name == "projection_inner":
        return studio.projection_inner_radius
    if name == "projection_outer":
        return studio.projection_outer_radius
    if name == "light_ring_radius":
        return studio.light_ring_radius
    if name == "temperature_c":
        return studio.temperature_c
    if name == "thermal_coupling":
        return studio.thermal_coupling
    if name == "vectorscope_hue":
        return studio.vectorscope_hue
    if name == "vectorscope_temperature_k":
        return studio.vectorscope_color_temperature_k
    if name == "vectorscope_saturation":
        return studio.vectorscope_saturation
    if name == "vectorscope_lightness":
        return studio.vectorscope_lightness
    if name == "vectorscope_rate":
        return studio.vectorscope_hue_rate
    if name == "vectorscope_rate_multiplier":
        return studio.vectorscope_hue_rate_multiplier
    if name == "vectorscope_strobe_rate":
        return studio.vectorscope_strobe_rate_hz
    if name == "vectorscope_beat_polarity":
        return studio.vectorscope_beat_polarity
    if name == "vectorscope_audio_band":
        return float(studio.vectorscope_audio_band)
    if name == "vectorscope_audio_window_ms":
        return studio.vectorscope_audio_window_ms
    if name == "uniform_hue":
        return studio.uniform_hue
    if name == "uniform_temperature_k":
        return studio.uniform_color_temperature_k
    if name == "uniform_saturation":
        return studio.uniform_saturation
    if name == "uniform_lightness":
        return studio.uniform_lightness
    if name == "uniform_rate":
        return studio.uniform_hue_rate
    if name == "uniform_rate_multiplier":
        return studio.uniform_hue_rate_multiplier
    if name == "uniform_strobe_rate":
        return studio.uniform_strobe_rate_hz
    if name == "uniform_beat_polarity":
        return studio.uniform_beat_polarity
    if name == "uniform_audio_band":
        return float(studio.uniform_audio_band)
    if name == "uniform_audio_window_ms":
        return studio.uniform_audio_window_ms
    if name == "uniform_hardware_speed":
        return float(studio.uniform_hardware_speed)
    if name == "output_rate_hz":
        return studio.refresh_rate
    if name == "source_angle":
        return studio.source_rotation_degrees
    if name == "source_angle_rate":
        return studio.source_rotation_rate_degrees
    if name == "source_angle_rate_multiplier":
        return studio.source_rotation_rate_multiplier
    if name == "lights_angle":
        if studio.engine == "sites" and array.geometry_kind == "ring":
            return 360.0 * array.site_rotation_steps / array.domain.count
        return array.placement_rotation_degrees
    if name == "lights_angle_rate":
        return array.placement_rotation_rate_degrees
    if name == "lights_angle_rate_multiplier":
        return array.placement_rotation_rate_multiplier
    if name == "manual_deployment_count":
        return float(studio.manual_deployment_count)
    if name == "timing_rate_hz":
        return studio.timing_rate_hz
    if name == "timing_phase_degrees":
        return studio.timing_phase_degrees
    if name == "timing_predelay_us":
        return studio.timing_predelay_ns / 1_000.0
    if name == "timing_pulse_ms":
        return studio.timing_pulse_width_ns / 1_000_000.0
    if name == "vectorscope_gain":
        return studio.vectorscope_gain
    if name == "vectorscope_level":
        return studio.vectorscope_level
    if name == "vectorscope_tail_ms":
        return studio.vectorscope_tail_milliseconds
    if name == "vectorscope_points":
        return float(studio.vectorscope_trace_points)
    if name == "vectorscope_fft_power":
        return float(round(math.log2(studio.vectorscope_fft_size)))
    if name == "vectorscope_oversample_power":
        return float(round(math.log2(studio.vectorscope_oversample)))
    if name == "vectorscope_low_hz":
        return studio.vectorscope_min_frequency_hz
    if name == "vectorscope_high_hz":
        return studio.vectorscope_max_frequency_hz
    family, raw_index = name.split(":")
    index = int(raw_index)
    if family.startswith("site_"):
        return float(studio.site_sources[index][{
            "site_hue": "hue", "site_saturation": "saturation",
            "site_temperature_k": "temperature_k",
            "site_lightness": "lightness", "site_rate": "hue_rate",
            "site_rate_multiplier": "hue_rate_multiplier",
            "site_strobe_rate": "strobe_rate_hz",
            "site_beat_polarity": "beat_polarity",
            "site_audio_band": "audio_band",
            "site_audio_window_ms": "audio_window_ms",
        }[family]])
    return {
        "hue": studio.hues,
        "temperature_k": studio.color_temperatures_k,
        "saturation": studio.saturations,
        "lightness": studio.lightnesses,
        "rate": studio.hue_rates,
        "rate_multiplier": studio.hue_rate_multipliers,
        "strobe_rate": studio.strobe_rates_hz,
        "beat_polarity": studio.beat_polarities,
        "audio_band": studio.audio_bands,
        "audio_window_ms": studio.audio_windows_ms,
        "mass": studio.masses,
        "spiro_hue": studio.spiro_hues,
        "spiro_temperature_k": studio.spiro_color_temperatures_k,
        "spiro_saturation": studio.spiro_saturations,
        "spiro_lightness": studio.spiro_lightnesses,
        "spiro_rate": studio.spiro_hue_rates,
        "spiro_rate_multiplier": studio.spiro_hue_rate_multipliers,
        "spiro_strobe_rate": studio.spiro_strobe_rates_hz,
        "spiro_beat_polarity": studio.spiro_beat_polarities,
        "spiro_audio_band": studio.spiro_audio_bands,
        "spiro_audio_window_ms": studio.spiro_audio_windows_ms,
        "ratio_n": studio.spiro_ratio_numerators,
        "ratio_d": studio.spiro_ratio_denominators,
        "spiro_angle": studio.spiro_angular_offsets,
        "spiro_angle_rate": studio.spiro_angular_rates,
        "spiro_angle_rate_multiplier": studio.spiro_angular_rate_multipliers,
        "manual_hue": studio.manual_hues,
        "manual_temperature_k": studio.manual_color_temperatures_k,
        "manual_saturation": studio.manual_saturations,
        "manual_lightness": studio.manual_lightnesses,
        "manual_rate": studio.manual_hue_rates,
        "manual_rate_multiplier": studio.manual_hue_rate_multipliers,
        "manual_strobe_rate": studio.manual_strobe_rates_hz,
        "manual_beat_polarity": studio.manual_beat_polarities,
        "manual_audio_band": studio.manual_audio_bands,
        "manual_audio_window_ms": studio.manual_audio_windows_ms,
    }[family][index]


def _set_slider_value(
    name: str, studio: StudioState, value: float,
    array: ArrayRuntime | None = None,
) -> None:
    if name == "falloff":
        studio.falloff_power = value
        return
    if name == "center":
        studio.center_contribution = value
        return
    if name == "spiro_speed":
        studio.spiro_speed = value
        return
    if name == "spiro_speed_multiplier":
        studio.spiro_speed_multiplier = _rate_multiplier(value)
        return
    if name == "projection_inner":
        studio.projection_inner_radius = min(value, studio.projection_outer_radius - 0.001)
        return
    if name == "projection_outer":
        studio.projection_outer_radius = max(value, studio.projection_inner_radius + 0.001)
        return
    if name == "light_ring_radius":
        studio.light_ring_radius = max(0.001, value)
        return
    if name == "temperature_c":
        studio.temperature_c = value
        return
    if name == "thermal_coupling":
        studio.thermal_coupling = value
        return
    if name == "vectorscope_hue":
        studio.vectorscope_hue = value
        return
    if name == "vectorscope_temperature_k":
        studio.vectorscope_color_temperature_k = value
        return
    if name == "vectorscope_saturation":
        studio.vectorscope_saturation = value
        return
    if name == "vectorscope_lightness":
        studio.vectorscope_lightness = value
        return
    if name == "vectorscope_rate":
        studio.vectorscope_hue_rate = value
        return
    if name == "vectorscope_rate_multiplier":
        studio.vectorscope_hue_rate_multiplier = _rate_multiplier(value)
        return
    if name == "vectorscope_strobe_rate":
        studio.vectorscope_strobe_rate_hz = value
        return
    if name == "vectorscope_beat_polarity":
        studio.vectorscope_beat_polarity = value
        return
    if name == "vectorscope_audio_band":
        studio.vectorscope_audio_band = max(0, round(value))
        return
    if name == "vectorscope_audio_window_ms":
        studio.vectorscope_audio_window_ms = value
        return
    if name == "uniform_hue":
        studio.uniform_hue = value
        return
    if name == "uniform_temperature_k":
        studio.uniform_color_temperature_k = value
        return
    if name == "uniform_saturation":
        studio.uniform_saturation = value
        return
    if name == "uniform_lightness":
        studio.uniform_lightness = value
        return
    if name == "uniform_rate":
        studio.uniform_hue_rate = value
        return
    if name == "uniform_rate_multiplier":
        studio.uniform_hue_rate_multiplier = _rate_multiplier(value)
        return
    if name == "uniform_strobe_rate":
        studio.uniform_strobe_rate_hz = value
        return
    if name == "uniform_beat_polarity":
        studio.uniform_beat_polarity = value
        return
    if name == "uniform_audio_band":
        studio.uniform_audio_band = max(0, round(value))
        return
    if name == "uniform_audio_window_ms":
        studio.uniform_audio_window_ms = value
        return
    if name == "uniform_hardware_speed":
        studio.uniform_hardware_speed = max(0, min(255, round(value)))
        return
    if name == "output_rate_hz":
        studio.refresh_rate = max(0.001, value)
        return
    if name == "source_angle":
        studio.source_rotation_degrees = value % 360.0
        return
    if name == "source_angle_rate":
        studio.source_rotation_rate_degrees = value
        return
    if name == "source_angle_rate_multiplier":
        studio.source_rotation_rate_multiplier = _rate_multiplier(value)
        return
    if name == "lights_angle":
        angle = value % 360.0
        if studio.engine == "sites" and array.geometry_kind == "ring":
            step = 360.0 / array.domain.count
            array.site_rotation_steps = round(angle / step) % array.domain.count
            array.site_rotation_phase_degrees = (
                360.0 * array.site_rotation_steps / array.domain.count
            )
            return
        array.placement_rotation_degrees = angle
        array.placement_rotation_phase_degrees = angle
        return
    if name == "lights_angle_rate":
        array.placement_rotation_rate_degrees = value
        return
    if name == "lights_angle_rate_multiplier":
        array.placement_rotation_rate_multiplier = _rate_multiplier(value)
        return
    if name == "manual_deployment_count":
        studio.manual_deployment_count = max(1, min(128, round(value)))
        return
    if name == "timing_rate_hz":
        studio.timing_rate_hz = value
        return
    if name == "timing_phase_degrees":
        studio.timing_phase_degrees = value % 360.0
        return
    if name == "timing_predelay_us":
        studio.timing_predelay_ns = max(0, round(value * 1_000.0))
        return
    if name == "timing_pulse_ms":
        studio.timing_pulse_width_ns = max(1, round(value * 1_000_000.0))
        return
    if name == "vectorscope_gain":
        studio.vectorscope_gain = value
        return
    if name == "vectorscope_level":
        studio.vectorscope_level = value
        return
    if name == "vectorscope_tail_ms":
        studio.vectorscope_tail_milliseconds = value
        return
    if name == "vectorscope_points":
        studio.vectorscope_trace_points = max(1, round(value))
        return
    if name == "vectorscope_fft_power":
        studio.vectorscope_fft_size = 1 << round(value)
        return
    if name == "vectorscope_oversample_power":
        studio.vectorscope_oversample = 1 << round(value)
        return
    if name == "vectorscope_low_hz":
        studio.vectorscope_min_frequency_hz = min(
            value, studio.vectorscope_max_frequency_hz - 1.0,
        )
        return
    if name == "vectorscope_high_hz":
        studio.vectorscope_max_frequency_hz = max(
            value, studio.vectorscope_min_frequency_hz + 1.0,
        )
        return
    family, raw_index = name.split(":")
    index = int(raw_index)
    if family.startswith("site_"):
        key = {
            "site_hue": "hue", "site_saturation": "saturation",
            "site_temperature_k": "temperature_k",
            "site_lightness": "lightness", "site_rate": "hue_rate",
            "site_rate_multiplier": "hue_rate_multiplier",
            "site_strobe_rate": "strobe_rate_hz",
            "site_beat_polarity": "beat_polarity",
            "site_audio_band": "audio_band",
            "site_audio_window_ms": "audio_window_ms",
        }[family]
        if family == "site_rate_multiplier":
            value = _rate_multiplier(value)
        elif family == "site_audio_band":
            value = max(0, round(value))
        studio.site_sources[index][key] = value
        return
    target = {
        "hue": studio.hues,
        "temperature_k": studio.color_temperatures_k,
        "saturation": studio.saturations,
        "lightness": studio.lightnesses,
        "rate": studio.hue_rates,
        "rate_multiplier": studio.hue_rate_multipliers,
        "strobe_rate": studio.strobe_rates_hz,
        "beat_polarity": studio.beat_polarities,
        "audio_band": studio.audio_bands,
        "audio_window_ms": studio.audio_windows_ms,
        "mass": studio.masses,
        "spiro_hue": studio.spiro_hues,
        "spiro_temperature_k": studio.spiro_color_temperatures_k,
        "spiro_saturation": studio.spiro_saturations,
        "spiro_lightness": studio.spiro_lightnesses,
        "spiro_rate": studio.spiro_hue_rates,
        "spiro_rate_multiplier": studio.spiro_hue_rate_multipliers,
        "spiro_strobe_rate": studio.spiro_strobe_rates_hz,
        "spiro_beat_polarity": studio.spiro_beat_polarities,
        "spiro_audio_band": studio.spiro_audio_bands,
        "spiro_audio_window_ms": studio.spiro_audio_windows_ms,
        "ratio_n": studio.spiro_ratio_numerators,
        "ratio_d": studio.spiro_ratio_denominators,
        "spiro_angle": studio.spiro_angular_offsets,
        "spiro_angle_rate": studio.spiro_angular_rates,
        "spiro_angle_rate_multiplier": studio.spiro_angular_rate_multipliers,
        "manual_hue": studio.manual_hues,
        "manual_temperature_k": studio.manual_color_temperatures_k,
        "manual_saturation": studio.manual_saturations,
        "manual_lightness": studio.manual_lightnesses,
        "manual_rate": studio.manual_hue_rates,
        "manual_rate_multiplier": studio.manual_hue_rate_multipliers,
        "manual_strobe_rate": studio.manual_strobe_rates_hz,
        "manual_beat_polarity": studio.manual_beat_polarities,
        "manual_audio_band": studio.manual_audio_bands,
        "manual_audio_window_ms": studio.manual_audio_windows_ms,
    }[family]
    if family in {
        "rate_multiplier", "spiro_rate_multiplier",
        "spiro_angle_rate_multiplier", "manual_rate_multiplier",
    }:
        target[index] = _rate_multiplier(value)
    else:
        target[index] = (
            round(value)
            if family in {"ratio_n", "ratio_d", "audio_band", "spiro_audio_band", "manual_audio_band"}
            else value
        )


def _slider_default_value(
    name: str, studio: StudioState, array: ArrayRuntime | None = None,
) -> float:
    """Return the canonical reset value for every slider in the studio."""

    del array
    if name in {
        "source_angle", "source_angle_rate", "lights_angle", "lights_angle_rate",
    }:
        return 0.0
    if name.startswith(("spiro_angle:", "spiro_angle_rate:")):
        return 0.0

    defaults = StudioState()
    scalar_defaults = {
        "falloff": defaults.falloff_power,
        "center": defaults.center_contribution,
        "spiro_speed": defaults.spiro_speed,
        "spiro_speed_multiplier": defaults.spiro_speed_multiplier,
        "projection_inner": defaults.projection_inner_radius,
        "projection_outer": defaults.projection_outer_radius,
        "light_ring_radius": defaults.light_ring_radius,
        "temperature_c": defaults.temperature_c,
        "thermal_coupling": defaults.thermal_coupling,
        "vectorscope_hue": defaults.vectorscope_hue,
        "vectorscope_temperature_k": defaults.vectorscope_color_temperature_k,
        "vectorscope_saturation": defaults.vectorscope_saturation,
        "vectorscope_lightness": defaults.vectorscope_lightness,
        "vectorscope_rate": defaults.vectorscope_hue_rate,
        "vectorscope_rate_multiplier": defaults.vectorscope_hue_rate_multiplier,
        "vectorscope_strobe_rate": defaults.vectorscope_strobe_rate_hz,
        "vectorscope_beat_polarity": defaults.vectorscope_beat_polarity,
        "vectorscope_audio_band": defaults.vectorscope_audio_band,
        "vectorscope_audio_window_ms": defaults.vectorscope_audio_window_ms,
        "uniform_hue": defaults.uniform_hue,
        "uniform_temperature_k": defaults.uniform_color_temperature_k,
        "uniform_saturation": defaults.uniform_saturation,
        "uniform_lightness": defaults.uniform_lightness,
        "uniform_rate": defaults.uniform_hue_rate,
        "uniform_rate_multiplier": defaults.uniform_hue_rate_multiplier,
        "uniform_strobe_rate": defaults.uniform_strobe_rate_hz,
        "uniform_beat_polarity": defaults.uniform_beat_polarity,
        "uniform_audio_band": defaults.uniform_audio_band,
        "uniform_audio_window_ms": defaults.uniform_audio_window_ms,
        "uniform_hardware_speed": defaults.uniform_hardware_speed,
        "output_rate_hz": defaults.refresh_rate,
        "source_angle_rate_multiplier": defaults.source_rotation_rate_multiplier,
        "lights_angle_rate_multiplier": 1.0,
        "manual_deployment_count": float(defaults.manual_deployment_count),
        "timing_rate_hz": defaults.timing_rate_hz,
        "timing_phase_degrees": defaults.timing_phase_degrees,
        "timing_predelay_us": defaults.timing_predelay_ns / 1_000.0,
        "timing_pulse_ms": defaults.timing_pulse_width_ns / 1_000_000.0,
        "vectorscope_gain": defaults.vectorscope_gain,
        "vectorscope_level": defaults.vectorscope_level,
        "vectorscope_tail_ms": defaults.vectorscope_tail_milliseconds,
        "vectorscope_points": float(defaults.vectorscope_trace_points),
        "vectorscope_fft_power": float(round(math.log2(defaults.vectorscope_fft_size))),
        "vectorscope_oversample_power": float(round(math.log2(defaults.vectorscope_oversample))),
        "vectorscope_low_hz": defaults.vectorscope_min_frequency_hz,
        "vectorscope_high_hz": defaults.vectorscope_max_frequency_hz,
    }
    if name in scalar_defaults:
        return float(scalar_defaults[name])

    family, raw_index = name.split(":")
    index = int(raw_index)
    if family.startswith("site_"):
        return {
            "site_hue": 360.0 * index / max(1, len(studio.site_sources)),
            "site_temperature_k": 5600.0,
            "site_saturation": 1.0, "site_lightness": 0.5,
            "site_rate": 0.0, "site_rate_multiplier": 1.0,
            "site_strobe_rate": 0.0, "site_beat_polarity": 1.0,
            "site_audio_band": 0.0, "site_audio_window_ms": 120.0,
        }[family]
    if family == "manual_hue":
        return 360.0 * index / max(1, len(studio.manual_hues))
    indexed_defaults = {
        "hue": defaults.hues,
        "temperature_k": defaults.color_temperatures_k,
        "saturation": defaults.saturations,
        "lightness": defaults.lightnesses,
        "rate": defaults.hue_rates,
        "rate_multiplier": defaults.hue_rate_multipliers,
        "strobe_rate": defaults.strobe_rates_hz,
        "beat_polarity": defaults.beat_polarities,
        "audio_band": defaults.audio_bands,
        "audio_window_ms": defaults.audio_windows_ms,
        "mass": defaults.masses,
        "spiro_hue": defaults.spiro_hues,
        "spiro_temperature_k": defaults.spiro_color_temperatures_k,
        "spiro_saturation": defaults.spiro_saturations,
        "spiro_lightness": defaults.spiro_lightnesses,
        "spiro_rate": defaults.spiro_hue_rates,
        "spiro_rate_multiplier": defaults.spiro_hue_rate_multipliers,
        "spiro_strobe_rate": defaults.spiro_strobe_rates_hz,
        "spiro_beat_polarity": defaults.spiro_beat_polarities,
        "spiro_audio_band": defaults.spiro_audio_bands,
        "spiro_audio_window_ms": defaults.spiro_audio_windows_ms,
        "ratio_n": defaults.spiro_ratio_numerators,
        "ratio_d": defaults.spiro_ratio_denominators,
        "spiro_angle_rate_multiplier": defaults.spiro_angular_rate_multipliers,
        "manual_rate": defaults.manual_hue_rates,
        "manual_temperature_k": defaults.manual_color_temperatures_k,
        "manual_saturation": defaults.manual_saturations,
        "manual_lightness": defaults.manual_lightnesses,
        "manual_rate_multiplier": defaults.manual_hue_rate_multipliers,
        "manual_strobe_rate": defaults.manual_strobe_rates_hz,
        "manual_beat_polarity": defaults.manual_beat_polarities,
        "manual_audio_band": defaults.manual_audio_bands,
        "manual_audio_window_ms": defaults.manual_audio_windows_ms,
    }
    values = indexed_defaults[family]
    return float(values[index % len(values)])


def _reset_slider(widget, studio: StudioState, array: ArrayRuntime) -> None:
    _kind, name, _rect, _limits = widget
    default_value = _slider_default_value(name, studio, array)
    if name == "projection_inner" and studio.projection_outer_radius <= default_value:
        studio.projection_outer_radius = StudioState().projection_outer_radius
    elif name == "projection_outer" and studio.projection_inner_radius >= default_value:
        studio.projection_inner_radius = StudioState().projection_inner_radius
    _set_slider_value(name, studio, default_value, array)


def _button(pygame, screen, font, rect, label, active=False, enabled=True):
    color = (62, 98, 132) if active else (42, 48, 58)
    if not enabled:
        color = (29, 32, 38)
    pygame.draw.rect(screen, color, rect, border_radius=5)
    pygame.draw.rect(screen, (110, 120, 135), rect, 1, border_radius=5)
    text = font.render(label, True, (225, 230, 238) if enabled else (105, 110, 118))
    screen.blit(text, text.get_rect(center=rect.center))


def _cycle_value(current: str, choices) -> str:
    choices = tuple(choices)
    return choices[(choices.index(current) + 1) % len(choices)]


def cycle_color_system(studio: StudioState, family: str, index: int = 0) -> None:
    if family == "physics":
        studio.color_systems[index] = _cycle_value(studio.color_systems[index], COLOR_SYSTEMS)
    elif family == "spiro":
        studio.spiro_color_systems[index] = _cycle_value(
            studio.spiro_color_systems[index], COLOR_SYSTEMS,
        )
    elif family == "manual":
        studio.manual_color_systems[index] = _cycle_value(
            studio.manual_color_systems[index], COLOR_SYSTEMS,
        )
    elif family == "vectorscope":
        studio.vectorscope_color_system = _cycle_value(
            studio.vectorscope_color_system, COLOR_SYSTEMS,
        )
    elif family == "uniform":
        studio.uniform_color_system = _cycle_value(
            studio.uniform_color_system, COLOR_SYSTEMS,
        )
    elif family == "site":
        source = studio.site_sources[index]
        source["color_system"] = _cycle_value(
            source.get("color_system", "hsl"), COLOR_SYSTEMS,
        )
    else:
        raise ValueError(f"unknown colour-control family {family!r}")


def cycle_white_point(studio: StudioState, family: str, index: int = 0) -> None:
    choices = tuple(WHITE_POINTS_XY)
    if family == "physics":
        studio.color_white_points[index] = _cycle_value(studio.color_white_points[index], choices)
    elif family == "spiro":
        studio.spiro_color_white_points[index] = _cycle_value(studio.spiro_color_white_points[index], choices)
    elif family == "manual":
        studio.manual_color_white_points[index] = _cycle_value(studio.manual_color_white_points[index], choices)
    elif family == "vectorscope":
        studio.vectorscope_color_white_point = _cycle_value(studio.vectorscope_color_white_point, choices)
    elif family == "uniform":
        studio.uniform_color_white_point = _cycle_value(studio.uniform_color_white_point, choices)
    elif family == "site":
        source = studio.site_sources[index]
        source["white_point"] = _cycle_value(source.get("white_point", "D65"), choices)
    else:
        raise ValueError(f"unknown white-point family {family!r}")


def _slider(
    pygame, screen, font, x, y, width, name, label, limits, studio, widgets,
    array=None,
):
    low, high = limits
    value = _slider_value(name, studio, array)
    track = pygame.Rect(x, y + 18, width, 8)
    if name == "timing_rate_hz":
        fraction = (
            math.log10(max(low, min(high, value))) - math.log10(low)
        ) / (math.log10(high) - math.log10(low))
    else:
        fraction = min(1.0, max(0.0, (value - low) / (high - low)))
    pygame.draw.rect(screen, (35, 39, 47), track, border_radius=4)
    pygame.draw.rect(screen, (78, 147, 194), (track.x, track.y, round(width * fraction), 8), border_radius=4)
    pygame.draw.circle(screen, (225, 230, 238), (round(track.x + width * fraction), track.centery), 6)
    displayed_value = (
        str(round(value))
        if name in {"manual_deployment_count", "uniform_hardware_speed"}
        or name.endswith("audio_band")
        or ":" in name and name.split(":", 1)[0].endswith("audio_band")
        or name.startswith(("ratio_n:", "ratio_d:"))
        else f"{value:.3f}"
    )
    screen.blit(font.render(f"{label}: {displayed_value}", True, (215, 220, 230)), (x, y))
    widgets.append(("slider", name, pygame.Rect(x, y, width, 32), limits))


def _color_system_row(
    pygame, screen, font, panel_x, y, studio, widgets,
    *, family: str, index: int, system: str, temperature_name: str,
    white_point: str,
):
    system_rect = pygame.Rect(panel_x, y, 145, 31)
    _button(pygame, screen, font, system_rect, f"colour: {system}")
    widgets.append(("button", f"color_system:{family}:{index}", system_rect, True))
    if system == "blackbody":
        _slider(
            pygame, screen, font, panel_x + 162, y, 308,
            temperature_name, "blackbody K", (1000.0, 40000.0), studio, widgets,
        )
    elif system == "white":
        white_rect = pygame.Rect(panel_x + 162, y, 308, 31)
        _button(pygame, screen, font, white_rect, f"white point: {white_point}")
        widgets.append(("button", f"white_point:{family}:{index}", white_rect, True))
    return y + 40


def _draw_direct_packet_table(pygame, screen, font, engine: str, array: ArrayRuntime) -> None:
    """Show the configured Polychrome byte packing without protocol jargon ambiguity."""

    if engine == "uniform":
        packed_order = (
            "R G B"
            if (array.channel_order == "GRB") != array.transport_rg_swap
            else "G R B"
        )
        title = "UNIFORM: one SET_ZONE packet for this light array"
        rows = (
            ("1", "10", "SET_ZONE command"),
            ("3", "array", "configured physical controller selector"),
            ("4", "00..0E", "selected native hardware effect"),
            ("5..7", packed_order, "configured color-byte order"),
            ("8", "E0", "speed byte, unused by STATIC"),
            ("9", "FF", "controller constant"),
            ("16", "00", "selected array only; 01 means every controller zone"),
            ("timed", "E3/E4", "scheduled ON/OFF uses direct frame packets"),
            ("other", "00", "zero-filled 65-byte HID report"),
        )
    elif engine == "sites":
        packed_order = (
            "R G B"
            if (array.channel_order == "GRB") != array.transport_rg_swap
            else "G R B"
        )
        title = "SITES: direct per-address packed frame"
        rows = (
            ("1", "10", "SET_ZONE/direct stream command"),
            ("3", "FF", "direct frame selector"),
            ("4", "E3/E4", "first / continuation report"),
            ("7", "FF", "first-report controller constant"),
            ("9..62", "54 B", "first packed per-address RGB payload"),
            ("5..61", "57 B", "continuation payload"),
            ("64", "65", "report terminator"),
            ("order", packed_order, "3 bytes per reported light site, addr 0..N"),
        )
    else:
        return
    x, y = 24, 700
    pygame.draw.rect(screen, (18, 22, 29), pygame.Rect(x, y, 625, 240), border_radius=6)
    screen.blit(font.render(title, True, (225, 230, 238)), (x + 10, y + 8))
    for index, (offset, value, meaning) in enumerate(rows):
        row_y = y + 32 + 22 * index
        screen.blit(font.render(f"{offset:>5}  {value:<10}  {meaning}", True, (175, 195, 215)), (x + 10, row_y))


def draw_studio(
    pygame, screen, font, small, array, arrays, field_runtime, logical_colors,
    widgets, temperature_enabled, transport_metrics, fan_rpm, timing_status,
):
    studio = field_runtime.studio
    physics = field_runtime.physics
    spirographs = field_runtime.spirographs
    widgets.clear()
    screen.fill((10, 12, 16))
    center = (335, 410)
    ring = 250
    _draw_direct_packet_table(pygame, screen, small, studio.engine, array)
    outer = round(ring * studio.projection_outer_radius / studio.light_ring_radius)
    inner = (
        round(ring * studio.projection_inner_radius / studio.light_ring_radius)
        if studio.engine == "physics" and studio.physics_geometry == "torus"
        else 0
    )
    if array.geometry_kind == "ring":
        pygame.draw.circle(screen, (35, 39, 47), center, outer)
        if inner > 0:
            pygame.draw.circle(screen, (10, 12, 16), center, inner)
        pygame.draw.circle(screen, (100, 108, 122), center, outer, 2)
        if inner > 0:
            pygame.draw.circle(screen, (100, 108, 122), center, inner, 2)
        receivers = receiver_points(array)
    else:
        receivers = receiver_points(array)
        pygame.draw.line(
            screen, (100, 108, 122),
            (
                round(center[0] + ring * float(receivers[0, 0])),
                round(center[1] - ring * float(receivers[0, 1])),
            ),
            (
                round(center[0] + ring * float(receivers[-1, 0])),
                round(center[1] - ring * float(receivers[-1, 1])),
            ), 3,
        )

    for address_index, (receiver, color) in enumerate(zip(receivers, logical_colors)):
        dx, dy = float(receiver[0]), float(receiver[1])
        pixel = (round(center[0] + ring * dx), round(center[1] - ring * dy))
        pygame.draw.circle(screen, color, pixel, 17)
        pygame.draw.circle(screen, (180, 185, 195), pixel, 18, 1)
        label = small.render(
            f"addr {address_index} {color[0]}/{color[1]}/{color[2]}",
            True, (215, 220, 230),
        )
        label_point = (round(center[0] + ring * 1.14 * dx), round(center[1] - ring * 1.14 * dy))
        screen.blit(label, label.get_rect(center=label_point))

    if studio.engine in {"physics", "manual"}:
        displayed_hues = studio.hues if studio.engine == "physics" else studio.manual_hues
        displayed_saturations = (
            studio.saturations if studio.engine == "physics" else studio.manual_saturations
        )
        displayed_lightnesses = (
            studio.lightnesses if studio.engine == "physics" else studio.manual_lightnesses
        )
        displayed_systems = (
            studio.color_systems if studio.engine == "physics" else studio.manual_color_systems
        )
        displayed_temperatures = (
            studio.color_temperatures_k if studio.engine == "physics"
            else studio.manual_color_temperatures_k
        )
        displayed_whites = (
            studio.color_white_points if studio.engine == "physics"
            else studio.manual_color_white_points
        )
        body_colors = [
            encoded_source_color(h, saturation, lightness, system, temperature, white)
            for h, saturation, lightness, system, temperature, white in zip(
                displayed_hues, displayed_saturations, displayed_lightnesses,
                displayed_systems, displayed_temperatures, displayed_whites,
            )
        ]
        for index, ((x, y), rgb, level) in enumerate(zip(
            field_runtime.latest_points, body_colors, field_runtime.latest_source_levels,
        )):
            color = tuple(round(255 * channel * level) for channel in rgb)
            pixel = (round(center[0] + ring * x), round(center[1] - ring * y))
            pygame.draw.circle(screen, color, pixel, 10)
            screen.blit(font.render(str(index), True, (240, 240, 245)), (pixel[0] + 12, pixel[1] - 10))
    elif studio.engine == "spirograph":
        body_colors = [
            encoded_source_color(h, saturation, lightness, system, temperature, white)
            for h, saturation, lightness, system, temperature, white in zip(
                studio.spiro_hues, studio.spiro_saturations, studio.spiro_lightnesses,
                studio.spiro_color_systems, studio.spiro_color_temperatures_k,
                studio.spiro_color_white_points,
            )
        ]
        for index, ((x, y), rgb, level) in enumerate(zip(
            field_runtime.latest_points, body_colors, field_runtime.latest_source_levels,
        )):
            color = tuple(round(255 * channel * level) for channel in rgb)
            pixel = (round(center[0] + ring * x), round(center[1] - ring * y))
            pygame.draw.circle(screen, color, pixel, 10)
            screen.blit(font.render(str(index), True, (240, 240, 245)), (pixel[0] + 12, pixel[1] - 10))
    elif studio.engine == "vectorscope":
        history = list(field_runtime.vectorscope_history)
        stride = max(1, math.ceil(len(history) / 64))
        sampled = history[::stride]
        if history and (not sampled or sampled[-1] is not history[-1]):
            sampled.append(history[-1])
        tail_seconds = studio.vectorscope_tail_milliseconds / 1000.0
        for older, newer in zip(sampled, sampled[1:]):
            _old_time, old_points, _old_hues, _old_power = older
            new_time, new_points, new_hues, new_power = newer
            fade = (
                math.exp(-(field_runtime.elapsed_seconds - new_time) / tail_seconds)
                if tail_seconds > 0.0 else 0.0
            )
            for old_point, new_point, hue, power in zip(
                old_points, new_points, new_hues, new_power,
            ):
                value = min(1.0, max(0.0, float(power) * fade))
                rgb = encoded_source_color(
                    hue, studio.vectorscope_saturation,
                    min(1.0, studio.vectorscope_lightness * value),
                    studio.vectorscope_color_system,
                    studio.vectorscope_color_temperature_k,
                    studio.vectorscope_color_white_point,
                )
                color = tuple(round(255 * channel) for channel in rgb)
                pygame.draw.line(
                    screen, color,
                    (round(center[0] + ring * float(old_point[0])),
                     round(center[1] - ring * float(old_point[1]))),
                    (round(center[0] + ring * float(new_point[0])),
                     round(center[1] - ring * float(new_point[1]))),
                    1,
                )
        if history:
            _time, current_points, current_hues, current_power = history[-1]
            for point, hue, power in zip(current_points, current_hues, current_power):
                rgb = encoded_source_color(
                    hue, studio.vectorscope_saturation,
                    min(1.0, studio.vectorscope_lightness * max(0.0, float(power))),
                    studio.vectorscope_color_system,
                    studio.vectorscope_color_temperature_k,
                    studio.vectorscope_color_white_point,
                )
                color = tuple(round(255 * channel) for channel in rgb)
                pygame.draw.circle(
                    screen, color,
                    (round(center[0] + ring * float(point[0])),
                     round(center[1] - ring * float(point[1]))),
                    2,
                )

    mouse = pygame.mouse.get_pos()
    if (
        array.geometry_kind == "ring"
        and 0.72 * ring
        <= math.hypot(mouse[0] - center[0], mouse[1] - center[1])
        <= 1.25 * ring
    ):
        pygame.draw.circle(screen, (95, 155, 205), center, ring, 2)

    for tab_index, (array_id, candidate) in enumerate(arrays.items()):
        rect = pygame.Rect(25 + tab_index * 190, 18, 175, 34)
        _button(
            pygame, screen, small, rect, candidate.label,
            active=array.array_id == array_id,
        )
        widgets.append(("button", f"tab:{array_id}", rect, True))

    panel_x = 690
    title = font.render("Native HID light-field studio", True, (240, 243, 248))
    screen.blit(title, (panel_x, 22))
    button_specs = [
        ("engine", f"engine: {studio.engine}", studio.engine == "vectorscope", True),
        (
            "direction",
            "order: forward" if array.identity_direction == 1 else "order: reverse",
            array.identity_direction == -1,
            array.geometry_kind == "ring",
        ),
        (
            "channel_order",
            "wire: GRB (swap R/G)" if array.channel_order == "GRB" else "wire: RGB (no swap)",
            array.channel_order == "GRB",
            True,
        ),
    ]
    if studio.engine == "physics":
        button_specs.extend((
            ("body_minus", "remove body", False, studio.body_count > 1),
            ("body_plus", "add body", False, studio.body_count < BODY_COUNT),
            ("physics_geometry", f"geometry: {studio.physics_geometry}", False, True),
        ))
    elif studio.engine == "manual":
        button_specs.append((
            "manual_deploy", "wipe + deploy colour wheel", False, True,
        ))
    elif studio.engine == "spirograph":
        button_specs.extend((
            ("body_minus", "remove source", False, studio.spiro_count > 1),
            ("body_plus", "add source", False, True),
            ("source_prev", "previous spiro source", False, studio.spiro_count > 1),
            ("source_next", "next spiro source", False, studio.spiro_count > 1),
        ))
    elif studio.engine == "sites":
        button_specs.extend((
            ("site_prev", "previous light site", False, len(studio.site_sources) > 1),
            ("site_next", "next light site", False, len(studio.site_sources) > 1),
        ))
    if studio.engine in {"physics", "spirograph", "manual", "vectorscope"}:
        normalize = active_mode_normalizes_source_count(studio)
        button_specs.append((
            "source_count_normalize",
            "source normalization: on" if normalize else "source normalization: off",
            normalize,
            True,
        ))
    for index, (action, label, active, enabled) in enumerate(button_specs):
        row, column = divmod(index, 2)
        rect = pygame.Rect(panel_x + column * 245, 58 + row * 40, 232, 31)
        _button(pygame, screen, small, rect, label, active, enabled)
        widgets.append(("button", action, rect, enabled))

    y = 65 + math.ceil(len(button_specs) / 2) * 40
    _slider(
        pygame, screen, small, panel_x, y, 470,
        "output_rate_hz", "controller output Hz", (1.0, 120.0), studio, widgets,
    )
    y += 42
    if studio.engine not in {"uniform", "sites"}:
        _slider(pygame, screen, small, panel_x, y, 470, "falloff", "distance falloff", (0.75, 8.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 470, "center", "center contribution", (0.0, 0.5), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 145, "source_angle", "sources angle", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, "source_angle_rate", "sources deg/s", (-360.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, "source_angle_rate_multiplier", "sources rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 42
    if studio.engine != "uniform":
        _slider(pygame, screen, small, panel_x, y, 145, "lights_angle", "lights angle", (0.0, 360.0), studio, widgets, array)
        _slider(pygame, screen, small, panel_x + 162, y, 145, "lights_angle_rate", "lights deg/s", (-360.0, 360.0), studio, widgets, array)
        _slider(pygame, screen, small, panel_x + 324, y, 145, "lights_angle_rate_multiplier", "lights rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets, array)
        y += 42
    if studio.engine not in {"uniform", "sites"}:
        _slider(pygame, screen, small, panel_x, y, 145, "projection_inner", "inner radius", (0.0, 4.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, "projection_outer", "outer radius", (0.01, 4.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, "light_ring_radius", "light radius", (0.05, 4.0), studio, widgets)
        y += 50
    if active_mode_uses_timed(studio):
        _slider(pygame, screen, small, panel_x, y, 220, "timing_rate_hz", "onset periodicity Hz", (0.1, 1_000_000_000.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "timing_phase_degrees", "onset phase deg", (0.0, 360.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "timing_predelay_us", "dispatch predelay us", (0.0, 250000.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "timing_pulse_ms", "requested duration ms", (1.0, 1000.0), studio, widgets)
        y += 42
    speed_limit = spirograph_speed_limit(studio)
    if studio.engine == "physics":
        if temperature_enabled:
            screen.blit(small.render(
                "temperature: processor sensor", True, (215, 220, 230)
            ), (panel_x, y))
        else:
            _slider(pygame, screen, small, panel_x, y, 220, "temperature_c", "temperature C", (0.0, 120.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "thermal_coupling", "entropy coupling /s", (0.0, 5.0), studio, widgets)
        y += 42
        for index in range(BODY_COUNT):
            active = index < studio.body_count
            screen.blit(font.render(f"body {index}{'' if active else ' (inactive)'}", True, (230, 233, 240)), (panel_x, y))
            y += 22
            y = _color_system_row(
                pygame, screen, small, panel_x, y, studio, widgets,
                family="physics", index=index, system=studio.color_systems[index],
                temperature_name=f"temperature_k:{index}",
                white_point=studio.color_white_points[index],
            )
            _slider(pygame, screen, small, panel_x, y, 145, f"hue:{index}", "hue deg", (0.0, 360.0), studio, widgets)
            _slider(pygame, screen, small, panel_x + 162, y, 145, f"rate:{index}", "hue deg/s", (-180.0, 180.0), studio, widgets)
            _slider(pygame, screen, small, panel_x + 324, y, 145, f"rate_multiplier:{index}", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
            y += 36
            _slider(pygame, screen, small, panel_x, y, 145, f"saturation:{index}", "saturation", (0.0, 1.0), studio, widgets)
            _slider(pygame, screen, small, panel_x + 162, y, 145, f"lightness:{index}", "lightness", (0.0, 1.0), studio, widgets)
            _slider(pygame, screen, small, panel_x + 324, y, 145, f"mass:{index}", "mass", (0.25, 12.0), studio, widgets)
            y += 36
            strobe_rect = pygame.Rect(panel_x, y, 145, 31)
            _button(pygame, screen, small, strobe_rect, f"strobe: {studio.strobe_waveforms[index]}")
            widgets.append(("button", f"strobe_physics:{index}", strobe_rect, True))
            if studio.strobe_waveforms[index] == "beat":
                _slider(pygame, screen, small, panel_x + 162, y, 145, f"beat_polarity:{index}", "beat +/-", (-1.0, 1.0), studio, widgets)
            elif studio.strobe_waveforms[index] == "audio":
                _slider(pygame, screen, small, panel_x + 162, y, 145, f"audio_band:{index}", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
                _slider(pygame, screen, small, panel_x + 324, y, 145, f"audio_window_ms:{index}", "average ms", (0.0, 2000.0), studio, widgets)
            elif studio.strobe_waveforms[index] != "timed":
                _slider(pygame, screen, small, panel_x + 162, y, 145, f"strobe_rate:{index}", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)
            y += 39
    elif studio.engine == "spirograph":
        _slider(
            pygame, screen, small, panel_x, y, 220, "spiro_speed",
            f"spiro angular speed (Nyquist ±{speed_limit:.2f})",
            (-speed_limit, speed_limit), studio, widgets,
        )
        _slider(
            pygame, screen, small, panel_x + 250, y, 220, "spiro_speed_multiplier",
            "spiro speed x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets,
        )
        y += 50
        index = studio.selected_spiro
        screen.blit(font.render(
            f"spiro source {index + 1} of {studio.spiro_count}", True, (230, 233, 240)
        ), (panel_x, y))
        y += 26
        y = _color_system_row(
            pygame, screen, small, panel_x, y, studio, widgets,
            family="spiro", index=index, system=studio.spiro_color_systems[index],
            temperature_name=f"spiro_temperature_k:{index}",
            white_point=studio.spiro_color_white_points[index],
        )
        _slider(pygame, screen, small, panel_x, y, 145, f"spiro_hue:{index}", "hue deg", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, f"spiro_rate:{index}", "hue deg/s", (-180.0, 180.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, f"spiro_rate_multiplier:{index}", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 40
        _slider(pygame, screen, small, panel_x, y, 220, f"spiro_saturation:{index}", "saturation", (0.0, 1.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, f"spiro_lightness:{index}", "lightness", (0.0, 1.0), studio, widgets)
        y += 40
        strobe_rect = pygame.Rect(panel_x, y, 220, 31)
        _button(pygame, screen, small, strobe_rect, f"strobe: {studio.spiro_strobe_waveforms[index]}")
        widgets.append(("button", f"strobe_spiro:{index}", strobe_rect, True))
        if studio.spiro_strobe_waveforms[index] == "beat":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"spiro_beat_polarity:{index}", "beat +/-", (-1.0, 1.0), studio, widgets)
        elif studio.spiro_strobe_waveforms[index] == "audio":
            _slider(pygame, screen, small, panel_x + 250, y, 105, f"spiro_audio_band:{index}", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
            _slider(pygame, screen, small, panel_x + 370, y, 100, f"spiro_audio_window_ms:{index}", "avg ms", (0.0, 2000.0), studio, widgets)
        elif studio.spiro_strobe_waveforms[index] != "timed":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"spiro_strobe_rate:{index}", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)
        y += 40
        _slider(pygame, screen, small, panel_x, y, 220, f"ratio_n:{index}", "ratio numerator (R-r)", (1.0, 12.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, f"ratio_d:{index}", "ratio denominator (r)", (1.0, 12.0), studio, widgets)
        y += 40
        _slider(pygame, screen, small, panel_x, y, 145, f"spiro_angle:{index}", "angular offset", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, f"spiro_angle_rate:{index}", "angular deg/s", (-720.0, 720.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, f"spiro_angle_rate_multiplier:{index}", "angular rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
    elif studio.engine == "manual":
        index = min(studio.selected_manual, len(studio.manual_positions) - 1)
        screen.blit(font.render(
            f"drag emitter {index + 1} of {len(studio.manual_positions)} directly in the field",
            True, (230, 233, 240),
        ), (panel_x, y))
        y += 30
        _slider(pygame, screen, small, panel_x, y, 220, "manual_deployment_count", "deployment count", (1.0, 128.0), studio, widgets)
        y += 40
        y = _color_system_row(
            pygame, screen, small, panel_x, y, studio, widgets,
            family="manual", index=index, system=studio.manual_color_systems[index],
            temperature_name=f"manual_temperature_k:{index}",
            white_point=studio.manual_color_white_points[index],
        )
        _slider(pygame, screen, small, panel_x, y, 145, f"manual_hue:{index}", "hue deg", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, f"manual_rate:{index}", "hue deg/s", (-180.0, 180.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, f"manual_rate_multiplier:{index}", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 40
        _slider(pygame, screen, small, panel_x, y, 220, f"manual_saturation:{index}", "saturation", (0.0, 1.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, f"manual_lightness:{index}", "lightness", (0.0, 1.0), studio, widgets)
        y += 40
        strobe_rect = pygame.Rect(panel_x, y, 220, 31)
        _button(pygame, screen, small, strobe_rect, f"strobe: {studio.manual_strobe_waveforms[index]}")
        widgets.append(("button", f"strobe_manual:{index}", strobe_rect, True))
        if studio.manual_strobe_waveforms[index] == "beat":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"manual_beat_polarity:{index}", "beat +/-", (-1.0, 1.0), studio, widgets)
        elif studio.manual_strobe_waveforms[index] == "audio":
            _slider(pygame, screen, small, panel_x + 250, y, 105, f"manual_audio_band:{index}", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
            _slider(pygame, screen, small, panel_x + 370, y, 100, f"manual_audio_window_ms:{index}", "avg ms", (0.0, 2000.0), studio, widgets)
        elif studio.manual_strobe_waveforms[index] != "timed":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"manual_strobe_rate:{index}", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)
    elif studio.engine == "vectorscope":
        screen.blit(font.render(
            "live speaker-output stereo vectorscope", True, (230, 233, 240)
        ), (panel_x, y))
        y += 28
        y = _color_system_row(
            pygame, screen, small, panel_x, y, studio, widgets,
            family="vectorscope", index=0, system=studio.vectorscope_color_system,
            temperature_name="vectorscope_temperature_k",
            white_point=studio.vectorscope_color_white_point,
        )
        _slider(pygame, screen, small, panel_x, y, 145, "vectorscope_hue", "hue deg", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, "vectorscope_rate", "hue deg/s", (-180.0, 180.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, "vectorscope_rate_multiplier", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "vectorscope_saturation", "saturation", (0.0, 1.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_lightness", "lightness", (0.0, 1.0), studio, widgets)
        y += 42
        strobe_rect = pygame.Rect(panel_x, y, 220, 31)
        _button(pygame, screen, small, strobe_rect, f"strobe: {studio.vectorscope_strobe_waveform}")
        widgets.append(("button", "strobe_vectorscope", strobe_rect, True))
        if studio.vectorscope_strobe_waveform == "beat":
            _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_beat_polarity", "beat +/-", (-1.0, 1.0), studio, widgets)
        elif studio.vectorscope_strobe_waveform == "audio":
            _slider(pygame, screen, small, panel_x + 250, y, 105, "vectorscope_audio_band", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
            _slider(pygame, screen, small, panel_x + 370, y, 100, "vectorscope_audio_window_ms", "avg ms", (0.0, 2000.0), studio, widgets)
        elif studio.vectorscope_strobe_waveform != "timed":
            _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_strobe_rate", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "vectorscope_gain", "XY gain", (0.1, 12.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_level", "field level", (0.0, 20.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "vectorscope_points", "log-band nodes", (8.0, 256.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_fft_power", "FFT power (2^n)", (9.0, 15.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "vectorscope_oversample_power", "overlap power (2^n)", (0.0, 5.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "vectorscope_low_hz", "lowest band Hz", (20.0, 500.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 470, "vectorscope_high_hz", "highest band Hz", (1000.0, 24000.0), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 470, "vectorscope_tail_ms", "tail decay ms", (0.0, 3000.0), studio, widgets)
    elif studio.engine == "uniform":
        screen.blit(font.render(
            "one object directly drives every address in this array",
            True, (230, 233, 240),
        ), (panel_x, y))
        y += 30
        hardware_mode_name = next(
            name for name, value in HARDWARE_MODES
            if int(value) == studio.uniform_hardware_mode
        )
        hardware_rect = pygame.Rect(panel_x, y, 470, 31)
        _button(
            pygame, screen, small, hardware_rect,
            f"hardware mode: {hardware_mode_name}",
        )
        widgets.append(("button", "uniform_hardware_mode", hardware_rect, True))
        y += 42
        uses_speed = studio.uniform_hardware_mode in HARDWARE_SPEED_MODES
        uses_color = studio.uniform_hardware_mode in HARDWARE_COLOR_MODES
        if studio.uniform_hardware_mode == int(ASRockEffectMode.MUSIC):
            needs = "needs: hardware audio level; standard speed is unused"
        elif uses_speed and uses_color:
            needs = "needs: colour + speed"
        elif uses_speed:
            needs = "needs: speed; hardware chooses colour"
        elif uses_color:
            needs = "needs: colour"
        else:
            needs = "needs: no adjustable effect parameter"
        screen.blit(small.render(needs, True, (175, 195, 215)), (panel_x, y))
        y += 22
        if uses_speed:
            _slider(
                pygame, screen, small, panel_x, y, 470,
                "uniform_hardware_speed", "hardware speed byte (0 fast, 255 slow)",
                (0.0, 255.0), studio, widgets,
            )
            y += 42
        y = _color_system_row(
            pygame, screen, small, panel_x, y, studio, widgets,
            family="uniform", index=0, system=studio.uniform_color_system,
            temperature_name="uniform_temperature_k",
            white_point=studio.uniform_color_white_point,
        )
        _slider(pygame, screen, small, panel_x, y, 145, "uniform_hue", "hue deg", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, "uniform_rate", "hue deg/s", (-180.0, 180.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, "uniform_rate_multiplier", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, "uniform_saturation", "saturation", (0.0, 1.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, "uniform_lightness", "lightness", (0.0, 1.0), studio, widgets)
        y += 42
        strobe_rect = pygame.Rect(panel_x, y, 220, 31)
        _button(pygame, screen, small, strobe_rect, f"strobe: {studio.uniform_strobe_waveform}")
        widgets.append(("button", "strobe_uniform", strobe_rect, True))
        if studio.uniform_strobe_waveform == "beat":
            _slider(pygame, screen, small, panel_x + 250, y, 220, "uniform_beat_polarity", "beat +/-", (-1.0, 1.0), studio, widgets)
        elif studio.uniform_strobe_waveform == "audio":
            _slider(pygame, screen, small, panel_x + 250, y, 105, "uniform_audio_band", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
            _slider(pygame, screen, small, panel_x + 370, y, 100, "uniform_audio_window_ms", "avg ms", (0.0, 2000.0), studio, widgets)
        elif studio.uniform_strobe_waveform != "timed":
            _slider(pygame, screen, small, panel_x + 250, y, 220, "uniform_strobe_rate", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)
    elif studio.engine == "sites":
        index = min(studio.selected_site_source, len(studio.site_sources) - 1)
        source = studio.site_sources[index]
        output_address = (
            index + array.identity_direction * array.site_rotation_steps
        ) % array.domain.count
        screen.blit(font.render(
            f"direct light site {index + 1} of {len(studio.site_sources)}"
            f" → address {output_address}",
            True, (230, 233, 240),
        ), (panel_x, y))
        y += 30
        y = _color_system_row(
            pygame, screen, small, panel_x, y, studio, widgets,
            family="site", index=index, system=source.get("color_system", "hsl"),
            temperature_name=f"site_temperature_k:{index}",
            white_point=source.get("white_point", "D65"),
        )
        _slider(pygame, screen, small, panel_x, y, 145, f"site_hue:{index}", "hue deg", (0.0, 360.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 162, y, 145, f"site_rate:{index}", "hue deg/s", (-180.0, 180.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 324, y, 145, f"site_rate_multiplier:{index}", "hue rate x", (1.0, ANGLE_RATE_MULTIPLIER_MAX), studio, widgets)
        y += 42
        _slider(pygame, screen, small, panel_x, y, 220, f"site_saturation:{index}", "saturation", (0.0, 1.0), studio, widgets)
        _slider(pygame, screen, small, panel_x + 250, y, 220, f"site_lightness:{index}", "lightness", (0.0, 1.0), studio, widgets)
        y += 42
        strobe_rect = pygame.Rect(panel_x, y, 220, 31)
        _button(pygame, screen, small, strobe_rect, f"strobe: {source['strobe_waveform']}")
        widgets.append(("button", f"strobe_site:{index}", strobe_rect, True))
        if source["strobe_waveform"] == "beat":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"site_beat_polarity:{index}", "beat +/-", (-1.0, 1.0), studio, widgets)
        elif source["strobe_waveform"] == "audio":
            _slider(pygame, screen, small, panel_x + 250, y, 105, f"site_audio_band:{index}", "FFT band", (0.0, max(0.0, studio.vectorscope_trace_points - 1.0)), studio, widgets)
            _slider(pygame, screen, small, panel_x + 370, y, 100, f"site_audio_window_ms:{index}", "avg ms", (0.0, 2000.0), studio, widgets)
        elif source["strobe_waveform"] != "timed":
            _slider(pygame, screen, small, panel_x + 250, y, 220, f"site_strobe_rate:{index}", "strobe Hz", (0.0, 0.5 * studio.refresh_rate), studio, widgets)

    temperature = float(physics.state.system_temperature_c[0])
    temperature_source = "processor" if temperature_enabled else "manual"
    status = (
        f"temperature {temperature:.1f} C ({temperature_source}) | "
        f"bath impulse {physics.last_thermal_impulse_rms:.4g} | "
        f"kinetic Δ {physics.last_thermal_energy_delta:+.5g}"
    )
    status_y = max(760, screen.get_height() - 100)
    screen.blit(small.render(status, True, (190, 205, 220)), (panel_x, status_y))
    if studio.engine == "vectorscope":
        if field_runtime.audio is None:
            engine_status = "native speaker/fftfree ingestion waiting"
        elif field_runtime.audio.error:
            engine_status = f"native speaker/fftfree error: {field_runtime.audio.error}"
        else:
            engine_status = (
                f"native WASAPI → fftfree {field_runtime.audio.sample_rate} Hz | "
                f"2^{round(math.log2(studio.vectorscope_fft_size))} / {studio.vectorscope_oversample}x overlap | "
                f"{len(field_runtime.latest_points)} log bands | "
                f"RMS {field_runtime.latest_audio_rms:.4f} power "
                f"{float(np.max(field_runtime.latest_band_power)) if len(field_runtime.latest_band_power) else 0.0:.3f}"
            )
    elif studio.engine == "spirograph":
        selected_period = (
            float(spirographs[studio.selected_spiro].state.repeat_period[0]) / math.pi
        )
        engine_status = f"selected spiro repeat {selected_period:.6g}π  | sources {studio.spiro_count}"
    elif studio.engine == "manual":
        engine_status = f"manual emitters {len(studio.manual_positions)} | click and drag any numbered body"
    elif studio.engine == "uniform":
        engine_status = "uniform one-object → one SET_ZONE color packet"
    elif studio.engine == "sites":
        engine_status = f"direct per-address objects {len(studio.site_sources)} | no field accumulation"
    else:
        engine_status = f"physics geometry {studio.physics_geometry} | bodies {studio.body_count}"
    if active_mode_uses_beat(studio):
        engine_status += f" | onset {field_runtime.latest_beat_strength:.3f}"
    if active_mode_uses_audio(studio) and field_runtime.latest_audio_levels:
        engine_status += " | audio " + ", ".join(
            f"{key}={value:.3f}"
            for key, value in sorted(field_runtime.latest_audio_levels.items())
        )
    if active_mode_uses_timed(studio) and timing_status:
        source = (
            f"fan {fan_rpm:.1f} RPM estimate"
            if fan_rpm is not None else f"clock {timing_status['reference_hz']:.3f} Hz"
        )
        engine_status += (
            f" | {source} phase {studio.timing_phase_degrees:.3f}° "
            f"ON {timing_status['pulse_width_ns'] / 1e6:.3f} ms "
            f"skip {timing_status['revolutions_to_skip']} rev"
        )
    screen.blit(small.render(engine_status, True, (190, 205, 220)), (panel_x, status_y + 20))
    if transport_metrics is None:
        refresh_status = f"requested {studio.refresh_rate:.3f} Hz | hardware inactive"
    else:
        refresh_status = (
            f"requested {studio.refresh_rate:.3f} Hz | HID completed "
            f"{transport_metrics['completion_hz']:.2f} Hz | "
            f"sent {transport_metrics['submitted']} done {transport_metrics['completed']} "
            f"drop {transport_metrics['dropped']} fail {transport_metrics['failed']} | "
            f"{'busy' if transport_metrics['busy'] else 'idle'}"
            f"{' + queued' if transport_metrics['pending'] else ''}"
        )
        if active_mode_uses_timed(studio):
            event_name = {
                1: "ON", 2: "OFF",
            }.get(int(transport_metrics.get("last_scheduled_event_kind", 0)), "waiting")
            refresh_status += (
                f" | timed q{transport_metrics.get('scheduled_queued', 0)} "
                f"ON {transport_metrics.get('scheduled_on_dispatched', 0)} "
                f"OFF {transport_metrics.get('scheduled_off_dispatched', 0)} "
                f"last {event_name} "
                f"dispatch Δ {transport_metrics.get('last_schedule_error_ns', 0)} ns"
            )
    screen.blit(small.render(refresh_status, True, (190, 205, 220)), (panel_x, status_y + 40))
    screen.blit(small.render(
        f"array {array.label} → field {array.field_id}  | offset {array.identity_offset}  | Space pauses",
        True, (190, 205, 220),
    ), (25, status_y + 78))
    if transport_metrics is not None:
        packet_mode = str(transport_metrics["packet_mode"])
        active_reports = int(transport_metrics["active_reports"])
        maximum_reports = int(transport_metrics["report_count"])
        acknowledgement_window = int(transport_metrics["acknowledgement_window"])
        packet_buttons = (
            ("packet_message", f"msg {transport_metrics['message_type']}", True),
            ("packet_mode", f"I/O {packet_mode}", True),
            ("packet_reports_minus", f"reports - ({active_reports})", active_reports > 1),
            ("packet_reports_plus", f"reports + ({active_reports})", active_reports < maximum_reports),
            ("packet_window_minus", f"window - ({acknowledgement_window})", packet_mode == "windowed" and acknowledgement_window > 2),
            ("packet_window_plus", f"window + ({acknowledgement_window})", packet_mode == "windowed" and acknowledgement_window < active_reports),
        )
        x = 25
        for action, label, enabled in packet_buttons:
            width = 96
            rect = pygame.Rect(x, status_y + 40, width, 29)
            _button(pygame, screen, small, rect, label, packet_mode == "burst" and action == "packet_mode", enabled)
            widgets.append(("button", action, rect, enabled))
            x += width + 7
    pygame.display.flip()
    return center, ring


def _update_slider_from_mouse(widget, mouse_x, studio, array):
    _kind, name, rect, limits = widget
    fraction = min(1.0, max(0.0, (mouse_x - rect.x) / rect.width))
    low, high = limits
    value = (
        10.0 ** (math.log10(low) + fraction * (math.log10(high) - math.log10(low)))
        if name == "timing_rate_hz"
        else low + fraction * (high - low)
    )
    _set_slider_value(name, studio, value, array)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device-name")
    parser.add_argument("--mode")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until the window closes")
    parser.add_argument("--brightness", type=int)
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--cpu-temperature", action="store_true")
    parser.add_argument("--temperature-assembly", default=str(DEFAULT_ASSEMBLY))
    parser.add_argument("--temperature-sensor", default=DEFAULT_SENSOR_IDENTIFIER)
    parser.add_argument("--fan-rpm", action="store_true")
    parser.add_argument("--fan-sensor", default="")
    parser.add_argument("--apply", action="store_true")
    return parser


def _send_all_off(controller) -> None:
    """Send one full-controller black frame through the established API."""

    cancel_scheduled = getattr(controller, "cancel_scheduled", None)
    if callable(cancel_scheduled):
        cancel_scheduled()
    controller.send_frame(controller.blank_frame())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reader = None
    fan_reader = None
    fan_rpm = None
    controller = None
    hardware_active = False
    pygame = None
    try:
        config_path = Path(args.config).resolve()
        config = load_or_create_config(config_path)
        refresh_rate = float(config["refresh_rate"] if args.fps is None else args.fps)
        if refresh_rate <= 0.0 or not math.isfinite(refresh_rate):
            raise ValueError("fps must be finite and positive")
        if args.frames < 0:
            raise ValueError("frames must be nonnegative")
        initial_temperature = 0.0
        if args.cpu_temperature:
            reader = CpuTemperatureReader(args.temperature_assembly, args.temperature_sensor)
            initial_temperature = reader.read_celsius()
        if args.fan_rpm:
            fan_reader = FanRpmReader(args.temperature_assembly, args.fan_sensor)
            fan_rpm = fan_reader.read_rpm()
        device_name = args.device_name or config["controller"]["device_name"]
        mode = args.mode or config["controller"]["mode"]
        transport_config = config["controller"].get("transport")
        controller = (
            connect_direct(config)
            if transport_config and transport_config.get("kind") == "direct_asrock_hid"
            else connect(device_name=device_name)
        )

        arrays: dict[str, ArrayRuntime] = {}
        configured_zone_types = {
            str(zone["name"]): int(zone["zone_type"])
            for zone in (transport_config or {}).get("zones", [])
        }
        configured_zone_swaps = {
            str(zone["name"]): bool(zone["rg_swap"])
            for zone in (transport_config or {}).get("zones", [])
        }
        for array_id, array_config in config["arrays"].items():
            zone = array_config["zone"]
            try:
                zone = int(zone)
            except (TypeError, ValueError):
                pass
            domain = address_domain_for(
                controller,
                zone,
                position_count=int(array_config["position_count"]),
            )
            geometry = array_config["geometry"]
            placement = array_config["placement"]
            arrays[array_id] = ArrayRuntime(
                array_id=array_id,
                label=str(array_config["label"]),
                domain=domain,
                geometry_kind=str(geometry["kind"]),
                identity_offset=int(geometry.get("identity_offset", 0)),
                identity_direction=int(geometry.get("identity_direction", 1)),
                channel_order=str(array_config["channel_order"]),
                field_id=str(array_config["field_id"]),
                transport_zone_type=configured_zone_types.get(str(array_config["zone"]), -1),
                transport_rg_swap=configured_zone_swaps.get(str(array_config["zone"]), False),
                placement_x=float(placement["x"]),
                placement_y=float(placement["y"]),
                placement_rotation_degrees=float(placement["rotation_degrees"]),
                placement_rotation_phase_degrees=float(placement["rotation_degrees"]),
                placement_rotation_rate_degrees=float(
                    placement.get("rotation_rate_degrees", 0.0)
                ),
                placement_rotation_rate_multiplier=_rate_multiplier(
                    placement.get("rotation_rate_multiplier", 1.0)
                ),
                placement_scale=float(placement["scale"]),
                site_rotation_steps=int(geometry.get("site_rotation_steps", 0))
                % domain.count,
                site_rotation_phase_degrees=(
                    360.0 * int(geometry.get("site_rotation_steps", 0)) / domain.count
                ) % 360.0,
            )
        state_domain = next(
            (array.domain for array in arrays.values() if array.domain.count == 8),
            None,
        )
        if state_domain is None:
            raise ValueError("the current managed field engines require one eight-lane state domain")

        three_body_pieces = {
            geometry: eager_three_body_pieces(geometry)
            for geometry in PHYSICS_GEOMETRIES
        }
        spiro_piece = _EagerEquationPiece(
            compile_spirograph_law(), SPIROGRAPH_LAW_NAME, 8,
        )

        def new_spirograph(field_runtime=None):
            brightness = 96 if field_runtime is None else field_runtime.brightness
            exposure = 3.0 if field_runtime is None else field_runtime.exposure
            return ManagedRadialSpirograph(
                state_domain,
                frames_per_second=refresh_rate,
                center_contribution=0.125,
                white_level=brightness,
                exposure=exposure,
                piece=spiro_piece,
            )

        fields: dict[str, FieldRuntime] = {}
        for field_id, field_config in config["fields"].items():
            studio = studio_state_from_config(field_config, refresh_rate)
            brightness = int(
                field_config["brightness"] if args.brightness is None else args.brightness
            )
            exposure = float(
                field_config["exposure"] if args.exposure is None else args.exposure
            )
            physics_systems = {
                geometry: ManagedThreeBodyField(
                    state_domain,
                    frames_per_second=refresh_rate,
                    center_contribution=studio.center_contribution,
                    falloff_power=studio.falloff_power,
                    brightness=brightness,
                    exposure=exposure,
                    system_temperature_c=(initial_temperature if reader is not None else studio.temperature_c),
                    thermal_coupling=studio.thermal_coupling,
                    geometry=geometry,
                    pieces=three_body_pieces[geometry],
                )
                for geometry in PHYSICS_GEOMETRIES
            }
            physics = physics_systems[studio.physics_geometry]
            runtime = FieldRuntime(
                field_id, studio, physics, physics_systems, [], brightness, exposure,
            )
            runtime.spirographs.extend(new_spirograph(runtime) for _ in range(studio.spiro_count))
            apply_studio_parameters(studio, physics, runtime.spirographs)
            for index, spirograph in enumerate(runtime.spirographs):
                spirograph.state.phase[:] = (
                    float(spirograph.state.repeat_period[0]) * index / len(runtime.spirographs)
                )
            fields[field_id] = runtime

        for array in arrays.values():
            configure_direct_sites(fields[array.field_id].studio, array.domain.count)

        active_array_id = str(config["ui"]["active_array"])

        def ensure_audio_capture(field_runtime: FieldRuntime) -> None:
            if field_runtime.audio is None:
                field_studio = field_runtime.studio
                field_runtime.audio = NativeAudioVectorscope(
                    fft_size=field_studio.vectorscope_fft_size,
                    oversample=field_studio.vectorscope_oversample,
                    band_count=field_studio.vectorscope_trace_points,
                    min_frequency_hz=field_studio.vectorscope_min_frequency_hz,
                    max_frequency_hz=field_studio.vectorscope_max_frequency_hz,
                )
                field_runtime.audio.start()

        def autosave() -> None:
            saved_refresh_rate = fields[arrays[active_array_id].field_id].studio.refresh_rate
            update_config_from_runtimes(
                config, arrays, fields, active_array_id, saved_refresh_rate,
            )
            config["controller"]["device_name"] = device_name
            config["controller"]["mode"] = mode
            save_config(config_path, config)

        autosave()
        for runtime in fields.values():
            if runtime.studio.engine == "vectorscope":
                ensure_audio_capture(runtime)
        if args.apply:
            controller.select_mode(mode)
            hardware_active = True

        import pygame

        pygame.init()
        screen = pygame.display.set_mode((1280, 1100), pygame.RESIZABLE)
        pygame.display.set_caption("Native HID managed light-field studio")
        font = pygame.font.Font(None, 25)
        small = pygame.font.Font(None, 18)
        widgets = []
        full_frame = controller.blank_frame()
        logical_colors = [(0, 0, 0)] * arrays[active_array_id].domain.count
        ring_center, ring_radius = (335, 410), 250
        frame_seconds = 1.0 / refresh_rate
        frame_number = 0
        next_deadline = time.perf_counter()
        timing_active = False
        timing_signature = None
        next_timing_target_ns = None
        last_timing_target_ns = None
        timing_rearm_ns = 0
        timing_status = {}
        transport_metrics = None
        native_uniform_arrays: set[str] = set()
        native_uniform_signatures: dict[
            str, tuple[int, int, int, tuple[int, int, int]]
        ] = {}
        double_click_interval_ms = int(ctypes.windll.user32.GetDoubleClickTime())
        last_slider_click_key = None
        last_slider_click_ms = -double_click_interval_ms - 1
        running = True
        while running and (args.frames == 0 or frame_number < args.frames):
            active_array = arrays[active_array_id]
            active_field = fields[active_array.field_id]
            studio = active_field.studio
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        studio.paused = not studio.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if studio.engine == "manual" and len(active_field.latest_points):
                        body_pixels = [
                            (
                                ring_center[0] + ring_radius * float(point[0]),
                                ring_center[1] - ring_radius * float(point[1]),
                            )
                            for point in active_field.latest_points
                        ]
                        hit = min(
                            range(len(body_pixels)),
                            key=lambda index: math.hypot(
                                event.pos[0] - body_pixels[index][0],
                                event.pos[1] - body_pixels[index][1],
                            ),
                            default=None,
                        )
                        if (
                            hit is not None
                            and math.hypot(
                                event.pos[0] - body_pixels[hit][0],
                                event.pos[1] - body_pixels[hit][1],
                            ) <= 22.0
                        ):
                            active_field.dragging_manual_index = hit
                            studio.selected_manual = hit
                            continue
                    radius = math.hypot(event.pos[0] - ring_center[0], event.pos[1] - ring_center[1])
                    if (
                        active_array.geometry_kind == "ring"
                        and 0.72 * ring_radius <= radius <= 1.25 * ring_radius
                    ):
                        active_array.dragging_ring = True
                        active_array.ring_drag_start = math.atan2(
                            ring_center[1] - event.pos[1], event.pos[0] - ring_center[0]
                        )
                        active_array.ring_drag_base_offset = active_array.identity_offset
                        active_array.ring_drag_delta = 0.0
                    else:
                        for widget in widgets:
                            if widget[2].collidepoint(event.pos):
                                if widget[0] == "slider":
                                    click_ms = pygame.time.get_ticks()
                                    click_key = (active_array_id, widget[1])
                                    is_double_click = (
                                        click_key == last_slider_click_key
                                        and 0 <= click_ms - last_slider_click_ms
                                        <= double_click_interval_ms
                                    )
                                    if is_double_click:
                                        _reset_slider(widget, studio, active_array)
                                        studio.active_slider = None
                                        last_slider_click_key = None
                                        last_slider_click_ms = -double_click_interval_ms - 1
                                        autosave()
                                    else:
                                        last_slider_click_key = click_key
                                        last_slider_click_ms = click_ms
                                        studio.active_slider = widget[1]
                                        _update_slider_from_mouse(
                                            widget, event.pos[0], studio, active_array,
                                        )
                                elif widget[3]:
                                    action = widget[1]
                                    if action.startswith("tab:"):
                                        active_array_id = action.split(":", 1)[1]
                                        active_array.dragging_ring = False
                                        active_array.ring_drag_delta = 0.0
                                    elif action == "engine":
                                        engines = (
                                            "physics", "spirograph", "vectorscope", "manual",
                                            "uniform", "sites",
                                        )
                                        studio.engine = engines[(engines.index(studio.engine) + 1) % len(engines)]
                                        if studio.engine == "vectorscope":
                                            ensure_audio_capture(active_field)
                                    elif action == "body_minus":
                                        if studio.engine == "physics":
                                            studio.body_count -= 1
                                        else:
                                            active_field.spirographs.pop(studio.selected_spiro)
                                            studio.spiro_hues.pop(studio.selected_spiro)
                                            studio.spiro_color_systems.pop(studio.selected_spiro)
                                            studio.spiro_color_temperatures_k.pop(studio.selected_spiro)
                                            studio.spiro_color_white_points.pop(studio.selected_spiro)
                                            studio.spiro_saturations.pop(studio.selected_spiro)
                                            studio.spiro_lightnesses.pop(studio.selected_spiro)
                                            studio.spiro_hue_rates.pop(studio.selected_spiro)
                                            studio.spiro_hue_rate_multipliers.pop(studio.selected_spiro)
                                            studio.spiro_strobe_waveforms.pop(studio.selected_spiro)
                                            studio.spiro_strobe_rates_hz.pop(studio.selected_spiro)
                                            studio.spiro_strobe_phases.pop(studio.selected_spiro)
                                            studio.spiro_beat_polarities.pop(studio.selected_spiro)
                                            studio.spiro_audio_bands.pop(studio.selected_spiro)
                                            studio.spiro_audio_windows_ms.pop(studio.selected_spiro)
                                            studio.spiro_ratio_numerators.pop(studio.selected_spiro)
                                            studio.spiro_ratio_denominators.pop(studio.selected_spiro)
                                            studio.spiro_angular_offsets.pop(studio.selected_spiro)
                                            studio.spiro_angular_rates.pop(studio.selected_spiro)
                                            studio.spiro_angular_rate_multipliers.pop(studio.selected_spiro)
                                            studio.spiro_count -= 1
                                            studio.selected_spiro = min(
                                                studio.selected_spiro, studio.spiro_count - 1
                                            )
                                    elif action == "body_plus":
                                        if studio.engine == "physics":
                                            studio.body_count += 1
                                        else:
                                            studio.spiro_count += 1
                                            studio.spiro_hues.append(
                                                (studio.spiro_hues[-1] + 137.507764) % 360.0
                                            )
                                            studio.spiro_color_systems.append("hsl")
                                            studio.spiro_color_temperatures_k.append(5600.0)
                                            studio.spiro_color_white_points.append("D65")
                                            studio.spiro_saturations.append(1.0)
                                            studio.spiro_lightnesses.append(0.5)
                                            studio.spiro_hue_rates.append(0.0)
                                            studio.spiro_hue_rate_multipliers.append(1.0)
                                            studio.spiro_strobe_waveforms.append("off")
                                            studio.spiro_strobe_rates_hz.append(0.0)
                                            studio.spiro_strobe_phases.append(0.0)
                                            studio.spiro_beat_polarities.append(1.0)
                                            studio.spiro_audio_bands.append(0)
                                            studio.spiro_audio_windows_ms.append(120.0)
                                            studio.spiro_ratio_numerators.append(2)
                                            studio.spiro_ratio_denominators.append(3)
                                            studio.spiro_angular_offsets.append(0.0)
                                            studio.spiro_angular_rates.append(0.0)
                                            studio.spiro_angular_rate_multipliers.append(1.0)
                                            active_field.spirographs.append(new_spirograph(active_field))
                                            studio.selected_spiro = studio.spiro_count - 1
                                            apply_spirograph_ratio(
                                                studio, active_field.spirographs[-1], studio.selected_spiro,
                                            )
                                    elif action == "manual_deploy":
                                        deploy_manual_emitters(studio)
                                    elif action == "physics_geometry":
                                        studio.physics_geometry = PHYSICS_GEOMETRIES[
                                            (PHYSICS_GEOMETRIES.index(studio.physics_geometry) + 1)
                                            % len(PHYSICS_GEOMETRIES)
                                        ]
                                        active_field.physics = active_field.physics_systems[
                                            studio.physics_geometry
                                        ]
                                    elif action == "direction":
                                        active_array.identity_direction *= -1
                                    elif action == "channel_order":
                                        active_array.channel_order = (
                                            "RGB" if active_array.channel_order == "GRB" else "GRB"
                                        )
                                    elif action == "source_count_normalize":
                                        toggle_active_mode_source_normalization(studio)
                                    elif action.startswith("color_system:"):
                                        _prefix, family, raw_index = action.split(":")
                                        cycle_color_system(studio, family, int(raw_index))
                                    elif action.startswith("white_point:"):
                                        _prefix, family, raw_index = action.split(":")
                                        cycle_white_point(studio, family, int(raw_index))
                                    elif action.startswith("strobe_physics:"):
                                        source = int(action.split(":", 1)[1])
                                        current = studio.strobe_waveforms[source]
                                        studio.strobe_waveforms[source] = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action.startswith("strobe_spiro:"):
                                        source = int(action.split(":", 1)[1])
                                        current = studio.spiro_strobe_waveforms[source]
                                        studio.spiro_strobe_waveforms[source] = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action.startswith("strobe_manual:"):
                                        source = int(action.split(":", 1)[1])
                                        current = studio.manual_strobe_waveforms[source]
                                        studio.manual_strobe_waveforms[source] = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action == "strobe_vectorscope":
                                        current = studio.vectorscope_strobe_waveform
                                        studio.vectorscope_strobe_waveform = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action == "strobe_uniform":
                                        current = studio.uniform_strobe_waveform
                                        studio.uniform_strobe_waveform = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action == "uniform_hardware_mode":
                                        values = [int(value) for _, value in HARDWARE_MODES]
                                        studio.uniform_hardware_mode = values[
                                            (values.index(studio.uniform_hardware_mode) + 1)
                                            % len(values)
                                        ]
                                    elif action.startswith("strobe_site:"):
                                        source = int(action.split(":", 1)[1])
                                        current = studio.site_sources[source]["strobe_waveform"]
                                        studio.site_sources[source]["strobe_waveform"] = STROBE_WAVEFORMS[
                                            (STROBE_WAVEFORMS.index(current) + 1)
                                            % len(STROBE_WAVEFORMS)
                                        ]
                                    elif action == "packet_mode":
                                        packet_modes = ("serial", "windowed", "burst")
                                        controller.configure_packets(
                                            mode=packet_modes[
                                                (packet_modes.index(controller.packet_mode) + 1)
                                                % len(packet_modes)
                                            ]
                                        )
                                    elif action == "packet_message":
                                        message_types = ("stream", "restart", "continue")
                                        controller.configure_packets(
                                            message_type=message_types[
                                                (message_types.index(controller.message_type) + 1)
                                                % len(message_types)
                                            ]
                                        )
                                    elif action == "packet_reports_minus":
                                        controller.configure_packets(
                                            report_count=controller.packet_report_count - 1,
                                        )
                                    elif action == "packet_reports_plus":
                                        controller.configure_packets(
                                            report_count=controller.packet_report_count + 1,
                                        )
                                    elif action == "packet_window_minus":
                                        controller.configure_packets(
                                            mode="windowed",
                                            window_size=controller.packet_window_size - 1,
                                        )
                                    elif action == "packet_window_plus":
                                        controller.configure_packets(
                                            mode="windowed",
                                            window_size=controller.packet_window_size + 1,
                                        )
                                    elif action == "source_prev":
                                        studio.selected_spiro = (
                                            studio.selected_spiro - 1
                                        ) % studio.spiro_count
                                    elif action == "source_next":
                                        studio.selected_spiro = (
                                            studio.selected_spiro + 1
                                        ) % studio.spiro_count
                                    elif action == "site_prev":
                                        studio.selected_site_source = (
                                            studio.selected_site_source - 1
                                        ) % len(studio.site_sources)
                                    elif action == "site_next":
                                        studio.selected_site_source = (
                                            studio.selected_site_source + 1
                                        ) % len(studio.site_sources)
                                    if action.startswith("packet_"):
                                        packet_config = config["controller"]["transport"]
                                        packet_config["packet_mode"] = controller.packet_mode
                                        packet_config["message_type"] = controller.message_type
                                        packet_config["report_count"] = controller.packet_report_count
                                        packet_config["window_size"] = controller.packet_window_size
                                    autosave()
                                break
                elif event.type == pygame.MOUSEMOTION:
                    if active_field.dragging_manual_index is not None:
                        displayed_x = (event.pos[0] - ring_center[0]) / ring_radius
                        displayed_y = (ring_center[1] - event.pos[1]) / ring_radius
                        angle = math.radians(-studio.source_rotation_degrees)
                        cosine, sine = math.cos(angle), math.sin(angle)
                        scale = _projection_scale(studio)
                        source_x = (cosine * displayed_x - sine * displayed_y) / scale
                        source_y = (sine * displayed_x + cosine * displayed_y) / scale
                        studio.manual_positions[active_field.dragging_manual_index] = [
                            source_x, source_y,
                        ]
                    elif active_array.dragging_ring:
                        angle = math.atan2(
                            ring_center[1] - event.pos[1], event.pos[0] - ring_center[0]
                        )
                        active_array.ring_drag_delta = angle - active_array.ring_drag_start
                    elif studio.active_slider is not None:
                        for widget in widgets:
                            if widget[0] == "slider" and widget[1] == studio.active_slider:
                                _update_slider_from_mouse(
                                    widget, event.pos[0], studio, active_array,
                                )
                                break
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    changed = studio.active_slider is not None
                    if active_field.dragging_manual_index is not None:
                        active_field.dragging_manual_index = None
                        changed = True
                    if active_array.dragging_ring:
                        step = 2.0 * math.pi / active_array.domain.count
                        active_array.identity_offset = (
                            active_array.ring_drag_base_offset
                            - round(active_array.ring_drag_delta / step)
                        ) % active_array.domain.count
                        active_array.dragging_ring = False
                        active_array.ring_drag_delta = 0.0
                        changed = True
                    studio.active_slider = None
                    if changed:
                        autosave()

            requested_refresh_rate = studio.refresh_rate
            if requested_refresh_rate != refresh_rate:
                refresh_rate = requested_refresh_rate
                frame_seconds = 1.0 / refresh_rate
                for field_runtime in fields.values():
                    field_runtime.studio.refresh_rate = refresh_rate
                next_deadline = time.perf_counter()

            if not running:
                break
            temperature = reader.read_celsius() if reader is not None else None
            fan_rpm = fan_reader.read_rpm() if fan_reader is not None else None
            for array_runtime in arrays.values():
                assigned_studio = fields[array_runtime.field_id].studio
                if not assigned_studio.paused:
                    if assigned_studio.engine == "uniform":
                        continue
                    if assigned_studio.engine == "sites" and array_runtime.geometry_kind == "ring":
                        array_runtime.site_rotation_phase_degrees = (
                            array_runtime.site_rotation_phase_degrees
                            + array_runtime.placement_rotation_rate_degrees
                            * array_runtime.placement_rotation_rate_multiplier
                            * frame_seconds
                        ) % 360.0
                        step = 360.0 / array_runtime.domain.count
                        array_runtime.site_rotation_steps = (
                            round(array_runtime.site_rotation_phase_degrees / step)
                            % array_runtime.domain.count
                        )
                        continue
                    array_runtime.placement_rotation_phase_degrees = (
                        array_runtime.placement_rotation_phase_degrees
                        + array_runtime.placement_rotation_rate_degrees
                        * array_runtime.placement_rotation_rate_multiplier
                        * frame_seconds
                    ) % 360.0
                    array_runtime.placement_rotation_degrees = (
                        array_runtime.placement_rotation_phase_degrees
                    )
            field_sources = {}
            for field_id, field_runtime in fields.items():
                field_studio = field_runtime.studio
                if not field_studio.paused:
                    field_runtime.elapsed_seconds += frame_seconds
                    for index in range(len(field_studio.hues)):
                        field_studio.hues[index] = (
                            field_studio.hues[index]
                            + field_studio.hue_rates[index]
                            * field_studio.hue_rate_multipliers[index]
                            * frame_seconds
                        ) % 360.0
                    for index in range(field_studio.spiro_count):
                        field_studio.spiro_hues[index] = (
                            field_studio.spiro_hues[index]
                            + field_studio.spiro_hue_rates[index]
                            * field_studio.spiro_hue_rate_multipliers[index]
                            * frame_seconds
                        ) % 360.0
                        field_studio.spiro_angular_offsets[index] = (
                            field_studio.spiro_angular_offsets[index]
                            + field_studio.spiro_angular_rates[index]
                            * field_studio.spiro_angular_rate_multipliers[index]
                            * frame_seconds
                        ) % 360.0
                    for index in range(len(field_studio.manual_hues)):
                        field_studio.manual_hues[index] = (
                            field_studio.manual_hues[index]
                            + field_studio.manual_hue_rates[index]
                            * field_studio.manual_hue_rate_multipliers[index]
                            * frame_seconds
                        ) % 360.0
                    field_studio.vectorscope_hue = (
                        field_studio.vectorscope_hue
                        + field_studio.vectorscope_hue_rate
                        * field_studio.vectorscope_hue_rate_multiplier
                        * frame_seconds
                    ) % 360.0
                    field_studio.uniform_hue = (
                        field_studio.uniform_hue
                        + field_studio.uniform_hue_rate
                        * field_studio.uniform_hue_rate_multiplier
                        * frame_seconds
                    ) % 360.0
                    for source in field_studio.site_sources:
                        source["hue"] = (
                            source["hue"] + source["hue_rate"]
                            * source["hue_rate_multiplier"] * frame_seconds
                        ) % 360.0
                    if field_studio.engine != "uniform":
                        field_studio.source_rotation_degrees = (
                            field_studio.source_rotation_degrees
                            + field_studio.source_rotation_rate_degrees
                            * field_studio.source_rotation_rate_multiplier
                            * frame_seconds
                        ) % 360.0
                    advance_strobe_phases(field_studio, frame_seconds)
                    apply_studio_parameters(
                        field_studio, field_runtime.physics, field_runtime.spirographs,
                    )
                    if temperature is not None:
                        field_runtime.physics.set_system_temperature(temperature)
                    if field_studio.engine == "physics":
                        field_runtime.physics.advance_frame()
                    elif field_studio.engine == "spirograph":
                        for spirograph in field_runtime.spirographs:
                            spirograph.advance_frame()
                    if (
                        field_studio.engine != "vectorscope"
                        and active_mode_uses_audio_capture(field_studio)
                    ):
                        ensure_audio_capture(field_runtime)
                        field_runtime.audio.configure(
                            fft_size=field_studio.vectorscope_fft_size,
                            oversample=field_studio.vectorscope_oversample,
                            band_count=field_studio.vectorscope_trace_points,
                            min_frequency_hz=field_studio.vectorscope_min_frequency_hz,
                            max_frequency_hz=field_studio.vectorscope_max_frequency_hz,
                        )
                        _beat_points, beat_power, beat_frequencies = (
                            field_runtime.audio.vectorscope_bands(gain=1.0)
                        )
                        field_runtime.latest_audio_rms = field_runtime.audio.last_rms
                        field_runtime.latest_band_power = beat_power
                        field_runtime.latest_frequencies = beat_frequencies
                        if active_mode_uses_beat(field_studio):
                            field_runtime.latest_beat_strength = field_runtime.beat_tracker.update(
                                beat_power, frame_seconds,
                            )
                if field_studio.engine == "physics":
                    points = projected_physics_points(field_runtime.physics, field_studio)
                    hues = field_studio.hues[:field_studio.body_count]
                    color_systems = field_studio.color_systems[:field_studio.body_count]
                    color_temperatures_k = field_studio.color_temperatures_k[:field_studio.body_count]
                    color_white_points = field_studio.color_white_points[:field_studio.body_count]
                    saturations = field_studio.saturations[:field_studio.body_count]
                    lightnesses = field_studio.lightnesses[:field_studio.body_count]
                    field_runtime.latest_audio_levels = {}
                    strengths = np.asarray([
                        strobe_level(
                            field_studio.strobe_waveforms[index],
                            field_studio.strobe_phases[index],
                            field_runtime.latest_beat_strength,
                            field_studio.beat_polarities[index],
                            windowed_audio_level(
                                field_runtime, f"body{index}", field_studio.audio_bands[index],
                                field_studio.audio_windows_ms[index],
                            ) if field_studio.strobe_waveforms[index] == "audio" else 0.0,
                        )
                        for index in range(field_studio.body_count)
                    ], dtype=np.float64)
                    off_strengths = strengths.copy()
                    for index in range(field_studio.body_count):
                        if field_studio.strobe_waveforms[index] == "timed":
                            off_strengths[index] = 0.0
                elif field_studio.engine == "spirograph":
                    points = spirograph_points(field_runtime.spirographs, field_studio)
                    hues = field_studio.spiro_hues
                    color_systems = field_studio.spiro_color_systems
                    color_temperatures_k = field_studio.spiro_color_temperatures_k
                    color_white_points = field_studio.spiro_color_white_points
                    saturations = field_studio.spiro_saturations
                    lightnesses = field_studio.spiro_lightnesses
                    field_runtime.latest_audio_levels = {}
                    strengths = np.asarray([
                        strobe_level(
                            field_studio.spiro_strobe_waveforms[index],
                            field_studio.spiro_strobe_phases[index],
                            field_runtime.latest_beat_strength,
                            field_studio.spiro_beat_polarities[index],
                            windowed_audio_level(
                                field_runtime, f"spiro{index}", field_studio.spiro_audio_bands[index],
                                field_studio.spiro_audio_windows_ms[index],
                            ) if field_studio.spiro_strobe_waveforms[index] == "audio" else 0.0,
                        )
                        for index in range(field_studio.spiro_count)
                    ], dtype=np.float64)
                    off_strengths = strengths.copy()
                    for index in range(field_studio.spiro_count):
                        if field_studio.spiro_strobe_waveforms[index] == "timed":
                            off_strengths[index] = 0.0
                elif field_studio.engine == "manual":
                    points = _project_planar_points(
                        field_studio.manual_positions,
                        field_studio,
                    )
                    hues = field_studio.manual_hues
                    color_systems = field_studio.manual_color_systems
                    color_temperatures_k = field_studio.manual_color_temperatures_k
                    color_white_points = field_studio.manual_color_white_points
                    saturations = field_studio.manual_saturations
                    lightnesses = field_studio.manual_lightnesses
                    field_runtime.latest_audio_levels = {}
                    strengths = np.asarray([
                        strobe_level(
                            field_studio.manual_strobe_waveforms[index],
                            field_studio.manual_strobe_phases[index],
                            field_runtime.latest_beat_strength,
                            field_studio.manual_beat_polarities[index],
                            windowed_audio_level(
                                field_runtime, f"manual{index}", field_studio.manual_audio_bands[index],
                                field_studio.manual_audio_windows_ms[index],
                            ) if field_studio.manual_strobe_waveforms[index] == "audio" else 0.0,
                        )
                        for index in range(len(field_studio.manual_positions))
                    ], dtype=np.float64)
                    off_strengths = strengths.copy()
                    for index, waveform in enumerate(field_studio.manual_strobe_waveforms):
                        if waveform == "timed":
                            off_strengths[index] = 0.0
                elif field_studio.engine == "vectorscope":
                    ensure_audio_capture(field_runtime)
                    if not field_studio.paused:
                        field_runtime.audio.configure(
                            fft_size=field_studio.vectorscope_fft_size,
                            oversample=field_studio.vectorscope_oversample,
                            band_count=field_studio.vectorscope_trace_points,
                            min_frequency_hz=field_studio.vectorscope_min_frequency_hz,
                            max_frequency_hz=field_studio.vectorscope_max_frequency_hz,
                        )
                        points, band_power, frequencies = field_runtime.audio.vectorscope_bands(
                            gain=field_studio.vectorscope_gain,
                        )
                        field_runtime.latest_unrotated_points = points
                        field_runtime.latest_audio_rms = field_runtime.audio.last_rms
                        field_runtime.latest_frequencies = frequencies
                        field_runtime.latest_band_power = band_power
                        field_runtime.latest_beat_strength = field_runtime.beat_tracker.update(
                            band_power, frame_seconds,
                        )
                        log_span = math.log(
                            field_studio.vectorscope_max_frequency_hz
                            / field_studio.vectorscope_min_frequency_hz
                        )
                        field_runtime.latest_hues = np.mod(
                            field_studio.vectorscope_hue
                            + 300.0 * np.log(
                                np.maximum(frequencies, field_studio.vectorscope_min_frequency_hz)
                                / field_studio.vectorscope_min_frequency_hz
                            ) / log_span,
                            360.0,
                        )
                    points = _project_planar_points(
                        field_runtime.latest_unrotated_points, field_studio,
                    )
                    hues = field_runtime.latest_hues
                    color_systems = [field_studio.vectorscope_color_system] * len(hues)
                    color_temperatures_k = [field_studio.vectorscope_color_temperature_k] * len(hues)
                    color_white_points = [field_studio.vectorscope_color_white_point] * len(hues)
                    saturations = np.full(len(hues), field_studio.vectorscope_saturation)
                    lightnesses = np.full(len(hues), field_studio.vectorscope_lightness)
                    field_runtime.latest_audio_levels = {}
                    vectorscope_audio_level = (
                        windowed_audio_level(
                            field_runtime, "vectorscope", field_studio.vectorscope_audio_band,
                            field_studio.vectorscope_audio_window_ms,
                        ) if field_studio.vectorscope_strobe_waveform == "audio" else 0.0
                    )
                    strengths = (
                        field_runtime.latest_band_power * field_studio.vectorscope_level
                        * strobe_level(
                            field_studio.vectorscope_strobe_waveform,
                            field_studio.vectorscope_strobe_phase,
                            field_runtime.latest_beat_strength,
                            field_studio.vectorscope_beat_polarity,
                            vectorscope_audio_level,
                        )
                    )
                    off_strengths = (
                        np.zeros_like(strengths)
                        if field_studio.vectorscope_strobe_waveform == "timed"
                        else strengths.copy()
                    )
                    field_runtime.latest_strengths = strengths
                elif field_studio.engine == "uniform":
                    points = np.zeros((0, 2), dtype=np.float64)
                    hues = [field_studio.uniform_hue]
                    color_systems = [field_studio.uniform_color_system]
                    color_temperatures_k = [field_studio.uniform_color_temperature_k]
                    color_white_points = [field_studio.uniform_color_white_point]
                    saturations = [field_studio.uniform_saturation]
                    lightnesses = [field_studio.uniform_lightness]
                    field_runtime.latest_audio_levels = {}
                    uniform_audio_level = (
                        windowed_audio_level(
                            field_runtime, "uniform", field_studio.uniform_audio_band,
                            field_studio.uniform_audio_window_ms,
                        ) if field_studio.uniform_strobe_waveform == "audio" else 0.0
                    )
                    strengths = np.asarray([strobe_level(
                        field_studio.uniform_strobe_waveform,
                        field_studio.uniform_strobe_phase,
                        field_runtime.latest_beat_strength,
                        field_studio.uniform_beat_polarity,
                        uniform_audio_level,
                    )])
                    off_strengths = (
                        np.zeros_like(strengths)
                        if field_studio.uniform_strobe_waveform == "timed"
                        else strengths.copy()
                    )
                else:
                    points = np.zeros((0, 2), dtype=np.float64)
                    hues = [source["hue"] for source in field_studio.site_sources]
                    color_systems = [source.get("color_system", "hsl") for source in field_studio.site_sources]
                    color_temperatures_k = [source.get("temperature_k", 5600.0) for source in field_studio.site_sources]
                    color_white_points = [source.get("white_point", "D65") for source in field_studio.site_sources]
                    saturations = [source["saturation"] for source in field_studio.site_sources]
                    lightnesses = [source["lightness"] for source in field_studio.site_sources]
                    field_runtime.latest_audio_levels = {}
                    strengths = np.asarray([
                        strobe_level(
                            source["strobe_waveform"], source["strobe_phase"],
                            field_runtime.latest_beat_strength, source["beat_polarity"],
                            windowed_audio_level(
                                field_runtime, f"site{index}", source["audio_band"],
                                source["audio_window_ms"],
                            ) if source["strobe_waveform"] == "audio" else 0.0,
                        )
                        for index, source in enumerate(field_studio.site_sources)
                    ], dtype=np.float64)
                    off_strengths = strengths.copy()
                    for index, source in enumerate(field_studio.site_sources):
                        if source["strobe_waveform"] == "timed":
                            off_strengths[index] = 0.0
                if active_mode_normalizes_source_count(field_studio):
                    source_scale = max(1.0, math.sqrt(len(strengths)))
                    strengths = np.asarray(strengths, dtype=np.float64) / source_scale
                    off_strengths = np.asarray(off_strengths, dtype=np.float64) / source_scale
                points = rotate_source_points(
                    points, field_studio.source_rotation_degrees,
                )
                field_runtime.latest_points = points
                field_runtime.latest_source_levels = np.asarray(strengths, dtype=np.float64)
                if field_studio.engine == "vectorscope" and not field_studio.paused:
                    visual_strobe = strobe_level(
                        field_studio.vectorscope_strobe_waveform,
                        field_studio.vectorscope_strobe_phase,
                        field_runtime.latest_beat_strength,
                        field_studio.vectorscope_beat_polarity,
                    )
                    field_runtime.vectorscope_history.append((
                        field_runtime.elapsed_seconds,
                        np.asarray(points, dtype=np.float64).copy(),
                        np.asarray(hues, dtype=np.float64).copy(),
                        np.asarray(field_runtime.latest_band_power, dtype=np.float64).copy()
                        * visual_strobe,
                    ))
                    tail_seconds = field_studio.vectorscope_tail_milliseconds / 1000.0
                    history_horizon = 6.25 * tail_seconds
                    while (
                        len(field_runtime.vectorscope_history) > 1
                        and field_runtime.elapsed_seconds
                        - field_runtime.vectorscope_history[0][0] > history_horizon
                    ):
                        field_runtime.vectorscope_history.popleft()
                field_sources[field_id] = (
                    points, hues, saturations, lightnesses, color_systems,
                    color_temperatures_k, color_white_points, strengths, off_strengths,
                    active_mode_uses_timed(field_studio),
                )

            logical_domain_frames = []
            wire_domain_frames = []
            logical_off_domain_frames = []
            wire_off_domain_frames = []
            array_timed_modes = {}
            wire_frames_by_array = {}
            for array in arrays.values():
                field_runtime = fields[array.field_id]
                (
                    points, hues, saturations, lightnesses, color_systems,
                    color_temperatures_k, color_white_points, strengths,
                    off_strengths, has_timed,
                ) = field_sources[array.field_id]
                array_timed_modes[array.array_id] = has_timed
                if field_runtime.studio.engine == "uniform":
                    color = direct_color(
                        hues[0], saturations[0], lightnesses[0], strengths[0],
                        field_runtime.brightness, color_systems[0],
                        color_temperatures_k[0], color_white_points[0],
                    )
                    off_color = direct_color(
                        hues[0], saturations[0], lightnesses[0], off_strengths[0],
                        field_runtime.brightness, color_systems[0],
                        color_temperatures_k[0], color_white_points[0],
                    )
                    domain_frame = direct_site_frame(
                        array.domain, [0.0] * array.domain.count,
                        [0.0] * array.domain.count, [0.0] * array.domain.count,
                        [1.0] * array.domain.count, 0,
                    )
                    off_domain_frame = list(domain_frame)
                    for address in array.domain.addresses:
                        domain_frame[address.frame_index] = color
                        off_domain_frame[address.frame_index] = off_color
                elif field_runtime.studio.engine == "sites":
                    domain_frame = direct_site_frame(
                        array.domain, hues, saturations, lightnesses, strengths,
                        field_runtime.brightness,
                        array.identity_direction * array.site_rotation_steps,
                        color_systems, color_temperatures_k, color_white_points,
                    )
                    off_domain_frame = direct_site_frame(
                        array.domain, hues, saturations, lightnesses, off_strengths,
                        field_runtime.brightness,
                        array.identity_direction * array.site_rotation_steps,
                        color_systems, color_temperatures_k, color_white_points,
                    )
                else:
                    domain_frame = spectral_source_frame(
                        array.domain, points, hues, receiver_points(array),
                        field_runtime.studio, field_runtime.brightness,
                        field_runtime.exposure, strengths,
                        saturations=saturations, lightnesses=lightnesses,
                        color_systems=color_systems,
                        color_temperatures_k=color_temperatures_k,
                        color_white_points=color_white_points,
                    )
                    off_domain_frame = spectral_source_frame(
                        array.domain, points, hues, receiver_points(array),
                        field_runtime.studio, field_runtime.brightness,
                        field_runtime.exposure, off_strengths,
                        saturations=saturations, lightnesses=lightnesses,
                        color_systems=color_systems,
                        color_temperatures_k=color_temperatures_k,
                        color_white_points=color_white_points,
                    )
                if field_runtime.studio.engine == "vectorscope" and not has_timed:
                    domain_frame = apply_receiver_tail(
                        array,
                        domain_frame,
                        frame_seconds,
                        field_runtime.studio.vectorscope_tail_milliseconds,
                    )
                else:
                    array.light_tail_rgb = np.zeros((0, 3), dtype=np.float64)
                logical_domain_frames.append((array.domain, domain_frame))
                logical_off_domain_frames.append((array.domain, off_domain_frame))
                wire_domain_frames.append((
                    array.domain,
                    _grb_wire_frame(domain_frame)
                    if array.channel_order == "GRB"
                    else domain_frame,
                ))
                wire_frames_by_array[array.array_id] = wire_domain_frames[-1][1]
                wire_off_domain_frames.append((
                    array.domain,
                    _grb_wire_frame(off_domain_frame)
                    if array.channel_order == "GRB"
                    else off_domain_frame,
                ))
            full_frame = merge_domain_frames(controller.led_count, logical_domain_frames)
            wire_frame = merge_domain_frames(controller.led_count, wire_domain_frames)
            off_full_frame = merge_domain_frames(controller.led_count, logical_off_domain_frames)
            off_wire_frame = merge_domain_frames(controller.led_count, wire_off_domain_frames)
            timing_studio = next(
                (
                    runtime.studio for runtime in fields.values()
                    if active_mode_uses_timed(runtime.studio)
                ),
                None,
            )
            next_native_uniform_arrays = {
                array.array_id for array in arrays.values()
                if fields[array.field_id].studio.engine == "uniform"
                and not array_timed_modes[array.array_id]
                and array.transport_zone_type >= 0
            }
            if args.apply and hasattr(controller, "send_uniform_color"):
                for array_id in native_uniform_arrays - next_native_uniform_arrays:
                    controller.send_uniform_color(NativeUniformColor.direct(
                        arrays[array_id].transport_zone_type,
                    ))
                    native_uniform_signatures.pop(array_id, None)
                for array_id in next_native_uniform_arrays:
                    array = arrays[array_id]
                    first_address = array.domain.addresses[0]
                    packed_color = tuple(
                        wire_frames_by_array[array_id][first_address.frame_index]
                    )
                    hardware_mode = fields[array.field_id].studio.uniform_hardware_mode
                    hardware_speed = fields[array.field_id].studio.uniform_hardware_speed
                    signature = (
                        array.transport_zone_type, hardware_mode,
                        hardware_speed, packed_color,
                    )
                    if native_uniform_signatures.get(array_id) == signature:
                        continue
                    controller.send_uniform_color(NativeUniformColor.uniform(
                        array.transport_zone_type,
                        packed_color,
                        hardware_mode,
                        hardware_speed,
                    ))
                    native_uniform_signatures[array_id] = signature
                native_uniform_arrays = next_native_uniform_arrays
            if args.apply and timing_studio is not None:
                if not all(hasattr(controller, name) for name in (
                    "clock_ns", "send_frame_at", "cancel_scheduled", "timing_metrics",
                )):
                    raise OpenRGBShellError(
                        "timed strobe requires the direct native HID transport"
                    )
                reference_hz = (
                    fan_rpm / 60.0 if fan_rpm is not None else timing_studio.timing_rate_hz
                )
                reference_period_ns = max(1, round(1_000_000_000.0 / reference_hz))
                pulse_width_ns = timing_studio.timing_pulse_width_ns
                clearance_ns = (
                    max(pulse_width_ns, GUARANTEED_HID_RESET_NS)
                    + GUARANTEED_HID_RESET_NS
                )
                firing_interval_cycles = max(
                    1,
                    math.floor(clearance_ns / reference_period_ns) + 1,
                )
                skipped_cycles = firing_interval_cycles - 1
                predelay_ns = timing_studio.timing_predelay_ns
                now_ns = controller.clock_ns()
                next_signature = (
                    id(timing_studio), predelay_ns,
                    timing_studio.timing_pulse_width_ns,
                    None if fan_rpm is not None else timing_studio.timing_rate_hz,
                    timing_studio.timing_phase_degrees,
                )
                if not timing_active or timing_signature != next_signature:
                    controller.cancel_scheduled()
                    phase_ns = round(
                        reference_period_ns
                        * timing_studio.timing_phase_degrees / 360.0
                    )
                    next_timing_target_ns = now_ns + phase_ns
                    if next_timing_target_ns - predelay_ns <= now_ns:
                        next_timing_target_ns += reference_period_ns
                    last_timing_target_ns = None
                    timing_rearm_ns = 0
                    timing_active = True
                    timing_signature = next_signature
                if now_ns >= timing_rearm_ns:
                    if next_timing_target_ns is None:
                        assert last_timing_target_ns is not None
                        next_timing_target_ns = (
                            last_timing_target_ns
                            + firing_interval_cycles * reference_period_ns
                        )
                    assert next_timing_target_ns is not None
                    while next_timing_target_ns - predelay_ns <= now_ns:
                        next_timing_target_ns += reference_period_ns
                    on_deadline_ns = next_timing_target_ns - predelay_ns
                    controller.send_frame_at(
                        wire_frame, on_deadline_ns,
                        event_kind="on",
                    )
                    controller.send_frame_at(
                        off_wire_frame,
                        on_deadline_ns + pulse_width_ns,
                        event_kind="off",
                    )
                    last_timing_target_ns = next_timing_target_ns
                    next_timing_target_ns = None
                    timing_rearm_ns = on_deadline_ns
                timing_status = {
                    "reference_hz": reference_hz,
                    "reference_period_ns": reference_period_ns,
                    "pulse_width_ns": pulse_width_ns,
                    "send_duration_ns": GUARANTEED_HID_RESET_NS,
                    "clearance_ns": clearance_ns,
                    "firing_interval_cycles": firing_interval_cycles,
                    "revolutions_to_skip": skipped_cycles,
                    "fan_locked": fan_rpm is not None,
                }
            elif args.apply:
                if timing_active:
                    controller.cancel_scheduled()
                    timing_active = False
                    timing_signature = None
                    next_timing_target_ns = None
                    last_timing_target_ns = None
                    timing_rearm_ns = 0
                    timing_status = {}
                if len(next_native_uniform_arrays) < len(arrays):
                    controller.send_frame(wire_frame)
            transport_metrics = (
                controller.metrics()
                if args.apply and hasattr(controller, "metrics")
                else None
            )
            frame_number += 1
            next_deadline += frame_seconds

            active_array = arrays[active_array_id]
            active_field = fields[active_array.field_id]
            displayed_frame = (
                off_full_frame
                if transport_metrics is not None
                and int(transport_metrics.get("last_scheduled_event_kind", 0)) == 2
                else full_frame
            )
            logical_colors = _logical_domain_frame(active_array.domain, displayed_frame)
            ring_center, ring_radius = draw_studio(
                pygame, screen, font, small, active_array, arrays, active_field,
                logical_colors, widgets, reader is not None, transport_metrics,
                fan_rpm, timing_status,
            )
            time.sleep(max(0.0, next_deadline - time.perf_counter()))
        return 0
    except (OpenRGBShellError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    finally:
        if hardware_active and controller is not None:
            try:
                if hasattr(controller, "send_uniform_color") and 'native_uniform_arrays' in locals():
                    for array_id in native_uniform_arrays:
                        controller.send_uniform_color(NativeUniformColor.uniform(
                            arrays[array_id].transport_zone_type, (0, 0, 0),
                        ))
                _send_all_off(controller)
            except Exception as exc:
                print(f"warning: failed to send all-off frame during exit: {exc}")
        if controller is not None and hasattr(controller, "close"):
            controller.close()
        if pygame is not None:
            pygame.quit()
        if reader is not None:
            reader.close()
        if fan_reader is not None:
            fan_reader.close()
        if 'fields' in locals():
            for field_runtime in fields.values():
                if field_runtime.audio is not None:
                    field_runtime.audio.close()


if __name__ == "__main__":
    raise SystemExit(main())
