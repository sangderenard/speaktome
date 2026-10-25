"""Persistent configuration for independently driven OpenRGB light arrays."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path


CONFIG_VERSION = 1
STROBE_WAVEFORMS = {"off", "sine", "square", "triangle", "saw", "beat", "audio", "timed"}
HARDWARE_MODES = set(range(0x00, 0x0F))
COLOR_SYSTEMS = {"hsl", "blackbody", "white"}
WHITE_POINTS = {"E", "D50", "D65"}


def _field_defaults() -> dict:
    return {
        "engine": "physics",
        "falloff_power": 2.0,
        "center_contribution": 0.125,
        "source_rotation_degrees": 0.0,
        "source_rotation_rate_degrees": 0.0,
        "source_rotation_rate_multiplier": 1.0,
        "timing": {
            "rate_hz": 1.0,
            "phase_degrees": 0.0,
            "predelay_ns": 0,
            "pulse_width_ns": 100000000,
        },
        "projection": {
            "inner_radius": 0.41,
            "outer_radius": 0.82,
            "light_ring_radius": 1.0,
        },
        "brightness": 96,
        "exposure": 3.0,
        "physics": {
            "geometry": "torus",
            "normalize_source_count": False,
            "body_count": 3,
            "hues": [180.0, 300.0, 60.0],
            "color_systems": ["hsl", "hsl", "hsl"],
            "color_temperatures_k": [5600.0, 5600.0, 5600.0],
            "color_white_points": ["D65", "D65", "D65"],
            "saturations": [1.0, 1.0, 1.0],
            "lightnesses": [0.5, 0.5, 0.5],
            "hue_rates": [0.0, 0.0, 0.0],
            "hue_rate_multipliers": [1.0, 1.0, 1.0],
            "strobe_waveforms": ["off", "off", "off"],
            "strobe_rates_hz": [0.0, 0.0, 0.0],
            "beat_polarities": [1.0, 1.0, 1.0],
            "audio_bands": [0, 0, 0],
            "audio_windows_ms": [120.0, 120.0, 120.0],
            "masses": [3.0, 4.0, 5.0],
            "temperature_c": 40.0,
            "thermal_coupling": 1.0,
        },
        "spirograph": {
            "speed": 1.0,
            "speed_multiplier": 1.0,
            "normalize_source_count": False,
            "selected_source": 0,
            "sources": [
                {"hue": 180.0, "color_system": "hsl", "temperature_k": 5600.0, "white_point": "D65", "saturation": 1.0, "lightness": 0.5, "hue_rate": 0.0, "hue_rate_multiplier": 1.0, "angular_offset_degrees": 0.0, "angular_rate_degrees": 0.0, "angular_rate_multiplier": 1.0, "strobe_waveform": "off", "strobe_rate_hz": 0.0, "beat_polarity": 1.0, "audio_band": 0, "audio_window_ms": 120.0, "ratio_numerator": 2, "ratio_denominator": 3},
                {"hue": 300.0, "color_system": "hsl", "temperature_k": 5600.0, "white_point": "D65", "saturation": 1.0, "lightness": 0.5, "hue_rate": 0.0, "hue_rate_multiplier": 1.0, "angular_offset_degrees": 120.0, "angular_rate_degrees": 0.0, "angular_rate_multiplier": 1.0, "strobe_waveform": "off", "strobe_rate_hz": 0.0, "beat_polarity": 1.0, "audio_band": 0, "audio_window_ms": 120.0, "ratio_numerator": 3, "ratio_denominator": 4},
                {"hue": 60.0, "color_system": "hsl", "temperature_k": 5600.0, "white_point": "D65", "saturation": 1.0, "lightness": 0.5, "hue_rate": 0.0, "hue_rate_multiplier": 1.0, "angular_offset_degrees": 240.0, "angular_rate_degrees": 0.0, "angular_rate_multiplier": 1.0, "strobe_waveform": "off", "strobe_rate_hz": 0.0, "beat_polarity": 1.0, "audio_band": 0, "audio_window_ms": 120.0, "ratio_numerator": 4, "ratio_denominator": 5},
            ],
        },
        "manual": {
            "normalize_source_count": False,
            "positions": [[0.55, 0.0], [-0.3, 0.52], [-0.3, -0.52]],
            "hues": [0.0, 120.0, 240.0],
            "color_systems": ["hsl", "hsl", "hsl"],
            "color_temperatures_k": [5600.0, 5600.0, 5600.0],
            "color_white_points": ["D65", "D65", "D65"],
            "saturations": [1.0, 1.0, 1.0],
            "lightnesses": [0.5, 0.5, 0.5],
            "hue_rates": [0.0, 0.0, 0.0],
            "hue_rate_multipliers": [1.0, 1.0, 1.0],
            "strobe_waveforms": ["off", "off", "off"],
            "strobe_rates_hz": [0.0, 0.0, 0.0],
            "beat_polarities": [1.0, 1.0, 1.0],
            "audio_bands": [0, 0, 0],
            "audio_windows_ms": [120.0, 120.0, 120.0],
            "selected_source": 0,
            "deployment_count": 3,
        },
        "vectorscope": {
            "hue": 180.0,
            "color_system": "hsl",
            "temperature_k": 5600.0,
            "white_point": "D65",
            "saturation": 1.0,
            "lightness": 0.5,
            "hue_rate": 0.0,
            "hue_rate_multiplier": 1.0,
            "strobe_waveform": "off",
            "strobe_rate_hz": 0.0,
            "beat_polarity": 1.0,
            "audio_band": 0,
            "audio_window_ms": 120.0,
            "gain": 6.0,
            "level": 4.0,
            "normalize_band_count": True,
            "tail_milliseconds": 600.0,
            "trace_points": 128,
            "fft_size": 4096,
            "oversample": 8,
            "min_frequency_hz": 27.5,
            "max_frequency_hz": 18000.0,
        },
        "uniform": {
            "hardware_mode": 0x01,
            "hardware_speed": 0xE0,
            "hue": 0.0,
            "color_system": "hsl",
            "temperature_k": 5600.0,
            "white_point": "D65",
            "saturation": 1.0,
            "lightness": 0.5,
            "hue_rate": 0.0,
            "hue_rate_multiplier": 1.0,
            "strobe_waveform": "off",
            "strobe_rate_hz": 0.0,
            "beat_polarity": 1.0,
            "audio_band": 0,
            "audio_window_ms": 120.0,
        },
        "sites": {"selected_source": 0, "sources": []},
    }


def default_config() -> dict:
    ring_field = _field_defaults()
    motherboard_field = deepcopy(ring_field)
    return {
        "version": CONFIG_VERSION,
        "refresh_rate": 60.0,
        "controller": {
            "device_name": "ASRock B650M Pro RS WiFi",
            "mode": "Direct",
            "transport": {
                "kind": "direct_asrock_hid",
                "vendor_id": 9934,
                "product_id": 418,
                "response_timeout_ms": 100,
                "packet_mode": "burst",
                "message_type": "stream",
                "report_count": 14,
                "window_size": 4,
                "hidapi_dll": "C:\\Users\\alber\\AppData\\Local\\Temp\\OpenRGB-1.0-81bbe18\\OpenRGB Windows 64-bit\\hidapi-hotplug.dll",
                "zones": [
                    {"name": "RGB LED 1 Header", "zone_type": 0, "count": 1, "maximum_count": 1, "rg_swap": False},
                    {"name": "Addressable Header 1", "zone_type": 2, "count": 80, "maximum_count": 100, "rg_swap": False},
                    {"name": "Addressable Header 2", "zone_type": 3, "count": 80, "maximum_count": 100, "rg_swap": False},
                    {"name": "PCB", "zone_type": 6, "count": 4, "maximum_count": 4, "rg_swap": True},
                    {"name": "Addressable Header 3/Audio", "zone_type": 7, "count": 80, "maximum_count": 100, "rg_swap": False},
                ],
            },
        },
        "arrays": {
            "ring": {
                "label": "Ring",
                "zone": "Addressable Header 3/Audio",
                "position_count": 8,
                "geometry": {
                    "kind": "ring",
                    "identity_offset": 1,
                    "identity_direction": 1,
                },
                "channel_order": "GRB",
                "field_id": "ring-field",
                "placement": {
                    "x": 0.0,
                    "y": 0.0,
                    "rotation_degrees": 0.0,
                    "rotation_rate_degrees": 0.0,
                    "rotation_rate_multiplier": 1.0,
                    "scale": 1.0,
                },
            },
            "motherboard": {
                "label": "Motherboard strip",
                "zone": "PCB",
                "position_count": 4,
                "geometry": {"kind": "vertical_strip"},
                "channel_order": "RGB",
                "field_id": "motherboard-field",
                "placement": {
                    "x": 0.0,
                    "y": 0.0,
                    "rotation_degrees": 0.0,
                    "rotation_rate_degrees": 0.0,
                    "rotation_rate_multiplier": 1.0,
                    "scale": 1.0,
                },
            },
        },
        "fields": {
            "ring-field": ring_field,
            "motherboard-field": motherboard_field,
        },
        "ui": {"active_array": "ring"},
    }


def validate_config(config: dict) -> None:
    if config.get("version") != CONFIG_VERSION:
        raise ValueError(f"field studio configuration version must be {CONFIG_VERSION}")
    arrays = config.get("arrays")
    fields = config.get("fields")
    controller = config.get("controller", {})
    transport = controller.get("transport")
    if transport is not None:
        if transport.get("kind") != "direct_asrock_hid":
            raise ValueError("controller.transport.kind must be direct_asrock_hid")
        zones = transport.get("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError("direct HID transport requires configured zones")
        for zone in zones:
            count = int(zone.get("count", 0))
            maximum = int(zone.get("maximum_count", 0))
            zone_type = int(zone.get("zone_type", -1))
            if not zone.get("name") or count <= 0 or maximum < count or not 0 <= zone_type < 8:
                raise ValueError("direct HID transport has an invalid zone layout")
        if transport.get("packet_mode", "burst") not in {"serial", "windowed", "burst"}:
            raise ValueError("direct HID packet mode must be serial, windowed, or burst")
        if transport.get("message_type", "stream") not in {"stream", "restart", "continue"}:
            raise ValueError("direct HID message type must be stream, restart, or continue")
        if int(transport.get("report_count", 14)) <= 0:
            raise ValueError("direct HID report count must be positive")
        if int(transport.get("window_size", 4)) <= 0:
            raise ValueError("direct HID window size must be positive")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("field studio configuration requires at least one array")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("field studio configuration requires at least one field")
    for array_id, array in arrays.items():
        if array.get("field_id") not in fields:
            raise ValueError(f"array {array_id!r} references an unknown field_id")
        if array.get("channel_order") not in {"RGB", "GRB"}:
            raise ValueError(f"array {array_id!r} channel_order must be RGB or GRB")
        if int(array.get("position_count", 0)) <= 0:
            raise ValueError(f"array {array_id!r} position_count must be positive")
        placement = array.get("placement", {})
        if float(placement.get("scale", 0.0)) <= 0.0:
            raise ValueError(f"array {array_id!r} placement scale must be positive")
        if float(placement.get("rotation_rate_multiplier", 1.0)) <= 0.0:
            raise ValueError(f"array {array_id!r} rotation multiplier must be positive")
    for field_id, field in fields.items():
        if field.get("engine") not in {"physics", "spirograph", "vectorscope", "manual", "uniform", "sites"}:
            raise ValueError(f"field {field_id!r} has an unknown engine")
        physics = field.get("physics", {})
        if float(field.get("source_rotation_rate_multiplier", 1.0)) <= 0.0:
            raise ValueError(f"field {field_id!r} source rotation multiplier must be positive")
        if physics.get("geometry", "torus") not in {"torus", "sphere", "cartesian"}:
            raise ValueError(f"field {field_id!r} has an unknown physics geometry")
        if not isinstance(physics.get("normalize_source_count", False), bool):
            raise ValueError(f"field {field_id!r} physics normalization must be boolean")
        physics_waves = physics.get("strobe_waveforms", ["off"] * 3)
        physics_rates = physics.get("strobe_rates_hz", [0.0] * 3)
        physics_polarities = physics.get("beat_polarities", [1.0] * 3)
        physics_saturations = physics.get("saturations", [1.0] * 3)
        physics_lightnesses = physics.get("lightnesses", [0.5] * 3)
        physics_color_systems = physics.get("color_systems", ["hsl"] * 3)
        physics_temperatures = physics.get("color_temperatures_k", [5600.0] * 3)
        physics_white_points = physics.get("color_white_points", ["D65"] * 3)
        physics_audio_bands = physics.get("audio_bands", [0] * 3)
        physics_audio_windows = physics.get("audio_windows_ms", [120.0] * 3)
        physics_hue_multipliers = physics.get("hue_rate_multipliers", [1.0] * 3)
        if (
            len(physics_waves) != 3 or len(physics_rates) != 3 or len(physics_polarities) != 3
            or len(physics_saturations) != 3 or len(physics_lightnesses) != 3
            or len(physics_color_systems) != 3 or len(physics_temperatures) != 3
            or len(physics_white_points) != 3
            or len(physics_audio_bands) != 3 or len(physics_audio_windows) != 3
            or len(physics_hue_multipliers) != 3
            or any(wave not in STROBE_WAVEFORMS for wave in physics_waves)
            or any(float(rate) < 0.0 for rate in physics_rates)
            or any(not -1.0 <= float(value) <= 1.0 for value in physics_polarities)
            or any(float(value) <= 0.0 for value in physics_hue_multipliers)
            or any(not 0.0 <= float(value) <= 1.0 for value in physics_saturations)
            or any(not 0.0 <= float(value) <= 1.0 for value in physics_lightnesses)
            or any(value not in COLOR_SYSTEMS for value in physics_color_systems)
            or any(float(value) <= 0.0 for value in physics_temperatures)
            or any(value not in WHITE_POINTS for value in physics_white_points)
            or any(int(value) < 0 for value in physics_audio_bands)
            or any(float(value) < 0.0 for value in physics_audio_windows)
        ):
            raise ValueError(f"field {field_id!r} has invalid physics strobe settings")
        for source in field.get("spirograph", {}).get("sources", []):
            if (
                source.get("strobe_waveform", "off") not in STROBE_WAVEFORMS
                or float(source.get("strobe_rate_hz", 0.0)) < 0.0
                or not -1.0 <= float(source.get("beat_polarity", 1.0)) <= 1.0
                or float(source.get("angular_rate_multiplier", 1.0)) <= 0.0
                or float(source.get("hue_rate_multiplier", 1.0)) <= 0.0
                or not 0.0 <= float(source.get("saturation", 1.0)) <= 1.0
                or not 0.0 <= float(source.get("lightness", 0.5)) <= 1.0
                or source.get("color_system", "hsl") not in COLOR_SYSTEMS
                or float(source.get("temperature_k", 5600.0)) <= 0.0
                or source.get("white_point", "D65") not in WHITE_POINTS
                or int(source.get("audio_band", 0)) < 0
                or float(source.get("audio_window_ms", 120.0)) < 0.0
            ):
                raise ValueError(f"field {field_id!r} has invalid spirograph strobe settings")
        if float(field.get("spirograph", {}).get("speed_multiplier", 1.0)) <= 0.0:
            raise ValueError(f"field {field_id!r} spirograph speed multiplier must be positive")
        if not isinstance(
            field.get("spirograph", {}).get("normalize_source_count", False), bool,
        ):
            raise ValueError(f"field {field_id!r} spirograph normalization must be boolean")
        projection = field.get("projection", physics)
        inner = float(projection.get("inner_radius", physics.get("projection_inner_radius", 0.41)))
        outer = float(projection.get("outer_radius", physics.get("projection_outer_radius", 0.82)))
        light_ring = float(projection.get("light_ring_radius", physics.get("light_ring_radius", 1.0)))
        temperature_c = float(physics.get("temperature_c", 40.0))
        thermal_coupling = float(physics.get("thermal_coupling", 1.0))
        if not 0.0 <= inner < outer or light_ring <= 0.0:
            raise ValueError(
                f"field {field_id!r} projection requires 0 <= inner < outer and light ring > 0"
            )
        if not 0.0 <= temperature_c < 150.0 or thermal_coupling < 0.0:
            raise ValueError(
                f"field {field_id!r} thermal controls require 0 <= C < 150 and coupling >= 0"
            )
        manual = field.get("manual", {})
        if not isinstance(manual.get("normalize_source_count", False), bool):
            raise ValueError(f"field {field_id!r} manual normalization must be boolean")
        positions = manual.get("positions", [[0.55, 0.0], [-0.3, 0.52], [-0.3, -0.52]])
        manual_hues = manual.get("hues", [360.0 * index / len(positions) for index in range(len(positions))])
        manual_saturations = manual.get("saturations", [1.0] * len(positions))
        manual_lightnesses = manual.get("lightnesses", [0.5] * len(positions))
        manual_color_systems = manual.get("color_systems", ["hsl"] * len(positions))
        manual_temperatures = manual.get("color_temperatures_k", [5600.0] * len(positions))
        manual_white_points = manual.get("color_white_points", ["D65"] * len(positions))
        manual_rates = manual.get("hue_rates", [0.0] * len(positions))
        manual_rate_multipliers = manual.get("hue_rate_multipliers", [1.0] * len(positions))
        manual_waves = manual.get("strobe_waveforms", ["off"] * len(positions))
        manual_strobe_rates = manual.get("strobe_rates_hz", [0.0] * len(positions))
        manual_polarities = manual.get("beat_polarities", [1.0] * len(positions))
        manual_audio_bands = manual.get("audio_bands", [0] * len(positions))
        manual_audio_windows = manual.get("audio_windows_ms", [120.0] * len(positions))
        if (
            not positions or len(positions) > 128
            or any(len(position) != 2 for position in positions)
            or len(manual_hues) != len(positions)
            or len(manual_saturations) != len(positions)
            or len(manual_lightnesses) != len(positions)
            or len(manual_color_systems) != len(positions)
            or len(manual_temperatures) != len(positions)
            or len(manual_white_points) != len(positions)
            or len(manual_rates) != len(positions)
            or len(manual_rate_multipliers) != len(positions)
            or any(float(value) <= 0.0 for value in manual_rate_multipliers)
            or len(manual_waves) != len(positions)
            or len(manual_strobe_rates) != len(positions)
            or len(manual_polarities) != len(positions)
            or len(manual_audio_bands) != len(positions)
            or len(manual_audio_windows) != len(positions)
            or any(not 0.0 <= float(value) <= 1.0 for value in manual_saturations)
            or any(not 0.0 <= float(value) <= 1.0 for value in manual_lightnesses)
            or any(value not in COLOR_SYSTEMS for value in manual_color_systems)
            or any(float(value) <= 0.0 for value in manual_temperatures)
            or any(value not in WHITE_POINTS for value in manual_white_points)
            or any(int(value) < 0 for value in manual_audio_bands)
            or any(float(value) < 0.0 for value in manual_audio_windows)
            or any(value not in STROBE_WAVEFORMS for value in manual_waves)
            or any(float(value) < 0.0 for value in manual_strobe_rates)
            or any(not -1.0 <= float(value) <= 1.0 for value in manual_polarities)
            or not 1 <= int(manual.get("deployment_count", len(positions))) <= 128
        ):
            raise ValueError(f"field {field_id!r} has invalid manual emitters")
        timing = field.get("timing", {})
        if (
            float(timing.get("rate_hz", 1.0)) <= 0.0
            or not 0.0 <= float(timing.get("phase_degrees", 0.0)) < 360.0
            or int(timing.get("predelay_ns", 0)) < 0
            or int(timing.get("pulse_width_ns", 100000000)) <= 0
        ):
            raise ValueError(f"field {field_id!r} has invalid timed-strobe settings")
        vectorscope = field.get("vectorscope", {})
        if (
            vectorscope.get("strobe_waveform", "off") not in STROBE_WAVEFORMS
            or float(vectorscope.get("strobe_rate_hz", 0.0)) < 0.0
            or not -1.0 <= float(vectorscope.get("beat_polarity", 1.0)) <= 1.0
            or float(vectorscope.get("hue_rate_multiplier", 1.0)) <= 0.0
        ):
            raise ValueError(f"field {field_id!r} has invalid vectorscope strobe settings")
        if float(vectorscope.get("tail_milliseconds", 600.0)) < 0.0:
            raise ValueError(f"field {field_id!r} vectorscope tail must be nonnegative")
        if int(vectorscope.get("trace_points", 128)) <= 0:
            raise ValueError(f"field {field_id!r} vectorscope trace_points must be positive")
        fft_size = int(vectorscope.get("fft_size", 4096))
        oversample = int(vectorscope.get("oversample", 8))
        if (
            fft_size < 256 or fft_size & (fft_size - 1)
            or oversample <= 0 or oversample & (oversample - 1)
            or oversample > fft_size
        ):
            raise ValueError(
                f"field {field_id!r} vectorscope FFT size and oversample must be powers of two"
            )
        low = float(vectorscope.get("min_frequency_hz", 27.5))
        high = float(vectorscope.get("max_frequency_hz", 18000.0))
        if not 0.0 < low < high:
            raise ValueError(f"field {field_id!r} vectorscope frequency range is invalid")
        for section_name in ("vectorscope", "uniform"):
            color_source = field.get(section_name, {})
            if (
                not 0.0 <= float(color_source.get("saturation", 1.0)) <= 1.0
                or not 0.0 <= float(color_source.get("lightness", 0.5)) <= 1.0
                or color_source.get("strobe_waveform", "off") not in STROBE_WAVEFORMS
                or int(color_source.get("audio_band", 0)) < 0
                or float(color_source.get("audio_window_ms", 120.0)) < 0.0
                or color_source.get("color_system", "hsl") not in COLOR_SYSTEMS
                or float(color_source.get("temperature_k", 5600.0)) <= 0.0
                or color_source.get("white_point", "D65") not in WHITE_POINTS
            ):
                raise ValueError(f"field {field_id!r} has invalid {section_name} color settings")
        if int(field.get("uniform", {}).get("hardware_mode", 0x01)) not in HARDWARE_MODES:
            raise ValueError(f"field {field_id!r} has invalid Uniform hardware mode")
        if not 0 <= int(field.get("uniform", {}).get("hardware_speed", 0xE0)) <= 0xFF:
            raise ValueError(f"field {field_id!r} has invalid Uniform hardware speed")
        for source in field.get("sites", {}).get("sources", []):
            if (
                not 0.0 <= float(source.get("saturation", 1.0)) <= 1.0
                or not 0.0 <= float(source.get("lightness", 0.5)) <= 1.0
                or source.get("color_system", "hsl") not in COLOR_SYSTEMS
                or float(source.get("temperature_k", 5600.0)) <= 0.0
                or source.get("white_point", "D65") not in WHITE_POINTS
                or source.get("strobe_waveform", "off") not in STROBE_WAVEFORMS
                or float(source.get("strobe_rate_hz", 0.0)) < 0.0
                or not -1.0 <= float(source.get("beat_polarity", 1.0)) <= 1.0
                or int(source.get("audio_band", 0)) < 0
                or float(source.get("audio_window_ms", 120.0)) < 0.0
            ):
                raise ValueError(f"field {field_id!r} has invalid direct-site settings")
    active = config.get("ui", {}).get("active_array")
    if active not in arrays:
        raise ValueError("ui.active_array must name a configured array")


def save_config(path: Path, config: dict) -> None:
    validate_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_or_create_config(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        config = default_config()
        save_config(path, config)
        return config
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


__all__ = ["CONFIG_VERSION", "default_config", "load_or_create_config", "save_config", "validate_config"]
