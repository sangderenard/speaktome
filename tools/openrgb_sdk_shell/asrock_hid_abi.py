"""Compiler-visible ctypes messages shared with the ASRock native C ABI.

The classes in this module are the layout authority on the Python side.  The
Turing compiler's established ctypes interception reads their live size,
alignment, and field offsets when they are present in ``python_bindings``.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum


ABI_VERSION = 2
EXTERNAL_ZONE_TYPE = 0x07


class ASRockEffectMode(IntEnum):
    OFF = 0x00
    STATIC = 0x01
    BREATHING = 0x02
    STROBE = 0x03
    SPECTRUM_CYCLE = 0x04
    RANDOM = 0x05
    MUSIC = 0x06
    WAVE = 0x07
    SPRING = 0x08
    STACK = 0x09
    CRAM = 0x0A
    SCAN = 0x0B
    NEON = 0x0C
    WATER = 0x0D
    RAINBOW = 0x0E
    DIRECT = 0x0F


class ASRockHIDRawColor(ctypes.Structure):
    """Three bytes in the controller's native zone-color order."""

    _fields_ = [
        ("byte_0", ctypes.c_uint8),
        ("byte_1", ctypes.c_uint8),
        ("byte_2", ctypes.c_uint8),
    ]

    def __init__(self, byte_0: int = 0, byte_1: int = 0, byte_2: int = 0):
        values = tuple(int(value) for value in (byte_0, byte_1, byte_2))
        if any(not 0 <= value <= 0xFF for value in values):
            raise ValueError("raw color components must be bytes")
        super().__init__(*values)

    @classmethod
    def white(cls) -> "ASRockHIDRawColor":
        return cls(0xFF, 0xFF, 0xFF)

    @property
    def bytes(self) -> tuple[int, int, int]:
        return int(self.byte_0), int(self.byte_1), int(self.byte_2)


class ASRockHIDExternalZoneEffect(ctypes.Structure):
    """Versioned semantic message for Audio/ARGB3 native effects.

    This structure intentionally does not expose a zone or all-zone field:
    both C entry points are restricted to physical zone 0x07 and always emit
    ``allzone=false``.
    """

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("mode", ctypes.c_uint8),
        ("speed", ctypes.c_uint8),
        ("color", ASRockHIDRawColor),
        ("reserved", ctypes.c_uint8 * 3),
    ]

    def __init__(
        self,
        mode: int | ASRockEffectMode = ASRockEffectMode.OFF,
        speed: int = 0xE0,
        color: ASRockHIDRawColor | tuple[int, int, int] = (0, 0, 0),
    ):
        mode_value = int(mode)
        speed_value = int(speed)
        if not 0 <= mode_value <= 0xFF or not 0 <= speed_value <= 0xFF:
            raise ValueError("effect mode and speed must be bytes")
        raw_color = color if isinstance(color, ASRockHIDRawColor) else ASRockHIDRawColor(*color)
        super().__init__(
            ABI_VERSION,
            ctypes.sizeof(type(self)),
            mode_value,
            speed_value,
            raw_color,
            (ctypes.c_uint8 * 3)(0, 0, 0),
        )

    @classmethod
    def strobe(
        cls,
        speed: int,
        color: ASRockHIDRawColor | tuple[int, int, int] = (0xFF, 0xFF, 0xFF),
    ) -> "ASRockHIDExternalZoneEffect":
        return cls(ASRockEffectMode.STROBE, speed, color)

    def validate(self) -> "ASRockHIDExternalZoneEffect":
        if int(self.abi_version) != ABI_VERSION:
            raise ValueError(f"effect message ABI version must be {ABI_VERSION}")
        if int(self.struct_size) != ctypes.sizeof(type(self)):
            raise ValueError("effect message struct_size does not match its ctypes layout")
        if any(int(value) != 0 for value in self.reserved):
            raise ValueError("effect message reserved bytes must be zero")
        return self

    @property
    def effect_mode(self) -> ASRockEffectMode:
        return ASRockEffectMode(int(self.mode))

    @property
    def color_bytes(self) -> tuple[int, int, int]:
        return self.color.bytes

    def copy(self) -> "ASRockHIDExternalZoneEffect":
        return type(self)(int(self.mode), int(self.speed), self.color_bytes)

    def __repr__(self) -> str:
        color = " ".join(f"{value:02X}" for value in self.color_bytes)
        return (
            f"{type(self).__name__}(mode=0x{int(self.mode):02X}, "
            f"speed=0x{int(self.speed):02X}, raw_color={color})"
        )


class ASRockHIDUniformColor(ctypes.Structure):
    """One-report request to uniformly color one configured physical zone."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("zone_type", ctypes.c_uint8),
        ("mode", ctypes.c_uint8),
        ("speed", ctypes.c_uint8),
        ("all_zones", ctypes.c_uint8),
        ("color", ASRockHIDRawColor),
        ("reserved", ctypes.c_uint8 * 1),
    ]

    def __init__(
        self,
        zone_type: int,
        mode: int | ASRockEffectMode = ASRockEffectMode.STATIC,
        color: ASRockHIDRawColor | tuple[int, int, int] = (0, 0, 0),
        *,
        speed: int = 0xE0,
        all_zones: bool = False,
    ):
        zone_value = int(zone_type)
        mode_value = int(mode)
        if not 0 <= zone_value <= 0xFF:
            raise ValueError("uniform color zone type must be a byte")
        if not int(ASRockEffectMode.OFF) <= mode_value <= int(ASRockEffectMode.DIRECT):
            raise ValueError("uniform color mode is outside the known Polychrome range")
        if not 0 <= int(speed) <= 0xFF:
            raise ValueError("uniform color speed must be a byte")
        raw_color = color if isinstance(color, ASRockHIDRawColor) else ASRockHIDRawColor(*color)
        super().__init__(
            ABI_VERSION,
            ctypes.sizeof(type(self)),
            zone_value,
            mode_value,
            int(speed),
            int(bool(all_zones)),
            raw_color,
            (ctypes.c_uint8 * 1)(0),
        )

    @classmethod
    def uniform(
        cls, zone_type: int, color: ASRockHIDRawColor | tuple[int, int, int],
        mode: int | ASRockEffectMode = ASRockEffectMode.STATIC,
        speed: int = 0xE0,
    ) -> "ASRockHIDUniformColor":
        return cls(zone_type, mode, color, speed=speed)

    @classmethod
    def uniform_all(
        cls, color: ASRockHIDRawColor | tuple[int, int, int],
    ) -> "ASRockHIDUniformColor":
        return cls(0xFF, ASRockEffectMode.STATIC, color, all_zones=True)

    @classmethod
    def direct(cls, zone_type: int) -> "ASRockHIDUniformColor":
        return cls(zone_type, ASRockEffectMode.DIRECT, (0, 0, 0))

    def validate(self) -> "ASRockHIDUniformColor":
        if int(self.abi_version) != ABI_VERSION:
            raise ValueError(f"uniform color ABI version must be {ABI_VERSION}")
        if int(self.struct_size) != ctypes.sizeof(type(self)):
            raise ValueError("uniform color struct_size does not match its ctypes layout")
        if not int(ASRockEffectMode.OFF) <= int(self.mode) <= int(ASRockEffectMode.DIRECT):
            raise ValueError("uniform color mode is outside the known Polychrome range")
        if not 0 <= int(self.speed) <= 0xFF:
            raise ValueError("uniform color speed must be a byte")
        if int(self.all_zones) not in {0, 1}:
            raise ValueError("uniform color all_zones must be zero or one")
        if any(int(value) != 0 for value in self.reserved):
            raise ValueError("uniform color reserved bytes must be zero")
        return self

    @property
    def color_bytes(self) -> tuple[int, int, int]:
        return self.color.bytes


if ctypes.sizeof(ASRockHIDRawColor) != 3:
    raise RuntimeError("ASRockHIDRawColor layout disagrees with the C ABI")
if ctypes.sizeof(ASRockHIDExternalZoneEffect) != 16:
    raise RuntimeError("ASRockHIDExternalZoneEffect layout disagrees with the C ABI")
if ctypes.sizeof(ASRockHIDUniformColor) != 16:
    raise RuntimeError("ASRockHIDUniformColor layout disagrees with the C ABI")


__all__ = [
    "ABI_VERSION",
    "EXTERNAL_ZONE_TYPE",
    "ASRockEffectMode",
    "ASRockHIDRawColor",
    "ASRockHIDExternalZoneEffect",
    "ASRockHIDUniformColor",
]
