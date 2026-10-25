"""Conservative ASRock Polychrome native-strobe experiment for ARGB3 only.

This tool deliberately exposes only the known STROBE mode and a fixed speed
allowlist.  It never selects Direct mode, updates a framebuffer, changes a
header layout, addresses another zone, or sends the firmware commit command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asrock_hid_transport import NativeZoneEffect, connect_direct
from openrgb_field_config import load_or_create_config


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "openrgb_field_studio_config.json"
EXTERNAL_ZONE_TYPE = 0x07
EXTERNAL_ZONE_NAME = "Addressable Header 3/Audio"
STROBE_MODE = 0x03
KNOWN_SPEEDS = (0xE0, 0xC0, 0xA0, 0x80, 0x60, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00)


def parse_speed(text: str) -> int:
    token = text.strip().removeprefix("0x").removeprefix("0X")
    try:
        value = int(token, 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("speed must be a hexadecimal byte such as E0 or 0x20") from exc
    if value not in KNOWN_SPEEDS:
        allowed = " ".join(f"{speed:02X}" for speed in KNOWN_SPEEDS)
        raise argparse.ArgumentTypeError(f"speed must be one of: {allowed}")
    return value


def format_effect(effect: NativeZoneEffect) -> str:
    color = " ".join(f"{value:02X}" for value in effect.color_bytes)
    return f"mode=0x{effect.mode:02X} speed=0x{effect.speed:02X} raw_color={color}"


def validate_target(config: dict) -> None:
    transport = config["controller"]["transport"]
    if transport.get("kind") != "direct_asrock_hid":
        raise RuntimeError("the configured controller is not the ASRock direct HID transport")
    matches = [
        zone for zone in transport["zones"]
        if int(zone["zone_type"]) == EXTERNAL_ZONE_TYPE
    ]
    if len(matches) != 1 or str(matches[0]["name"]) != EXTERNAL_ZONE_NAME:
        raise RuntimeError(
            "refusing to write: physical zone 0x07 is not uniquely configured as "
            f"{EXTERNAL_ZONE_NAME!r}"
        )


def recovery_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".native-strobe-restore.json")


def save_recovery(path: Path, effect: NativeZoneEffect) -> None:
    payload = {
        "zone_type": EXTERNAL_ZONE_TYPE,
        "mode": effect.mode,
        "speed": effect.speed,
        "color_bytes": list(effect.color_bytes),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_recovery(path: Path) -> NativeZoneEffect:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("zone_type", -1)) != EXTERNAL_ZONE_TYPE:
        raise RuntimeError(f"recovery file {path} does not describe zone 0x07")
    color = tuple(int(value) for value in payload["color_bytes"])
    if len(color) != 3:
        raise RuntimeError(f"recovery file {path} has an invalid raw color")
    return NativeZoneEffect(int(payload["mode"]), int(payload["speed"]), color)


def apply_speed(controller, speed: int) -> None:
    effect = NativeZoneEffect(STROBE_MODE, speed, (0xFF, 0xFF, 0xFF))
    controller.write_external_zone_effect(effect)
    print(f"ARGB3 native strobe active: speed=0x{speed:02X}; observe/measure optical output now.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--speed", type=parse_speed, help="apply one allowlisted native speed byte")
    actions.add_argument("--sweep", action="store_true", help="step interactively through every allowlisted speed")
    actions.add_argument("--restore", action="store_true", help="restore a state left by an interrupted experiment")
    parser.add_argument(
        "--apply", action="store_true",
        help="authorize zone 0x07 writes; without this flag the command is read-only",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_or_create_config(config_path)
    validate_target(config)
    saved_path = recovery_path(config_path)
    wants_experiment = args.speed is not None or args.sweep

    if (wants_experiment or args.restore) and not args.apply:
        parser.error("--speed, --sweep, and --restore require --apply")
    if wants_experiment and saved_path.exists():
        raise RuntimeError(
            f"an interrupted run left {saved_path}; restore it first with --restore --apply"
        )

    controller = connect_direct(config)
    try:
        current = controller.read_external_zone_effect()
        print(f"Current ARGB3 state: {format_effect(current)}")
        if args.restore:
            if not saved_path.exists():
                raise RuntimeError(f"no recovery state exists at {saved_path}")
            previous = load_recovery(saved_path)
            controller.write_external_zone_effect(previous)
            saved_path.unlink()
            print(f"Restored ARGB3 state: {format_effect(previous)}")
            return 0
        if not wants_experiment:
            print("Read-only inspection complete; no controller state was changed.")
            return 0

        save_recovery(saved_path, current)
        restored = False
        try:
            if args.speed is not None:
                apply_speed(controller, args.speed)
                input("Press Enter to restore the captured ARGB3 state... ")
            else:
                for index, speed in enumerate(KNOWN_SPEEDS, start=1):
                    input(f"[{index}/{len(KNOWN_SPEEDS)}] Press Enter to apply speed 0x{speed:02X}... ")
                    apply_speed(controller, speed)
                    input("Measure frequency and pulse width, then press Enter for the next value... ")
        finally:
            controller.write_external_zone_effect(current)
            restored = True
            print(f"Restored captured ARGB3 state: {format_effect(current)}")
            if restored:
                saved_path.unlink(missing_ok=True)
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
