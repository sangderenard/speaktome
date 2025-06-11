#!/usr/bin/env python3
"""Interactive dev environment setup with dynamic codebase discovery.

This script presents a menu asking which codebases you want to work on and
which optional dependency groups to install for each one.  It can simply print
the selections or install them automatically when ``--install`` is passed.  Use
``--json`` to output the selections in machine readable form.

Agents may bypass all prompts by providing ``--codebases`` and ``--groups``
arguments.  ``--list`` prints available codebases/groups while ``--show-active``
reveals the path used to record selections.  See ``--help`` for examples.
"""
from __future__ import annotations

try:
    # Standard library imports
    import argparse
    import json
    import os
    import re
    import subprocess
    import sys
    import threading
    import tempfile
    from pathlib import Path
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
except Exception:
    import sys
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    print(ENV_SETUP_BOX)
    sys.exit(1)

# Platform-specific input handling
if os.name == 'nt':  # Windows
    import msvcrt

    def getch_timeout(timeout: float) -> str | None:
        """Get a single character with timeout on Windows."""
        result = []
        # Event to signal thread completion or error
        finished_event = threading.Event()

        def _input_thread_target():
            try:
                # msvcrt.getch() blocks until a key is pressed and returns bytes.
                char_bytes = msvcrt.getch()
                # For simple y/n prompts, we expect single ASCII/UTF-8 chars.
                # Special keys (arrows, F-keys) might return multiple bytes
                # or non-decodable sequences, which errors='ignore' handles.
                if char_bytes:
                    decoded_char = char_bytes.decode('utf-8', errors='ignore')
                    if decoded_char: # Ensure decoding was successful and non-empty
                        result.append(decoded_char)
            except Exception:
                # Optionally log the exception here, e.g.:
                # import logging
                # logging.exception("Error reading character in getch_timeout")
                pass # Let the main thread handle timeout or empty result
            finally:
                finished_event.set() # Signal that the thread is done

        thread = threading.Thread(target=_input_thread_target)
        thread.daemon = True  # Allow main program to exit even if thread is running
        thread.start()

        finished_event.wait(timeout=timeout)  # Wait for the event or timeout

        if result:
            return result[0]
        return None # Timeout occurred or thread failed to get a character
# --- END HEADER ---

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "AGENTS" / "CODEBASE_REGISTRY.md"
MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"
ACTIVE_ENV = "SPEAKTOME_ACTIVE_FILE"


def discover_codebases(registry_path: Path) -> list[Path]:
    """Return valid codebase directories listed in ``CODEBASE_REGISTRY.md``."""
    pattern = re.compile(r"- \*\*(.+?)\*\*")
    codebases: list[Path] = []
    if not registry_path.exists():
        return codebases
    for line in registry_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        path = ROOT / match.group(1)
        if path.is_dir():
            codebases.append(path)
    return codebases


def extract_group_packages(toml_path: Path) -> dict[str, list[str]]:
    """Return optional dependency mapping from ``pyproject.toml``."""
    try:
        data = tomllib.loads(toml_path.read_text())
    except Exception:
        return {}
    return data.get("project", {}).get("optional-dependencies", {})


def build_codebase_groups() -> tuple[dict[str, dict[str, list[str]]], dict[str, Path]]:
    """Return mapping of codebase name to group->packages and name->path."""
    mapping: dict[str, dict[str, list[str]]] = {}
    paths: dict[str, Path] = {}
    for cb_path in discover_codebases(REGISTRY):
        groups: dict[str, list[str]] = {}
        for toml_file in cb_path.rglob("pyproject.toml"):
            groups = extract_group_packages(toml_file)
            if groups:
                break
        mapping[cb_path.name] = groups
        paths[cb_path.name] = cb_path
    return mapping, paths


def load_codebase_map(map_path: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, Path]]:
    """Load codebase mapping from ``map_path`` or fall back to discovery."""
    if map_path.exists():
        try:
            data = json.loads(map_path.read_text())
            mapping = {k: v.get("groups", {}) for k, v in data.items()}
            paths = {k: ROOT / v.get("path", k) for k, v in data.items()}
            return mapping, paths
        except Exception:
            pass
    return build_codebase_groups()


CODEBASES, CODEBASE_PATHS = load_codebase_map(MAP_FILE)


def interactive_selection() -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """Return selected codebases mapped to chosen packages."""

    selected_codebases: list[str] = []
    for cb in CODEBASES:
        if ask(f"Work on codebase '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
            selected_codebases.append(cb)

    selected: dict[str, dict[str, list[str]]] = {}
    for cb in selected_codebases:
        groups = CODEBASES[cb]
        selected[cb] = {}
        for group, pkgs in groups.items():
            if (
                ask(
                    f"Install group '{group}' for '{cb}'? [Y/n] (auto-accept in 3s): ",
                    default="y",
                )
                == "y"
            ):
                selected[cb][group] = []
                for pkg in pkgs:
                    if (
                        ask(
                            f"  Install package '{pkg}'? [y/N] (auto-skip in 3s): "
                        )
                        == "y"
                    ):
                        selected[cb][group].append(pkg)

    return selected_codebases, selected


def noninteractive_selection(
    cb_arg: str | None, group_args: list[str] | None
) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """Parse command line selections without prompting."""

    selected_codebases = cb_arg.split(',') if cb_arg else list(CODEBASES)
    for cb in selected_codebases:
        if cb not in CODEBASES:
            raise SystemExit(f"Unknown codebase: {cb}")

    selected: dict[str, dict[str, list[str]]] = {cb: {} for cb in selected_codebases}
    if group_args:
        for spec in group_args:
            if ':' not in spec:
                raise SystemExit(f"Invalid group spec: {spec}")
            cb, groups = spec.split(':', 1)
            if cb not in CODEBASES:
                raise SystemExit(f"Unknown codebase: {cb}")
            for grp in groups.split(','):
                if grp and grp in CODEBASES[cb]:
                    selected.setdefault(cb, {})[grp] = list(CODEBASES[cb][grp])
                elif grp:
                    raise SystemExit(f"Unknown group '{grp}' for {cb}")

    return selected_codebases, selected


def install_selections(
    selections: dict[str, dict[str, list[str]]], *, pip_cmd: str = "pip"
) -> None:
    """Install selected packages for each codebase using ``pip_cmd``."""

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    for cb, groups in selections.items():
        cb_path = CODEBASE_PATHS.get(cb, ROOT / cb)
        if not cb_path.is_dir():
            continue
        subprocess.run([
            pip_cmd,
            "install",
            "--no-build-isolation",
            "-e",
            str(cb_path),
        ], check=False, env=env)
        for pkgs in groups.values():
            for pkg in pkgs:
                subprocess.run([pip_cmd, "install", pkg], check=False, env=env)


def interactive_menu_selection() -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """
    Show a console menu letting the user toggle each codebase and its groups.
    Pressing the assigned letter toggles a selection. Pressing 'c' continues.
    The final result is returned as (list_of_codebases, {cb: {grp: [pkgs]}}).
    """
    selected_codebases = set()
    selected_groups: dict[str, dict[str, list[str]]] = {}

    codebases = sorted(CODEBASES.keys())
    idx_map = {}  # letter -> (type, name)  type='cb' or 'grp'

    # Assign letters to codebases and groups
    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_idx = 0
    for cb in codebases:
        idx_map[letters[letter_idx]] = ("cb", cb)
        letter_idx += 1
        for grp in sorted(CODEBASES[cb]):
            idx_map[letters[letter_idx]] = ("grp", f"{cb}:{grp}")
            letter_idx += 1
            if letter_idx >= len(letters):
                break
        if letter_idx >= len(letters):
            break

    timeout = 5.0  # Start with 5 seconds

    while True:
        print("\nSelect codebases and optional groups (press letter to toggle, 'c' to continue, or 'q' to quit):")
        for letter, (typ, name) in idx_map.items():
            if typ == "cb":
                cb = name
                mark = "[X]" if cb in selected_codebases else "[ ]"
                print(f"  ({letter}) {mark} Codebase: {cb}")
                # Indent next lines for groups
                for k, (grpTyp, grpName) in idx_map.items():
                    if grpTyp == "grp" and grpName.startswith(f"{cb}:"):
                        groupOnly = grpName.split(":", maxsplit=1)[1]
                        isSelected = (cb in selected_groups and groupOnly in selected_groups[cb])
                        grpMark = "[X]" if isSelected else "[ ]"
                        print(f"       ({k}) {grpMark} Group: {groupOnly}")

        choice = getch_timeout(timeout)
        if not choice:
            print("No input, continuing...")
            break
        
        # Any key pressed increases timeout to 10 seconds for subsequent iterations
        timeout = 10.0
        
        choice = choice.lower()
        if ord(choice) == 27:
            print("No selections made. Exiting.")
            sys.exit(0)
        if ord(choice) == 13:
            print("Continuing with current selections.")
            break
        if choice in idx_map:
            (typ, name) = idx_map[choice]
            if typ == "cb":
                if name in selected_codebases:
                    selected_codebases.remove(name)
                    selected_groups.pop(name, None)
                else:
                    selected_codebases.add(name)
                    selected_groups.setdefault(name, {})
            else:  # group
                cb, grp = name.split(":", 1)
                if cb not in selected_codebases:
                    print(f"Select codebase '{cb}' first.")
                else:
                    if grp in selected_groups[cb]:
                        selected_groups[cb].pop(grp)
                    else:
                        selected_groups[cb][grp] = CODEBASES[cb][grp]
        else:
            print(f"Unknown command: {choice}")

    final_codebases = sorted(selected_codebases)
    final_groups: dict[str, dict[str, list[str]]] = {}
    for cb in final_codebases:
        gSel = selected_groups.get(cb, {})
        final_groups[cb] = {}
        for grp, pkgs in gSel.items():
            final_groups[cb][grp] = pkgs
    return final_codebases, final_groups


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available codebases and groups then exit",
    )
    parser.add_argument(
        "--show-active",
        action="store_true",
        help="Print the active selection file path and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output selections as JSON instead of human readable text",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install selected codebases and packages using $PIP_CMD",
    )
    parser.add_argument(
        "--codebases",
        metavar="LIST",
        help="Comma-separated codebases for non-interactive mode",
    )
    parser.add_argument(
        "--groups",
        action="append",
        metavar="CB:GRP1,GRP2",
        help="Group selections for a codebase (repeatable)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="Write selections to PATH (default from SPEAKTOME_ACTIVE_FILE)",
        nargs="?",
        const=os.environ.get(ACTIVE_ENV, str(Path(tempfile.gettempdir()) / "speaktome_active.json")),
    )
    args = parser.parse_args(argv)

    if args.list:
        for cb, groups in CODEBASES.items():
            print(cb)
            for grp in groups:
                print(f"  - {grp}")
        return

    if args.show_active:
        print(args.record or os.environ.get(ACTIVE_ENV, str(Path(tempfile.gettempdir()) / "speaktome_active.json")))
        return

    # If user passed --codebases/--groups on CLI, skip menu. Otherwise, show it.
    if args.codebases or args.groups:
        cbs, selections = noninteractive_selection(args.codebases, args.groups)
    else:
        cbs, selections = interactive_menu_selection()

    if args.install:
        pip_cmd = os.environ.get("PIP_CMD", "pip")
        install_selections(selections, pip_cmd=pip_cmd)

    if args.record:
        path = Path(args.record)
        try:
            path.write_text(json.dumps({"codebases": cbs, "packages": selections}))
        except OSError as exc:  # pragma: no cover - file write errors
            print(f"[WARN] Could not write selections to {path}: {exc}")

    if args.json:
        print(json.dumps({"codebases": cbs, "packages": selections}))
    else:
        print("Selected codebases:", cbs)
        print("Selected packages:", selections)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
