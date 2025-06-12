#!/usr/bin/env python3
"""Generate a JSON map of codebase paths and optional dependency groups."""
from __future__ import annotations

try:
    import argparse
    import json
    from pathlib import Path
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

ROOT = Path(__file__).resolve().parents[2]


def discover_codebases(root: Path) -> list[Path]:
    """Return directories containing a top-level `pyproject.toml`."""
    returnvals = [p.parent for p in root.glob('*/pyproject.toml')]
    # Always include AGENTS/tools as a path:
    special_path = root / 'AGENTS' / 'tools'
    if special_path.is_dir():
        returnvals.append(special_path)
    return returnvals


def extract_groups(toml_path: Path) -> dict[str, list[str]]:
    """Read optional dependency mapping from `pyproject.toml`."""
    try:
        data = tomllib.loads(toml_path.read_text())
    except Exception:
        return {}

    groups = data.get('project', {}).get('optional-dependencies')
    if groups:
        return groups

    # Poetry style
    tool = data.get('tool', {}).get('poetry', {})
    group_section = tool.get('group', {})
    if group_section:
        out: dict[str, list[str]] = {}
        for name, meta in group_section.items():
            deps = meta.get('dependencies', {})
            items = []
            for pkg, spec in deps.items():
                if isinstance(spec, str):
                    items.append(f"{pkg}{spec if spec != '*' else ''}")
                else:
                    items.append(pkg)
            out[name] = items
        return out

    extras = tool.get('extras')
    if extras:
        return extras

    return {}


def build_map(root: Path) -> dict[str, dict[str, object]]:
    """Assemble codebase name to path and group mapping."""
    mapping: dict[str, dict[str, object]] = {}
    for cb in discover_codebases(root):
        groups = extract_groups(cb / 'pyproject.toml')
        mapping[cb.name] = {
            # Use POSIX separators for cross-platform consistency
            'path': cb.relative_to(root).as_posix(),
            'groups': groups,
        }
    return mapping


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', help='Write JSON to OUTPUT instead of stdout')
    args = parser.parse_args(argv)

    mapping = build_map(ROOT)
    text = json.dumps(mapping, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == '__main__':  # pragma: no cover - CLI entry
    raise SystemExit(main())
