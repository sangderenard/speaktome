#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    import your_modules
except Exception:
    import os
    import sys
    from pathlib import Path
    import json

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensorprinting",
            "timesync",
            "AGENTS",
            "fontmapper",
            "tensors",
            "testenv",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    ROOT = _find_repo_root(Path(__file__))
    MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"

    def guess_codebase(path: Path, map_file: Path = MAP_FILE) -> str | None:
        """Return codebase name owning ``path``.

        Tries ``codebase_map.json`` first, then falls back to scanning path
        components for known codebase names.
        """
        try:
            data = json.loads(map_file.read_text())
        except Exception:
            data = None

        if data:
            for name, info in data.items():
                cb_path = ROOT / info.get("path", name)
                try:
                    path.relative_to(cb_path)
                    return name
                except ValueError:
                    continue
        else:
            candidates = {
                "speaktome",
                "laplace",
                "tensorprinting",
                "timesync",
                "fontmapper",
                "tensors",
                "testenv",
                "tools",
            }
            for part in path.parts:
                if part in candidates:
                    return part

        return None

    def parse_pyproject_groups(pyproject: Path) -> list[str]:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # Python < 3.11
                import tomli as tomllib
        except Exception:
            return []
        try:
            data = tomllib.loads(pyproject.read_text())
        except Exception:
            return []

        groups = set()
        groups.update(data.get("project", {}).get("optional-dependencies", {}).keys())
        tool = data.get("tool", {}).get("poetry", {})
        groups.update(tool.get("group", {}).keys())
        groups.update(tool.get("extras", {}).keys())
        return sorted(groups)

    if "ENV_SETUP_BOX" not in os.environ:
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = _find_repo_root(Path(__file__))
        pyproject = root / "pyproject.toml"
        groups = parse_pyproject_groups(pyproject)
        codebase = guess_codebase(Path(__file__))
        base_cmd = [
            sys.executable,
            "-m",
            "AGENTS.tools.auto_env_setup",
            str(root),
        ]
        if codebase:
            base_cmd.append(f"-codebases={codebase}")
        subprocess.run(base_cmd, check=False)
        for grp in groups:
            subprocess.run(base_cmd + [f"-groups={grp}"], check=False)

    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"[HEADER] import failure in {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


EXCLUDE_DIRS = {
    "archive",
    "third_party",
    "laplace",
    "training",
    "tensorprinting",
    "tensorprinting",
}

HEADER_START = "# --- BEGIN HEADER ---"
HEADER_END = "# --- END HEADER ---"
IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"


def in_tensorprinting_inspiration(path: Path) -> bool:
    parts = path.parts
    return "tensorprinting" in parts and "inspiration" in parts


HEADER_START_SENTINEL = HEADER_START
HEADER_END_SENTINEL = HEADER_END

IMPORT_RE = re.compile(r"^(from\s+\S+\s+import|import\s+\S+)")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    for d in EXCLUDE_DIRS:
        if d in parts:
            return True
    if in_tensorprinting_inspiration(path):
        return True
    return False


def iter_py_files(root: Path):
    """Yield Python files skipping directories in ``EXCLUDE_DIRS``."""
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).parts
        if any(d in parts for d in EXCLUDE_DIRS):
            dirnames[:] = []
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def fix_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    # Skip when duplicate header sentinels are present to avoid infinite loops
    if text.count(HEADER_START_SENTINEL) > 1 or text.count(HEADER_END_SENTINEL) > 1:
        print(f"[auto_fix_headers] duplicate header in {path}")
        return
    if HEADER_END_SENTINEL in text:
        if all(
            token in text
            for token in (
                "import sys",
                "import os",
                "ENV_SETUP_BOX = os.environ[\"ENV_SETUP_BOX\"]",
                "print(ENV_SETUP_BOX)",
                "sys.exit(1)",
                IMPORT_FAILURE_PREFIX,
                HEADER_START_SENTINEL,
            )
        ):
            return
        lines = text.splitlines()
        sentinel_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip() == HEADER_END_SENTINEL),
            None,
        )
        if sentinel_idx is None:
            return
        except_idx = None
        try_idx = None
        for i in range(sentinel_idx - 1, -1, -1):
            if lines[i].strip().startswith("except"):
                except_idx = i
                break
        for i in range(sentinel_idx):
            if lines[i].strip().startswith("try:"):
                try_idx = i
                break
        if except_idx is None:
            return
        insert_idx = except_idx + 1
        indent = " " * (len(lines[except_idx]) - len(lines[except_idx].lstrip()) + 4)
        region = lines[except_idx:sentinel_idx]
        modified = False
        header_lines = [
            f"{indent}import sys",
            f"{indent}print(f'{IMPORT_FAILURE_PREFIX} {{__file__}}')",
            f"{indent}print(ENV_SETUP_BOX)",
            f"{indent}sys.exit(1)",
        ]
        if lines[insert_idx:sentinel_idx] != header_lines:
            lines[insert_idx:sentinel_idx] = header_lines
            modified = True
        if try_idx is not None:
            try_region = lines[try_idx:except_idx]
            if not any(
                "ENV_SETUP_BOX = os.environ[\"ENV_SETUP_BOX\"]" in ln
                for ln in try_region
            ):
                indent_try = " " * (
                    len(lines[try_idx]) - len(lines[try_idx].lstrip()) + 4
                )
                lines.insert(try_idx + 1, f"{indent_try}import os")
                lines.insert(
                    try_idx + 2,
                    f"{indent_try}ENV_SETUP_BOX = os.environ['ENV_SETUP_BOX']",
                )
                modified = True
        if HEADER_START_SENTINEL not in lines[:3]:
            insert_idx = 1 if lines and lines[0].startswith("#!") else 0
            lines.insert(insert_idx, HEADER_START_SENTINEL)
            modified = True
        if modified:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = text.splitlines()

    out_lines: list[str] = []
    idx = 0

    # Preserve shebang
    if lines and lines[0].startswith("#!"):
        out_lines.append(lines[0])
        idx = 1
    out_lines.append(HEADER_START_SENTINEL)

    # Capture leading comments or encoding declarations
    while (
        idx < len(lines)
        and lines[idx].startswith("#")
        and not IMPORT_RE.match(lines[idx])
    ):
        out_lines.append(lines[idx])
        idx += 1

    # Capture module docstring
    if idx < len(lines) and (
        lines[idx].startswith('"""') or lines[idx].startswith("'''")
    ):
        quote = lines[idx][:3]
        out_lines.append(lines[idx])
        idx += 1
        while idx < len(lines):
            out_lines.append(lines[idx])
            if lines[idx].endswith(quote) and len(lines[idx]) >= 3:
                idx += 1
                break
            idx += 1

    out_lines.append("from __future__ import annotations")
    out_lines.append("")
    out_lines.append("try:")

    # Move imports into try block
    while idx < len(lines):
        line = lines[idx]
        if IMPORT_RE.match(line):
            out_lines.append("    " + line)
            idx += 1
            continue
        if not line.strip():
            out_lines.append("    " + line)
            idx += 1
            continue
        break

    out_lines.append("except Exception:")
    out_lines.append("    import os")
    out_lines.append("    import sys")
    out_lines.append("    from pathlib import Path")
    out_lines.append("    def _find_repo_root(start: Path) -> Path:")
    out_lines.append("        current = start.resolve()")
    out_lines.append("        required = {")
    out_lines.append("            'speaktome',")
    out_lines.append("            'laplace',")
    out_lines.append("            'tensorprinting',")
    out_lines.append("            'timesync',")
    out_lines.append("            'AGENTS',")
    out_lines.append("            'fontmapper',")
    out_lines.append("            'tensors',")
    out_lines.append("        }")
    out_lines.append("        for parent in [current, *current.parents]:")
    out_lines.append("            if all((parent / name).exists() for name in required):")
    out_lines.append("                return parent")
    out_lines.append("        return current")
    out_lines.append("    if 'ENV_SETUP_BOX' not in os.environ:")
    out_lines.append("        root = _find_repo_root(Path(__file__))")
    out_lines.append("        box = root / 'ENV_SETUP_BOX.md'")
    out_lines.append("        try:")
    out_lines.append("            os.environ['ENV_SETUP_BOX'] = f\"\\n{box.read_text()}\\n\"")
    out_lines.append("        except Exception:")
    out_lines.append("            os.environ['ENV_SETUP_BOX'] = 'environment not initialized'")
    out_lines.append("        print(os.environ['ENV_SETUP_BOX'])")
    out_lines.append("        sys.exit(1)")
    out_lines.append("    import subprocess")
    out_lines.append("    root = _find_repo_root(Path(__file__))")
    out_lines.append(
        "    subprocess.run([sys.executable, '-m', 'AGENTS.tools.auto_env_setup', str(root)], check=False)"
    )
    out_lines.append("    ENV_SETUP_BOX = os.environ['ENV_SETUP_BOX']")
    out_lines.append(f"    print(f'{IMPORT_FAILURE_PREFIX} {{__file__}}')")
    out_lines.append("    print(ENV_SETUP_BOX)")
    out_lines.append("    sys.exit(1)")
    out_lines.append(HEADER_END_SENTINEL)

    out_lines.extend(lines[idx:])
    Path(path).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(".")
    for path in iter_py_files(root):
        if should_skip(path):
            continue
        fix_file(path)


if __name__ == "__main__":
    main()
