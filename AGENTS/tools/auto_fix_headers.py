#!/usr/bin/env python3
"""Automatically ensure standard headers across the repository."""
from __future__ import annotations

try:
    import os
    import re
    from pathlib import Path
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"
    HEADER_START = "# --- BEGIN HEADER ---"
    HEADER_END = "# --- END HEADER ---"
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


EXCLUDE_DIRS = {
    "archive",
    "third_party",
    "laplace",
    "training",
    "tensor printing",
    "tensor_printing",
}


def in_tensor_printing_inspiration(path: Path) -> bool:
    parts = path.parts
    return "tensor printing" in parts and "inspiration" in parts


HEADER_START_SENTINEL = HEADER_START
HEADER_END_SENTINEL = HEADER_END

IMPORT_RE = re.compile(r"^(from\s+\S+\s+import|import\s+\S+)")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    for d in EXCLUDE_DIRS:
        if d in parts:
            return True
    if in_tensor_printing_inspiration(path):
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
    out_lines.append("    import subprocess")
    out_lines.append("    root = Path(__file__).resolve()")
    out_lines.append("    for parent in [root, *root.parents]:")
    out_lines.append("        if (parent / 'pyproject.toml').is_file():")
    out_lines.append("            root = parent")
    out_lines.append("            break")
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
