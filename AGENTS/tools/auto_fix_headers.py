#!/usr/bin/env python3
"""Automatically ensure standard headers across the repository."""
from __future__ import annotations

try:
    import re
    from pathlib import Path
    from .header_utils import ENV_SETUP_BOX
except Exception:
    import sys

    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


EXCLUDE_DIRS = {
    "archive",
    "third_party",
    "laplace",
    "training",
}


def in_tensor_printing_inspiration(path: Path) -> bool:
    parts = path.parts
    return "tensor printing" in parts and "inspiration" in parts


HEADER_SENTINEL = "# --- END HEADER ---"

IMPORT_RE = re.compile(r"^(from\s+\S+\s+import|import\s+\S+)")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    for d in EXCLUDE_DIRS:
        if d in parts:
            return True
    if in_tensor_printing_inspiration(path):
        return True
    return False


def fix_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if HEADER_SENTINEL in text:
        if all(
            token in text
            for token in ("import sys", "print(ENV_SETUP_BOX)", "sys.exit(1)")
        ):
            return
        lines = text.splitlines()
        sentinel_idx = lines.index(HEADER_SENTINEL)
        except_idx = None
        for i in range(sentinel_idx - 1, -1, -1):
            if lines[i].strip().startswith("except"):
                except_idx = i
                break
        if except_idx is None:
            return
        insert_idx = except_idx + 1
        indent = " " * (len(lines[except_idx]) - len(lines[except_idx].lstrip()) + 4)
        region = lines[except_idx:sentinel_idx]
        block: list[str] = []
        if not any("import sys" in ln for ln in region):
            block.append(f"{indent}import sys")
        if not any("print(ENV_SETUP_BOX)" in ln for ln in region):
            block.append(f"{indent}print(ENV_SETUP_BOX)")
        if not any("sys.exit(1)" in ln for ln in region):
            block.append(f"{indent}sys.exit(1)")
        if block:
            lines[insert_idx:insert_idx] = block
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = text.splitlines()

    out_lines: list[str] = []
    idx = 0

    # Preserve shebang
    if lines and lines[0].startswith("#!"):
        out_lines.append(lines[0])
        idx = 1

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
    out_lines.append("    import sys")
    out_lines.append("    print(ENV_SETUP_BOX)")
    out_lines.append("    sys.exit(1)")
    out_lines.append(HEADER_SENTINEL)

    out_lines.extend(lines[idx:])
    Path(path).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(".")
    for path in root.rglob("*.py"):
        if should_skip(path):
            continue
        fix_file(path)


if __name__ == "__main__":
    main()
