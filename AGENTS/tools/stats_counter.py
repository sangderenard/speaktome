#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Collect repository line statistics with a friendly summary."""
from __future__ import annotations

try:
    import re
    from collections import Counter
    from pathlib import Path
except Exception:
    import os
    import sys
    from pathlib import Path

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
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

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
        subprocess.run(
            [sys.executable, "-m", "AGENTS.tools.auto_env_setup", str(root)],
            check=False,
        )
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

CURSE_WORDS = ["fuck", "fucking", "shit", "damn", "bitch"]


def find_repo_root(start: Path) -> Path:
    """Return repository root from a starting path."""
    for parent in [start, *start.parents]:
        if (parent / "AGENTS").exists() and (parent / "speaktome").exists():
            return parent
    return start


def gather_files(pattern: str, *, exclude_archives: bool) -> list[Path]:
    root = find_repo_root(Path(__file__).resolve())
    files = []
    for path in root.rglob(pattern):
        if exclude_archives and any(p in {"archive", "BACKUPS"} for p in path.parts):
            continue
        files.append(path)
    return files


def count_lines(paths: list[Path]) -> tuple[int, int]:
    total = 0
    unique: set[str] = set()
    for p in paths:
        try:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                total += 1
                unique.add(line)
        except Exception:
            continue
    return total, len(unique)


def count_curses(paths: list[Path]) -> Counter:
    patt = re.compile("|".join(fr"\b{re.escape(w)}\b" for w in CURSE_WORDS), re.IGNORECASE)
    hits = Counter()
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for w in patt.findall(text):
            hits[w.lower()] += 1
    return hits


def main(argv: list[str] | None = None) -> None:
    root = find_repo_root(Path(__file__).resolve())

    py_paths = gather_files("*.py", exclude_archives=True)
    py_total, py_unique = count_lines(py_paths)

    md_paths = list(root.glob("AGENTS*.md"))
    for p in (root / "AGENTS").rglob("*.md"):
        if "experience_reports" in p.parts:
            continue
        md_paths.append(p)
    md_total, md_unique = count_lines(md_paths)

    all_py_paths = gather_files("*.py", exclude_archives=False)
    py_all_total, py_all_unique = count_lines(all_py_paths)

    curses = count_curses(all_py_paths)
    f_bombs = curses.get("fuck", 0) + curses.get("fucking", 0)

    lines = [
        "\U0001F41B Ticking quietly in the workshop:",
        f"  - Python lines: {py_total} total / {py_unique} unique",
        f"  - Markdown guides: {md_total} total / {md_unique} unique",
        f"  - Python with history: {py_all_total} total / {py_all_unique} unique",
    ]
    if f_bombs > 5:
        lines.append(f"  - Curses spotted: {sum(curses.values())} (f-bombs: {f_bombs})")
    print("\n".join(lines))


class StatsCounter:
    """Expose :func:`main` for header tests."""

    HEADER = "Repository statistics counter"

    @staticmethod
    def test() -> None:
        main([])


if __name__ == "__main__":
    main()
