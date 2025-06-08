#!/usr/bin/env python3
"""Update the AGENTS.md table of contents.

Reads ``markdown_descriptions.json`` and rewrites the section between
``<!-- TOC START -->`` and ``<!-- TOC END -->`` in ``AGENTS/AGENTS.md``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- END HEADER ---

START_TAG = "<!-- TOC START -->"
END_TAG = "<!-- TOC END -->"


def load_descriptions(path: Path) -> dict[str, str]:
    """Return filename-to-description mapping."""
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def generate_toc(agents_dir: Path, desc: dict[str, str]) -> str:
    """Create a Markdown bullet list for files in ``agents_dir``."""
    lines = []
    for md_file in sorted(agents_dir.glob("*.md")):
        name = md_file.name
        description = desc.get(name, "unknown")
        lines.append(f"- [{name}]({name}) - {description}")
    return "\n".join(lines)


def replace_toc(md_path: Path, toc: str) -> None:
    """Insert ``toc`` between placeholder tags in ``md_path``."""
    text = md_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"{START_TAG}.*?{END_TAG}", re.DOTALL)
    new_section = f"{START_TAG}\n{toc}\n{END_TAG}"
    if pattern.search(text):
        text = pattern.sub(new_section, text)
    else:
        text = f"{new_section}\n\n{text}"
    md_path.write_text(text, encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Directory containing AGENTS markdown files",
    )
    parser.add_argument(
        "--desc",
        type=Path,
        default=Path(__file__).parent / "markdown_descriptions.json",
        help="JSON file with markdown descriptions",
    )
    args = parser.parse_args()

    descriptions = load_descriptions(args.desc)
    toc = generate_toc(args.agents_dir, descriptions)
    md_path = args.agents_dir / "AGENTS.md"
    replace_toc(md_path, toc)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
