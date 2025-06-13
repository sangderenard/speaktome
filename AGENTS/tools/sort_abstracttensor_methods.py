#!/usr/bin/env python3
"""Sort AbstractTensor methods according to reference documentation."""
from __future__ import annotations

try:
    import ast
    import re
    from pathlib import Path
    from typing import Iterable
except Exception:  # pragma: no cover - environment guard
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:  # pragma: no cover - env missing
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def parse_reference(md_path: Path) -> list[str]:
    """Return method order from ``abstraction_functions.md``."""
    pattern = re.compile(r"-\s*(?:\w+\.)?(\w+)\(")
    order: list[str] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            order.append(m.group(1))
    return order


def sort_methods(methods: list[ast.FunctionDef], order: list[str]) -> list[ast.FunctionDef]:
    name_map = {m.name: m for m in methods}
    used: set[str] = set()
    sorted_nodes: list[ast.FunctionDef] = []
    for name in order:
        node = name_map.get(name)
        if node is not None:
            sorted_nodes.append(node)
            used.add(name)
    for m in methods:
        if m.name not in used:
            sorted_nodes.append(m)
    return sorted_nodes


def write_sorted(src: Path, dest: Path, order: list[str]) -> None:
    """Write a new file with ``AbstractTensor`` methods sorted by ``order``."""
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AbstractTensor":
            others = [n for n in node.body if not isinstance(n, ast.FunctionDef)]
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            node.body = others + sort_methods(methods, order)
            break
    dest.write_text(ast.unparse(tree), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sort AbstractTensor methods")
    parser.add_argument("--src", default="tensors/abstraction.py", type=Path)
    parser.add_argument(
        "--doc", default="tensors/abstraction_functions.md", type=Path
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Path to write sorted source file"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    order = parse_reference(args.doc)
    if args.output is None:
        for name in order:
            print(name)
    else:
        write_sorted(args.src, args.output, order)


if __name__ == "__main__":
    main()
