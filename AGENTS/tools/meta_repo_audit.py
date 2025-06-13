#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Run a sequence of repository maintenance tasks with pretty logs."""
from __future__ import annotations

try:
    import argparse
    import json
    import subprocess
    import time
    from pathlib import Path
    from AGENTS.tools.pretty_logger import PrettyLogger
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def sanitize_filename(name: str) -> str:
    """Sanitize a name for safe filesystem usage."""
    return name.replace(" ", "_").replace("/", "_")


def load_identity(path: Path) -> dict[str, object] | None:
    """Load an agent identity JSON file if it exists."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if content.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    try:
        return json.loads(content)
    except Exception:
        return None


def prompt_identity() -> tuple[Path, dict[str, object]]:
    """Interactively collect identity details and write a JSON file."""
    print("=== Create Agent Identity ===")
    name = input("Agent name: ").strip() or "UnknownAgent"
    date_of_identity = str(int(time.time()))
    nature_of_identity = input(
        "Nature of identity (human/llm/script/hybrid/system_utility): "
    ).strip() or "script"
    entry_point = input("Entry point (optional): ").strip()
    description = input("Description (optional): ").strip()
    created_by = input("Created by (optional): ").strip()
    tags = input("Tags (comma-separated, optional): ").strip()
    notes = input("Notes (optional): ").strip()

    data: dict[str, object] = {
        "name": name,
        "date_of_identity": date_of_identity,
        "nature_of_identity": nature_of_identity,
    }
    if entry_point:
        data["entry_point"] = entry_point
    if description:
        data["description"] = description
    if created_by:
        data["created_by"] = created_by
    if tags:
        data["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if notes:
        data["notes"] = notes

    dest = Path("AGENTS/users")
    dest.mkdir(parents=True, exist_ok=True)
    filename = f"{date_of_identity}_{sanitize_filename(name)}.json"
    path = dest / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"Agent profile saved to {path}")
    return path, data


def run(cmd: list[str] | str, logger: PrettyLogger, report: list[str], cwd: Path | None = None) -> int:
    """Execute ``cmd`` and log output."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    logger.info(f"$ {' '.join(cmd if isinstance(cmd, list) else [cmd])}")
    report.append(f"$ {' '.join(cmd if isinstance(cmd, list) else [cmd])}")
    if proc.stdout:
        for line in proc.stdout.splitlines():
            logger.info(line)
            report.append(line)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            logger.info(line)
            report.append(line)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("AGENTS/experience_reports"))
    parser.add_argument("--identity", type=Path, help="Path to agent identity JSON")
    args = parser.parse_args(argv)

    identity_path = args.identity
    identity = None
    if identity_path:
        identity = load_identity(identity_path)

    if identity is None:
        resp = input("Agent identity JSON path (leave blank to create): ").strip()
        if resp:
            identity_path = Path(resp)
            identity = load_identity(identity_path)

    if identity is None:
        identity_path, identity = prompt_identity()

    log = PrettyLogger("meta")

    report_lines: list[str] = []

    with log.context("Agent Identity"):
        log.info(f"file: {identity_path}")
        report_lines.append(f"identity_file: {identity_path}")
        for key, value in identity.items():
            log.info(f"{key}: {value}")
            report_lines.append(f"{key}: {value}")

    with log.context("Update codebase map"):
        run(["python", "AGENTS/tools/update_codebase_map.py", "-o", "AGENTS/codebase_map.json"], log, report_lines)

    with log.context("Auto fix headers"):
        run(["python", "AGENTS/tools/auto_fix_headers.py"], log, report_lines)

    with log.context("Validate headers"):
        run(["python", "AGENTS/tools/validate_headers.py", ".", "--rewrite"], log, report_lines)

    with log.context("Update pyproject deps"):
        for root in Path(".").glob("*/pyproject.toml"):
            for py in Path(root.parent).rglob("*.py"):
                run([
                    "python",
                    "AGENTS/tools/add_imports_to_pyproject.py",
                    str(py),
                    "--pyproject",
                    str(root),
                ], log, report_lines)

    with log.context("Generate docstring maps"):
        for root in Path(".").glob("*/pyproject.toml"):
            for py in Path(root.parent).rglob("*.py"):
                run(["python", "AGENTS/tools/docstring_map.py", str(py)], log, report_lines)

    with log.context("Find stubs"):
        run(["python", "AGENTS/tools/stubfinder.py"], log, report_lines)

    with log.context("Update AGENTS TOC"):
        run(["python", "AGENTS/tools/update_agents_toc.py"], log, report_lines)

    with log.context("Run tests"):
        run(["python", "testing/test_hub.py"], log, report_lines)

    report_path = args.report / "DOC_Meta_Audit.md"
    report_path.write_text("\n".join(report_lines))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
