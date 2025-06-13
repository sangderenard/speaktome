#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Run a sequence of repository maintenance tasks with pretty logs."""
from __future__ import annotations

try:
    import argparse
    import subprocess
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
    args = parser.parse_args(argv)

    log = PrettyLogger("meta")

    report_lines: list[str] = []

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
