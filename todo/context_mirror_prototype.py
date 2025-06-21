# ########## STUB: context_mirror ##########
# PURPOSE: Provide a high-level summary of project state by inspecting
#          test logs, recent commits, and open tasks.
# EXPECTED BEHAVIOR: Collect information from multiple sources and
#          generate a reflective report for agents.
# INPUTS: paths to logs, git history, or todo files.
# OUTPUTS: Markdown document capturing key metrics and recommendations.
# KEY ASSUMPTIONS/DEPENDENCIES: integrates with other prototypes such as
#          clarity_engine and log_interpreter.
# TODO:
#   - Gather recent commit messages and test results.
#   - Format a consolidated status report.
#   - Hook into interactive agent workflows.
# NOTES: Captures the "Context Mirror" idea suggested by GPT-4o.
# ###########################################################################

import subprocess
from pathlib import Path


def reflect_state() -> str:
    """Return a Markdown overview of the repository's recent state.

    This implementation fetches the five most recent git commit messages and
    reports whether any pytest log files exist under ``testing/logs``. It is a
    lightweight placeholder demonstrating the envisioned behaviour of the
    Context Mirror utility.
    """

    try:
        commits = subprocess.check_output(
            ["git", "log", "-5", "--pretty=%h %s"],
            text=True,
        ).strip()
    except Exception as exc:
        commits = f"Failed to read git log: {exc}"

    log_dir = Path("testing/logs")
    logs = list(log_dir.glob("pytest_*.log")) if log_dir.exists() else []

    lines = ["## Recent Commits", commits or "(none)"]
    lines.append("\n## Pytest Logs")
    if logs:
        for path in logs[-5:]:
            lines.append(f"- {path.name}")
    else:
        lines.append("No logs found.")

    return "\n".join(lines)


if __name__ == "__main__":
    print(reflect_state())
