"""SpeakToMe beam search package."""
from __future__ import annotations

try:
    from .tensors.faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV
except Exception:
    print(
        "\n"
        "+-----------------------------------------------------------------------+\n"
        "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
        "| project and module you plan to use. Missing packages mean setup was |\n"
        "| skipped or incomplete.                                             |\n"
        "+-----------------------------------------------------------------------+\n"
    )
    raise
# --- END HEADER ---

__all__ = ["Faculty", "DEFAULT_FACULTY", "FORCE_ENV"]
