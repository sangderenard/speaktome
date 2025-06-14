"""Header utilities and templates."""

__all__ = [
    "auto_fix_headers",
    "dump_headers",
    "dynamic_header_recognition",
    "header",
    "header_audit",
    "header_guard_precommit",
    "header_template",
    "header_utils",
    "run_header_checks",
    "test_all_headers",
    "validate_headers",
]

from importlib import import_module


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)

