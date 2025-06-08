import importlib
import subprocess
import sys
from functools import lru_cache
# --- END HEADER ---

@lru_cache(maxsize=None)
def lazy_import(module_name: str):
    """Import a module only when needed and cache the result."""
    return importlib.import_module(module_name)


@lru_cache(maxsize=None)
def optional_import(module_name: str):
    """Attempt to import a module, returning ``None`` if it is missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


@lru_cache(maxsize=None)
def lazy_install(module_name: str, package_name: str | None = None):
    """Import a module, installing it with pip if it's missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg = package_name or module_name.split('.')[0]
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        return importlib.import_module(module_name)
