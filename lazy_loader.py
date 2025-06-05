import importlib
from functools import lru_cache

@lru_cache(maxsize=None)
def lazy_import(module_name: str):
    """Import a module only when needed and cache the result."""
    return importlib.import_module(module_name)
