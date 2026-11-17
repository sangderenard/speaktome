"""No module of speaktome_mcp may share a standard-library name.

An MCP client may start a server as a script (``python speaktome_mcp/api.py``); Python then puts the package folder
first on the import path, and a module there named like a standard-library module replaces it. ``http.py`` did
exactly that (2026-11-15): ``rest.py``'s ``import http.client`` found it, and every agent run that started the server
that way had no tools.
"""
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "speaktome_mcp"


def test_no_module_shadows_the_standard_library():
    names = {p.stem for p in PACKAGE.glob("*.py")} | {p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert not names & set(sys.stdlib_module_names), sorted(names & set(sys.stdlib_module_names))

