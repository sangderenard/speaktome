# Standard Header

The Python module `AGENTS.tools.headers.header_template` defines the canonical
header used across this repository.  Each line ends with sentinel comments
that mark the start or end of major sections.  Tools read these comments to
generate or validate headers programmatically.

```python
#!/usr/bin/env python3  # <shebang>
# --- BEGIN HEADER ---  # <header:start>
"""Template for SPEAKTOME module headers."""  # <docstring:start> <docstring:end>
from __future__ import annotations  # <future>

try:  # <try:start>
    import your_modules  # <import>
except Exception:  # <try:end> <except:start>
    import os  # <import>
    import sys  # <import>
    from pathlib import Path  # <import>

    def _find_repo_root(start: Path) -> Path:  # <find-root:start>
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensorprinting",
            "timesync",
            "AGENTS",
            "fontmapper",
            "tensors",
            "testenv",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    ROOT = _find_repo_root(Path(__file__))
    MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"

    def guess_codebase(path: Path, map_file: Path = MAP_FILE) -> str | None:
        """Return codebase name owning ``path``."""
        try:
            data = json.loads(map_file.read_text())
        except Exception:
            data = None

        if data:
            for name, info in data.items():
                cb_path = ROOT / info.get("path", name)
                try:
                    path.relative_to(cb_path)
                    return name
                except ValueError:
                    continue
        else:
            candidates = {
                "speaktome",
                "laplace",
                "tensorprinting",
                "timesync",
                "fontmapper",
                "tensors",
                "testenv",
                "tools",
            }
            for part in path.parts:
                if part in candidates:
                    return part

        return None

    if "ENV_SETUP_BOX" not in os.environ:  # <env-check:start>
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])  # <print-env>
        sys.exit(1)  # <env-check:end>
    import subprocess  # <import>
    try:  # <setup-call:start>
        root = _find_repo_root(Path(__file__))
        subprocess.run(
            [sys.executable, '-m', 'AGENTS.tools.auto_env_setup', str(root)],
            check=False,
        )  # <setup-call>
    except Exception:  # <setup-call:end>
        pass
    try:  # <env-fetch:start>
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]  # <env-fetch>
    except KeyError as exc:  # <env-fetch-error:start>
        raise RuntimeError("environment not initialized") from exc  # <env-fetch-error:end> <env-fetch:end>
    print(f"[HEADER] import failure in {__file__}")  # <print-failure>
    print(ENV_SETUP_BOX)  # <print-env>
    sys.exit(1)  # <exit> <except:end>
# --- END HEADER ---  # <header:end>
```

## Dynamic Header Recognition

The module `AGENTS.tools.headers.dynamic_header_recognition` provides a skeleton
implementation for parsing and comparing headers using a tree structure.
It exposes :class:`HeaderNode` and helpers like :func:`parse_header` to
serve as building blocks for future validation logic.  The helper
:func:`load_template_tree` reads ``header_template.py`` and returns a
parsed tree so other tools can compare against the canonical layout.
