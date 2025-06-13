# Standard Header

The Python module `AGENTS.tools.header.Header` defines the canonical
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
    import subprocess  # <import>
    try:  # <setup-call:start>
        root = Path(__file__).resolve()
        for parent in [root, *root.parents]:
            if (parent / "pyproject.toml").is_file():
                root = parent
                break
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
