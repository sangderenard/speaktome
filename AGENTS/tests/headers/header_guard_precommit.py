#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Pre-commit hook enforcing HEADER, tests, and the end sentinel."""

from __future__ import annotations

try:
    import subprocess
    import ast
    import os
    from pathlib import Path
except Exception:
    import os
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensor printing",
            "time_sync",
            "AGENTS",
            "fontmapper",
            "tensors",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    if "ENV_SETUP_BOX" not in os.environ:
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = _find_repo_root(Path(__file__))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "AGENTS.tools.auto_env_setup",
                str(root),
            ],
            check=False,
        )
    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"
    HEADER_START = "# --- BEGIN HEADER ---"
    HEADER_END = "# --- END HEADER ---"
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

FRIENDLY_GUIDANCE = """
# A Gentle Guide to Code Headers

Dear fellow explorer,

If the pre-commit hook caught your changes, here's a friendly checklist:

1. Every Python file needs:
   - A `HEADER` (as docstring or constant)
   - A `@staticmethod test()` method in each class
    - `from __future__ import annotations` before the `try` block
    - Imports wrapped in a `try` block
    - An `except` block that imports `sys`, prints guidance to consult `ENV_SETUP_OPTIONS.md`, then calls `sys.exit(1)`
    - A `# --- END HEADER ---` sentinel after the `except` block
    - The `# --- BEGIN HEADER ---` sentinel after the shebang

2. Example structure:
   ```python
   #!/usr/bin/env python3
   # --- BEGIN HEADER ---
   \"\"\"Optional module docstring.\"\"\"
   from __future__ import annotations

   try:
       import your_modules
   except Exception:
       import sys
       print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
       print(ENV_SETUP_BOX)
       sys.exit(1)
   # --- END HEADER ---
   ```

3. Quick tips:
   - The sentinel must be exactly `# --- END HEADER ---` (three dashes each side)
   - Copy-paste the sentinel line to avoid sneaky whitespace issues
   - Place it after your last import statement

Need to bypass temporarily? Use:
    SKIP_HEADER_GUARD=1 git commit -m "your message"

For full standards, see the `AGENTS/` directory documentation.
"""


def get_staged_py_files():
    """Return a list of staged Python files (relative paths)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    files = [f for f in result.stdout.splitlines() if f.endswith(".py")]
    return files


def check_class_header_and_test(filepath):
    """Return a list of (lineno, classname, missing) for classes missing HEADER or test()."""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except Exception as e:
        return [(-1, None, f"Could not parse file: {e}")]
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for HEADER (either as a docstring or HEADER attribute)
            docstring = ast.get_docstring(node)
            has_header_attr = any(
                isinstance(stmt, ast.Assign)
                and any(getattr(t, "id", None) == "HEADER" for t in stmt.targets)
                and isinstance(stmt.value, ast.Str)
                for stmt in node.body
            )
            has_header = bool(docstring) or has_header_attr
            # Check for @staticmethod test()
            has_test = False
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "test":
                    for deco in stmt.decorator_list:
                        if isinstance(deco, ast.Name) and deco.id == "staticmethod":
                            has_test = True
            missing = []
            if not has_header:
                missing.append("HEADER")
            if not has_test:
                missing.append("@staticmethod test()")
            if missing:
                errors.append((node.lineno, node.name, ", ".join(missing)))
    return errors


def check_end_header(filepath: Path) -> list[str]:
    """Ensure ``# --- END HEADER ---`` appears after the final global import."""
    sentinel = HEADER_END
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    sentinel_line = next(
        (i + 1 for i, ln in enumerate(lines) if ln.strip() == sentinel), None
    )
    if sentinel_line is None:
        return [f"Missing sentinel '{sentinel}'"]

    try:
        tree = ast.parse("".join(lines), filename=str(filepath))
    except Exception as exc:  # pragma: no cover - parse failure
        return [f"Parse error: {exc}"]

    last_import = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = max(last_import, node.end_lineno or node.lineno)

    if sentinel_line <= last_import:
        return [f"Sentinel before last import (line {last_import})"]

    return []


def check_try_header(filepath: Path) -> list[str]:
    """Verify header elements such as shebang, docstring, and ``try`` block."""
    sentinel = HEADER_END
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    try_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("try:")), None
    )
    sentinel_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == sentinel), None
    )
    future_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("from __future__")),
        None,
    )
    except_idx = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith("except")
            and (sentinel_idx is None or i < sentinel_idx)
        ),
        None,
    )

    env_print = False
    sys_import = False
    sys_exit = False
    run_call = False
    env_check = False
    if except_idx is not None:
        search_end = sentinel_idx if sentinel_idx is not None else len(lines)
        region = lines[except_idx:search_end]
        env_print = any("print(ENV_SETUP_BOX)" in ln for ln in region)
        sys_import = any("import sys" in ln for ln in region)
        sys_exit = any("sys.exit(" in ln for ln in region)
        run_call = any("auto_env_setup" in ln for ln in region)
        env_check = any("ENV_SETUP_BOX" in ln and "not in os.environ" in ln for ln in region)

    errors = []
    if HEADER_START not in lines[:3]:
        errors.append("Missing '# --- BEGIN HEADER ---'")
    if not lines or not lines[0].startswith("#!"):
        errors.append("Missing shebang")
    try:
        tree = ast.parse("".join(lines), filename=str(filepath))
        if not ast.get_docstring(tree):
            errors.append("Missing module docstring")
    except Exception as exc:  # pragma: no cover - parse failure
        errors.append(f"Parse error: {exc}")
        return errors
    if future_idx is None or (try_idx is not None and future_idx > try_idx):
        errors.append("Missing '__future__' import before try block")
    if try_idx is None or (future_idx is not None and try_idx <= future_idx):
        errors.append("Missing 'try:' at start of header")
    if except_idx is None:
        errors.append("Missing 'except' block for header")
    else:
        if not sys_import:
            errors.append("Missing 'import sys' in except block")
        if not env_print:
            errors.append("Missing 'print(ENV_SETUP_BOX)' in except block")
        if not sys_exit:
            errors.append("Missing 'sys.exit(1)' in except block")
        if not run_call:
            errors.append("Missing call to auto_env_setup in except block")
        if not env_check:
            errors.append("Missing ENV_SETUP_BOX check in except block")
    return errors


def main():
    if os.getenv("SKIP_HEADER_GUARD"):
        return
    failed = False
    for relpath in get_staged_py_files():
        path = Path(relpath)
        if not path.exists():
            continue
        class_errors = check_class_header_and_test(path)
        header_errors = check_end_header(path) + check_try_header(path)
        for lineno, classname, missing in class_errors:
            print(
                f"[AGENT_ACTIONABLE_ERROR] {relpath}"
                f"{f':{lineno}' if lineno > 0 else ''} "
                f"Class '{classname or '?'}' missing: {missing}"
            )
            failed = True
        for msg in header_errors:
            print(f"[AGENT_ACTIONABLE_ERROR] {relpath}: {msg}")
            failed = True
    if failed:
        print(
            "\nCommit blocked: ensure each file has the try/except header, "
            "class HEADER strings, and @staticmethod test() methods."
        )
        print("\nNeed help? Here's a friendly guide:")
        print(FRIENDLY_GUIDANCE)
        sys.exit(1)


if __name__ == "__main__":
    main()
