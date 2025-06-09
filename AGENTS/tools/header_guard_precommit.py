try:
    """Pre-commit hook enforcing HEADER, tests, and the end sentinel."""
    import sys
    import subprocess
    import ast
    import os
    from pathlib import Path
except Exception:
    print(
        "\n*** Did you run setup_env_dev with the correct codebases? "
        "Is the virtual environment active?***\n"
    )
    raise
# --- END HEADER ---

FRIENDLY_GUIDANCE = """
# A Gentle Guide to Code Headers

Dear fellow explorer,

If the pre-commit hook caught your changes, here's a friendly checklist:

1. Every Python file needs:
   - A `HEADER` (as docstring or constant)
   - A `@staticmethod test()` method in each class
   - Imports and header content wrapped in a `try` block
   - An `except` block printing guidance about running `setup_env_dev` and activating the virtual environment
   - A `# --- END HEADER ---` sentinel after the `except` block

2. Example structure:
   ```python
   try:
       import sys
       import your_modules
   except Exception:
       print(
           "\n*** Did you run setup_env_dev with the correct codebases? "
           "Is the virtual environment active?***\n"
       )
       raise
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
        stdout=subprocess.PIPE, text=True, check=True
    )
    files = [f for f in result.stdout.splitlines() if f.endswith('.py')]
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
                isinstance(stmt, ast.Assign) and
                any(getattr(t, 'id', None) == 'HEADER' for t in stmt.targets) and
                isinstance(stmt.value, ast.Str)
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
    sentinel = "# --- END HEADER ---"
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    sentinel_line = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == sentinel), None)
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
    """Verify the header starts with ``try:`` and includes an ``except`` block."""
    sentinel = "# --- END HEADER ---"
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    try_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("try:")), None)
    sentinel_idx = next((i for i, ln in enumerate(lines) if ln.strip() == sentinel), None)
    except_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("except") and (sentinel_idx is None or i < sentinel_idx)),
        None,
    )

    errors = []
    if try_idx is None or try_idx > 1:
        errors.append("Missing 'try:' at start of header")
    if except_idx is None:
        errors.append("Missing 'except' block for header")
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
