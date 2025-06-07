# Pyproject Setup

**Date/Version:** 2025-06-17 v1
**Title:** Pyproject Setup

## Overview
Created a minimal `pyproject.toml` so the `speaktome` package can be installed without manual path adjustments.

## Prompts
```
set up pyproject.toml so we don't have to fight the imports within our project please
```

## Steps Taken
1. Reviewed `AGENTS.md` instructions and coding standards.
2. Added `pyproject.toml` defining project metadata and optional dependencies.
3. Ran `python AGENTS/validate_guestbook.py`.
4. Executed `python testing/test_hub.py` to ensure tests pass.

## Observed Behaviour
- Guestbook validation reported no issues.
- Test suite succeeded with all existing tests.

## Lessons Learned
A pyproject file makes local imports easier by installing the package in editable mode or using `pip install .`.

## Next Steps
- Consider publishing to PyPI once the API stabilizes.
