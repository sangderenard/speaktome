# Agents Tools Codebase

**Date/Version:** 1749445669 v1
**Title:** Agents Tools Codebase

## Overview
Moved the repository's agent tooling metadata into the `AGENTS/tools` directory and formally registered it as its own codebase.

## Prompts
- "move the root level pyproject.toml to AGENTS/tools where you will also need to make an init file and then register the codebase AGENTS/tools"

## Steps Taken
1. Relocated `pyproject.toml` into `AGENTS/tools/`.
2. Created `AGENTS/tools/__init__.py` with a minimal header.
3. Updated `ensure_pyproject_deps.py` to look for the file in its new location.
4. Added `AGENTS/tools` to `CODEBASE_REGISTRY.md`.
5. Fixed `time_sync` export to satisfy tests.
6. Ran `python testing/test_hub.py` to verify all tests pass.

## Observed Behaviour
All tests passed after updating the package export.

## Lessons Learned
The tools directory was previously a namespace package without an `__init__` file. Explicitly adding one helps clarify its intent and simplifies future extensions.

## Next Steps
Continue ensuring all helper scripts remain discoverable under this directory and document future additions in the registry.
