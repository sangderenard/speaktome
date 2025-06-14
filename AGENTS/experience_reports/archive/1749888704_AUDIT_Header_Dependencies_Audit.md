# Header Dependency Automation Audit

**Date:** 1749888704
**Title:** Survey of scripts that parse pyproject files for dynamic header setup

## Scope
Examine existing utilities for detecting codebases and optional dependency groups from `pyproject.toml` files. Document how these tools integrate with header initialization and identify areas for improvement.

## Methodology
- Searched the repository for tools interacting with `pyproject.toml` or optional groups.
- Reviewed `auto_env_setup.py`, `update_codebase_map.py`, and `dev_group_menu.py` for relevant logic.
- Inspected `lazy_loader.py` for standardized optional imports.
- Consulted `AGENTS.md` and conceptual flag documents for historical context.

## Detailed Observations
- `auto_env_setup.py` contains `parse_pyproject_dependencies` which extracts optional groups from a project's `pyproject.toml` and runs the setup script with each group.
- `update_codebase_map.py` scans all codebases to create `AGENTS/codebase_map.json` mapping paths and groups.
- `dev_group_menu.py` loads this map to present interactive installation choices and records selected groups.
- Several scripts (`header_audit.py`, `ensure_pyproject_deps.py`) read the `SPEAKTOME_GROUPS` environment variable to call `auto_env_setup.py` with the necessary groups.
- `lazy_loader.py` exposes an `optional_import` helper for gracefully handling missing packages, used in modules like `speaktome/core/scorer.py`.

## Analysis
The tooling already parses project metadata to discover codebases and groups. However, header initialization still relies on manual selection via environment variables. A missing group triggers a header failure message, but there is no direct link between that failure and the correct group to install. Incorporating `parse_pyproject_dependencies` into header wrappers could automatically suggest groups or invoke `auto_env_setup.py` with defaults based on the module's location.

## Recommendations
1. Extend the header template to call a helper that determines the current codebase from its file path and consults `codebase_map.json` for required groups.
2. Integrate `optional_import` with this helper so modules can lazily import heavy dependencies while providing clear instructions when absent.
3. Document a standard pattern for optional imports across codebases to maintain consistency and reduce header errors.

## Prompt History
See conversation excerpts requesting an audit of existing scripts and guidance on dynamic dependency detection.
