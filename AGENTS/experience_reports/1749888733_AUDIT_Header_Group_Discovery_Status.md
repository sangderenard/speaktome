# Audit Report

**Date:** 1749888733
**Title:** Header_Group_Discovery_Status

## Scope
Examine existing scripts for detecting codebases and dependency groups from `pyproject.toml` files, particularly for automating header error resolution.

## Methodology
- Searched the repository for utilities referencing `pyproject.toml` or optional dependency groups.
- Reviewed documentation in `AGENTS/conceptual_flags` and prior experience reports.
- Inspected modules under `AGENTS/tools` that handle environment setup and header validation.

## Detailed Observations
- `AGENTS/tools/update_codebase_map.py` builds `AGENTS/codebase_map.json` by scanning each codebase for `pyproject.toml` and extracting optional dependency groups.
- `AGENTS/tools/dev_group_menu.py` loads this map to offer interactive installation of groups per codebase.
- `AGENTS/tools/auto_env_setup.py` parses a project's `pyproject.toml` to determine group order when running setup scripts.
- `AGENTS/tools/dynamic_header_recognition.py` provides a skeleton for parsing and comparing header structures but does not currently infer groups.
- Documentation under `AGENTS/conceptual_flags/Dynamic_Codebase_Group_Discovery.md` outlines the design for dynamic discovery using the above scripts.

## Analysis
While codebase and group information can be derived via `update_codebase_map.py` and used by the menu and setup scripts, no single tool links a script's header to its owning codebase or automatically applies the correct groups when the header fails. Header recognition utilities exist, but integration with group detection remains unfinished.

## Recommendations
- Extend `dynamic_header_recognition.py` or a new helper to resolve a script's codebase from its path and consult `codebase_map.json` for required groups.
- Integrate this helper into the header template's except block to prompt automatic installation when imports fail.
- Provide tests ensuring the detection works across all registered codebases.

## Prompt History
- "okay, the task for us right now..."
- "always check the files in the repo ecosystem for your benefit..."
