# Audit Report

**Date:** 1749888705
**Title:** Search for Header Group Detection Utilities

## Scope
Investigate existing repository utilities that parse `pyproject.toml` files to automatically determine codebase membership and optional dependency groups. This functionality is needed to empower header scripts to self-detect required groups and handle optional imports safely.

## Methodology
- Scanned the `AGENTS/tools` directory for scripts related to environment setup and group selection.
- Searched for mentions of dynamic header recognition across the repository.
- Reviewed `pyproject.toml` and `AGENTS/codebase_map.json` for group definitions.
- Examined previous experience reports and TODO notes to understand prior work.

## Detailed Observations
- `AGENTS/tools/dev_group_menu.py` provides an interactive menu for selecting codebases and optional groups. It discovers codebases via `CODEBASE_REGISTRY.md` and loads group information from each project's `pyproject.toml` files using `load_codebase_map()`.
- `AGENTS/tools/update_codebase_map.py` builds `AGENTS/codebase_map.json` by scanning for `pyproject.toml` files and extracting optional dependency groups.
- `AGENTS/tools/auto_env_setup.py` parses optional dependency groups from a given `pyproject.toml` and invokes platform setup scripts for each group.
- Dynamic header utilities reside in `AGENTS.tools.dynamic_header_recognition` with a basic `HeaderNode` class and stubbed parsing functions.
- The repository's `pyproject.toml` defines several optional groups (`dev`, `cpu-torch`, `gpu-torch`, `projects`), while `codebase_map.json` lists groups for each subproject.
- Notes under `todo/dynamic_header_recognition_detailed_todo.md` outline a plan for a robust header parsing system but do not yet implement group detection.

## Analysis
Existing scripts already gather codebase and group information from `pyproject.toml` files. However, integration with header validation is minimal. The dynamic header recognition module remains a stub and does not yet reference the codebase map. A unified utility could leverage `update_codebase_map.py` to map a file path to its codebase and retrieve the relevant dependency groups. This would allow header scripts to determine required optional imports and report missing groups consistently.

## Recommendations
1. Extend `dynamic_header_recognition` to use `AGENTS/codebase_map.json` for mapping file paths to codebases.
2. Provide a helper that inspects a module's `pyproject.toml` to list groups needed for full functionality, falling back to optional imports when groups are missing.
3. Document the intended workflow in `headers.md` so new scripts understand how to detect and handle optional dependencies.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago... please generate an audit experience report without implementing any changes in the commit other than that report"
- "always check the files in the repo ecosystem for your benefit..."
