# Audit Report

**Date:** 1749888727
**Title:** Status of Dynamic Header and Codebase Mapping Utilities

## Scope
Examine existing scripts related to dynamic header recognition and automatic codebase
mapping from `pyproject.toml`. Assess current functionality and identify missing pieces needed for
fully automated header environment detection.

## Methodology
- Searched repository for header utilities and codebase discovery scripts using `grep` and directory listings.
- Reviewed `dynamic_header_recognition.py`, `update_codebase_map.py`, and related documentation.
- Checked historical experience reports for context on prior attempts.

## Detailed Observations
- `AGENTS/tools/dynamic_header_recognition.py` contains a stub `HeaderNode` class and helper
  functions for parsing headers but lacks a full implementation. Lines 58-115 show placeholder logic
  and TODO markers.
- `AGENTS/tools/update_codebase_map.py` scans each project directory for `pyproject.toml` files
  and writes a JSON mapping of codebase paths and optional dependency groups. This mapping is stored
  in `AGENTS/codebase_map.json` for consumption by other utilities.
- `AGENTS/tools/dev_group_menu.py` uses this JSON map to present an interactive installer for
  optional dependency groups.
- Header templates rely on `ENV_SETUP_BOX` and `auto_env_setup` to bootstrap missing dependencies.
  Several tools like `ensure_pyproject_deps.py` use the `SPEAKTOME_GROUPS` environment variable
  to decide which groups to install during error handling.
- Past experience reports such as `1749847853_DOC_Dynamic_Header_Skeleton.md` record the initial
  creation of the header recognition skeleton.

## Analysis
Current utilities partly address automatic dependency installation through group detection, but
integration with header parsing remains unfinished. The `dynamic_header_recognition.py` stub lacks
logic to read a header, determine its owning codebase, and infer required groups from the
`pyproject.toml` data. The codebase map infrastructure is in place, yet individual modules do not
currently call into it when import errors occur.

## Recommendations
- Complete the parser in `dynamic_header_recognition.py` to build a tree from existing headers.
- Extend error handlers in the standard header template to call a helper that looks up the
  module's codebase and associated groups using `codebase_map.json`.
- Provide a uniform wrapper for optional imports that consults this mapping and attempts automatic
  installation via `auto_env_setup`.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created ..."
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos..."
