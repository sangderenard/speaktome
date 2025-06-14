# Dynamic Header Group Detection Audit

**Date:** 1749888709
**Title:** Dynamic Header Group Detection Audit

## Scope
Examine existing utilities for automatically resolving dependency groups from `pyproject.toml` to address header initialization failures. The goal is to help each script determine its codebase context and required optional groups.

## Methodology
- Searched repository for header utilities and toml parsing helpers.
- Reviewed `AGENTS/tools/auto_env_setup.py` and `ensure_pyproject_deps.py`.
- Explored `AGENTS/tools/dynamic_header_recognition.py` for future header parsing logic.
- Ran `python testing/test_hub.py` to observe environment setup behaviour.

## Detailed Observations
- `auto_env_setup.py` parses optional dependencies with `parse_pyproject_dependencies`, installing groups sequentially using the repo's setup scripts.
- `ensure_pyproject_deps.py` wraps this behaviour and can be invoked when `ENV_SETUP_BOX` is missing to bootstrap optional packages.
- The `AGENTS.tools.dynamic_header_recognition` module currently contains a minimal tree parser stub.
- Running the test hub without initialization leads to `Skipped: Environment not initialized. See ENV_SETUP_OPTIONS.md`.

## Analysis
These utilities collectively move toward automatically detecting which dependency groups are needed. However, there is no single script that reads the root `pyproject.toml` to configure headers dynamically at runtime. The header template still prints `ENV_SETUP_BOX` and exits on failure. Optional hardware-specific imports are not consistently guarded.

## Recommendations
- Expand `auto_env_setup.parse_pyproject_dependencies` to prioritize groups based on context or a mapping in `CODEBASE_REGISTRY.md`.
- Enhance `dynamic_header_recognition.py` to parse headers and suggest missing groups when an import fails.
- Integrate these helpers so header stubs can automatically invoke the correct setup groups without user intervention.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago ... It was to be an integral part of the header template that each script be empowered to find out what codebase it's in and what groups it needs for what level of function ..."
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos ..." 
