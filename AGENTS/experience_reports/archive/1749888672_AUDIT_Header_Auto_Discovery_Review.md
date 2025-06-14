# Audit Report

**Date:** 2025-06-14
**Title:** Auto Discovery Utilities Overview

## Scope
Examine existing scripts that detect optional dependency groups from `pyproject.toml` files and integrate them with header utilities. Identify how these tools help scripts determine required packages and codebase context.

## Methodology
- Searched the repository for tooling related to dynamic group discovery and header processing.
- Reviewed documentation such as `AGENTS/conceptual_flags/Dynamic_Codebase_Group_Discovery.md`.
- Inspected utilities within `AGENTS/tools/` including `dev_group_menu.py`, `auto_env_setup.py`, `ensure_pyproject_deps.py`, and `update_codebase_map.py`.
- Checked references in `AGENTS/headers.md` and existing patches in `AGENTS/messages/outbox`.

## Detailed Observations
- **Dynamic group discovery** is implemented in `dev_group_menu.py`. Functions like `discover_codebases` and `extract_group_packages` read `CODEBASE_REGISTRY.md` and each codebase’s `pyproject.toml` to build a map of optional dependency groups【F:AGENTS/tools/dev_group_menu.py†L48-L101】【F:AGENTS/tools/dev_group_menu.py†L113-L149】.
- `auto_env_setup.py` parses the root `pyproject.toml` to list available groups via `parse_pyproject_dependencies` and then invokes the setup script with those groups【F:AGENTS/tools/auto_env_setup.py†L33-L69】【F:AGENTS/tools/auto_env_setup.py†L70-L112】.
- `ensure_pyproject_deps.py` calls `auto_env_setup` for every optional group defined in its own `pyproject.toml`【F:AGENTS/tools/ensure_pyproject_deps.py†L60-L88】.
- `update_codebase_map.py` scans all registered codebases and writes a JSON file mapping each codebase to its optional groups, which `dev_group_menu.py` can read for faster startup【F:AGENTS/tools/update_codebase_map.py†L21-L69】【F:AGENTS/tools/update_codebase_map.py†L70-L101】.
- The standard header in `AGENTS/headers.md` explains how scripts call `auto_env_setup` when imports fail, allowing them to bootstrap required groups automatically【F:AGENTS/headers.md†L9-L32】【F:AGENTS/headers.md†L33-L50】.

## Analysis
These utilities provide the building blocks for a self-configuring environment. When a script using the standard header encounters missing imports, it triggers `auto_env_setup`, which reads `pyproject.toml` to determine available dependency groups. `dev_group_menu.py` and `update_codebase_map.py` further automate discovery across multiple codebases, reducing manual maintenance. Together they move toward the desired capability where each script can derive its required groups directly from project configuration.

## Recommendations
- Integrate `auto_env_setup.py` more tightly with header logic so that scripts can specify which groups are essential versus optional.
- Document the environment variable `SPEAKTOME_GROUPS` usage for tests and demos.
- Continue expanding dynamic discovery to handle hardware-specific groups (e.g., `gpu-torch`) gracefully.

## Prompt History
- "there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material...we need a script capable of automatically recognizing from the toml of the project root in the monorepo which codebase and what groups from it are necessary for the present header error"
- Root `AGENTS.md` directive: "After adding or updating a report, run `python AGENTS/validate_guestbook.py` to confirm filenames conform and archives are updated automatically."
