# Auto Group Detection Scripts Audit

**Date:** 1749888668
**Title:** Auto Group Detection Scripts Audit

## Scope
Survey utilities that examine `pyproject.toml` files to determine which codebases and optional dependency groups are available. Evaluate how these scripts might integrate with the standard header to resolve missing dependencies automatically.

## Methodology
- Ran `grep` and `find` across `AGENTS/tools` for references to optional groups and header utilities.
- Inspected `auto_env_setup.py`, `dev_group_menu.py`, `update_codebase_map.py`, and `dynamic_header_recognition.py`.
- Reviewed the root `pyproject.toml` to confirm group definitions.

## Detailed Observations
- `auto_env_setup.py` includes `parse_pyproject_dependencies` to list optional groups and `run_setup_script` to invoke `setup_env.sh` with each group sequentially.
- `dev_group_menu.py` discovers codebase directories from `CODEBASE_REGISTRY.md` and maps each to its optional dependency groups via `build_codebase_groups`.
- `update_codebase_map.py` generates a JSON file describing codebase paths and groups for external tools.
- `dynamic_header_recognition.py` defines a skeleton parser using `HeaderNode` that can detect sections of the standard header.
- The root `pyproject.toml` lists groups like `tools`, `dev`, `cpu-torch`, `gpu-torch`, and `projects` containing subproject paths.

## Analysis
Existing scripts collectively provide the pieces needed to detect optional groups and link them to codebases. However, the header template currently lacks automated logic for determining which groups a module requires. Wrapping these utilities into a lightweight library would allow the header to resolve optional dependencies contextually.

## Recommendations
1. Unify group discovery and environment setup into a shared module that headers can import after environment initialization.
2. Rely on the JSON mapping from `update_codebase_map.py` to identify the current codebase and its available groups.
3. Extend `dynamic_header_recognition.py` to verify that modules specify required groups in their header metadata and attempt installation when missing.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago, there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material, it doesn't get pushed and breaks repos. anyhow I think at some point we were prepping a suite of utilities to integrate with other header utilities and a great aspiration in that is what would fix the present moment, a script capable of automatically recognizing from the toml of the project root in the monorepo which codebase and what groups from it are necessary for the present header error. It was to be an integral part of the header template that each script be empowered to find out what codebase it's in and what groups it needs for what level of function, so we need probably a way to standardize wrapping extra functionality from problematic or hardware specific optional imports. This is a complex issue so please generate an audit experience report without implementing any changes in the commit other than that report"
