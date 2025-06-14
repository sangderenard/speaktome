# Dynamic Group Discovery Utilities Audit

**Date:** 1749888702
**Title:** Dynamic Group Discovery Utilities Audit

## Scope
Examine existing scripts that detect optional dependency groups from `pyproject.toml` files and interact with header utilities.

## Methodology
- Reviewed documentation under `AGENTS/experience_reports` and `AGENTS/messages/outbox` for historical context.
- Inspected current tools with `grep` and manual browsing to identify code parsing dependency groups or linking headers to environment setup.

## Detailed Observations
- `AGENTS/tools/update_codebase_map.py` scans the repository for `pyproject.toml` files and writes a JSON mapping of codebase paths and groups【F:AGENTS/tools/update_codebase_map.py†L26-L79】.
- `AGENTS/tools/dev_group_menu.py` loads this map to let users select codebases and groups for installation, dynamically extracting groups from each TOML file【F:AGENTS/tools/dev_group_menu.py†L120-L165】.
- `AGENTS/tools/auto_env_setup.py` reads optional dependency groups from the project root and invokes the setup script for each group to initialize the environment【F:AGENTS/tools/auto_env_setup.py†L60-L123】.
- Past reports describe building a minimal wheelhouse using Git LFS, but the directory `AGENTS/proposals/wheelhouse_repo` referenced in that report is absent.

## Analysis
These utilities collectively provide dynamic discovery of optional dependencies and programmatic setup. However, they are not yet fully integrated into the header template to automatically determine which groups should be installed when an import fails. The missing wheelhouse repository suggests a past attempt to store wheels via Git LFS that may have been reverted.

## Recommendations
- Develop a standard wrapper for optional imports that consults `codebase_map.json` to request missing groups via `auto_env_setup`.
- Reinstate or document the wheelhouse repository approach for offline installs without tripping Git LFS.
- Extend `dynamic_header_recognition` to note required groups per file so headers can prompt automated setup when needed.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago, there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material, it doesn't get pushed and breaks repos. anyhow I think at some point we were prepping a suite of utilities to integrate with other header utilities and a great aspiration in that is what would fix the present moment, a script capable of automatically recognizing from the toml of the project root in the monorepo which codebase and what groups from it are necessary for the present header error. It was to be an integral part of the header template that each script be empowered to find out what codebase it's in and what groups it needs for what level of function, so we need probably a way to standardize wrapping extra functionality from problematic or hardware specific optional imports. This is a complex issue so please generate an audit experience report without implementing any changes in the commit other than that report"
