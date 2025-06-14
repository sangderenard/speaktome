# Audit Report: TOML Group Discovery Utilities

**Date:** 1749888674
**Title:** Audit of header-aware codebase/group detection scripts

## Scope
Examine existing utilities that parse `pyproject.toml` files to automatically detect codebases and optional dependency groups. Review integration with header tools and document any gaps or improvements needed.

## Methodology
- Searched repository for scripts handling dynamic codebase or group discovery.
- Read documentation in `AGENTS.md` and referenced design proposals in `AGENTS/messages/outbox`.
- Inspected modules in `AGENTS/tools` such as `dev_group_menu.py` and `update_codebase_map.py`.

## Detailed Observations
- `AGENTS.tools.dev_group_menu` dynamically reads codebases from `AGENTS/CODEBASE_REGISTRY.md` and parses optional groups from each `pyproject.toml`. The script supports both interactive and non-interactive modes, and can install selections via `poetry` and `pip`.
- `AGENTS.tools.update_codebase_map` builds a JSON mapping of codebase paths and their groups by scanning for `pyproject.toml` files. It extracts groups whether defined under the `project` table or within Poetry's `tool.poetry.group` section.
- `AGENTS.tools.auto_env_setup` loads optional groups from the top-level `pyproject.toml` and runs the setup script for each group sequentially.
- The root `pyproject.toml` defines several optional groups (`tools`, `dev`, `cpu-torch`, `gpu-torch`, `projects`).【F:pyproject.toml†L1-L30】
- Design notes under `AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md` outline the motivation for a self-maintaining discovery system that removes hardcoded group lists.【F:AGENTS/messages/outbox/1749496178_Proposal_Dynamic_Codebase_Group_Discovery_System.md†L1-L37】

## Analysis
The repository already includes utilities that identify codebases and groups programmatically. `dev_group_menu.py` provides user-facing selection and installation logic, while `update_codebase_map.py` offers a means to cache the mapping. Together these tools enable scripts to introspect the project structure and determine which groups are available for installation. The header utilities reference these modules indirectly to avoid premature imports during environment setup.

However, integration with header templates is still minimal. The current dynamic header recognition module (`AGENTS.tools.dynamic_header_recognition`) remains mostly a stub, lacking full parsing or enforcement logic. Automating error handling based on detected groups will require expanding these modules and possibly standardizing how optional imports are wrapped.

## Recommendations
1. Expand `dynamic_header_recognition.py` to parse headers completely and expose information about optional imports.
2. Document how setup utilities should interface with header modules to determine required groups during execution.
3. Consider unifying `update_codebase_map.py` output with `dev_group_menu.py` to avoid duplication.
4. Ensure large binary artifacts are not introduced by accident (e.g., via wheels) to prevent LFS issues.

## Prompt History
- "okay, the task for us right now, then, is to look around for scripts we might have already created a while ago, there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material, it doesn't get pushed and breaks repos. ... please generate an audit experience report without implementing any changes in the commit other than that report"
- Custom instructions: "always check the files in the repo ecosystem ..."
