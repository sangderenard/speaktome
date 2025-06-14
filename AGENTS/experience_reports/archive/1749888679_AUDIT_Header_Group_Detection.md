# Audit Report

**Date:** 1749888679
**Title:** Search for Header-Driven Dependency Recognition Utilities

## Scope
Review existing scripts that detect codebases and optional dependency groups from `pyproject.toml` files to integrate with header utilities.

## Methodology
- Scanned `AGENTS/tools/` for scripts referencing codebases, groups, or header utilities.
- Examined documentation and previous experience reports for context about Git LFS issues.
- Ran the automated test hub to observe environment setup behavior.

## Detailed Observations
- `update_codebase_map.py` builds a JSON map of codebases and their optional groups by parsing each `pyproject.toml`.
- `dev_group_menu.py` loads that map and lets users select codebases/groups interactively for environment setup.
- `auto_env_setup.py` reads the main `pyproject.toml` to determine groups to install when setting up the environment.
- `dynamic_header_recognition.py` offers a skeleton for parsing the standard header format.
- Historical reports document a Git LFS lock-in disaster caused by downloading a wheel; the repository history had to be rewritten to remove LFS pointers.
- Running `testing/test_hub.py` still fails because no codebases are selected, showing the auto-setup logic waits for selection data.

## Analysis
These utilities partially satisfy the goal of detecting which packages should install based on project metadata. The codebase map generator extracts groups cleanly, but there is no direct link from header errors to these utilities yet. Further automation could parse header output to choose relevant groups from the map and invoke environment setup.

## Recommendations
- Integrate `dynamic_header_recognition` with `update_codebase_map.py` so header analysis can request the appropriate dependency groups.
- Document the LFS incident prominently to avoid accidental wheel downloads in future automation.
- Consider enhancing `auto_env_setup.py` to reference the codebase map for multi-project installs.

## Prompt History
```
"okay, the task for us right now, then, is to look around for scripts we might have already created a while ago, there was a big disaster when I had you download a wheel and it tripped lfs and you can't upload lfs material, it doesn't get pushed and breaks repos. anyhow I think at some point we were prepping a suite of utilities to integrate with other header utilities and a great aspiration in that is what would fix the present moment, a script capable of automatically recognizing from the toml of the project root in the monorepo which codebase and what groups from it are necessary for the present header error. It was to be an integral part of the header template that each script be empowered to find out what codebase it's in and what groups it needs for what level of function, so we need probably a way to standardize wrapping extra functionality from problematic or hardware specific optional imports. This is a complex issue so please generate an audit experience report without implementing any changes in the commit other than that report"
```
