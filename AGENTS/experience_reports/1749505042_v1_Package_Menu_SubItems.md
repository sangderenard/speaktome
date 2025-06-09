# Package Menu SubItems

**Date/Version:** 1749505042 v1
**Title:** Add package-level selection to dev_group_menu

## Overview
Implemented nested prompts in `dev_group_menu.py` so users can choose individual packages within each optional dependency group. Groups and packages are dynamically read from each project's `pyproject.toml`.

## Prompts
```
the python script that allows users to select their packages to install needs to have sub-items for all groups in each project, dynamically fetched from the pyproject.toml
```

## Steps Taken
1. Updated `extract_group_packages` to return a mapping of groups to packages.
2. Modified `build_codebase_groups` and `interactive_selection` to support nested package prompts.
3. Adjusted installation logic to install selected packages individually.
4. Ran `python testing/test_hub.py` after installing `cffi` and `setuptools` to satisfy optional dependencies.

## Observed Behaviour
Tests passed: `33 passed, 30 skipped`.

## Lessons Learned
Providing package-level granularity gives developers more control over environment setup while keeping options synced with `pyproject.toml`.

## Next Steps
None.
