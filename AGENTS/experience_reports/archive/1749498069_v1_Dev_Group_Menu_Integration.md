# Dev Group Menu Integration

**Date/Version:** 1749498069 v1
**Title:** Integrate dynamic group menu into dev setup scripts

## Overview
Updated developer environment setup scripts to use `AGENTS/tools/dev_group_menu.py`. The tool now supports `--install` and `--json` modes for easier automation.

## Prompts
```
fix the start_env_dev.ps1 and start_env_dev.sh scripts to use the new tool in agents/tools for obtaining the menu that individually loads the groups and only offers groups from codebases selected.
```

## Steps Taken
1. Extended `dev_group_menu.py` with argument parsing, JSON output and automatic installation capability.
2. Modified `setup_env_dev.sh` and `setup_env_dev.ps1` to call the tool with `--install` so users select codebases and groups once.
3. Updated interactive menus accordingly and ran formatting.
4. Installed missing `cffi` and `setuptools` to run tests.
5. Executed `python testing/test_hub.py` ensuring all tests pass.

## Observed Behaviour
- `dev_group_menu.py` now lists available codebases from `AGENTS/CODEBASE_REGISTRY.md` and installs chosen extras via pip.
- Tests reported `31 passed, 30 skipped`.

## Lessons Learned
Leveraging a single dynamic tool simplifies maintenance and keeps extras in sync with project configuration.

## Next Steps
- Consider documenting the new `--install` flag in repository docs.
