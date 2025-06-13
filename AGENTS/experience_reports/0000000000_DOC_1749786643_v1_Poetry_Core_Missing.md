# Poetry Core Missing During Setup

## Overview
Investigated why `setup_env.sh` failed with `ModuleNotFoundError: No module named 'poetry.core.masonry.api'` when installing editable packages.

## Prompts
- "investigate the reason powtry.core.masonry.api is not available, determine if we need a pre-poetry install of poetry"

## Steps Taken
1. Checked `poetry --version` (2.1.3) and verified `pip show poetry-core` reported none.
2. Re-ran `poetry install` and captured stack trace showing the missing module.
3. Installed `poetry-core` manually with `pip install poetry-core>=1.5.0`.
4. Confirmed `import poetry.core.masonry.api` succeeds.

## Observed Behaviour
- Without `poetry-core`, editable builds of `tensors` and other packages fail during setup.
- Installing `poetry-core` resolves the import error.

## Lessons Learned
- The repo's pyproject files require `poetry-core>=1.5.0` as the build backend.
- Some environments bundle `poetry` without `poetry-core`; ensure it is installed before running `setup_env.sh`.

## Next Steps
- Updated `setup_env.sh` to auto-install `poetry-core` if missing.
- Documented this requirement in `ENV_SETUP_OPTIONS.md`.
