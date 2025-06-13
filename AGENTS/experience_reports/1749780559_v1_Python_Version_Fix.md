# Experience Report: Python Version Fix

**Date/Version:** 1749780559 v1
**Title:** Adjust Python requirements for numpy

## Overview
Addressed user feedback that previous Python requirements were arbitrary and caused
`poetry install` to fail due to numpy's version constraints.

## Prompts
- Root `AGENTS.md` reminder to run setup scripts and record findings.
- User instruction: "the requirements for python were invented by ai working on this, they have no meaning, your task you must complete is to fix that, find the right versions, do what we need to do to get everything to work."

## Steps Taken
1. Reviewed `pyproject.toml` files across all packages.
2. Set `python = ">=3.10"` to match numpy's needs.
3. Restricted numpy dependencies to `<2.3` to avoid Python 3.11+ constraints.
4. Re-ran `bash setup_env_dev.sh -n` to verify solver errors disappeared (still blocked by network restrictions).

## Observed Behaviour
- Poetry previously reported conflicts where numpy required Python >=3.9.
- After updating requirements, dependency resolution progressed further but network
  access prevented full installation.

## Lessons Learned
Explicit version ranges must reflect actual upstream packages. Aligning Python and
numpy versions eliminates solver conflicts.

## Next Steps
- Retry full environment setup once network access permits downloading packages.
