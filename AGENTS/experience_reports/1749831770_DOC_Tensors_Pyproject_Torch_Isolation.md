# Tensors Pyproject Torch Isolation

**Date:** 1749831770
**Title:** Check optional torch group configuration

## Overview
Verified that `tensors/pyproject.toml` isolates PyTorch using an optional `torch` group, consistent with other projects.

## Steps Taken
- Inspected `tensors/pyproject.toml` and `fontmapper/pyproject.toml`.
- Ran `python testing/test_hub.py`.

## Observed Behaviour
- `tensors` defines the group:
  ```toml
  [tool.poetry.group.torch]
  optional = true
  [tool.poetry.group.torch.dependencies]
  torch = "*"
  ```
- `fontmapper` follows the same pattern with extra dependencies.
- Test run failed because the environment was not initialized:
  ``Skipped: Environment not initialized. See ENV_SETUP_OPTIONS.md``

## Lessons Learned
Consistent optional groups avoid PyTorch installation unless explicitly requested.
Tests require the setup scripts to initialize `ENV_SETUP_BOX`.

## Next Steps
None for now.

## Prompt History
- "make sure the tensors project pyproject.toml correctly isolates torch in a group identical to other projects isolating torch as optional"
- "Do not be maliciously vague, explain what failed, there is no ordinary scenario designed to produce any network failure unless you explicitly ask for groups including torch or torch dependent content."
