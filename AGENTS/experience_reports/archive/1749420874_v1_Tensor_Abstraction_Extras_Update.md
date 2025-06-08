# Template User Experience Report

**Date/Version:** 1749420874 v1
**Title:** Tensor Abstraction Extras Update

## Overview
Add optional dependencies for each tensor backend in `pyproject.toml` except torch which remains installed via `setup_env`.

## Prompts
- "modify pyproject to accomodate requirements through all the tensor abstraction types, with the exception of torch, which needs to retain its careful selection through the setpu_env process."

## Steps Taken
1. Edited `pyproject.toml` to define extras `numpy`, `jax`, and `ctensor`.
2. Ran `pytest -q` which failed due to missing torch.
3. Attempted to `pip install torch==1.13.1+cpu` but network restrictions blocked the download.

## Observed Behaviour
- Test collection error for `tests/test_ngram_diversity.py` because `torch` was unavailable.
- Installation attempt reported `Tunnel connection failed: 403 Forbidden`.

## Lessons Learned
Without torch installed certain tests fail immediately. Optional extras ensure other backends can be installed easily.

## Next Steps
Consider providing CPU torch wheels in the repo or adjust tests to skip when torch is missing.
