# User Experience Report

**Date/Version:** 2025-06-07 v5
**Title:** Aggregated To-Do Items from Prior Reports

## Overview
This summary collects all outstanding "Next Steps" from earlier experience
reports so contributors can see remaining work in one place.

## To-Do
- Continue refactoring modules to adopt the new docstring style and develop the
  mailbox-based scoring pipeline.
- Maintain PyTorch for core algorithms but design a thin tensor wrapper to allow
  NumPy fallbacks for lightweight demos.
- Prototype a `SimpleTensor` helper supporting common operations and replace
  direct `torch.topk` calls.
- Expand the CPU demo with vocabulary handling and optional scoring strategies.
- Document the correct entry command (`python -m speaktome.speaktome`) and guard
  optional imports so missing libraries route to `cpu_demo`.
- Encourage use of helper scripts for setup and consider Windows batch files.
- Investigate how NumPy alternatives could fully replace PyTorch for demos.
- Extend the generic tensor wrapper and swap implementations via CLI flag.
- Add example configurations for reinstall/automation scripts once CPU backend
  stabilizes.
- Grow testing utilities to cover more components as they stabilize.
- Trim verbose AI-generated comments and provide a short table of scoring
  functions.
- Improve unit test coverage for the LookaheadController.
- Ensure `setup_env.ps1` and `fetch_models.ps1` run before demos on Windows.
- Refactor imports so missing Transformers fall back to `cpu_demo` and expand the
  lazy loader to catch `ImportError`. Explore a single cross-platform setup
  script.
