# User Experience Report

**Date/Version:** 2025-06-07 v4
**Title:** Torch/Transformers Flag Investigation and Graceful Failure Gaps

## Overview
Researched whether the project can fully bypass heavy dependencies when only the CPU demo is desired. The goal was a one-step setup after cloning without manually downloading models or installing PyTorch/Transformers.

## Steps Taken
1. Reviewed previous reports describing the CPU demo (`v3_CPU_Demo_Mode` and `v4_CPU_Demo_Fallback_Issue`).
2. Examined `speaktome.py`, `lazy_loader.py`, and `scorer.py` for dependency checks.
3. Attempted to run `run` on a clean Windows machine without installing packages.
4. Observed the `ModuleNotFoundError` for `transformers.GPT2LMHeadModel` as documented in the prior report.

## Observed Behaviour
- Importing `transformers` happens at module import time before the CPU fallback logic runs.
- If PyTorch is installed but Transformers is not, the program aborts before reaching the demo mode.
- The fetch scripts (`fetch_models.sh`/`ps1`) still require manual execution to download the language models.

## Lessons Learned
- Optional imports are only partially guarded. `lazy_import()` raises an error instead of returning `None`, so missing packages stop execution.
- The CPU demo path triggers only when PyTorch itself is absent, not when Transformers is missing. This explains why we cannot simply hide heavy modules behind a flag today.
- A cross‑platform Python downloader could replace the shell/PowerShell scripts to achieve a "git to go" experience.

## Next Steps
- Refactor imports so that missing Transformers results in a clear message and automatic fall back to `cpu_demo` when requested.
- Expand `lazy_loader.py` to catch `ImportError` and optionally install or skip features.
- Explore a universal setup script that creates the virtual environment and fetches models without requiring PowerShell.
