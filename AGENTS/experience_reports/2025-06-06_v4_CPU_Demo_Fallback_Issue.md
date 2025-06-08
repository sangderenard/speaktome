# CPU Demo Fallback Issue

**Date/Version:** 2025-06-06 v4
**Title:** Trouble running after reinstall and missing CPU fallback

## Overview
Attempted to run the program on Windows after wiping and reinstalling the environment. Expected the script to gracefully fall back to `cpu_demo` when dependencies were missing. Instead, it failed with `ImportError: attempted relative import with no known parent package`.

## Steps Taken
1. Ran `powershell -ExecutionPolicy Bypass -File reinstall_env.ps1` which completed successfully.
2. Executed `python speaktome.py` from the repository root.
3. Encountered the import error before any CPU demo launched.
4. Tried `python -m speaktome.speaktome` and saw the program start up and load PyTorch before failing on missing `transformers` modules.

## Observed Behaviour
- Directly executing `speaktome.py` caused Python's relative imports to fail because the module was not run as part of the package.
- Running with `-m speaktome.speaktome` invoked the proper module but still attempted to import optional libraries before showing any fallback.

## Lessons Learned
The README examples use `python speaktome.py`, yet the codebase expects the module path invocation. Also, `lazy_loader.py` does not handle missing Transformers gracefully when Torch is installed, so the CPU demo is bypassed.

## Next Steps
- Update documentation to clarify the correct entry command (`python -m speaktome.speaktome`).
- Consider catching `ModuleNotFoundError` for optional libraries and routing to `cpu_demo` when advanced models are absent.
- Ensure the CPU-only path truly avoids importing heavy modules unless requested.
