# Environment Setup Attempt

**Date/Version:** 1749636030 v1
**Title:** Environment Setup Attempt

## Overview
Attempted to follow project guidelines to create the virtual environment and run the pytest suite.

## Prompts
- "See if you can figure out setting up the environment for yourself well enough and by the rules enough to run pytests"

## Steps Taken
1. Executed `bash setup_env_dev.sh --prefetch` with `CODEBASES=""` to avoid an unbound variable error.
2. The script created `.venv` but failed to install dependencies due to network restrictions (403 errors from PyTorch URL).
3. Activated the environment and ran `python testing/test_hub.py`.

## Observed Behaviour
- `setup_env.sh` reported repeated `ProxyError` messages and could not download `torch`.
- Running tests failed immediately because `pytest` was not installed and `ENV_SETUP_BOX` was undefined.

## Lessons Learned
The setup scripts expect network access to fetch packages. Without it, they leave the environment incomplete, so tests cannot run. Manual `pip` use is discouraged, so a cached wheel archive or offline installer might be needed.

## Next Steps
Look for instructions or tools that allow installing dependencies in an offline environment. Consider filing an issue about `setup_env.sh` referencing `CODEBASES` before initialization.

## Follow‑Up
Discovered that `AGENTS_DO_NOT_PIP_MANUALLY.md` suggested using
`setup_env.sh --from-dev` when running headless. This was misleading.
Updated the document to instead recommend:

```bash
bash setup_env_dev.sh --prefetch
python AGENTS/tools/dev_group_menu.py --install --codebases speaktome --groups speaktome:dev
```

Refer to `AGENTS/CODEBASE_REGISTRY.md` for available codebase names.
