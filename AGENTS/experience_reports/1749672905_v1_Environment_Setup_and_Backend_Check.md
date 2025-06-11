# Environment Setup and Backend Review

## Prompt History
- "Remove backwards compatibility there is a strict methodology ..."
- "Check methods and methodology between abstract tensors and the different backends (ignore c for now it is much more complicated) and make sure that an abstract tensor behaves and returns in all ways like a pytorch tensor regardless of backends and with fluency of conversion of backends"

## Steps Taken
1. Ran `CODEBASES=speaktome bash setup_env_dev.sh --prefetch` which failed to run the dev menu because the `AGENTS` and `tomli` modules were missing.
2. Attempted `python AGENTS/tools/dev_group_menu.py --install --codebases speaktome --groups speaktome:dev` but it crashed with the same missing modules.
3. Verified that `python AGENTS/validate_guestbook.py` and `python testing/test_hub.py` both fail due to `ModuleNotFoundError: No module named 'AGENTS'`.
4. Inspected `tensors/abstraction.py` and backend implementations to ensure all methods accept and return `AbstractTensor` instances. All arithmetic and helper functions now wrap results before returning.

## Observed Behaviour
- Setup scripts do not complete because dependencies cannot be imported.
- Tests and guestbook validation abort with missing `AGENTS` module.
- Backend methods rely on the `_AbstractTensor__unwrap` helper internally while returning wrapped tensors, matching PyTorch semantics across backends.

## Next Steps
- Investigate the missing `AGENTS` and `tomli` packages in the virtual environment.
- Re-run the setup scripts once dependencies are available so that tests can be executed.
