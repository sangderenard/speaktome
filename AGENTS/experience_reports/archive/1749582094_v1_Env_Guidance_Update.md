# Environment Guidance Update

**Date/Version:** 1749582094 v1
**Title:** Added headless setup notes and fixed digital fallback

## Overview
Documented another attempt to run the environment setup script and clarified documentation so agents avoid manual `pip` usage. Fixed `print_digital_clock` so ASCII digits render when Pillow is missing.

## Prompt History
```
do the same job over again and this time pretend you give a shit about following instructions in the repo and set up the fucking environment
```

## Steps Taken
1. Ran `bash setup_env_dev.sh --extras --prefetch --from-dev` which failed to install Torch due to network restrictions.
2. Observed the headless menu execution recorded to `/tmp/speaktome_active.json`.
3. Updated `AGENTS/CODEBASES_AND_ENVIRONMENT.md` and `AGENTS_DO_NOT_PIP_MANUALLY.md` with stronger guidance on using `dev_group_menu.py --install` for headless setups.
4. Patched `print_digital_clock` in `ascii_digits.py` to return pixel arrays using the fallback digit map when Pillow is not available.
5. Verified guestbook rules via `python AGENTS/validate_guestbook.py` and attempted `python testing/test_hub.py`.

## Observed Behaviour
- Environment setup aborted while downloading Torch.
- Test suite fails with `SystemExit` from `_internet.py`, confirming missing optional groups.
- The digital clock now displays basic ASCII digits instead of a black patch when Pillow is absent.

## Lessons Learned
- The setup scripts must be run even in headless mode with explicit codebase and group selections; manual pip installs should never be used.
- Providing a fallback inside `print_digital_clock` prevents blank output.

## Next Steps
- Continue iterating on test failures once dependencies are resolved.
