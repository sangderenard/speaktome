# Clock Demo CL Mode

**Date/Version:** 1749578532 v1

## Overview
Attempted to run the clock demo in command line mode after following repository environment setup guidance.

## Prompts
```
follow guidance and attempt to use the clock demo in its cl mode, investigate and correct any problems along the way, with special attention paid to your responsibility to read about and understand environmental setup and execute the apropriate commands to prep and enter the venv before running anything
```

## Steps Taken
1. Executed `bash setup_env_dev.sh --extras --prefetch --from-dev` which created `.venv` but failed to install `torch` due to 403 errors.
2. Activated the virtual environment with `source .venv/bin/activate`.
3. Installed the `time_sync` codebase via `python AGENTS/tools/dev_group_menu.py --install --codebases time_sync` and installed the optional GUI extras with `pip install 'time_sync[gui]'`.
4. Installed `numpy` as it was required by the demo but not declared in `time_sync` dependencies.
5. Ran `python -m time_sync.clock_demo --no-analog --no-digital-system --no-digital-internet --no-stopwatch --no-offset --refresh-rate 0.1`.
6. Encountered a `ValueError` from `PixelFrameBuffer.update_render` complaining about shape `(35,120,4)` versus `(35,120,3)`.
7. Patched `clock_demo.render_fn` to convert the processed image back to `RGB` before returning a numpy array.
8. Re-ran the demo with `timeout 5 python -m time_sync.clock_demo ...` to verify no crash.
9. Ran the test suite using `PYTHONPATH=. python testing/test_hub.py` which produced internal errors in the test harness.

## Observed Behaviour
- The environment setup script warns of torch installation failure but continues.
- After patching, the clock demo runs for a short time and exits cleanly when timed out.
- `testing/test_hub.py` fails with a `TypeError` in `tests/conftest.py`.

## Lessons Learned
Converting the post-processed image back to RGB avoids framebuffer channel mismatch. Tests require additional environment configuration beyond the default dev setup.

## Next Steps
- Update `time_sync` dependencies to include `numpy`.
- Investigate the faculty comparison error in `tests/conftest.py`.
