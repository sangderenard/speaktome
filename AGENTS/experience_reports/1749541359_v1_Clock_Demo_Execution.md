# Clock Demo Execution

**Date/Version:** 1749541359 v1

## Overview
Follow repository instructions to run the time sync clock demo. The setup script failed to install `torch` due to network restrictions but the virtual environment was created and the `time_sync` package was installed manually.

## Prompts
- "in time sync use the clock demo by running it as a module from the repo root after running the setup env dev script selecting y for all the options that come up yes or no and then activating the venv"

## Steps Taken
1. Executed `bash setup_env_dev.sh` and attempted to select `y` for prompts. Torch installation failed due to network restrictions.
2. Activated the virtual environment with `source .venv/bin/activate`.
3. Installed missing packages manually: `pip install ntplib colorama pillow numpy`.
4. Installed the `time_sync` package in editable mode: `pip install -e time_sync`.
5. Ran the demo with `python -m time_sync.clock_demo --no-analog --no-digital-system --no-digital-internet --no-stopwatch --no-offset --refresh-rate 0.1`.

## Observed Behaviour
- `setup_env_dev.sh` reported an error fetching PyTorch due to a 403 proxy response.
- The clock demo started successfully and printed terminal graphics until interrupted.

## Lessons Learned
Editable installs ensure module imports resolve correctly. Network limitations may block dependency retrieval from certain hosts.

## Next Steps
Consider providing pre-installed wheels or skipping GPU packages when network restricted.
