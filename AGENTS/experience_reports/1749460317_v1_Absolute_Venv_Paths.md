# Absolute Venv Paths Fix

**Date/Version:** 1749460317 v1

## Overview
`setup_env_dev.sh` failed when changing into the `speaktome` directory because
relative paths like `./.venv/bin/python` no longer pointed to the virtual
environment. This produced errors such as `No such file or directory`. The fix
resolves the repository root and uses absolute paths for the venv.

## Prompt History
- "ok, interesting, not bad work, also, not addressing the problem that you have - you specifically - where YOU are unable to use setup_end_dev.sh because YOU get the error every time that you can't install because the venv doesn't exist. YOU are supposed to monitor the startup script output and understand it is part of your task"

## Steps Taken
1. Reproduced the issue with `setup_env_dev.sh --extras` and observed missing
   `.venv` errors when inside the `speaktome` subdirectory.
2. Updated `setup_env_dev.sh` to resolve `SCRIPT_ROOT` and construct venv paths
   using this absolute location.
3. Adjusted the script to call `setup_env.sh` via `SCRIPT_ROOT` as well.

## Observed Behaviour
Running the developer setup script now locates the virtual environment
correctly, even after `pushd speaktome`. Subsequent pip and Python commands work
as expected.

## Lessons Learned
Using absolute paths prevents directory changes from breaking helper scripts.
