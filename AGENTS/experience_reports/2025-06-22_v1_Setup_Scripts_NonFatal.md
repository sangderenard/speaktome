# Template User Experience Report

**Date/Version:** 2025-06-22 v1
**Title:** Setup Scripts NonFatal

## Overview
Updated environment setup scripts to continue running even if a command fails. This prevents abrupt session termination during `pytest` introduction.

## Prompts
```
one thing i need is the setup env scripts to not bubble up an error code because of any individual failing. it kills your session and we're doing pytest in that because you get an immediate introduction to our needs through it. you may see in pyproject though i'm not sure if that's exactly the right place, a bunch of commented out lines that previously would run a setup env script - we can only use that if we capture and contain any error codes and only bubble up what we think is vital, which at this early stage is nothing, but don't let that stop you from some exploring
```

## Steps Taken
1. Modified Bash setup scripts (`setup_env.sh`, `setup_env_dev.sh`, `reinstall_env.sh`, `fetch_models.sh`) to use a `safe_run` helper and removed `set -e`.
2. Updated PowerShell equivalents (`setup_env.ps1`, `reinstall_env.ps1`, `fetch_models.ps1`) to continue on errors using a `Safe-Run` function.
3. Created this experience report and validated the guestbook.

## Observed Behaviour
Commands that fail now print warnings but do not abort the entire script, allowing subsequent steps to run.

## Lessons Learned
Graceful handling of setup failures makes iterative testing smoother. Capturing exit codes without halting provides visibility while keeping the environment usable.

## Next Steps
Review the new warnings for any recurring failures and consider logging them to a diagnostics file for deeper analysis.
