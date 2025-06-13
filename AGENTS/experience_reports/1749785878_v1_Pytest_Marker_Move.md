# Pytest Marker Move

**Date/Version:** 1749785878 v1
**Title:** Marker file moved to base setup scripts

## Overview
Shifted `.venv/pytest_enabled` creation from developer convenience scripts to the primary environment setup scripts. Also fixed leftover environment variable causing unintended torch installs.

## Prompts
- "move the marker file to basic setup scripts in bash and powershell"
- "investigate how it was possible that your use of the script with no arguments led to any attempt to install torch, that should never happen unless arguments are supplied asking for them"

## Steps Taken
1. Added marker creation logic to `setup_env.sh` and `setup_env.ps1`.
2. Removed the same logic from `setup_env_dev.sh` and `setup_env_dev.ps1`.
3. Passed `SPEAKTOME_POETRY_ARGS` to `setup_env.sh` only for the command invocation instead of exporting it.

## Observed Behaviour
- Without unsetting, the Bash dev script left `SPEAKTOME_POETRY_ARGS` in the environment. Running `setup_env.sh` afterward attempted to install torch groups. Passing the variable locally prevents this.

## Next Steps
Verify test scripts detect the marker from base setup and ensure future documentation reflects the change.
