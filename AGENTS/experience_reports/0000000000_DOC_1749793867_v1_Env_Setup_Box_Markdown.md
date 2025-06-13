# Environment Setup Box Markdown

## Overview
Moved the static ENV_SETUP_BOX string from the setup scripts to a shared markdown file. Updated Bash and PowerShell setup scripts to read this file when exporting the ENV_SETUP_BOX variable. Extended `auto_env_setup.py` with a simple CLI so tests can pass the module root explicitly.

## Prompts
- "take the static definitions for the environmental error warning out of the setup scripts for bash and powershell and relocate it to a markdown file that both setup scripts reference to generate the environmental variable. If such a thing exists. Have the auto-setup script be passed as an argument the root folder of the module (wherever the pyproject.toml is) so it can automatically configure itself to construct exactly what is necessary."

## Steps Taken
1. Created `ENV_SETUP_BOX.md` containing the warning message.
2. Modified `setup_env.sh`, `setup_env.ps1`, `setup_env_dev.sh`, and `setup_env_dev.ps1` to export the variable from this file.
3. Added a command line interface to `AGENTS/tools/auto_env_setup.py` to accept a root path argument.
4. Executed `python testing/test_hub.py` which skipped tests due to missing environment.

## Observed Behaviour
- The environment message is now stored in a single markdown file.
- Test harness skipped because environment setup failed in this container.

## Lessons Learned
Externalizing common strings reduces duplication and keeps setup messages consistent across platforms.

