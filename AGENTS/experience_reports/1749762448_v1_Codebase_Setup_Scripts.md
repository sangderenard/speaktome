# Codebase Setup Scripts

## Overview
Implemented wrapper scripts in each codebase directory to simplify environment setup.

## Prompts
- "Investigate setup_env and the call to dev_something something, the python script that accepts headless install of codebases and options, I want to put a setup_env in each codebase root(not the repo root) that properly changes to the repo root and runs the dev script with every group that doesn't depend on torch, i might have to make the actual group and codebase details for the right command line arguments, butt technically what we'll do is have these independent scripts look themselves and their groups up to issue a full or non torch or minimal install, defaulting to non torch. There's a codebase map in agents"
- "always check the files in the repo ecosystem for your benefit..." (developer instruction)

## Steps Taken
1. Created a generic setup script template parsing `AGENTS/codebase_map.json`.
2. Added `setup_env.sh` to each codebase root using that template.
3. Documented work in this experience report.

## Observed Behaviour
Scripts call the repository root `setup_env_dev.sh` with default `-notorch` and headless flags. Groups are automatically selected from the map unless `-full` or `-minimal` is provided.

## Lessons Learned
Mapping optional groups to avoid torch enables quicker setup for codebases that don't require heavy dependencies.

## Next Steps
- Verify cross-platform PowerShell equivalents if needed.
