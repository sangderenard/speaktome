# Default Codebases Update

**Date/Version:** 1749447284 v1
**Title:** Default Codebases Update

## Overview
Adjusted the setup scripts so the default editable installs include all core
codebases.

## Prompts
- "the default should include agents/tools, speaktome, and the time related project I forget the directory of"

## Steps Taken
1. Modified `setup_env.sh` and `setup_env.ps1` to initialize the `CODEBASES`
   variable with `speaktome`, `AGENTS/tools`, and `time_sync`.
2. Ran the validation script to ensure experience report formatting.

## Observed Behaviour
Both setup scripts now install all three codebases in editable mode when run
with no extra options.

## Lessons Learned
Explicit defaults prevent confusion about which projects are active when setting
up a fresh environment.

## Next Steps
Continue monitoring for issues when new codebases are added.

