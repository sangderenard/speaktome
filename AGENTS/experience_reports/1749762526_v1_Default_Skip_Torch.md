# Default Skip Torch

**Date/Version:** 1749762526 v1
**Title:** Default_Skip_Torch

## Overview
Adjusted environment setup scripts to only install PyTorch if explicitly requested with `-torch` or `-gpu` flags.

## Prompts
- "Adjust the setup env scripts in bash and powershell to by default skip torch install attempts, make the decision logic just, did they say torch or gpu torch or if nothing, skip even trying"
- Root repository instructions on maintaining experience reports and running `validate_guestbook.py`.

## Steps Taken
1. Updated `setup_env.sh` and `setup_env.ps1` to parse `-torch` and `-gpu` options and skip installation otherwise.
2. Simplified developer scripts to forward arguments without special torch handling.
3. Added explanations in script comments and created this experience report.

## Observed Behaviour
- Scripts now emit an informational message when torch is not requested.

## Lessons Learned
Defaulting to skip heavy dependencies avoids failures in restricted environments while still allowing explicit installs when needed.

## Next Steps
Run the environment setup with `-torch` in a clean environment to verify the install path works when network access allows.
