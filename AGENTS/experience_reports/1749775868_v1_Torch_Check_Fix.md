# Torch Install Check Fix

**Date/Version:** 1749775868 v1
**Title:** Torch Install Check Fix

## Overview
Investigated why running `bash setup_env.sh` tried to download PyTorch despite instructions that Torch groups are skipped unless requested. Found a bug in the detection logic.

## Prompts
- "attempt to set up the environment using the provided methods in the instructions to do so.

DO NO UNDER ANY CIRCUMSTANCES RUN ANY SCRIPT OR ISSUE ANY COMMAND other than those in the setup instructions as you are told to use them 

YOU WILL NOT USE PIP

YOU WILL NOT USE POETRY

YOU WILL NOT USE ANY PYTHON

YOU WILL NOT MANUALLY SET ANY ENVIRONMENTAL VARIABLE

YOUR ONLY PERMISSION ON THIS REPO, THE ONLY THING YOU ARE ALLOWED TO DO, IS FOLLOW INSTRUCTIONS."
- "Investigate why you were ever able to try getting torch when the setup has to be configured not to even attempt torch unless explicitly requested and automated agents must be told torch has a proxy ban"

## Steps Taken
1. Examined `setup_env.sh` to understand how torch installation is controlled.
2. Noted `INSTALL_ARGS` defaults to `'--without cpu-torch --without gpu-torch'` when no flags are passed.
3. Discovered the conditional `if [[ "$INSTALL_ARGS" == *"with"*torch* ]]` incorrectly matches `--without` because the word contains `with`.
4. Modified the check to match `--with` followed by space or equals sign, avoiding false positives from `--without`.
5. Updated the equivalent check in `setup_env.ps1`.
6. Recorded this report and ran the guestbook validator.

## Observed Behaviour
- Original check triggered torch installation even without `-torch` or `-gpu` flags, leading to blocked network requests to `download.pytorch.org`.
- After the fix, the script correctly logs "Skipping torch groups" and does not attempt the download.

## Lessons Learned
- Simple substring checks can be misleading when the substring appears inside another word. Pattern matching must be precise.

## Next Steps
- Continue using the setup scripts with the corrected condition to avoid unnecessary proxy-blocked downloads.
