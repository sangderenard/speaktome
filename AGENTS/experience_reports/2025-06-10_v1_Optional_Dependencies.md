# Template User Experience Report

**Date/Version:** 2025-06-10 v1
**Title:** Optional Dependencies Simplification

## Overview
Testing minimal installation after moving heavy packages to `optional_requirements.txt`.

## Prompts
"There are issues with our soft dependency wrangling and initial install. We must not require to set up the environment torch or transformers..."

## Steps Taken
1. Created a clean virtual environment.
2. Installed `requirements.txt` only.
3. Verified `torch` and `transformers` are absent.
4. Ran `python -m speaktome.speaktome -h` to confirm CPU demo runs.
5. Executed `python AGENTS/validate_guestbook.py` to ensure filename correctness.

## Observed Behaviour
- The help message displayed and the CPU demo started without error.
- Guestbook validator reported no issues.

## Lessons Learned
- With heavy packages optional, the project runs basic demos on pure NumPy.
- Full beam search still requires installing the extras.

## Next Steps
- Document the new dependency split in the README.
- Ensure fetch scripts warn if dependencies are missing.
