# Template User Experience Report

**Date/Version:** 2025-06-14 v3
**Title:** Gitignore Auto Update Script

## Overview
Added Bash and PowerShell scripts that scan the repository for files exceeding a configurable size limit and append them to `.gitignore`.

## Prompts
"can you please give me a script in training used to auto update a git ignore for large files or any files beyond a certain set cap, powershell and bash if you can."

## Steps Taken
1. Created `training/gitignore_large_files.sh`.
2. Created `training/gitignore_large_files.ps1`.
3. Ran `pytest -v` and `python testing/test_hub.py` to ensure tests pass.
4. Validated guestbook entries with `python AGENTS/validate_guestbook.py`.

## Observed Behaviour
- Both scripts list files larger than the limit and add their relative paths to `.gitignore` if not already present.
- Test suite succeeded.

## Lessons Learned
Automating `.gitignore` updates avoids accidental commits of large artifacts.

## Next Steps
- Verify the scripts on a clone containing large training archives.
