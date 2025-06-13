# Repository Root Detection Fix

**Date:** 1749831115
**Title:** Update root path logic across tools

## Overview
Implemented a helper `find_repo_root` that searches parent directories for a
`pyproject.toml` file. This replaces all hardcoded `parents[1]` assumptions.

## Steps Taken
- Added `AGENTS/tools/path_utils.py` with `find_repo_root`.
- Updated tools and tests to call this helper.
- Modified header template and doc to compute root dynamically.
- Ran `python AGENTS/validate_guestbook.py` and a focused pytest run.

## Observed Behaviour
Auto environment setup still fails due to dependency issues, causing tests to
skip. The new logic correctly resolves the repository root regardless of module
location.

## Lessons Learned
Avoid hardcoding path depths in a monorepo. Searching for a known directory
makes scripts more robust.

## Next Steps
Investigate the failing environment setup and refine the automation scripts.

## Prompt History
- "fix the naive use of parents[1], the monorepo does not require all modules run from one specific filesystem depth within their project folders"
