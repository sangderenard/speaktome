# Testing Folder Disambiguation

**Date/Version:** 2025-06-07 v2
**Title:** Added local guidance for top-level directories

## Overview
Implemented additional documentation to clarify the purpose of the `testing/` folder and how it differs from `tests/`. Added brief `AGENTS.md` files for other top-level directories for future contributors.

## Prompts
"disambiguate the testing folder linguistically - accentuate what the difference is intuitively"

## Steps Taken
1. Created `testing/AGENTS.md` explaining the folder's exploratory nature and linking to `test_hub.py`.
2. Added lightweight `AGENTS.md` files to `models/`, `speaktome/`, `training/`, and `todo/`.
3. Updated the `Project Layout` section in `README.md` to describe `tests/` versus `testing/`.
4. Ran `pytest -q` and `python testing/test_hub.py`.

## Observed Behaviour
- All tests passed and `testing/stub_todo.txt` listed remaining stub entries.

## Lessons Learned
Providing explicit guidance at the folder level helps new contributors quickly identify where to run formal tests versus where to place ad-hoc scripts.

