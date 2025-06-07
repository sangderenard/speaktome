# Test vs Testing Guidance

**Date/Version:** 2025-06-07 v1
**Title:** Clarifying tests and testing directories

## Overview
Examined the repository layout to understand whether the `tests/` and `testing/` directories serve separate purposes. Also reviewed existing `AGENTS.md` guidance and considered additional folder-level instructions.

## Prompts
"are test and testing redundant or unique? Please provide agents.md human and LLM guidance for any top level directory, and when agent guidance already exists, examine if it is sufficient, or if folder local guidance is also warranted."

## Steps Taken
1. Searched for `AGENTS.md` files with `find` to locate existing guidance.
2. Read the main `AGENTS.md` and the `tests/AGENTS.md` instructions.
3. Ran `pytest -q` and `python testing/test_hub.py` to confirm the suite passes and stub list generation works.
4. Surveyed other top-level folders (`models`, `speaktome`, `testing`, `training`, `todo`) for any missing guidance.

## Observed Behaviour
- `pytest` reported all tests passing with several marked as skipped.
- `test_hub.py` generated `testing/stub_todo.txt` listing remaining stub tests.
- Only `AGENTS/` and `tests/` currently contain `AGENTS.md` files.

## Lessons Learned
`tests/` is the formal automated test suite and includes guidelines for running pytest and maintaining stub tests. The `testing/` folder holds demonstration scripts and helper utilities, so it is not redundant with `tests/`.

## Next Steps
Adding lightweight `AGENTS.md` files to folders like `testing`, `training`, `todo`, `models`, and `speaktome` could clarify their intent for future contributors.
