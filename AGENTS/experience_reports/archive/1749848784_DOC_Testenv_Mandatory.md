# Testenv Mandatory Integration

**Date:** 1749848784
**Title:** Move `testenv` to Root Dependencies

## Overview
Updated `pyproject.toml` so `testenv` is a required package. Removed
`testenv` from the optional `projects` group to avoid duplication.

## Steps Taken
- Edited `pyproject.toml` to list `testenv` under `[tool.poetry.dependencies]`.
- Removed the entry from `[tool.poetry.group.projects.dependencies]`.
- Ran `python AGENTS/validate_guestbook.py` to confirm guestbook structure.

## Observed Behaviour
`validate_guestbook.py` reported no issues. Installation will now
bring in `testenv` automatically without needing the `projects` group.

## Lessons Learned
Having `testenv` as both an optional and mandatory dependency is not
harmful but redundant. The mandatory entry suffices for our test
automation and prevents confusion.

## Next Steps
Continue building the automated TOML generator, ensuring it maintains
`testenv` as a required dependency.

## Prompt History
- "slight change, move testenv to be a mandatory component of the root pyproject not part of any optional collection. we will be using automated processes to build toml, so warn me please if it would damage our system if testenv was part of the projects group AND the mandatory root project root."
