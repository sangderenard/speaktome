# Ignore Markup Log Files

**Date/Version:** 2025-06-07 v1
**Title:** Ignore markup logs in testing directory

## Overview
Added a `.gitignore` file inside `testing/logs` so generated Markdown logs are not tracked by Git.

## Prompts
```
add markup logs to gitignore in the log directory
```

## Steps Taken
1. Reviewed repository guidelines in `AGENTS.md`.
2. Created `testing/logs/.gitignore` with a rule to ignore `*.md` files.
3. Wrote this experience report and ran the guestbook validator.
4. Executed `pytest -v` to ensure the test suite still passes.

## Observed Behaviour
- `pytest` completed without failures.
- New `.gitignore` prevents future `.md` logs from being added to version control.

## Lessons Learned
Small ignore rules keep the repository clean, especially for auto-generated logs.

## Next Steps
None.
