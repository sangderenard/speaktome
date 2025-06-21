# Documentation Report

**Date:** 1749924190
**Title:** Embed guess_codebase in header template

## Summary
- Added full `guess_codebase` implementation directly to `AGENTS/header_template.py`.
- Mirrored the change in `AGENTS/tests/headers/header_template.py` for testing.
- Updated `AGENTS/headers.md` to show the new helper in the canonical header.

## Prompt History
```
a prototype, guess codebase, has been put in the tools folder, and implemented once or twice, but it violates a fundamental principle of our scripts and header template, being that it is forbidden to import from the project anything, because then there is a tautological dependency. Please examine if guess codebase can be put in a form we can place in the header template directly, the whole function all the code, because it won't truly matter if we bloat our headers if the tools to clean and create them know what to expect. it will just look like we tried harder than we actually did
```
