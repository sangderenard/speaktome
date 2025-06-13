# Documentation Report

**Date:** 1749801602
**Title:** Header Template Externalization

## Overview
Centralized the Python header definition so tools share a single source. Added a fenced code block in `AGENTS/headers.md` containing the template and updated utilities to load it.

## Steps Taken
- Edited `AGENTS/headers.md` with the full header block.
- Updated `header_utils.py` to read the block.
- Modified `auto_fix_headers.py` and `header_guard_precommit.py` to import the template.
- Prepared a guestbook entry.

## Observed Behaviour
Utilities now pull header lines from the markdown file. The auto fixer inserts the template verbatim when needed.

## Lessons Learned
Keeping a single markdown representation simplifies updates and keeps scripts consistent.

## Next Steps
Monitor for unexpected header formatting issues across the codebase.

## Prompt History
Validation only needs to be done every now and then, we can skip it. Do you think you could externalize the header template from all agent tools that deal with header templates, so that a single markdown file will contain the full description of all components and their placement and nesting in a highly human readable format using a sentinel syntax that permits dynamic apprehension of the template's nature, providing arguments for each section and implying with whitespace like python when a component is contained by another component and which pieces come before and after.
