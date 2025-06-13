# Documentation Report

**Date:** 1749802889
**Title:** Rewrite Header Template

## Overview
Rewrote the header template file as structured pseudocode. Removed all direct imports of `header_utils`.

## Steps Taken
- Replaced `AGENTS/headers.md` content with pseudocode format.
- Updated `header_template.py` to drop `header_utils` import.
- Modified `auto_fix_headers.py` and `header_guard_precommit.py` to load the template directly.
- Adjusted tests to avoid importing `header_utils`.

## Observed Behaviour
Tools now read the pseudocode template without relying on a helper module.

## Lessons Learned
Centralizing the template in an opaque format keeps header logic independent from any runtime helpers.

## Next Steps
Iterate on `header.py` to translate the pseudocode into live headers.

## Prompt History
importing header_utils is strictly not allowed but I want your experience reports and the headers.md file but I want you to rewrite the headers.md into something with no human explanation at all, using markdown procedural pseudocode where headers contain arguments and objects must mark their start and their name and arguments and then somehwere else at the correct nesting level their end. We will use this to construct a data structure of the header that is programmatic, a header.py is already in development that will be using the pseudocode template in markdown and managing its translation to python objects and python code headers fully realized as well as eventually also taking python headers and producing their template or object, with some rudimentary pattern matching for repairing or creating from scratch. But what I need from you is to use nested pseudocode using non programmatic non programmer non os convention pseudocode that works with ###(number of hashes is the nesting)TYPE(start, end, loop start, string start) Descriptive Name Of Feature arg1 arg2 arg3
