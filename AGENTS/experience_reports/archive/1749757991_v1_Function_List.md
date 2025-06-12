# Function and Operator Listing

## Overview
Generated a list of all functions and operators defined in `tensors/abstraction.py`.
Appended a section listing incomplete tensor operations across the backends.

## Prompt History
- "Dump all functions and operators in a file added to the repo and then in that document add functions that need to be realized for tensor math and organization completeness, do not ignore my commands"

## Steps Taken
1. Parsed `tensors/abstraction.py` using Python AST to enumerate classes, methods, and functions.
2. Wrote the results to `tensors/abstraction_functions.md`.
3. Searched for `NotImplementedError` in `tensors/` and appended missing features.
4. Updated the guestbook with this report.

## Observations
- Several backend functions remain unimplemented, primarily in `c_backend.py` and `pure_backend.py`.
- Tests still fail due to missing `AGENTS` module.

## Next Steps
- Implement the unimplemented backend operations.
- Fix environment so that test suite can be executed.
