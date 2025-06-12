# Docstring Map Tool

**Date/Version:** 1749761217 v1
**Title:** Docstring_Map_Tool

## Overview
Created a new utility `docstring_map.py` that prints an indented outline of classes and functions along with their docstrings.

## Prompts
- "Can you make me a tool in agents/tools that takes a file location and makes an indented map of classes functions and their docstrings"
- Repository instructions from `AGENTS.md` about guestbook and tool placement.

## Steps Taken
1. Implemented `AGENTS/tools/docstring_map.py` following the standard header conventions.
2. Attempted to run tests with `python testing/test_hub.py` but encountered missing dependency errors.
3. Tried creating a virtual environment and installing packages; installation failed when building editable wheels.

## Observed Behaviour
- The new tool works locally to outline a file's classes and functions.
- Testing commands failed because required packages were not available in the environment.

## Lessons Learned
Dependency setup is tightly controlled by repository scripts and may not succeed with direct pip commands. Test failures can occur if these scripts are not run.

## Next Steps
Explore how to properly bootstrap the environment using provided setup scripts so automated tests can run without manual intervention.
