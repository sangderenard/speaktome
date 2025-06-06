# Harvest Script Added

**Date/Version:** 2025-06-15 v2
**Title:** Harvest Script Added

## Overview
Implemented `training/harvest_alpha_classes.py` which collects Python files from
`training/archive/Alpha`, renames them according to their modification time and
contained class names, and copies them to a user-specified directory.

## Prompts
- "The training archive cleaning script is now searching for duplicates, what I need it to do is search for .py files from achive/Alpha recursively and put them into a new folder outside alpha, where their filenames have been changed to their modified timestamp in epoch followed by all class names inside the python files in alphabetical order."
- "Check the experience logs to see if anyone has actually gone through the python deep inside that filestructure."

## Steps Taken
1. Reviewed repository `AGENTS.md` and existing guestbook entries.
2. Implemented `harvest_alpha_classes.py` with class name parsing and unique filename logic.
3. Searched experience reports for references to `archive/Alpha` but found none.
4. Prepared this guestbook entry and ran the validation script.

## Observed Behaviour
- No prior experience reports mention inspecting the Python files inside the Alpha archive.
- The new script correctly traverses the directory and prints planned copy operations during a dry run.

## Lessons Learned
- The archive has not yet been explored in depth, so this script provides a foundation for systematic analysis.

## Next Steps
- Run the script locally to generate the reorganized dataset.
- Explore the harvested files for unique class implementations.

