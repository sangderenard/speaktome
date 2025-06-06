# Harvest Script Deduplication Update

**Date/Version:** 2025-06-15 v3
**Title:** Harvest Script Deduplication Update

## Overview
Enhanced `harvest_alpha_classes.py` to detect and skip duplicate files using SHA256 hashes while preserving the timestamp-and-class naming scheme. The user also requested a listing of Python files in Level 16 of the Alpha archive.

## Prompts
- "The training archive cleaning script is now searching for duplicates, what I need it to do is search for .py files from achive/Alpha recursively and put them into a new folder outside alpha, where their filenames have been changed to their modified timestamp in epoch followed by all class names inside the python files in alphabetical order."
- "Can you recursively list the python files, probably focus on Level 16."

## Steps Taken
1. Reviewed existing harvest script and repository guidelines.
2. Added SHA256-based duplicate detection to avoid copying identical files.
3. Generated a listing of all `.py` files within `Level 16` using `find`.
4. Created this report and validated the guestbook.

## Observed Behaviour
- Duplicate files are reported and skipped during the harvest run.
- The Level 16 directory contains a variety of networking, device, renderer, and primitive scripts.

## Lessons Learned
- Deep levels of the archive hold many experimental files; deduplication is necessary to keep the harvested set manageable.

## Next Steps
- Run the updated script locally to produce a deduplicated dataset.
- Further analyze the Level 16 modules for useful class implementations.

