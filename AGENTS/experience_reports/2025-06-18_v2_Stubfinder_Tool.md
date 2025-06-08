# Stubfinder Tool

**Date/Version:** 2025-06-18 v2
**Title:** Implement stubfinder utility

## Overview
Created a command line script to locate stub blocks across Python
files. The tool scans directories recursively and prints each stub with
its file path and starting line number separated by thin lines.

## Prompts
```
Use a few clues with regex to match any stubs in any .py files recursively through any target folder and put it in the tools directory please, call it the stubfinder, put thin lines between stubs that identify file and beginning line number, if you could.
```

## Steps Taken
1. Reviewed repository guidelines and coding standards.
2. Added `AGENTS/tools/stubfinder.py` with header comment and CLI.
3. Tested the script on `speaktome/core` directory.
4. Logged this report and validated filenames.

## Observed Behaviour
- Script correctly prints stub comment blocks with separators.

## Lessons Learned
Writing a simple regex-based search script helps document remaining
work across the project.

## Next Steps
- Consider integrating stubfinder with other validation tools or CI.
