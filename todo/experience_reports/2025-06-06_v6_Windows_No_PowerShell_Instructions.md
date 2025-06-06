# Windows Without PowerShell Instructions

**Date/Version:** 2025-06-06 v6
**Title:** Added instructions for users avoiding PowerShell

## Overview
This update documents how to set up and run the project on Windows without relying on PowerShell. A simple `run.cmd` script was added alongside manual command steps in the README.

## Steps Taken
1. Created `run.cmd` which calls `.venv\Scripts\python.exe` just like the PowerShell wrapper.
2. Expanded the README with a new "Windows Without PowerShell" section describing manual setup commands.
3. Updated examples to show both `run.ps1` and `run.cmd`.

## Observed Behaviour
CMD users can now run `run.cmd` directly or follow the step-by-step commands without needing PowerShell. The script mirrors the behaviour of `run.ps1`.

## Lessons Learned
Providing explicit instructions for alternative workflows keeps the barrier to entry low for a wider range of Windows users.

## Next Steps
- Consider adding batch equivalents for `setup_env.ps1` if more users request them.
