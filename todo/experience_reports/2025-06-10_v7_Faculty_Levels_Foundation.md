# Template User Experience Report

**Date/Version:** 2025-06-10 v7
**Title:** Faculty Levels Foundation

## Overview
Implemented a basic faculty detection system and integrated it across key modules. Added enumerations for NumPy, Torch and PyGeo tiers and updated documentation and tests.

## Prompts
"Please modify so we can create three classes of faculty and resource demand, work through all project files to weave this in on the ground floor. Levels are: Numpy (research demo of algorithm) Torch (performant production faculty) PyGeo (NN programmable smart search)"

## Steps Taken
1. Created `faculty.py` with `Faculty` enum and detection logic.
2. Updated configuration and CLI to report the detected level.
3. Added requirements constants to major modules.
4. Adjusted tests to handle optional Torch dependency.
5. Documented the new faculty system in the README.

## Observed Behaviour
- CLI prints the active faculty tier at startup.
- Tests pass even when Torch is not installed thanks to skipping logic.

## Lessons Learned
Foundation support for multiple compute tiers makes it easier to scale between lightweight demos and full PyGeo features.

## Next Steps
- Expand faculty awareness to remaining modules.
- Explore stubbing Torch to run the CPU demo without installing heavy dependencies.
