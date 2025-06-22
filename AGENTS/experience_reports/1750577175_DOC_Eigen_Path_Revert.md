# Eigen Path Revert

## Overview
Adjusted asciioscilliscope headers to reference the in-tree Eigen copy and restored `Projection2D` API accuracy. Added value range checks in unit tests.

## Steps Taken
- Ran `setup_env.sh`
- Attempted `python testing/test_hub.py`
- Attempted CMake configure/build

## Observed Behaviour
- Python tests skipped due to environment not initialized.
- CMake failed since submodules were missing.

## Lessons Learned
Project uses submodules for Eigen and GoogleTest; failing to update them leads to build errors.

## Next Steps
- Ensure submodules are initialized before building.

## Prompt History
Fix eigen paths and ensure unit tests align with header expectations.
