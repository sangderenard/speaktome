# C Library Fetch Update

## Prompt History
- User: upgrade the script to obtain c library dependencies by adding every single listed library here
- Developer: always check the files in the repo ecosystem for your benefit...

## Overview
Extended the helper script and build.zig to automatically fetch all recommended C libraries when missing.

## Steps Taken
1. Edited `AGENTS/tools/fetch_tensor_sources.py` to include each library URL.
2. Updated `build.zig` to clone all libraries before building.

## Observed Behaviour
No tests exercise the clone logic directly but the modified script now lists all repositories.

## Lessons Learned
Vendoring is simplified when the build script handles fetching every dependency.

## Next Steps
Add continuous integration steps to cache these clones.
