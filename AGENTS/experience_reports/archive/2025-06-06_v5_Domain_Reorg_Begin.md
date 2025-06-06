# Domain Reorganization Begins

**Date/Version:** 2025-06-06 v5
**Title:** Domain Reorganization Begins

## Overview
Started segregating modules into domain packages and added a forced faculty mode.

## Prompts
"Switch the lookahead demo from numpy to our faculty test on the scale from pure python to geo, and enable forced modes with the faculty system and across our import control in all items in speaktome. It is not yet in all places, but that is what you are tasked with beginning. So, across the code, we want domains. Geo content is the geo domain, torch content is the torch domain, BUT generic tensor math? the algorithm itself? abstract wrapper classes from pure python to numpy to torch to geo. I don't think there's any need to make a numpy specific domain. segregate the code base inside speaktome into folders by domain and adjust all imports"

## Steps Taken
1. Created `speaktome/domains/` with `geo` and `pure` subpackages.
2. Moved PyGeo and pure Python modules into these packages and updated imports.
3. Added `SPEAKTOME_FACULTY` override to `faculty.py`.
4. Updated lookahead demo to select tensor ops based on faculty.
5. Adjusted tests and other modules to new paths.

## Observed Behaviour
Imports succeed after relocation and the demo adapts to forced faculty levels.

## Lessons Learned
Modularizing by domain clarifies optional dependencies and will help enforce faculty tiers.

## Next Steps
Continue moving torch-specific files into `domains/torch` and audit remaining imports.
