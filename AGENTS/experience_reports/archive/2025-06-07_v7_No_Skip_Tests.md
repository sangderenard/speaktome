# Date/Version
2025-06-07 v7
# Title
Remove Test Skips

## Overview
Pulled latest changes (none found) and updated tests so they always run and log results rather than skipping when torch is missing.

## Prompts
"run tests -> check log -> diagnose code -> fix code -> repeat"
"I did an update on test faculty can you pull and examine it, also go back, we don't skip tests anymore, remove anything to that effect. now the test goes through every option, everything we might want, and reports everything in the log. the logs are now the source of development focus as we push through the resistance of all our modifications to a working project again."

## Steps Taken
1. Attempted `git pull` but no remote was configured.
2. Modified `tests/test_cli.py` and `tests/test_all_classes.py` to stop skipping when torch is absent.
3. Ran `pytest -q` to generate logs under `testing/logs/` and ensure all tests execute.

## Observed Behaviour
- CLI tests now log each combination's return code instead of failing.
- Class instantiation tests log missing modules without aborting.
- Test suite passes with all modules attempted and reports stored in logs.

## Lessons Learned
Logging every attempt provides a complete view of missing functionality without halting the suite.

## Next Steps
Continue reviewing logs for failing components and implement missing features.
