# Development Setup Script

**Date/Version:** 2025-06-18 v3
**Title:** Add setup_env_dev script

## Overview
Created a developer-oriented environment setup script that runs the standard
`setup_env.sh` and then prints repository metadata. This includes header dumps,
stub listings, and key documents for quick reference.

## Prompts
```
create a dev version of the setup_env that runs the standard setup_env and then dumps all headers and then all stubs and then dumps the agent constitution and the main agents file followed by the license and then then our coding standards.
```

## Steps Taken
1. Added `setup_env_dev.sh` which calls `setup_env.sh` and executes
   `dump_headers.py` and `stubfinder.py`.
2. Appended output of critical documentation files in order.
3. Logged this experience report and validated guestbook entries.

## Observed Behaviour
The new script successfully sets up the environment and prints the expected
information.

## Lessons Learned
Providing a single command to view important documents helps new contributors
understand repository structure faster.

## Next Steps
Consider adding a Windows PowerShell equivalent if needed.
