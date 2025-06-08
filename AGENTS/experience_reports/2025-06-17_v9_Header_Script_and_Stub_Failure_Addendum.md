# Header Script and Stub Failure Addendum

**Date/Version:** 2025-06-17 v9
**Title:** Header Script and Stub Failure Addendum

## Overview
Addressed new instructions to update the coding conventions with an addendum on
stubs, emphasizing that they may fail loudly. Added a standard header marker and
created a helper script to dump file headers.

## Prompt History
```
While I've got you, I need you to put an addendum in coding conventions regarding stubs that while stubs must be conspicuously commented and marked as currently noted, they must also be allowed to commit hard failures. Additionally, please prepare a script that finds .py files recursively in speaktome and prints until it finds a standardized header end comment I'll need to you enshrine. We're going to dump the headers of everything on every log as single document context, please.
```

## Steps Taken
1. Updated `AGENTS/CODING_STANDARDS.md` with guidance on allowing stub failures
   and documenting the new `# --- END HEADER ---` marker.
2. Created `dump_headers.py` at the repository root to collect headers from all
   Python files.
3. Recorded this report and ran `python AGENTS/validate_guestbook.py` to verify
   naming conventions.

## Observed Behaviour
- Documentation updated successfully.
- The new script walks the project and prints headers as intended.
- Guestbook validation confirmed correct filenames.

## Lessons Learned
Keeping stub failures visible surfaces missing functionality early. A consistent
header marker makes it trivial to gather context across the entire codebase.

## Next Steps
- Encourage contributors to include the header marker in all Python files.
- Integrate the header dump into automated logs.
