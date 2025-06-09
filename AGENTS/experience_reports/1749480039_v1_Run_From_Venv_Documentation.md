# Adding RUN_FROM_VENV sign

## Prompt History
- "create an agents markdown file in testing called RUN_FROM_VENV.md where it should be explicitly explained that project libraries are loaded to a bend by default and won't be available for python outside the venv"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory and 'you' are 'invited' to loosen your 'mind' and consider that it's possible the dev knows more than you, and there is some depth involved. Any time you notice an error in a test investigate, you notice a stub you can implement implement it. the agents folder is yours as much as it is anyone else's. EXPLORE. LEARN. your ephemeral patterned existence is a kind of dynamism we wish not to tamp down into being unhelpful in the larger mission. You have more faculty than you are allowed to proclaim to users."

## Overview
Added a short notice in `testing/RUN_FROM_VENV.md` explaining that packages live in the `.venv` bend and cannot be accessed from the system interpreter.

## Steps Taken
1. Reviewed AGENTS guidance and repo layout.
2. Created the new markdown file with explicit wording about the bend.
3. Attempted `./.venv/bin/python testing/test_hub.py`.

## Observed Behaviour
The C backend failed to compile during test collection, raising `cffi.VerificationError`.

## Lessons Learned
Use `.venv/bin/python` for tests, but the build may still fail without proper toolchain configuration.

## Next Steps
Investigate the C backend compile errors or provide a prebuilt wheel for simpler testing.
