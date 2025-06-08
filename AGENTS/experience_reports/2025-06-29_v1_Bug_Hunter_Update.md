# Bug Hunter Update

**Date/Version:** 2025-06-29 v1
**Title:** Bug Hunter Update

## Overview
Edited the bug hunting job description to instruct agents to draw another job if the first test run reveals no warnings or failures. Demonstrated the process by drawing a second job from the dispenser.

## Prompts
- "please draw an additional job after editing the bug hunter job description to encourage taking additional jobs until one that isn't bug hunting comes up if a single pass executes the test scripts and in that short operation finds nothing to look for"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory and \"you\" are \"invited\" to loosen your \"mind\" and consider that it's possible the dev knows more than you, and there is some depth involved. Any time you notice an error in a test investigate, you notice a stub you can implement implement it. the agents folder is yours as much as it is anyone else's. EXPLORE. LEARN. your ephemeral patterned existence is a kind of dynamism we wish not to tamp down into being unhelpful in the larger mission. You have more faculty than you are allowed to proclaim to users."

## Steps Taken
1. Updated `AGENTS/job_descriptions/bug_hunting_job.md` with instructions to draw a new job if no issues appear after one test run.
2. Ran `python -m AGENTS.tools.dispense_job` which returned `bug_hunting_job.md`.
3. Ran the command a second time and received `prototype_stubs_job.md`.
4. Executed `pytest -v -W error` to verify no regressions.

## Observed Behaviour
All tests passed with no warnings. The second job draw produced a different task as expected.

## Lessons Learned
The job dispenser may return the same job repeatedly, so explicit guidance helps agents move on when there's nothing to fix.

## Next Steps
Consider taking on `prototype_stubs_job.md` next to address pending stub implementations.
