# Bug Hunting Job

Identify, reproduce, and fix bugs across the project. Pay special attention to runtime warnings and deprecation notices that hint a feature will be removed in a future version. The goal is a clean test suite with no warnings.

**Stub Design Policy Reminder**: Bug hunters should **never** implement or modify existing stubs unless they are elevating them to the fully documented format defined in `AGENTS/job_descriptions/prototype_stubs_job.md`. Always consult that job description and `AGENTS/CODING_STANDARDS.md` before touching stub blocks.

Steps:
1. Run `pytest -v` and note any failures or warnings.
2. Reproduce reported issues and write regression tests if missing.
3. Fix the underlying problems or open issues describing the root cause.
4. If a fix is not immediately possible:
   - File a ticket detailing the bug and current investigation status.
   - Create a stub that documents the bug's nature following the format in `AGENTS/CODING_STANDARDS.md`.
   - Mark especially bad issues for human review.
5. **Do not modify the algorithm.** Never replace a parallel operation with a loop under any circumstances.
6. Eliminate warnings by updating calls or replacing deprecated APIs.
7. Document each fix or ticket in an experience report and update the CHANGELOG if user facing.
8. If the first test run surfaces **no** warnings or failures, request another job by running
   `python -m AGENTS.tools.dispense_job`. Keep drawing jobs until one that isn't this
   bug hunting task appears.
