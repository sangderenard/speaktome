# Bug Hunting Job

Identify, reproduce, and fix bugs across the project. Pay special attention to runtime warnings and deprecation notices that hint a feature will be removed in a future version. The goal is a clean test suite with no warnings.

Steps:
1. Run `pytest -v` and note any failures or warnings.
2. Reproduce reported issues and write regression tests if missing.
3. Fix the underlying problems or open issues describing the root cause.
4. Eliminate warnings by updating calls or replacing deprecated APIs.
5. Document each fix in an experience report and update the CHANGELOG if user facing.
