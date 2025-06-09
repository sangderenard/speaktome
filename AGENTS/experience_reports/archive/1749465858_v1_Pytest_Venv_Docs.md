# Pytest Venv Docs Update

**Date/Version:** 1749465858 v1

## Overview
Standardised documentation to reference `pytest` from the repository's virtual
environment. This avoids missing dependencies when users run the tests.

## Prompt History
- System: "You are ChatGPT, a large language model trained by OpenAI."
- Developer: "always check the files in the repo ecosystem for your benefit..."
- User: "make sure any use of pytest in the code uses pytest on the venv to avoid missing dependencies"

## Steps Taken
1. Searched the repository for `pytest` usages.
2. Updated README, CONTRIBUTING, testing guidelines, job descriptions, workflow
   config, and tests documentation to call `.venv/bin/pytest` (or the Windows
   equivalent).
3. Ensured no other files required changes.

## Observed Behaviour
No functional changes; documentation now explicitly calls the venv's
`pytest` executable in all instructions.

## Lessons Learned
Documentation consistency helps avoid environment mismatches when running
the test suite.
