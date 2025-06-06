# Central Test Hub

**Date/Version:** 2025-06-11 v2
**Title:** Central Test Hub and Stubs

## Overview
Added placeholder tests for all classes and a script that reports which tests are marked as stubs.

## Prompts
"Can you do me a favor and put in a test for every single class there is, and for cases where it is prohibitively complex to create a real test, use your best professional instinct for making a stub for the following purpose: a centralized test hub that will dynamically determine which of the tests have been internally marked as stubs and then offering a todo list of stub tests and most recent outputs, create an agents instructions file for the test environment and procedures,"

## Steps Taken
1. Created `tests/AGENTS.md` explaining how to run tests and handle stubs.
2. Added `tests/conftest.py` to register the `stub` marker.
3. Implemented `tests/test_all_classes.py` with real tests for lightweight classes and stub entries for complex ones.
4. Wrote `testing/test_hub.py` which executes pytest and records stubbed tests in `testing/stub_todo.txt`.
5. Ran `AGENTS/validate_guestbook.py` to verify this report.

## Observed Behaviour
The suite runs successfully using the new hub script and outputs a list of stub tests.

## Lessons Learned
Stub markers help keep track of incomplete coverage while allowing continuous integration to run.
