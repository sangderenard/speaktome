# Tensor Classmethod Refactor

**Date:** 1749887536
**Title:** Tensor Classmethod Refactor

## Overview
Implemented classmethod wrappers for common tensor creation routines so `AbstractTensor` mirrors PyTorch's module level API. Updated scorer logic and tests to use the new style.

## Steps Taken
- Converted `full`, `zeros`, `arange`, and `tensor_from_list` to class methods
- Updated `speaktome.core.scorer` to call `AbstractTensor.arange` and `AbstractTensor.zeros`
- Modified `tests/test_next_token_logprob.py` to use the classmethod constructors
- Ran `python testing/test_hub.py` *(fails: Environment not initialized)*

## Observed Behaviour
Compilation succeeds and unit tests still abort due to missing environment setup.

## Lessons Learned
Providing classmethod access preserves the existing ops-based usage while enabling a torch-like API.

## Next Steps
Continue migrating other tensor utilities and expand coverage when the environment can be initialized.

## Prompt History
- "I hate to ask but can you take another look at this ... get out of the ops paradigm"
- "always check the files in the repo ecosystem for your benefit..."
