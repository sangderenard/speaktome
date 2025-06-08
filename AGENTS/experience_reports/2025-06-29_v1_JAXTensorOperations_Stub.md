# JAXTensorOperations Stub

## Overview
Added a stub class for a future JAX backend by following the "prototype stubs job" instructions.

## Prompt History
- "draw job and perform task"
- "Agents unsure what to work on can request a task via the job dispenser" (from AGENTS.md)

## Steps Taken
1. Ran `python -m AGENTS.tools.dispense_job` which returned `prototype_stubs_job.md`.
2. Implemented `JAXTensorOperations` stub in `speaktome/core/tensor_abstraction.py`.
3. Added a `test()` method verifying the stub raises `NotImplementedError`.
4. Documented progress in this report.

## Observed Behaviour
The new stub class raises `NotImplementedError` when instantiated, as expected.

## Lessons Learned
Stub templates clarify future intentions and keep development organized.

## Next Steps
Integrate JAX detection logic and implement the operations.
