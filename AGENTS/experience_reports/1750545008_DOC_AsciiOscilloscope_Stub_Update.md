# AsciiOscilloscope stub update

## Overview
Created stub implementations for the C++ oscilloscope to match header
APIs and documented them in new guidance files. Initialized the Eigen
submodule and verified compilation of a small demo.

## Steps Taken
- Read repository guidance and coding standards.
- Initialized `asciioscilliscope/eigen` submodule.
- Replaced implementations in `src/` with stubbed versions.
- Added agent and human guidance documents under `AGENTS/guidance/`.
- Created a minimal `main.cpp` and compiled with g++.
- Ran `python testing/test_hub.py` (tests skipped due to setup).

## Observed Behaviour
The resulting `osc_demo` binary runs and prints "Stub renderer executed".

## Lessons Learned
Updating code to align with header expectations requires explicit
template arguments. Submodule initialization may be necessary before
compilation.

## Next Steps
- Flesh out the Eigen-based implementations.
- Integrate real rendering and signal input.

## Prompt History
```
please go through the asciioscilliscope project, you may need to update the submodule for eigen, look through the headers, they are the definitive coding standard and their comments should never be reduced but stand enshrining the intentions. Please consolidate them to agent and human guidance files, and put the expectations of the headers into the cpp, making sure to fit in any code that can stay with a comment block defining any adjustments it needs, and then otherwise heavily stubbed to reflect header expectations. I want the headers and cpp files to all compile and produce a program that is just dumbly passing things around definitions with stub comments saying what will eventually happen
```
