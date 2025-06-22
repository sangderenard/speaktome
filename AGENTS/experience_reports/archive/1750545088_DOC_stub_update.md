# Stub Update for Ascii Oscilloscope

## Overview
Converted C++ sources in `asciioscilliscope` to stub implementations that align
with the documented APIs in the headers. Added guidance files for humans and
agents explaining that header comments are canonical and implementations are
lightweight placeholders.

## Steps Taken
- Initialized the Eigen submodule.
- Created `AGENTS.md` and `HUMAN_GUIDANCE.md` in the project directory.
- Replaced existing source files with stub code matching the headers.
- Added new source files for previously unimplemented headers.
- Compiled the project with g++ to ensure stubs build.

## Observed Behaviour
Compilation succeeded producing a minimal executable `osc_demo` which simply
instantiates a `Renderer` and exits.

## Lessons Learned
Headers serve as design documents. Keeping them intact while using stubs allows
future work without losing intent.

## Next Steps
- Flesh out the stubbed methods when real rendering logic is needed.

## Prompt History
- "please go through the asciioscilliscope project ... I want the headers and cpp files to all compile and produce a program that is just dumbly passing things around definitions with stub comments saying what will eventually happen"
