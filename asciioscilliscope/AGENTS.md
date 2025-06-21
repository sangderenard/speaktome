# AsciiOscilliscope

This folder contains a prototype ASCII oscilloscope implemented in C++.
The headers in `include/asciioscilliscope` document the intended design
and should be preserved verbatim. Implementations may currently be stubs
but must compile.

## Quick Setup

1. Ensure the `eigen` submodule is initialized: `git submodule update --init asciioscilliscope/eigen`.
2. Build the demo with a C++17 compiler. For example:
   ```bash
   g++ -std=c++17 -Iinclude -Ieigen src/*.cpp -o osc_demo
   ```

## Non-Interactive Setup

No additional steps beyond compiling the `osc_demo` example.
