# Ascii Oscilloscope Human Guidance

The C++ headers located in `include/asciioscilliscope/` outline a proposed CRT
simulation pipeline. They document responsibilities for each component and are
treated as the reference design. Implementation files currently contain minimal
stub logic so the project can compile. Contributions should preserve the header
comments and extend the stubs rather than rewriting the public interface.

To build a small demo:

```bash
g++ -std=c++17 -Iinclude -I./eigen src/*.cpp main.cpp -o osc_demo
```

This compiles the stubs and produces `osc_demo`, a placeholder program that does
nothing beyond constructing a `Renderer` and immediately exiting.
