# Ascii Oscilloscope

`ultimate.cpp` implements the active renderer. Historical experiments live under `archive/`.
This version introduces a fully double-buffered design:
1. High-resolution render buffer (double buffered)
2. Phosphor grid (double buffered)
3. Diff buffer for change detection
4. Character classification buffer (double buffered)
5. Terminal display buffer (double buffered)

`signal_input.h` now provides an inline `start_signal_reader` helper.
