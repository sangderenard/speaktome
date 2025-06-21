# AsciiOscilliscope Agent Guidance

This document summarizes the intent preserved in the C++ headers under
`asciioscilliscope/include/asciioscilliscope`. The headers are
considered canonical for style and behaviour. When exploring or stubbing
implementations keep the commentary intact and mirror the documented
expectations in the source files.

Key components:

- **PixelFrameBuffer** – 5D Eigen buffer for spatiotemporal data. Provides
  `updateRender` and `getDiffAndSwap` for producer/consumer pipelines.
- **CharClassifier** – maps RGB values to ASCII symbols. Future versions
  may support tensor based batch operations.
- **CharDisplay** – double buffered character grid with a queue of
  time-stamped frames.
- **Renderer** – orchestrates downsampling, buffer updates and terminal
  output using the above primitives.

Stub implementations should compile but may do minimal work. Use the
`STUB:` block format from `AGENTS/CODING_STANDARDS.md` to describe the
planned behaviour inside each source file.
