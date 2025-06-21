# AsciiOscilliscope Human Guidance

Developers working on the C++ oscilloscope should treat the header
comments in `asciioscilliscope/include/asciioscilliscope` as definitive
specifications. Do not trim their text. Implementations may be stubs but
must preserve the described APIs.

When extending the code:

1. Keep the templated Eigen based structures intact even if current
   behaviour is minimal.
2. Document incomplete areas using the stub comment format defined in
   `AGENTS/CODING_STANDARDS.md`.
3. Update this guidance if new headers appear.
