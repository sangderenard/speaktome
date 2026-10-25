# Continue AbstractTensor compression residency

- Replace eager GLSL tensor materialization with resident tensor storage.
- Convert remaining generic decode-only Python prefix scans to tensor state.
- Restrict `.tolist()`/byte serialization to final file I/O boundaries.
- Add a live Mandelbrot recording regression with progress and residency checks.
