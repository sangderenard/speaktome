# Agent Reference: asciioscilloscope Interfaces

The asciioscilloscope headers define the structure that implementations must follow.  Automated agents should consult this summary when generating code.

*Keep header comments intact.*  When implementing methods use the high‑visibility STUB format documented in `AGENTS/CODING_STANDARDS.md`.

## Key Components
- `BeamFocus` – focus or diffuse an electron gun mask.
- `DiffusionKernel` – apply circular diffusion over a tensor.
- `ElectronGun` – generate high definition diff masks per time step.
- `PhosphorScreen` – queue excitations, apply decay envelopes, and optionally diffuse.
- `PixelFrameBuffer` – 5D double buffer providing sparse diffs.
- `CharDisplay` – diff character frames for terminal output.
- `CharClassifier` – map RGB values to ASCII ramp.
- `Interpolator` – resample tensors to new dimensions.
- `SampleSiteGrid` – map HD tensors to a reduced sampling grid.
- `Renderer` – tie everything together for basic rendering.

Implementations may be stubs but should respect these interfaces so that future work can expand them without breaking consumers.
