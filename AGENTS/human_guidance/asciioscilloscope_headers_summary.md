# AsciiOscilloscope Header Overview

This document mirrors the intentions expressed in the asciioscilloscope headers.  It is provided as a quick reference for human contributors.

- **BeamFocus.h** – wraps a `PixelFrameBuffer` to apply beam shaping and spatial focusing onto a high‑definition tensor.
- **DiffusionKernel.h** – utilities for circular diffusion of energy values across a grid.
- **ElectronGun.h** – high‑definition electron beam simulation producing incremental diff masks.
- **PhosphorScreen.h** – multi‑channel phosphor screen with envelope based excitation and optional spatial diffusion.
- **PixelFrameBuffer.h** – generic 5D double‑buffered tensor manager for time‑sliced data streams.
- **CharDisplay.h** – double‑buffered character frames with diff generation and optional full print mode.
- **CharClassifier.h** – maps RGB triples to ASCII characters using a brightness ramp.
- **Interpolator.h** – resamples tensors between grids using configurable interpolation modes.
- **SampleSiteGrid.h** – defines reduced sampling sites over an HD plane with metadata.
- **Renderer.h** – orchestrates image ingestion, phosphor simulation, classification, and display output.

Each header contains extensive comments that must remain intact.  Implementations should include matching STUB blocks describing intended behavior.
