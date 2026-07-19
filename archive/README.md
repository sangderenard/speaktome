# Ascii Oscilloscope

`ultimate.cpp` implements the active renderer. Historical experiments live under `archive/`.
This version introduces a fully double-buffered design:
1. High-resolution render buffer (double buffered)
2. Phosphor grid (double buffered)
3. Diff buffer for change detection
4. Character classification buffer (double buffered)
5. Terminal display buffer (double buffered)

## New Standard: Material-Resolved Volumetric Field Solver

### Overview
The project now incorporates a material-resolved volumetric field solver for simulating beam deflection systems as expressive constraints. This approach models materials rather than behavior, enabling realistic CRT emulation.

### Key Features
1. **Ferromagnetic Topology**:
   - Model ferrite geometry, permeability tensors, and coil windings.
   - Define material-space properties for accurate field simulation.

2. **Field Computation**:
   - Use Biot–Savart Law and magnetostatics solvers to compute B-fields.
   - Resolve 3D B-field at every grid point in the neck volume.

3. **Beam Motion Simulation**:
   - Fire real beamlets and track their motion using Runge-Kutta or Boris push methods.
   - Discover emergent distortions and beam edge smearing.

4. **Dual Modes**:
   - "Convenient" mode for fast, approximated field simulation.
   - "True Simulation" mode for full B-field computation and realistic distortions.

### Implementation Notes
- Use Eigen for FEM-style grid-based solvers.
- Store B(x, y, z) in a sparse grid.
- Expose ∇·B = 0 as a constraint.
- Enable batch ray-pushing for beams.

### Optional Features
- Ferrous sculpting mode for real-time field visualization.
- Field microscope to track divergence of electrons.
- Misalignment minimap for convergence error visualization.

`signal_input.h` now provides an inline `start_signal_reader` helper.
