# Ascii Oscilloscope Agent Guidance

This subproject experiments with C++ rendering utilities. The headers under
`include/asciioscilliscope/` describe the canonical API and their comments are
considered part of the specification. Do **not** trim or rewrite those header
comments.

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

Implementation files in `src/` are intentionally stubbed. Follow the repository
coding standards for stub blocks when expanding functionality. Mirror the
expectations outlined in the headers and keep the code compiling with minimal
behavior.

