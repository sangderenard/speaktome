# Ascii Oscilloscope Human Guidance

The C++ headers located in `include/asciioscilliscope/` outline a proposed CRT
simulation pipeline. They document responsibilities for each component and are
treated as the reference design. Implementation files currently contain minimal
stub logic so the project can compile. Contributions should preserve the header
comments and extend the stubs rather than rewriting the public interface.

## New Standard: Material-Resolved Volumetric Field Solver

### Overview
The project now supports a material-resolved volumetric field solver for simulating beam deflection systems. This approach models materials rather than behavior, enabling realistic CRT emulation.

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

To build a small demo:
g++ -std=c++17 -Iinclude -I./eigen src/*.cpp main.cpp -o osc_demo
This compiles the stubs and produces `osc_demo`, a placeholder program that does
nothing beyond constructing a `Renderer` and immediately exiting.
