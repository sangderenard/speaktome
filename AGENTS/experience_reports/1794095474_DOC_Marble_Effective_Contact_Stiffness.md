# Marble effective contact stiffness

**Date:** 2026-11-07

## Overview

Marble Madness and the Studio compiled preview now expose effective soft-sphere contact stiffness presets while retaining the existing 3D law and time step.

## Steps Taken

- Read `fountain3d.py` normal contact, `stiffness_of`, CFL guard, and `ComputeRuntime` uniform ownership.
- Added glass-like (150), metal-like (180), and hard ceramic-like (195) contact-frequency presets. The law's ceiling is 200. The Hertz blend remains zero for these presets, so the selected frequency is not silently reduced by its stability clamp.
- Exposed custom frequency 40–200 in both Marble Madness and Studio. Studio changes restart its particle preview. Marble Madness changes restart the run, because browser runtime inputs ignore later server defaults within the same epoch.
- Verified focused Python tests (4 passed), JavaScript syntax, the isolated Studio preview contract check, and `git diff --check`. The existing combined Node preview check could not import `compute.js` because the generated `turing-wasm-shell.js` is absent; no Wasm build was attempted.

## Observed Behaviour

The 3D law uses `kn = m_eff * omega²` between particles and `kw = m * omega²` at walls. `omega` is in the simulator's units; it is not a SI Young modulus. At `DT = 1/360`, explicitly integrating real stiff contacts at marble scale would demand a smaller step or different contact model. The supplied presets order effective DEM compliance within the present stability range and do not claim measured material stiffness or different optical roughness. Saved designs do not yet store contact frequency; Studio's control applies to its live preview only.

## Next Steps

If physically calibrated metal, glass, or ceramic contacts are required, derive the relevant contact-scale stiffness and collision time from material and geometry data, then validate a smaller step or implicit/contact-constraint method against impact and rolling measurements. Track this separately from the present stable preview presets.

## Prompt History

> make sure the sim has the stiffness of marbles apropriate to metal or glass or ceramic

> User requests marble sim stiffness appropriate to metal/glass/ceramic. Audit 3D law omega/hertz/restitution/CFL limit and saved material physical defaults. Implement truthful optional contact presets/controls for Marble Madness and Studio if feasible without compiler rebuild, distinguishing effective DEM stiffness from physical Young's modulus. Keep within stability guard and document mapping; do not pretend exact material stiffness if impossible. Coordinate with studio_test_drive files if needed (agent completed). Focused tests and report; no Wasm. User low credit, finish promptly.
