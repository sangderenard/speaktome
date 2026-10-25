# Follow-up: bake reduced reproductions from the full arch/gun reference

- Treat the 3,414-free-DOF beam/joint system as reference truth.
- Record reference impulse, proof-load, droop, strain and moment trajectories.
- Bake smaller reproduction matrices at requested time/space scales and check
  them against those trajectories; do not fit new independent physics.
- Resolve the two outer stand-leg anchor connectivity warnings.
- Resume EngineCycleSim's full-native typed-state ABI work after this viewer
  integration is stable.
