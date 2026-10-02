# Turing vehicle whole-graph SSA parity

Migrate `solveMechanicalGraph` and the plastic/fracture state update from the
worker-authored JavaScript in `turing/src/compiler/state_loop_deployment.py`
into one shared repository-SSA graph that can emit both the game worker target
and the native C double-double target. Preserve all 241 nodes, 339 edges,
hardpoint identities, bushing dissipation, link-length actuators, and damage
state. This remains useful for native damage-state fidelity and tuning, but is
not a presentation launch gate. The user's parity boundary is the authored
API/ABI entering the compiler; a read-only scientific shader may visualize
compiled state before this optional consolidation is complete.

Source report:
`../1787892021_DOC_Mechanical_Creature_One_Sided_Terrain_And_Tire_Volume_Support.md`.
