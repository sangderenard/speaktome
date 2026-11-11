# Camera trigger bus through the original graph and mixer

Source-only Spectral work implements `camera_software/trigger_bus.py` using the
original ExposureFrame, RoutingGraph, RoutingGridView, materializer compile_nodes
entry and GraphSolver. The panel holds the same graph that is resolved. No audio
voices, replacement UI, replacement solver or scheduler were introduced.

The camera sources are Sync, Flash (existing slice weight) and Gate (sensor
integration state). Instantaneous wires preserve named receivers and original
causality identities. Outputs remain complex control signals, not photon energy.
Unsupported delayed/nonlinear/feedback/parameter/meta/wide-transfer settings are
refused. No optical-slave/peanut model is asserted.

Baseline focused suites passed 50 tests. The new 15-case suite plus those gates
passed 65 tests. The actual CPU Pygame RoutingGridView was rendered outside the
pytest GUI stubs; right-click disconnected, redraw preserved the deletion, and
double-click reconnected with each state verified through the real solver.
The screenshot `spectral-analyzer/build/camera-trigger-bus-panel.png` was inspected.
No GPU context, service restart or web publication was performed.

## Next Steps

See `todo/camera_trigger_graph_consumer.stub.md` and the fuller boundaries in
`spectral-analyzer/docs/CAMERA_TRIGGER_BUS_2026-11-11.md`. Native Studio camera
housing is already placed but its optical consumer remains analytical thin lens.
Connecting fixture emission, thick-lens software and calibrated shutter/ISO
remains unfinished; preserve the existing camera timeline and barriers.

## Prompt History

User: "spectral analyzer has a produce that is a signal graph resolver we could buy ourselve a LOT by using it for the flash busses"

User: "we can use its interface directly, just the mixer board gui is enough for a dumb wire bus"
