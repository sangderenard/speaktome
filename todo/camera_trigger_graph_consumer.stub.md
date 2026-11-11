# Connect the existing camera trigger graph to optical consumers

Continue `spectral-analyzer/camera_software/trigger_bus.py` and its handoff
`docs/CAMERA_TRIGGER_BUS_2026-11-11.md`. The existing mixer and GraphSolver resolve
instantaneous camera controls with original exposure identities. Connect actual
fixture identities and emission at source under existing camera barriers; apply
flash energy once, retain natural emission, and do not treat control weights as
new radiant power. Define optical detection/rearm behavior before claiming a
peanut slave. A delayed consumer needs existing persistent time integration.

Studio has a placed native housing but analytical thin-lens optics. Integrate
the existing thick-lens software rather than a replacement. Photographic shutter
duration must stay separate from render wall time, and ISO requires calibrated
sensor/back response. No production deployment has been performed by this seam.
