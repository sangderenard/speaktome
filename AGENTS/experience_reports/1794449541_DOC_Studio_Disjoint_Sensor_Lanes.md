# Studio disjoint sensor lanes

**Date:** 2026-11-11 UTC. The owner required the tracer and system to use
disjoint forward-ray origins and sensors, with N lanes rather than a hardcoded
camera pair. The optical shaders were left unchanged. The Spectral timed worker
now owns a native root population, field state and sensor accumulator per lane;
it admits a lighting slice to all lanes and commits only after all transports
and receptions complete. Native roots from distinct seeds are checked for
duplicate position/direction/wavelength identities. Sensor strip seeds use a
unique pass/lane/strip namespace. Lanes may select the existing RGB or
monochrome transport basis independently. The built-in calibration uses RGB
and monochrome lanes in one job per declared light case; its earlier sequential
two-job path was removed.

Turing admits 1–32 resolved camera lanes for a timed untiled RGB Studio photo.
The existing browser icosphere supplies a twenty-face-center surround pattern
in one photo job, and a gallery can inspect authenticated per-lane image/HDR
artifacts. A `camera_visible` setting includes or excludes authored camera
geometry in the Spectral BVH while leaving the analytical sensor active. The
Studio Phong preview still has no camera render mesh; the existing native
housing endpoint declares mount-only geometry. Thus the requested Studio
preview visibility half of that setting is not completed.
The authored peanut shell and clear/frosted dome also remain pending.

Checks: Spectral 19 focused CPU tests passed, including native root energy and
N-lane barrier ordering. Turing 55 focused photo/pose tests passed, including
two new lane admission/artifact tests. Browser module syntax and pure
icosahedron face-center checks passed. No GPU exposure was run; the field
information prerequisite in `turing/AGENT_OBSTINANCE.md` remains open.

See `spectral-analyzer/docs/STUDIO_DISJOINT_SENSOR_LANES.md` for the current
contract and limits.

## Prompt History

> disjoint forward rays and sensors

> you are working on making the tracer, the system, work on disjoint forward ray origins and sensors

> no other way is permitted even slightly

> but n lanes not 2 right

> this is just packing and unpacking, the algorithm doesn't care where the sensor sites are

> make a setting at some point while this is cooking that makes the camera visible or invisible in both the studio and in exposures, and then we can also make a demo with multiple cameras where we expose an entire geodesic faces centers or whatever camera positions, a surround pattern, in one pass
