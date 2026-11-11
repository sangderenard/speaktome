# Studio vector field and immediate material passes

Integrated the existing Spectral tracer with Turing Studio through its owned
photo field route. The optional immutable v3 bundle carries actual exterior
segment track-length throughput density, measured propagation vectors, native
triangle/material classes, source records and explicit authored material lineage.
Scalar transport remains byte-identical. There are no new rays or atmospheric
effects. The primary and volumetric/reflectance/glass passes run in one frame,
with linear composition and one canonical display conversion.

Initial real-browser gain-isolation testing exposed that legacy fog deposits
were exactly zero around the subject. More rays did not fix this. The corrected
recorder measures existing travelling packet energy independently rather than
modifying extinction, amplifying gain or inventing atmospheric scattering.
Actual unchanged-scene testing then demonstrated a field contribution at gain1.
The configurable six/twelve-bin, 32-cubed result remains coarse and uncalibrated; it is not a
complete mirror reconstruction. Depth-and-material visibility also retains a
known coplanar equal-depth tie limitation.

Source and test contracts are documented in
`turing/docs/STUDIO_SECONDARY_FIELD_2026-11-11.md`,
`spectral-analyzer/camera_designer/STUDIO_DIRECTIONAL_FIELD.md`, and
`spectral-analyzer/docs/STUDIO_SECONDARY_FIELD_SHADER.md`.
The actual browser receipt is maintained under
`turing/build/dot-secondary-browser-20261111/browser-receipt.json`.
Focused producer, API, Node lifecycle/orchestration, shader export and native GPU
checks passed. This is not a full-suite claim.

Concurrent source edits were preserved. The authorized coordinated v3 release
published37 changed assets, verified108 served hashes and74 retained products,
and gracefully saved/restarted through the existing supervisor. Boot28e70e0353c1
restored sessions with unchanged launch settings; other instances were untouched.
Receipt: turing/build/dot-secondary-release-20261111/verification.json. Shared
Studio/capture edits remain uncommitted rather than committing others' work.

The twelve-bin image exposed repeated source-energy accumulation:4464 fixed
roots replayed unevenly across131072jobs, yielding about29.44 times energy.
Root beta already includes area/spectral normalization. Correcting each root's
actual replay count, then averaging full-coverage pumps, is necessary rather
than arbitrary display gain or division by all rays. Sparse fixed wavelength
paths separately cause chromatic noise. V4 root accounting and64³ spatial
selection are being validated; they are not included in the live v3 receipt.

## Next Steps

The v6 implementation is now staged but not published. It adds a full-geometry
subject ROI at exact 0.5 mm pitch, configurable margin, category reception,
per-root replay normalization, cumulative fresh angular populations, explicit
internal Beer segment recording, and a field-only Live presentation. The actual
Live run completed 22 populations and 1441792 jobs with clean GL and no sensor
overlay requests, but its 64-cubed subject atlas stayed empty. A regenerated
source/rolling hash audit confirms zero direct root intersections; that is not
an indirect-path audit. Native seed changes rotate existing disk directions,
not independent origin/radius samples.

The user requested actual legacy/glass-rod comparisons and all-bounce arrival
records. Matched original 256-square sensor images, one sample per pixel and
one shared fixed exposure, show different material behavior; both field grids
remain zero. Images and raw sensor data are in Spectral repo-cache under
subject-engine-images-matched. A separate observational all-bounce ledger is
being checked against the exact Live bounce seeds. Initial controlled seed0
diagnostic records 93359 actual segments with bounce counts65536/27771/50/2,
zero ROI crossings before exclusions, no queued reactive rays and all paths
ultimately below the0.002 threshold. This is not a claim that camera/NEE data
has been included in the forward field. It also exposes an actual white-plaster
legacy surface loss that needs mixed diffuse/specular transport review.

See `todo/studio_vector_field_release.stub.md` for the explicit release boundary
and remaining calibrated-quality limitations. Final material correspondence and
browser evidence are recorded in the feature note and browser receipt.

## Prompt History

User: "Make a secondary shader and ensure it can run after the primary in immediate succession, it will do three passes, volumetric, all reflectance, glass the material has it so the light field is used to derive the scene painted onto that material, and maybe a phong specular override"

User: "The light field can discriminate in recording all the details of material area from metadata in the delivered product so the where to do what work can be anded with visible triangles"

User: "Yeah make the light field a server side vector bundle with metadata"

User: "don't force curpuscule rays just to see them, we need to work that out with particle sim for different atmospheric particles"

User: "man push your work forward faster this is rapid development, make the real one or make the demo the real one"

User: "I can't make sense of the view from memory of the room, let's put a real beefy stencil on and advance to that stage"

User: "Let's try to work in a denser grid with that stencil and examine the way we're making odd colors"

User: "Let's have a setting for division scale and make it opt in by category, subject, stage, room"

User: "Try the subject at .5 mm"

User: "You need to find that and expose it's levers and use the live mode but without the ray trace overlay just feeding the light field"

User: "Is that inner shadow caused by the engine used for glass transfer? We need the volume March to obtain complete internal volume field"

User: "It is fucking absurd that there would be zero energy so you have not satisfied my invariants, where are the volume records with the ray arrival record on the studied field area"
