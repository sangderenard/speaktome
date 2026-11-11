# Studio forward field routed to the existing Phong volume

Implemented in the canonical Turing and Spectral repositories. The explicit
receive_light_field API option routes immutable copies of the actual
three-line forward field, via existing job ownership authorization, into
Studio's existing uFieldVolume/uFieldGain input. No fake lights, separate
renderer or estimator change. Default jobs retain their old digest and do
no field readback or artifact writing.

The producer stores direct 650/550/450 nm energy, not camera response RGB.
The export reuses existing CIE/D65 conversion without film inversion or
normalization. The existing additive glow semantics are explicit: this is
not directional irradiance, surface BRDF integration or refracted lighting.
Still fields keep existing additive history; Live retains its fresh field
per sensor sweep. Browser replaces each snapshot, never sums it again.

Scene coordinates use the existing affine inverse and shader vPosV with
physical unit/origin/bounds. Outside-field samples contribute zero. Legacy
shader behavior is unchanged unless the new transform flag is enabled.
Room/Exposure mismatches, moving/rolled subjects and stale camera/scene
responses are withheld. Opt-out restores the original complete zero-volume
sampler. Existing scene packet consumers reject active unsupported resources.

Evidence: 153 focused Turing Python tests, 55 Spectral worker tests passed
(one Windows symlink creation test skipped); real
RTX 3060 production field/pump export; 45 real native Phong mapping checks
across three variants; project IR and five exported GLSL validations. Node
receiver/packing and existing photo/Live lifecycle regressions pass. The
unchanged old lighting default assertion expects obsolete A-off/B-half-power
values and also fails on untouched HEAD e92e60f9 in the detached baseline
worktree C:/dev/pf-field-baseline. Full-suite success is not claimed.

Actual signed-in Studio UI on an isolated real Sessions/API and GPU worker
has received a completed 96 cubed, 10,616,832-byte field. Detailed browser
receipt/screen verification lives in Turing build/dot-studio-field-20261111.
Actual draw gain 1, transform enabled and texture unit 8 were observed;
opt-out disabled field sampling without WebGL errors. The browser fixture
was stopped and its tab closed. No appearance improvement is claimed:
on/off variation was absent or comparable to off/off variation. Live's
browser lifecycle was not exercised.
Source/built material mismatch (missing ribbon rows) was repaired only in
that isolated fixture through the canonical registrar/MaterialDatabase.
No production restart or publication was performed. Turing documentation:
docs/STUDIO_FORWARD_FIELD_2026-11-11.md and related hazard receipts.

## Prompt History

User: "can we figure out light field accumulation for the phong shader and test it by making the studio automatically route that data, having opted in to receive it from the ray trace api"
