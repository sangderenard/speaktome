# Studio native camera preview visibility

The exposure-side `camera_visible` switch is wired to the Spectral native scene
builder. Studio's Phong preview has no camera render mesh; its existing
`/api/marble_camera/housing` contract explicitly says `renderMesh: false` and
exports only position vertices for mount calculations. Add native face/material
ownership to that existing endpoint, then render the actual housing in Studio
under the same setting, without substituting a marker or deleting the physical
camera from unrelated light paths. Validate the Phong and exposure settings
against one resolved camera pose. Do not run a GPU picture until the separate
field-information condition in `turing/AGENT_OBSTINANCE.md` is satisfied.
