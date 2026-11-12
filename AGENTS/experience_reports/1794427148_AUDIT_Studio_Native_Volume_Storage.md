# Native volume and sparse storage inspection

Source audit and authorized storage-only repair in spectral-analyzer on
2026-11-11 UTC. No GPU context, driver compilation, dispatch, render,
screenshot, service reload or deployment was attempted. CPU/source tests
included offline glslang validation. No rendered result is claimed.

## Findings

- The Studio camera source in spectral-analyzer/demo_pluck_gl.py already
  computes native per-band material/region/dye Beer attenuation and boundary
  Fresnel/Snell/TIR. The optional response rows copy these evaluated results.
  Its march_volume_layers helper is explicitly an exterior-air display march,
  not an all-material interior scattering solver. Region composition restricts
  it to valid exterior air; marble_preview_worker.py sets vol_alpha=0.
- The optional directional interior observer records actual refracted segments
  before the native Beer update, retaining a canonical Beer-integrated track
  estimate. It remains separate from the unchanged global scalar integration
  and is not complete angular radiance. ExtinctionGrid combined with this
  observer is explicitly refused.
- The inspected SceneResponseStore._put gate omitted rows lacking a surface
  recipient, a copied position, or in-bounds support. Virtual partition hits
  and no-hit completed samples can therefore be omitted from raw retention.
  The authorized repair separates raw retention from surface eligibility.
  One eighth of the existing row allocation is a bounded native-only FIFO;
  the remainder retains surface rows. Raw-only metadata admission cannot evict
  surface records. Evictions, out-of-scope rows and native-only context-capacity
  discards are explicit. Actual known endpoint ROI/category restrictions
  remain; misses have no fabricated endpoint or spatial-support claim.
  Namespace-zero records stay unavailable to the application stage.
- Native triangle groups support explicit per-triangle UVs, barycentric hit
  mapping, and CPU/GPU UV accumulation. That accumulator merges counts,
  source/tag bitfields, normals and complex spectral amplitudes. It does not
  preserve every ray's exact position, direction and identity. The existing
  manifold/noodle LUT interpolates surface-plus-direction keys and is also
  not a lossless record replacement.
- Turing's marble_response_application.js already packs sparse uint32 response
  rows into a two-dimensional RGBA word table without changing payload bits.
  A 2D storage address can retain explicit 3D vector columns. Physical surface
  UVs alone cannot identify interior depth or distinguish overlapping layers.
- The separate native pipeline has PathVertex and EndpointRecord contracts,
  complex regular/k-d FieldGrid storage and T4 field-plane copying. These are
  existing native mechanisms, but this inspection found no direct connection
  from them to the current Studio GLSL response consumer. They must not be
  presented as a newly integrated universal volume marcher.
- Array retention/texture layout and shader compiler working memory are
  different budgets. This source inspection does not identify the cause of
  the reported NVIDIA compile-only memory exhaustion.

## Verification

The focused CPU/source command passed 74 tests:

`python -m pytest tests/test_studio_scene_responses.py tests/test_studio_response_factory.py tests/test_studio_response_lineage.py tests/test_studio_completed_samples.py tests/test_studio_copied_components.py tests/test_studio_forward_responses.py -q`

New checks preserve bit-exact virtual/native-only camera rows, forward interior
Beer and region data, escape/no-endpoint semantics, category/ROI omission,
surface retention and unavailable denominators under native-only floods,
metadata pressure, and partition-receipt identity. Shared binary strides are
unchanged. Receiver retention validation is coordinated with its existing
owner. The guestbook validator passed in read-only mode.

## Prompt History

User: "we could splat to anything that could encode vector locations in a 2d surface in 3d"

User: "server side volume march until we can compile more of the ray tracer client side?"

User: "you understand if you rederive anything from the data package for the second phong pass you're doing it wrong"

User: "all you have to do is include extra rows for extra data the ray tracer ALREADY FUCKING HAS"

User: "Do you understand i didn't say only use forward rays"

User: "every fucking volume of any fucking material is part of the ray trace and when it isn't permeated by the field data but it has transparency YOU DIDN'T DO IT RIGHT"
