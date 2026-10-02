# Turing Segmented WebAssembly Homepage Audit

**Date:** 2026-08-01
**Title:** Review and completion of the Turing segmented-WASM homepage

## Scope

Audited the Turing repository's new WASM class-graph handoff and prior agent
changes. The intended product was the existing reduced Mandelbrot homepage as
one compiled program object and API, with ProcessGraph-visible but privately
segmented browser execution. AVI/JPEG compilation was explicitly out of scope.

## Methodology

Read the workspace and Turing agent guidance, audited the handoff and commit
diff, traced ProcessGraph/FusedProgram/WASM ownership boundaries, added focused
regressions, repeatedly built the real 160-iteration homepage, served its
generated directory over HTTP, and executed the page in isolated headless Edge
through the browser developer protocol. The final browser assertion required
all three named output arrays to be finite and spatially varying.

## Detailed Observations

- The handoff had introduced backend-private interpretation of
  `tensor_from_list`, rejected genuine tensor constants, and reduced the class
  runner to one element.
- The first real build exposed a second constant case: two uniform constant
  operands in a binary ProcessGraph node. One is now preserved as a tensor
  constructor while the other occupies `right_scalar`.
- Varying constants are now packed into each WASM data segment; runtime arrays
  start after `reserved_bytes` and no longer overwrite tables/constants.
- The public `render` API now owns all four inputs and three RGB outputs.
  Logical output bindings are independent of which private region produces
  them.
- The initial generated page embedded about 12.8 MB of region base64. Following
  the user's correction, the page now references eight
  `site/v1/wasm/*.wasm`
  assets which are fetched and instantiated on first execution.
- Browser execution initially succeeded but produced flat output. Live feed
  inspection showed coordinate arrays were varying; the manifest had paired
  boundary-set order with module input names even though the emitted WASM ABI
  uses first-use order. Reusing `program_feed_order(spec.program)` repaired the
  silent operand routing bug.
- Final Edge run: 32x32, eight region URLs loaded, status good, 1,024 finite
  values in each RGB array, with 111/110/110 distinct values.
- The generated site was installed and published from the actual root
  `nogodsnomasters` Pages repository. Its root-only ignore rule received one
  narrow `site/` exception. GitHub Pages built commit `5be901d`; a final Edge
  run directly from the public HTTPS URL loaded all eight `site/v1/wasm/`
  regions and produced finite, spatially varying RGB outputs.
- Focused suite: 75 passed. A wider relevant suite had 131 passes and seven
  unrelated legacy failures in AST/bit-op schema tests and the explicitly
  excluded AVI-region compiler test. Repository-wide collection is already
  blocked by unrelated FluxSpring syntax/export errors.

## Analysis

The most important boundary is semantic: segmentation is deployment metadata,
not a new public object model. Tensor constructor meaning belongs in the shared
IR, while packing and loading belong in the backend. Likewise, graph boundary
sets are structural facts but are not automatically ABI parameter order. A
runtime can accept correctly shaped arrays and still be numerically wrong, so
browser validation must test output variation/content rather than only module
instantiation and a success status.

## Recommendations

No follow-up is required for this completed scope. If direct cross-module WASM
calls are revisited later, design a shared-memory ABI separately; do not bolt
them onto the independently-memoried host-runner deployment. Keep AVI/JPEG and
FluxSpring repair as independent tasks.

## Prompt History

> there's a new handoff document but i want you to be careful with it and the work the agent did, it might be right, but at the end instead of teaching process graph how to interpret tensor from list or maybe they were failing to get constant tensors working, or tensor from list isn't defined yet for web assembly, anyway they started doing bespoke crap so i wanted to call you in to review and complete the work

> nobody was ever supposed to try to compile that the avi jpeg encoding breaks the will of compilers, just the reduced little mandelbrot toy but now expressed with the segmented backend to run the example and the graph tab integration

> do you understand it's supposed to be both the object and its api and everything and just the way it runs on the site will behind there scenes be the segmented compile, i'm a little nervous if their handoff was flawed

> can you make this a goal and run auto to completion, when your done i want to be able to go to my github page and see the new content working

> i was thinking the segmentation plus coding file loads for pieces, lazy loading, we should make the html simple but load the web assembly being the scenes, don't let me interrupt you though

> i'm pretty sure they existed already i wouldn't worry about them unless you have extra time to just clean up
