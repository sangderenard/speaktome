# Studio scene response consumer, source only

The canonical shader no longer reconstructs lights/material response from
direction moments or suppresses primary Phong by material flags. Prepared
direct lookup of the native 48-byte application rows, exact recipient/cell/
stencil/current-query matching, and independent original-sample availability.
The volume slot retains pre-first-recipient light separately. An integer
recipient/depth copy gates both downstream slots to the actual nearest
fragment. Sorted other layers retain ordinary fallback. The final resolve
adds volume once. Float32 storage/blending and an explicit target allocation
budget reject unsupported devices/viewports without changing scene geometry.

The response-copy control is separately default-off. Typed snapshots replace
atomically through the existing authenticated field route, without fetching
the unused legacy atlas in response mode. Current producer/API/decoder
application-ready flags remain false. This is not a rendering-completeness
claim. No GPU context, render, screenshot, exposure, deployment or restart
was run. Four Node CPU checks and three actual project-IR/offline glslang
tests passed. Ordinary fragment preprocessor tokens match the prior HEAD.

Source contract, exact commands and support limits are in
`turing/docs/STUDIO_SCENE_RESPONSE_CONSUMER.md`; the corresponding receipt is
`turing/docs/hazards/20261111T194448Z_studio-scene-response-consumer.md`.

## Prompt History

User: "also you need to alpha to the phong in sparsity of data. an empty region is not black, it's no information"

User: "you understand if you rederive anything from the data package for the second phong pass you're doing it wrong"

User: "you fucking have the info when it is made into a fucking response by the fucking ray tracer before fucking integration ... make copies there"

User: "all you have to do is include extra rows for extra data the ray tracer ALREADY FUCKING HAS"
