# Existing divider request boundary

2026-11-11 UTC. Read-only architecture inspection and existing CPU/admission
tests on DESKTOP-LCKQH36. No compiler source, adapter, live job, service,
shader, transport, or runtime configuration was changed. No real compiler,
Nodus submission, GPU context, GPU link, dispatch, render, or restart ran.

## Observed state

- Canonical Turing: `C:/dev/Powershell/turing`, main `0485ded4` at inspection.
  The canonical Turing and Nodus trees contain substantial unrelated dirty
  work. Those edits were left alone.
- The artifact-accounting implementation remains in clean worktrees
  `turing-site-compile-artifacts-20261111` (`fabc2d6a`) and
  `C:/Users/alber/nodus-site-compiler-artifacts`. Its document explicitly says
  source implementation, not live deployment.
- Read-only SQL inspection of original upload `a7257bbac893` found 377
  pieces, all requested as wasm: 121 `needs-annotation` and 256 `queued`.
  No attempt was leased, complete, or refused. The job's `running` label
  therefore is not evidence that a compiler is running.
- The immutable uploaded `demo_pluck_gl.py` has SHA-256
  `e5c56348070922b3bcf2631a82f02b4b573ddc162b4c1dd04284462d913c274f`.
  Pieces 55 (`_prog`), 81 (`_gpu_ray_field`), and 238
  (`_gpu_ray_field_prebuilt`) still have missing annotations in that upload.
  Their original-source counterparts now have annotations. No upload or job
  was silently replaced or requeued.

## Existing mechanisms and exact boundary

The existing source divider uses `authored_compilation_growth_plan` for cheap
AST triage before the compiler's real cut. It is not a post-GLSL splitter.
`divider_nodus_bridge._express` only admits one plain module-level function
whose positional parameters and return are all annotated AbstractTensor.
It rejects variadics and keyword-only arguments and writes rank-one f64
columns. The bridge serves c/llvm/fortran, not the upload's wasm backend.

This is the adapter's restriction, not a lack of generic Nodus transport:

- `nodus/include/generic_object.h` already defines GenericObject, class-bound
  methods, and object-reference SlotValue fields.
- `nodus/include/table_abi.h` already exposes typed raw-byte and opaque-pointer
  edge FIFO publish/consume operations.
- `turing/src/compiler/oop_schema.py` and `tests/test_oop_interchange.py`
  already define and exercise the SCHEMA V1 class interchange.

The Nodus row request and `_buffer_abi` in
`nodus/tools/turing_row_tool_compiler.py` consume numeric Buffer ABI V2.
`_nodus_dtype` spells numeric/bool dtypes, not a generic object/reference ABI.
Neither the request schema nor its compiler.lower call carries root_calls.
The compiler already accepts genuine call records at its sanctioned entry:
`called_inventory.post_root_calls` consumes actual Python values/classes and
TensorClass/MappingClass/RecordClass descriptors. Raw JSON round-tripping
tuples or classes is not that contract.

The actual `_prog` accepts variadic `(str, int)` shader pairs and returns an
integer. `_gpu_ray_field` and `_gpu_ray_field_prebuilt` mix arrays, objects,
optional scalars, mappings, and non-tensor results. Their existing callsites
were inspected, including the prebuilt call in
`camera_designer/marble_preview_worker.py`. Renaming these inputs
AbstractTensor or deleting the precheck would not establish their real ABI.

No implemented binding from the generic transports into this row-request
schema was found in the inspected canonical and artifact-accounting paths.
That is an inspection limit, not a claim that no other worktree contains one.
The unresolved decision is recorded in
`todo/divider_request_boundary_contract.stub.md`; no schema was invented.

## Verification

- `tests/test_meowdoku_divider_annotation_probe.py`: 3 passed in 0.75 s.
  These are scoped admission/adapter checks, not compiler acceptance.
- Spectral `test_pluck_ray_annotations.py` plus
  `test_studio_gl_ownership.py`: 19 passed in 2.30 s. AST and mocked GL only.
- First probe invocation had three setup errors because packaged LOCALAPPDATA
  resolved to a nonexistent parent. Repeating with the verified explicit
  `C:/Users/alber/AppData/Local/Nodus/UserFiles` root passed. It did not
  require or change a product source file. All pytest temporary state was
  directed outside the repositories; no cache provider was enabled.
- No full suite, actual compiler inference, cross-language binary, or live
  Site-to-Nodus acceptance is claimed by these tests.

## Prompt History

> delete your task specific stage splitter

> use the existing source divider and report any error and FUCKING STOP DOING SHIT FROM CREATIVE IMPULSE TO DEFY ME

> okay the source splitter, if you fucking look at it, is not for after it is glsl

> use a test for that and in the meantime just annotate the source please

> there is a reference or generic object and also a byte string fifo

## Authorized isolated implementation and real trial (2026-11-12 UTC)

The opening audit above describes the initial read-only phase. Later explicit
direction authorized a default-off unstable permission and an actual
non-AbstractTensor source attempt. Owned worktrees were created from the
existing artifact-accounting integration, leaving canonical dirty work alone:

- Turing: `C:/Users/alber/turing-divider-unstable-20261111`, branch
  `codex/divider-unstable-permission-20261111`, base `fabc2d6a`.
- Nodus: `C:/Users/alber/nodus-unstable-source-20261111`, branch
  `fix/unstable-source-only`, base `3cd6a33`, now through `f5bf718`.
- Existing API-v2 dependency: detached SpeakToMe `86721dd` at
  `C:/Users/alber/speaktome-divider-unstable-api2`.

The grouped `compile_unstable` permission is granted by no default account
role or access tier. Operator enablement and per-upload opt-in permit the
existing Site queue and native Nodus host to call the existing source divider
and lowerer without synthetic tensor columns. A separate explicit unchecked
inference-admission option lets unannotated pieces reach the compiler without
assigning types or changing source bytes. Current grants are rechecked at
dispatch. Source-only success is explicitly emitted-not-built and cannot be
adopted; no runtime generic ABI is claimed.

The complete current annotated source SHA
`903dba7c938a77e7881058b0c8db0732524955440407ea040fcfe57fed1e88b1`
was uploaded unchanged through an isolated authenticated Site. The existing
queue selected `main.<locals>._doc_id`. An initial adapter name-validation
error was corrected using existing qualified-name conventions. The next real
SSA failure exposed an existing partition bug: it dropped the authored
`_doc_misc_ids: dict[str, int] = {}` declaration. A scoped lexical dependency
repair preserves that declaration without inventing types. Partition tests
pass, and the mapping identity audit has zero unsourced facts/identities.

The next integrated trial reached the captured input's unresolved dtype.
Evidence is retained under
`C:/Users/alber/AppData/Local/Nodus/UserFiles/original-source-trial-bb841e6f357342729b2d9e46722d8624`.
It is a failed SSA-lowering attempt, not successful compilation. Further
inspection found enclosing mapping column contracts are not inherited along
with closure aggregate kinds; that narrow evidence propagation is under
investigation. Existing mapping contracts, not invented homogeneous tensor
types or call values, are authoritative.

Checkpoint verification: 171 focused Turing tests passed, 8 partition tests
passed, mapping audit 0 findings. Nodus 85 passed, 4 compiler-dependent tests
skipped, native host regression tests passed with aborting adoption stubs.
The CPU-only native trial harness was built for this check. No original-source
native/GPU product, plugin adoption, rendering, production grant/configuration,
live restart, push, or full-suite run occurred. All original live jobs remain
unchanged. Full implementation receipt lives in the owned Turing worktree:
`docs/SITE_UNSTABLE_SOURCE_2026-11-12.md`.

Additional prompt history:

> can you turn off the annotation check to see if there is a real error, that sounds like avoiding letting the compiler resolve types

> it occurred to me that type resolution should be a privlege/right gate not always available

> So you're saying the nodus is really conservative with what it allows ita not open for the unstable features, let's group permission for undoing that limitation as an unstable option because it doesn't have to be abstract tensor

> So do try to use it without abstract tensor

Final code checkpoint: Turing `4010f365` contains the tested source/permission
adapter and source-backed closure/partition/planner repairs; Nodus remains
`f5bf718`. The combined gate passed 192 tests; 33 focused compiler checks and
the zero-finding mapping audit passed. These remain isolated, not merged,
pushed or deployed. Ignored test caches were retained outside the worktree.

The full original trial `original-source-trial-011a8ab141cc48d188d31c5535060e48`
reaches final SSA concordance. Its three table-helper failures are a real
string-key ABI mismatch: authored key:str is a glyph span, while map keys are
scalar string-table tokens. Existing full-content string equality forbids
hash-only substitution. No safe dynamic conversion covering insertion and
lookup was established; no detector bypass or invented span output was added.
The compiler records shared sequence-arena references across helpers, not a
dictionary copy per call. This is not a tested runtime alias of the original
CPython dictionary: Nodus receives a compile-contract ID plus source JSON, and
no source product was emitted, linked, adopted or rendered.
