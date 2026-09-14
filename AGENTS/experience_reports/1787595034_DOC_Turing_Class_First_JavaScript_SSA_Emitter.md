# Documentation Report

**Date:** 1787595034
**Title:** Turing class-first JavaScript SSA emitter

## Overview

Started the JavaScript destination for Turing's generic repository SSA. The
first-class surface is deliberately `SSAClassTable`: preserved classes become
real JavaScript classes whose fields retain their numeric SSA slots and whose
methods invoke their linked SSA function bodies. The same module also prints
generic free functions and arbitrary CFGs through a block dispatcher, forming
the host-language foundation for later WGSL and WebAssembly regions.

## Steps Taken

- Read the workspace, Turing, test-hazard, and guestbook guidance plus the
  relevant compiler, WASM, WebGPU, class-schema, and prior deployment reports.
- Added `src/compiler/ssa_javascript_backend.py` with explicit artifact and
  shortfall types, class emission, call-closure emission, memory helpers,
  scalar operations, and a dependency-free ESM ABI.
- Made class-only modules valid outputs with no fabricated root function.
- Mapped numeric receiver-field addresses through immutable emitted
  `fieldLayout` metadata to named JavaScript properties.
- Recovered planned-region returns from call-site `output_ids` contracts.
- Added `src/compiler/class_emission_plan.py` as a destination-neutral join of
  class-table identities/layouts, FunctionTable logical contracts, and physical
  SSA method bodies. JavaScript consumes it now; C++ and Java can consume the
  same plan later.
- Preserved class, field, method, and logical parameter order of appearance.
  Numeric slots and reordered SSA formal positions remain explicit ABI
  coordinates rather than becoming sort keys for the emitted source surface.
- Corrected target-neutral lowering so nested instance calls retain their
  receiver according to the FunctionTable record-alias contract, and returns
  are rebound to the real producer before duplicate SSA ids are freshened.
- Added a canonical JavaScript numeric-operator table covering all 57 portable
  elementwise operations. Both direct repository opcodes and named
  `Call[tensor_operation=...]` instructions resolve through that one table.
- Exposed the Python oracle's actual 46-operation elementwise inventory and
  made parity an executable assertion; JavaScript is now present in the shared
  backend inventory alongside C, LLVM, Fortran, and Wasm.
- Fixed repository `And`/`Or`/`Xor`/`Not` emission to remain integer-bitwise;
  Boolean `LAnd`/`LOr`/`LNot` retain their separate semantics.
- Added executable Node tests for a class constructor/method, a missing method
  body, a class from the canonical source compiler, a Phi-driven typed-array
  loop, and an unsupported instruction.
- Documented the boundary in `turing/docs/JAVASCRIPT_SSA_EMITTER.md`.

## Observed Behaviour

`python -m pytest tests/test_ssa_javascript_backend.py -q --tb=short` passes
all five original tests. The expanded class/emitter gate passes eight tests,
and the combined class/emitter/entrypoint gate passes twelve. The emitted
JavaScript was imported and executed by Node
22.16.0 in the tests. A canonical `lower_ast_source_to_ssa` class constructor
correctly wrote through SSA field slot 0 into the emitted object's `value`
property. A translated `Counter.twice` made two nested `self.bump` calls and
produced `8` with resident state `5`; a second instance independently produced
and retained `15`.

A deliberate ABI regression presents physical formals as `(second, receiver,
first)` while the FunctionTable contract remains `(self, first, second)`. The
public plan correctly stays `(first, second)` and records physical positions
`(2, 0)` for invocation.

A generated Node probe now executes every one of the 57 portable elementwise
operations in a single emitted SSA function. The combined operator,
JavaScript-ingestion, Python-oracle, class-plan, and Node gate passed 52 tests;
the final focused inventory/emitter/oracle gate passed 17 tests. The broader
SSA contract file has one unrelated stale assertion expecting 117 `Handler`
members while the current enum contains 118; both modified inventory tests in
that file pass.

`tests/test_process_graph_function_linking.py` has seven pre-existing failures
and seventeen passes, reproduced identically in a clean worktree at `57b5e25`;
this newly measured baseline is now recorded in `TEST_BASELINE_AND_HAZARDS.md`.
`tests/test_precompile_to_ssa.py` matched its documented single-failure
baseline with 56 passes. Ruff was unavailable: the system interpreter has no
module and the repository `.venv` points to a removed Python 3.10 installation.

The existing minimal `SSAClassDefinition` does not carry inheritance, field
types/defaults, static-method status, or full signatures. Optional
`ClassSchema` data can now enrich the shared plan only after its SSA projection
is proved to agree; absent facts are not synthesized.

## Lessons Learned

The class table cannot be treated as decorative API metadata. Compiler-lowered
methods address receiver state by numeric slots, so a JavaScript class needs a
real slot-to-property bridge or its method bodies and visible fields describe
two different objects.

Planned scalar region functions can legitimately omit `Ret`; their output
record lives at aggregate call sites. A generic emitter must reconcile that
contract before deciding a block is unterminated.

## Next Steps

- Emit the now-planned inheritance, defaults, static methods, and types where
  each destination can express them faithfully.
- Add shader/Wasm deployment-region imports and resident-memory seams around
  the JavaScript host module.
- Expand exact numeric coverage, especially a stated BigInt strategy for i64
  and an explicit precision policy for JavaScript `Number`.
- Define JavaScript lowering contracts for non-elementwise Python-oracle work:
  reductions, tensor construction/fill, and then shape/index operations.
- Add the JavaScript backend to the target-selection/publishing surfaces once
  their input type can represent a complete `IRModule`, not only a flattened
  `FusedProgram`.

## Prompt History

> "we need to start making the javascript emitter, so we can web host using a quality template for holding a program in javascript and web shaders, assembly. but before we do any shaders or assembly, we need to be able to print any generic program in javascript, it's going to be a huge project but it's quick for us to get what we need"

> "actually I'd like if you made it your first priority to grasp classes, which we have an ssa representatoin for even if it often gets lowered away"

> "feel free to use node for now, let's try to translate some simple python to javascript and inspect how our compiler handles the classes involved, see if we need a little new wiring. and I want you to look out for the fact that this is kind of also paving the way for when we emit c++ and java too, so, try to do things generally and professionally, harmonizing with my repo but bringing what's needed"

> "oh I've been thinking about that, we have deterministic positions and we do because that solved an ambiguity problem, won't we conform to general coding standards if we just maintain order of appearance"

> "I'm sure there are plenty of operators that need to be filled out in translation tables, we're of course going to have to define at the very least the translation of the pure python backend essential operators to javascript, if nothing else"
