# Token Orchard reference core

**Date:** 2026-10-24

**Title:** Dual-domain integer token core, native fingerprint kernel, snapshots, and writer service

## Overview

Implemented the first executable Token Orchard foundation in the directly
tracked root project `C:\dev\Powershell\string_core`. The work converts the
prior architecture receipt into a collision-safe, binary-aware,
encoding-explicit restricted-Python reference without modifying Turing, Nodus,
or Spectral Analyzer.

## Steps Taken

- Read the root, shared, local, testing, dependency, guestbook, architecture,
  continuation, and prior cross-repository glyph-token audit guidance.
- Wrote the executable Phase 0 integer and allocation contract.
- Implemented exact Unicode-scalar and arbitrary-byte domains, arithmetic atom
  IDs, aggregate SoA storage, fingerprint candidate indexes, exact collision
  checks, deterministic covers, flat batches, and core operations.
- Added strict encoding adapters and dual-token transform receipts.
- Replaced boxed authoritative lists with contiguous `array` buffers.
- Added deterministic little-endian SHA-256 snapshots, structural validation,
  derived-index rebuilding, and continuation after reload.
- Added a bounded multi-producer/single-writer queue prototype with grouped
  drains, request tickets, backpressure, and clean shutdown.
- Refactored the live two-lane FNV-1a step into a shared compiler-safe kernel
  using two `u32` limbs per lane and bounded signed-`i64` intermediates.
- Lowered that exact source through Turing's sanctioned public compiler entry,
  then compiled and executed it with both C and LLVM backends.
- Added greedy FIFO fusion for compatible queue requests and routed fused
  length buckets through the established Turing `AbstractTensor` substrate.
  Atom predicates and both fingerprint lanes are tensor operations; exact
  comparison, cover selection, and commits remain deterministically ordered.
- Updated the architecture, README, implementation receipt, and continuation
  prompt to distinguish completed work from open phases.
- Added the first versioned SSA-linkable query, `token_kind_batch`, using the
  established postponed, `TYPE_CHECKING`-guarded `AbstractTensor` annotation
  convention rather than an eager compiler dependency or alternate function.
- Centralized its physical Turing ProgramABI contract and made the native gate
  verify compiler-normalized parameters, result span accounting, output extent
  derivation, and the emitted runtime ABI before accepting numerical parity.
- Added an optional continuous embedding attachment downstream of ordered
  commits. Flat parent/cover/offset deltas feed a bounded worker whose numerical
  update is one AbstractTensor gather/scatter batch.
- Replaced the prototype's undifferentiated seed vectors with explicit fixed
  n-dimensional unit-sphere positions, a shared Turing AbstractNN residual
  adapter, and meaning vectors defined by their sum. The adapter is the
  repository's `Linear -> Tanh -> Linear` network trained by its Adam
  optimizer; cover learning never overwrites the positional space.
- Replaced permutation-blind cover means with ordered COO targets carrying
  edge ordinal and cover length. Added an integer-only live-program observation
  ABI and vectorized typed-context target reduction for source, runtime,
  decompiler, glyph, outcome, and temporal facts.
- Declared six strict optional semantic depths (position through query) with an
  inspectable state ABI. Position-only operation constructs no network or Adam
  state. The first three preparation stages are standalone AbstractTensor
  roots with a depth-pruned Turing ProgramABI; later native training still
  waits on the recorded Turing VJP provenance repair.
- Added the non-instantiable `token_orchard.api.String` namespace. Live Python
  bytes ingress now uses one lazy global TokenService and returns only integer
  tokens/receipts. The exact compiler identity contract forbids instances,
  host-string ingress, name-based matching, and compilation of Python queue or
  lock machinery.
- Added a focused Turing extraction audit. It reproducibly shows current
  pursuit crossing the intended namespace boundary and reaching
  `token_orchard.service.TokenService` as
  `default:unknown / provenance_not_declared`; the compiler-team handoff now
  has an exact command and acceptance boundary.
- Added a generational retention contract: batched logical-time notices,
  structure-of-arrays lifecycle state, and a single AbstractTensor stale-row
  mask. A live vocabulary never drops or reuses a token ID. Reclamation waits
  for pinned readers, rotates vocabulary identity, and emits a remap; learned
  state carry is explicitly cold, retained, or conditioned-out.
- Audited Turing's existing Python string seams and added an exact capture
  manifest. Source glyph recipes, StringTable projections, `str.strip`,
  explicit UTF-8 encode aliases, and byte joins are incumbent compiler facts;
  lifted CPython Unicode scalar helpers are differential oracles. Ready token
  operations are distinguished from transforms still lacking exact opcodes.
- Added a profiled text-edit learning demo that exercises the real token and
  learner queues, graph/context objectives, backward, and Adam. Retesting the
  current merged Turing main still fails native Adam at the generated
  `forward_vjp` source-cell declaration, so the demo records native training
  as unavailable and a compiler-team handoff preserves the exact repro.
- Verified all four Turing captured-training-step laws. The focused compiled
  Adam padding test currently stops before LLVM because the generated
  `engine_dendrite_learner__forward_vjp` function declaration has no source
  cells for function-address concordance. Plain graph VJP, parameter-group
  training, and scalar-loss adjoints fail at the identical boundary, proving
  this is graph-reverse packaging lag rather than Adam arithmetic. It was
  recorded for the compiler team rather than bypassed in Token Orchard.

## Observed Behaviour

Final verification from `string_core/`:

```text
python -m unittest discover -s tests -v
45 tests passed

python -m compileall -q token_orchard tests benchmarks tools
completed successfully

python tools/test_turing_compile.py
C compile+execute: 4/4 exact limb pairs
LLVM compile+execute: 4/4 exact limb pairs
gate exited successfully

python tools/test_turing_ssa_api.py --backend all
runtime-sized int64 span query compiled and executed 6/6 cases in C and LLVM
turing-runtime-abi-v1 input/output contract verified

final standard-library suite
72 tests passed

python tools/test_turing_retention.py --backend all
vectorized AbstractTensor retention mask compiled through C/LLVM, 4/4 rows each

python -m benchmarks.tensor_preflight_baseline
16,384 x 16 atoms, eager tensor/scalar ratios:
NumPy 0.86x / Nodus 0.78x / Torch 0.71x

python -m benchmarks.reference_baseline
2,000 items / 32,000 atoms in 1.7766 s
1,125.8 items/s / 18,012.3 atoms/s
1,254,184 traced peak bytes
```

The benchmark is a Python reference baseline only. A prior boxed-list run
peaked at 1,582,280 bytes; contiguous typed arrays remain materially smaller.
The current scalar limb kernel is slower in CPython than the original
unbounded-integer expression, but it gives the compiler a portable arithmetic
contract; short-run timing is not treated as significant.

Forced-total-collision, malformed encoding, forged snapshot digest, malformed
snapshot structure, same-sequence producer races, bounded-queue backpressure,
and clean-shutdown paths are included in the green suite.

## Lessons Learned

- Binary awareness needs a distinct byte identity domain, not an overloaded
  Unicode token or an assumption that every byte sequence decodes.
- Arithmetic atom tokens preserve deterministic constant-time identity without
  allocating more than one million Python scalar objects.
- “Structure of arrays” is not memory-smart if its fields are still boxed
  Python lists; typed buffers materially improved the measured footprint.
- A hash-independent exact row is essential: constant fingerprint mode is a
  simple proof that the lookup structure is not semantic authority.
- Service-thread completion epochs are not yet immutable reader publication;
  the implementation and docs deliberately avoid that claim.
- A native claim must stay at the compiled seam: C/LLVM parity proves the live
  fingerprint atom step, not the Python-only vocabulary or service around it.
- Tensorized is not synonymous with faster. Replacing per-length Python
  buckets with one padded batch and masked AbstractTensor lanes produced a
  small eager NumPy crossover at 16,384x16 atoms (1.04x versus the scalar
  oracle; Nodus 0.91x, Torch 0.79x). Fusion still belongs inside whole-core
  compilation; making the fingerprint seam a separately dispatched
  accelerator would introduce an artificial boundary and is not the target.
- A fully tensor-resident offsets-to-COO experiment was rejected by
  measurement: `AbstractTensor.searchsorted` is a portable broadcast
  compare+sum, so using it for atom row IDs made packing O(items*atoms) and
  reached about 53 seconds on the largest NumPy case. Keep the current single
  O(atoms) topology pack until ingress carries validated COO coordinates; do
  not confuse a tensor expression with an efficient algorithm.
- Commit parallelism is intentionally constrained: fingerprinting and validity
  screening commute across items, while token mint order and prior-token cover
  selection are semantic and must retain one serialization point.
- Queue entry count and fusion work are bounded, but aggregate queued payload
  bytes are not yet governed by an atom-credit ledger. That must land before
  the service can honestly claim a strict memory bound.
- Turing's C artifact retains its loaded DLL for process lifetime on Windows;
  persistent ignored build output is the correct compile-gate lifetime, not a
  temporary directory that tries to unlink the live module.
- The holistic audit now pursues the real `Vocabulary.intern_batch` graph,
  including exact lookup, cover construction, and ordered minting. Explicit
  scalar/sequence annotations, fixed batch outputs, and total cover-choice
  state cleared earlier compiler refusals without changing any of 43 laws.
  Turing now reports one honest frontier: persistent dynamic arena growth needs
  typed native pointer cells, capacities, an explicit heap owner, and arena
  replacement. This is evidence for a state ABI, not permission to split out
  another partial runtime kernel.
- An import-only compiler submission is not an authored root. Submitting the
  actual function source, with `AbstractTensor` present in its postponed
  annotation and imported only under `TYPE_CHECKING`, gives ordinary Python
  and source ingestion one canonical function while allowing the ProgramABI to
  state physical width and runtime rank separately.
- Turing PR #923 repaired runtime-sized signed-`int64` LLVM arithmetic. The
  authored token-kind query now passes C/LLVM 6/6, and the vectorized retention
  mask passes C/LLVM 4/4. Logical Token Orchard identifiers remain unsigned,
  but the currently proven native ProgramABI is explicitly signed; unsupported
  unsigned declarations must be refused rather than silently type-erased.
- PR #923 did not repair generated training-motion provenance. The focused
  compiled Adam test still reaches `FunctionTable.declare` without source cells
  for `engine_dendrite_learner__forward_vjp`, before LLVM emission. The native
  NN demo therefore remains honestly blocked at that separate compiler seam.
- Swapping Adam for SGD does not avoid that blocker: both require the same
  generated backward/VJP.
- A proposed optimizer-free relaxation was rejected because it compiled only
  one isolated update while constructing its targets eagerly. The code, ABI,
  demo, tests, and native claims were removed. A subsequent audit kept glyph
  positioning, ordered COO composition, repeated updates, and loss in one
  reachable AbstractTensor graph; Turing refused that whole graph at missing
  module-lane `reshape`, `scatter`, `to_dtype`, and `sum` spellings plus
  runtime storage provenance. This is a compiler receipt, not a compiled
  product. No divided native-training path is authorized: success means one
  whole reachable string-core product through the sanctioned compiler entry.
- The next string-operation breadth is deliberately read-only and exact:
  containment, find, startswith, endswith, and count over same-domain tokens
  with explicit bounds. Their compiler captures are recorded as pending batch
  opcodes; Unicode transforms still wait for versioned table semantics.
- An optional learner must be lossy only explicitly. Its writer-side offer is
  nonblocking; queue overflow, an epoch gap, or a sink exception marks the
  learner desynchronized. The authoritative request still completes normally.
  A published embedding snapshot therefore includes epoch, policy, dense row
  map, fixed spherical positions, learned residuals, meaning vectors,
  desynchronization, and failure receipts instead of presenting stale vectors
  as current.

## Next Steps

- Implement immutable pinned epoch views and release/reclamation semantics.
- Complete integer batch opcodes for the read/query operation surface.
- Add cancellation and idempotent retry behavior to the queue service.
- Measure and replace the first-atom prefix bucket with a compact index while
  preserving cover ordering.
- Define the capacity-bearing native vocabulary-state ABI and clear the
  retained `tools/audit_turing_whole_core.py` dynamic-growth refusal. Then add
  C/LLVM/Fortran execution parity while retaining PR #906 discipline.
- Lower the whole reachable string core as one Turing compilation product;
  move immutable published reader arrays onto AbstractTensor and parallelize
  indexed read/query batches against pinned epochs. Historical isolated gates
  are evidence about compiler capabilities only, never acceptance criteria or
  separately linkable pieces of the requested product.

The matching continuation stub is
`todo/1792852825_token_orchard_immutable_epoch_and_native.stub.md`.

## Prompt History

> Did you prioritize vectorized memory smart binary aware token systems and encoding aware

> Okay please proceed along this course implementing your own, our own, vision

> Keep going along the plan until you can test compile it

> Use abstract tensor instead of bare python

> Why fuse part of it when the entire core is going to be compiled down and linked for programs using strings

> well, so, you're talking I think about the model that sits between our positional encoding in n-d spherical space to our "meaning" embeddings, right?

> can we, using abstract tensor, trusting in the compiler, can we knock that out with a swift bit of change

> okay, so, can you move on this, abstract tensor has networks we can use that are known to compile and train with adam in the examples

> okay, I want you to try to act on this, this is the next optional layer, we keep them so optional that the ssa only includes the depth of NN prep it's going to need. versions of what we make now may become important elements of general programmatic reasoning and source ingestion, decompiling training, a variety of things

> we will not have string objects, we're not going to make string code, we have to be very disciplined. we make token code. tokens have glyph sets. you have String and that means you can send byte strings through global string methods, that have some contrivance in our ssa that the String core, being invoked, has it's threads running and genuine queing, I don't know exactly If I'm saying this right, but the idea is not to make string objects and it's not to encourage string programming. it's to give comfortable ergonomic interface with the always present self growing vocabulary

> don't let me interrupt but while doing this, wouldn't hurt to make sure the vocabulary can release a node after a certain amount of time without being noticed again, we could condition the previous impact out or just train from where it left off, to adapt to a new internment

> two questions, can you make a demo program yet that shows profiled text modification with all that nn activity that makes it to native,
> and then, see if we can capture string related functions that might currently go to a known engine in python, that we could slip in and hijack as automatically requiring our string core and using it in place

> I gave all your last response to the turing team and they have included it in a recent PR

> While we wait do we need to add more string related functions, or more token training with glyph sequences related material, are we able to make a demo script work that doesn't use Adam

> What do you mean eager abstract tensor work. That is the act phrase "i did something fake"

> Do you realize you have zero authorization to compile in any way that is in pieces while doing what i told you to do

The governing root instruction also required preserving established systems,
using the sanctioned compiler entry, and not adding substitutes beside existing
mechanisms. Turing was used read-only as the compiler and behavioral gate;
Turing, Nodus, and Spectral Analyzer source remained unmodified.
