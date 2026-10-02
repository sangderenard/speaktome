# Direct x86 vocabulary to SSA

**Date:** 2026-08-02
**Project:** `C:\dev\Powershell\turing`

## Overview

Replaced the primary lifting path's disassembly-text dependency with a direct,
numeric x86-64 reference vocabulary. The bounded region decoder recognizes the
controlled `IMUL r32,r/m32`, `LEA r32,m`, and near `RET` forms, including
handwritten ModRM, SIB, REX extension bits, signed displacement, base-less SIB,
and RIP-relative decoding.

Added one start-to-finish Python entry point,
`raise_binary_region_to_ssa`, which accepts the user binary region plus an
explicit maximum file-size boundary. It returns repository SSA only after a
complete decode and lowering, alongside the safely decoded prefix, structured
failed-vocabulary records, and coverage/count statistics.

Bulk byte validation, checksum reduction, and instruction/semantic token
counts use the repository's `AbstractTensor` operations. Instruction-boundary
walking and ModRM/SIB decoding remain scalar because x86 length decoding is
stateful. A tensor backend failure is exposed in statistics and falls back to
exact host reductions without being mislabeled as missing ISA vocabulary.

## Steps Taken

- Added append-only numeric instruction and semantic token enums.
- Added structured register/effective-address operands and exact byte
  provenance.
- Implemented direct ModRM/SIB and displacement decoding without objdump or a
  third-party machine decoder.
- Added explicit rejection for legacy prefixes, duplicate REX prefixes,
  `REX.W` under 32-bit tokens, REX-prefixed `RET`, malformed byte values,
  truncated opcodes/instructions, trailing bytes, and missing return.
- Added fail-closed prefix reports; no attempt is made to resynchronize after
  an unknown variable-length instruction.
- Added structured token lowering for integer multiply and effective-address
  arithmetic into the repository's `Function`/`BasicBlock`/`Instr` SSA.

## Observed Behaviour

No compiler suite or bespoke test was run. After the user requested a
substantial input, the complete 289,792-byte Windows `cmd.exe` PE image was
passed directly to `raise_binary_region_to_ssa` with an exactly matching
maximum-file-size boundary. The function returned no SSA and one structured
failure at offset/address zero: the file begins with the PE/DOS container
signature bytes `4d 5a`, for which the ISA vocabulary correctly has no
instruction token. `AbstractTensor` processed the entire accepted region and
reported 289,792 valid bytes, byte sum 25,608,421, zero decoded instructions,
zero byte coverage, and no tensor error. This distinguishes successful bounded
file ingestion from executable-container parsing and machine-code coverage.

A subsequent PE-specific pass parsed that same file as AMD64 PE32+, validated
seven bounded sections, and mapped entry RVA `0x18f50` into executable `.text`
at file offset `0x18350` (virtual address `0x140018f50`). A 4,096-byte bounded
entry window then reached the ISA vocabulary and stopped honestly at
`48 83 ec 28 e8 2b 06 00`: the current vocabulary does not yet contain the
`REX.W + 83 /5 ib` stack adjustment beginning cmd's startup code. Tensor
statistics covered all 4,096 extracted code bytes without error.

The observed startup forms were then added as append-only numeric vocabulary
tokens. `SUB_R64_IMM8` requires `REX.W`, matches opcode-group extension `/5`,
accepts only a register destination, and returns a signed structured immediate.
`CALL_REL32` returns both its signed displacement and resolved next-instruction-
relative virtual target. Replaying the entry bytes decoded both instructions
and then stopped at `48 83 c4 28`, now reported precisely as the uncovered
opcode-group form `83 /0`. ModRM extension is part of `InstructionSpec`
identity so future `83 /0` ADD and `83 /7` CMP rows can coexist correctly.

Added `src/compiler/x86_tensor_read_head.py`, a general configurable x86
read-head state machine over `AbstractTensor`. Dense tensor tables own opcode
maps, escape maps, opcode-group dispatch, prefix policy, immediate widths, and
per-token constraints. Tensor lane state owns cursor, instruction origin,
phase/status/failure, token, prefixes, REX presence and bits, opcode, ModRM,
SIB, displacement/immediate accumulators, and resolved relative target. Each
transition consumes at most one byte per active lane through tensor masks and
gathers; emitted lanes are explicitly acknowledged before reading the next
instruction. A preset compiles the current controlled vocabulary into those
tables. This is a discrete compiler kernel rather than a managed-dt engine.

After the goal resumed, the real cmd entry stub was completed through
`ADD_R64_IMM8` and terminal `JMP_REL32`. It now raises into repository SSA as
`Const/Sub/Call/Const/Add/Br`, with an implicit 64-bit RSP argument and explicit
call-effect shortfalls. PE exception-directory parsing then read 751 validated
AMD64 runtime-function records from `.pdata`; the entry call target maps to the
exact 212-byte range `[0x19584, 0x19658)`.

Vocabulary growth against that real function added bidirectional MOV forms,
masked register PUSH, group-encoded AND, MOV imm64, CMP, and JNE. Its first
44-byte basic block now raises successfully into versioned register and memory
SSA, including stack stores, RIP-relative load, read-modify-write, a boolean
comparison, and `CondBr`. The branch successors are `0x140019643` and
`0x1400195b0`; these are the next CFG worklist frontier.

PE container work now lives in `src/compiler/binary_ingestion.py`. It provides
strict PE parsing, entry-region extraction, a binary-facing API facade, and
validated immutable equivalence tables spanning PE structure, ISA tokens,
machine semantics, and repository SSA handlers.

## Lessons Learned

An ISA vocabulary is not a mnemonic list. Its meaningful unit is an encoding
form with width, prefix policy, operand grammar, and a machine-state semantic
token. ModRM/SIB belongs inside that vocabulary boundary, while tensor math is
well suited to bulk coverage accounting rather than sequential instruction
framing.

## Next Steps

- Move token-path `MultiDiGraph` lineage into the precompiler meta/evolution
  package rather than leaving it lifting-local.
- Continue the `.pdata`-bounded function through both observed CFG successors.
- Materialize general flags state beyond the current CMP/JNE predicate fusion.
- Install the tensor read head after emitted-record provenance is implemented.

The region raiser now also accepts `full_vocabulary_report=True`. Its primary
decode remains fail-closed and proves only the contiguous prefix. A separate
bytewise diagnostic audit then covers the entire accepted region, reports
every unclassified span and its leading byte signature, and labels all decoded
islands after a gap as resynchronization candidates rather than proven
instruction boundaries. Any reported gap withholds SSA.

On the 212-byte `.pdata`-bounded `cmd.exe` function at RVA `0x19584`, the
strict pass proved 9 instructions / 44 bytes. The whole-region census found 25
candidate instructions covering 118 bytes and 9 gaps covering the other 94
bytes. The returned function was `None`, as required for incomplete
vocabulary coverage. This was a direct ingestion run against the system
binary; the compiler test suite was not run.

The convergence pass then added exact forms for 64-bit LEA, indirect CALL,
32-bit MOV with architectural zero-extension, both 64-bit XOR directions,
immediate SHL, register-source AND, short JNE, NOT, and POP. The scalar
handwritten decoder now proves 49 instructions covering all 212 bytes with no
diagnostic gaps.

Whole-function lowering now partitions the decoded region at branch targets
and fallthroughs. The resulting five-block acyclic CFG carries versioned
register and memory state through Phi nodes. Indirect Windows-x64 calls return
one opaque call-state SSA value; existing Load nodes project RAX, every
volatile argument/scratch register, and the post-call memory version. This
models one call event with multiple machine-state effects without duplicating
the call. Direct ingestion now returns a non-null complete SSA Function with
49 decoded instructions, five blocks, six indirect calls, and zero failures.
The system binary was exercised directly; the compiler test suite remained
unrun as requested.

## Program-graph architecture correction (2026-08-03)

Repository SSA is a derived analysis/lowering view, not the owning program IR.
The direct path is now:

`PE bytes -> machine vocabulary tokens -> token-ID MultiDiGraph -> optional SSA
projection -> existing ProcessGraph/control planning -> existing whole-program
AOT C shell assembly`.

`TokenPathAtlas` now lives in `src/compiler/evolution_metagraph.py`, where
components may carry stable integer token identities and relationship events
may carry integer role identities. `EvolutionMetaGraph.to_token_multidigraph`
retains every relationship event as a keyed edge, including parallel edges,
instead of collapsing the program into textual labels or a simple graph.

`src/compiler/machine_program_graph.py` directly raises the PE image and every
AMD64 `.pdata` runtime-function range into that representation. Program,
section, function, instruction, operand, vocabulary-failure, sequential-flow,
local-control-target, and resolved internal-call facts are explicit graph
objects. Instruction labels remain diagnostics; identity comes from the token
atlas. SSA ingestion is deliberately absent from this owning layer.

The scalar decoder vocabulary grew from 59 to 83 exact encoding forms. Legacy
prefixes are now part of `InstructionSpec` identity and read-head state, so
`66`-selected 16-bit forms coexist with unprefixed 32-bit and `REX.W` 64-bit
forms. The additions include common arithmetic, compare/test, move-extension,
conditional branch/set, and group-opcode forms; ModRM and SIB remain decoded
locally without an external disassembler.

The latest direct `C:\Windows\System32\cmd.exe` census (not a compiler test)
records 751 runtime functions, 343 completely vocabulary-proven functions,
18,397 proven instructions, and 73,646 proven bytes. The remaining 408
functions stop at an explicit first unknown instruction. The executable
section has 200,704 raw bytes; `.pdata` describes 190,979 of them. The other
9,725 bytes are now represented as 603 explicit unclassified executable ranges
rather than silently disappearing. Consequently `MachineProgramGraph.complete`
cannot become true merely because all `.pdata` functions decode: all executable
ranges must also be classified and all described code bytes proven.

The existing whole-program C handoff was identified as `ControlProgram` plus
`RegionCode` into `render_c_shell`/`compile_cffi_shell`. No new SSA-to-C backend
was invented. The outstanding compiler work is a projection from the machine
token multigraph into the existing ProcessGraph/control/numeric abstractions
that already feed that assembler.

### Continued whole-program convergence

Three further direct `cmd.exe` vocabulary cycles expanded the handwritten
encoding vocabulary from 83 to 170 forms. Strict coverage progressed through
418, 516, 616, and finally 679 complete runtime functions out of 751. The
latest graph contains 42,033 proven instructions and 168,873 proven bytes;
72 functions retain explicit first-failure nodes. The token multigraph has
109,969 nodes and 157,871 edges.

This growth added the remaining common integer widths and directions, group
immediates, flag-consuming branches and conditional moves/sets, shifts,
rotates, multiply/divide and bit-test families, indirect control, FS-relative
loads, repeated word stores, and aligned/unaligned XMM moves. XMM registers and
legacy AH/CH/DH/BH byte registers now have operand types distinct from general
registers. Mandatory-prefix SSE forms are separate numeric vocabulary tokens;
they are not inferred from mnemonic strings. Redundant `66` padding prefixes
are retained in instruction provenance.

The next observed boundary is 72 functions spanning further conditional
moves, lock-prefixed atomics, repeated string moves, additional shift/group
forms, and a VEX-prefixed encoding. These remain failures rather than being
misread as legacy opcodes.

### Reachable CFG completion and encoding implications

Continued convergence expanded the vocabulary to 219 forms, including locked
atomic compare/exchange and exchange/add, string scan/move, scalar-double and
XMM/GPR transfers, further conditional moves, multi-destination division,
interrupts, and the remaining observed bit/shift forms. Direct linear decoding
then reached 748/751 runtime functions and 190,569/190,979 bytes. Inspection of
the last three failures proved they followed unconditional jumps into table-like
regions; treating their bytes as new opcodes would have corrupted the
vocabulary.

`machine_program_graph` now performs reachable worklist decoding per runtime
function. Direct conditional targets, direct jumps, and in-range direct calls
become worklist entries; returns, traps, interrupts, and indirect jumps close a
path. Non-reached ranges are explicit graph components instead of presumed
instructions. With this correction, all 751 function entries have zero
reachable vocabulary failures: 39,263 reachable instructions and 157,513
reachable instruction bytes are proven. The graph retains 596 unreached ranges
covering 33,466 bytes inside runtime-function extents, plus 9,725 executable
bytes outside `.pdata`. `MachineProgramGraph.complete` remains false until
those ranges are resolved as indirect targets, handlers, padding, or data.

A global fixed-point pass now propagates every direct relative call and branch
target into the owning `.pdata` range, including targets that enter a runtime
range away from its nominal begin RVA. This reduced unreached runtime bytes
from 33,466 to 1,486 across 460 small ranges while retaining zero vocabulary
failures. The current proof covers 46,834 reachable instructions and 189,493
of 190,979 runtime-described bytes. Remaining classification is concentrated
in padding, exception-handler/funclet entries, indirect switch targets, and
data rather than ordinary direct-control code.

For a native project encoding, the observed operator families argue for a
token selecting semantic transformation plus explicit fields for operand
width/topology, address space, predicate input, memory order, and result/effect
arity. Flags should be ordinary state outputs; branches and conditional moves
consume predicate tokens. Calls, division, atomics, and string operations need
multiple state outputs rather than hidden side effects. Vector lane shape and
scalar/vector transfer width should be operands or types, not mnemonic text.
This retains compact numeric identity without reproducing x86 prefix and
ModRM ambiguity.

### Read-head, structure inspection, and execution orchestration

The tensor read head now supports prepared batches that cache flattened bytes
and lane offsets, and the head caches flattened opcode/group/escape tables.
Fixed `transition_block` construction avoids per-microstep host synchronization
when captured by an AOT backend. `X86ReadHeadProfile` provides immutable,
composable row/prefix/map configurations. Bounded `DECODE`, `TRACE`, and
`EMULATE` modes share the same transition kernel; emulation requires an
explicit emitted-instruction executor and therefore cannot silently invent
unsupported machine effects.

During a direct terminal probe, the read head exposed an older assumption that
`AbstractTensor.__and__`/`__or__` were integer bitwise operations; on this
abstraction they are logical operations. Encoding-flag, REX-mask, and legacy-
prefix tests now use arithmetic bit extraction, and prefix-mask union is
explicit. A one-byte `RET` now halts cleanly in three microsteps, while
`48 83 ec 28; c3` emits the expected SUB and RET events and halts without a
failure. Terminal events are retained in trace mode.

Added `src/compiler/binary_structure_graph.py`. It creates nested file/header,
section, runtime-function, instruction, unreached, and unclassified-executable
regions, then partitions the entire file at every boundary. Each partition
records all covering regions, making raw-file offsets, RVA mappings, overlaps,
and ownership directly inspectable. On `cmd.exe` the graph contains 48,662
regions and 47,910 non-overlapping partitions covering all 289,792 bytes with
zero uncovered bytes; the resulting graph has 96,572 nodes and 287,580 edges.
The entry byte is simultaneously identified as file, executable section,
runtime function, and instruction.

Added `src/compiler/machine_execution.py`. Its orchestration modes return the
decoded program, build the inspection graph, or schedule emulation from the PE
entrypoint. Direct control, call stacks, returns, traps, and bounded execution
are structural. Arithmetic, memory, predicates, imported calls, and indirect
targets require numeric semantic-token handlers. A no-handler `cmd.exe` run
stops exactly at entry with `BLOCKED_EFFECT` on `INTEGER_SUBTRACT`, demonstrating
that execution is enabled but fail-closed rather than falsely emulated.

## Turing micro-IR and concurrent read/write heads (2026-08-03)

The checked-in Turing calculus is a credible normalization micro-IR for the
pure value portion of the machine program.  The complete backend hook surface
has eight names (`nand`, `sigma_L`, `sigma_R`, `concat`, `slice`, `mu`,
`length`, and `zeros`), while the remembered five-operator reduction appears
directly in `Turing.ripple_add`: NAND, the left/right motion family, concat,
slice, and selector/mux.  `length` and `zeros` are shape/constructor services,
and counting the two directions as one parameterized motion family explains
the useful five-family presentation.

This is not sufficient as the sole whole-program IR.  It has no first-class
terminators, memory/world ordering, capability or handle lifetime, calls,
traps, exceptions, or concurrency.  It is nevertheless a good *inner*
calculus: x86 register/flag/value transformations can be expanded into it,
while the surrounding token multigraph retains control, memory, effects, and
capabilities.  Existing `abstract_tensor_bitops.py` and
`bitops_process_graph.py` already demonstrate both backend independence and
splicing primitive provenance into ProcessGraph.

The natural traversal is two coordinated heads:

1. The x86 read head performs breadth-first reachability from PE entrypoints,
   direct branch targets, and discovered callable entries.  Its frontier
   elements must include address plus the incoming abstract machine-state
   version, not address alone.
2. For each decoded instruction event, the Turing writer appends immutable
   normalized value nodes and emits the outgoing state version(s).  The writer
   writes a graph/tape history; it must not destructively rewrite one global
   tape.
3. At a control join, incoming register, flag, memory, and capability versions
   are merged explicitly (phi/select where semantically valid).  A changed
   block-entry state re-enqueues that block until the graph reaches a fixed
   point.
4. Indirect control remains an unresolved frontier/effect node until target-set
   analysis supplies successors.  It must not be guessed from linear bytes.

The existing reachable-function traversal already has a worklist but calls
`worklist.pop()`, so it is currently LIFO/depth-first.  The tensor read head
already exposes immutable emitted states and an executor callback.  Those are
the two concrete attachment points for a FIFO program frontier and a Turing
normalization writer.

## Direct binary-to-Turing graph proof slice (2026-08-03)

Implemented `src/compiler/machine_turing_graph.py` as a deliberately pre-SSA
bridge.  `decode_reachable_region` now owns a FIFO block-entry frontier and an
optional instruction sink.  The Turing writer is attached to that sink, so
accepted instructions are normalized while reachability discovery is still in
progress rather than after construction of repository SSA.

The proof currently covers 32/64-bit-width-matched register-immediate writes,
integer add/subtract/multiply, bitwise and/or/xor/not, no-op, return, and direct
jump.  It emits stable numeric tokens for input, constant, NAND, both motion
directions, concat, slice, select, length, and zeros.  The pure provenance DAG
and machine control graph remain distinct layers in one result.  Unknown
opcodes stop before the writer, while unsupported semantics are retained as
addressed failure records.

The first conditional branch fixture deliberately exposes the next boundary:
the decoder produces correct fallthrough and branch edges, but the writer
reports that block-entry state versions and a join are required.  It does not
pretend that the fallthrough arithmetic executes unconditionally.  Direct
calls similarly require continuation/effect state before they can be marked
complete.

`reduce_turing_operator_graph` hash-conses equivalent pure nodes in topological
order and remaps register outputs.  On the focused 32-bit arithmetic fixture,
the fully expanded graph was large (thousands of primitive provenance nodes),
and reduction removed substantial repeated structure.  This confirms the need
to retain compact semantic nodes with a verifiable expansion rather than
serializing every NAND occurrence by default.

`turing_operator_graph_to_ssa` projects either the original or reduced Turing
DAG into the repository's ordinary `Instr`/`SSAValue` representation.  This
keeps the required ordering explicit: binary -> Turing calculus -> topological
reduction -> SSA, rather than using SSA as an undocumented intermediate on the
way into the calculus.

Focused verification:

```text
py -3.11 -m pytest -q tests/test_machine_turing_graph.py
3 passed in 5.00s
```

The checked-in `.venv` is stale and points to a removed Python 3.10 executable;
the available Python 3.11 installation was used without changing dependencies.

## Versioned joins and quotient verification (2026-08-03)

The direct Turing writer no longer applies discovery-order instructions to one
linear register map.  Its decoder callback appends immutable instruction
events; after reachable control topology is known, an acyclic state solver
processes each instruction with its predecessor state.  Splits copy a state
version.  Joins compare incoming register carriers and emit one Turing `mu` per
differing register, sharing a control-selector carrier and recording join and
predecessor addresses in provenance metadata.

This makes the conditional fixture complete without inventing a concrete
branch outcome.  The final RAX producer is a `mu` over the direct-branch value
and fallthrough-add value.  Cyclic control remains intact in the control graph
and fails explicitly with `cyclic control requires loop-header state fixed
points`; no linear execution is fabricated.

Topological reduction now returns a checkable quotient witness.  Verification
requires total original-node coverage, surjectivity onto reduced nodes,
operator-token preservation, an image edge with the same argument position for
every original dependency, and exact register-output remapping.  The focused
test removes a reduced edge and confirms the witness is rejected.

Focused verification after these changes:

```text
py -3.11 -m pytest -q tests/test_machine_turing_graph.py
4 passed in 5.26s
```

Switch semantics were reported as concurrent work by another agent.  No switch
vocabulary was added or modified here; the control graph continues to use
generic labeled successor edges so that work can attach without altering the
Turing value calculus.

## Clarification: the intended Turing computer is the audio cassette machine

The user clarified that “Turing computer” referred to the repository's
audio-storage survival computer (`TapeMachine` + `CassetteTapeBackend`), not
only the NAND/shift/selector calculus.  The intended outcome is for the lifted,
de-Windowsized `cmd` program to be compiled onto that machine and for its
physical tape traversal and analog operations to constitute the program's
experienced execution.

The accidentally built micro-IR is still directly relevant.  `TapeCompiler`
already maps the same eight provenance operation names to tape opcodes, and the
new reduced graph can project to repository SSA.  However, this is not yet a
working end-to-end connection: `compile_ssa` does not accept input/constant
instructions, initial data and spills are incomplete, the compiler packs its
16-bit words as four 4-bit fields while `TapeMachine._fetch_decode` interprets
them as 4/2/2/2/6-bit fields, and the tape ISA has no branch, call/return,
capability, or environment-effect operations.  `TapeMachine.run` currently
executes a fixed sequential instruction count.

The corrected target path is therefore:

```text
cmd PE -> reachable machine/effect graph
       -> remove Windows implementation and bind imports/handles to capabilities
       -> Turing-reduce pure value regions
       -> lower control, memory, and capabilities into an extended tape VM ISA
       -> assemble BIOS + code + initial data onto CassetteTapeBackend
       -> boot TapeMachine and experience execution as generated audio
```

The smallest honest next proof is to connect one lifted x86 fixture—not `cmd`
yet—through the new graph and repaired tape encoding into `TapeMachine`, then
use the exact failures to grow control/effect coverage toward `cmd`.

## Tape-computer reduction catalog inventory

The tape target has several concentric catalogs that must not be conflated:

- The physical ISA has `SEEK`, `READ`, `WRITE`, `NAND`, `SIGL`, `SIGR`,
  `CONCAT`, `SLICE`, `MU`, `LENGTH`, `ZEROS`, and `HALT`.
- `Turing` derives NOT/AND/OR/XOR/XNOR, mux, rotation, half/full/ripple add,
  successor, bit writes, and head movement exclusively from its eight carrier
  hooks.
- `BitOpsTranslator` further derives fixed-width bitwise operations, shifts,
  add, subtract, multiply, divide, and modulo.  The ProcessGraph expansion
  registry currently exposes only bitwise and/or/xor/not plus add/sub/mul;
  shift/div/mod recipes exist but are not registered in that graph pass.
- Loop machinery can rewrite provenance backedges to `mu`; the generic SSA
  builder can represent scheduled backedges as `phi`; `TapeCompiler` aliases a
  phi result to its first operand register rather than emitting an opcode.
- `bitops.py` contains a large representational catalog: integer, rational,
  float, complex, tensor, manifold, graph/DAG, schema/serialization, and
  if/match/loop/switch/goto/call/return structures.  These are not uniformly
  reductions to tape instructions.
- `GrayTableOps` supplies host-built lookup semantics for integer add/sub/mul/
  div/mod and composite float, complex, and rational arithmetic.  These methods
  use Python integers and tables rather than recording Turing provenance, so
  they are recipes/candidates rather than currently compilable tape graphs.
- `AbstractTensorBitOpsTranslator` is a second carrier implementation for the
  same eight-hook calculus, allowing primitive provenance to be spliced into a
  tensor ProcessGraph.

The existing intended demonstration is already the desired shape:
BitOpsTranslator multiplication -> ProvenanceGraph -> TapeCompiler -> tape
image -> CassetteTapeBackend -> TapeMachine.  It does not yet prove the current
implementation runs end to end because the survival demo has stale tuple
unpacking and the compiler/machine instruction layouts disagree.

The most useful next adapter is a tape-feasibility classifier over the new
`TuringSSAProjection`: input/constant instructions become initialized tape data;
the eight primitives become instructions; phi becomes allocation/coalescing;
control/effect nodes remain named shortfalls.  That will expose how much of each
lifted x86 fixture can already become audible execution before extending the
ISA.

## The unscaffolded bridge: universal OOP to physical tape reduction

The user clarified the larger tendril.  Python is bootstrap notation, not a
runtime constraint; the intended system ultimately bootstraps the repository
into binary.  `bitops.py` is not merely a collection of integer helpers.  It is
the self-describing universalized OOP language: types, schemas, objects,
functions, graphs, control structures, and storage describe themselves in
lower-level elements, and each element can recursively reduce into still lower
elements.

Because BitOps structures also inhabit ProcessGraph, the intended path is:

```text
source OOP/object system
  -> ProcessGraph identity and dependency topology
  -> self-describing BitOps objects and schemas
  -> recursive BitOps reduction subgraphs
  -> Turing super-reduction (NAND/motion/concat/slice/mu + construction)
  -> tape placement, scheduling, and instruction encoding
  -> physical cassette reads, writes, seeks, wave operations, and audio
```

The missing artifact is therefore a recursive reduction catalog plus a graph
rewriter, not a one-off TapeCompiler adapter.  Each catalog entry must carry:

- the higher-level type/operator token and its self-description;
- its lowering rule into lower BitOps elements;
- input/output role, shape, storage, control, and effect contracts;
- a termination/rank measure proving that recursive lowering reaches the tape
  primitive basis;
- the parent-to-child graph morphism needed to retain deep provenance and
  equivalence witnesses;
- physical cost contributions: tape distance, seeks, traversed/read/written
  frames, bit transitions, motor acceleration, lane concurrency, elapsed audio
  time, storage extent, and uncertainty/noise.

The same multigraph must retain the entire ancestry chain from source object or
method through every reduction level to each physical tape event.  Placement
and scheduling may then optimize memory-distance and conservative energy costs
without erasing semantic identity.  The new x86-to-Turing graph work supplies a
first concrete vertical specimen of this recursive bridge, but the general
catalog/rewriter/cost algebra remains unscaffolded.

## Recursive reduction bridge scaffold (2026-08-03)

A renewed goal now targets the general self-hosting bridge rather than the x86
specimen alone.  `src/compiler/recursive_reduction.py` introduces:

- numeric reduction namespaces and BitOps rule tokens;
- `ReductionRank`, which requires every rule and journey stage to descend
  strictly toward physical execution;
- a self-describing `ReductionRule`/`ReductionCatalog` carrying role contracts,
  terminal token possibilities, reducer identity, and diagnostics;
- seven live BitOps rules (and/or/xor/not/add/sub/mul) whose reducer is the
  existing `BitOpsTranslator.apply_bits`, not duplicated arithmetic;
- `ReductionStage`, `ReductionMorphism`, and `ReductionJourney`, which compose
  parent-to-child ancestry across arbitrarily many adjacent reduction stages;
- cross-stage component/handoff recording through the existing append-only
  `EvolutionMetaGraph`;
- `TapePlacement`, `TapeFeasibilityReport`, and a vector `TapeCostVector` with
  distance, seeks, reads, writes, operator events, transition upper bounds,
  storage, latency, mechanical work units, signal-energy frame units, noise
  exposure, and peak lane concurrency;
- serial and parallel cost composition;
- terminal Turing-op to physical `Opcode` bindings;
- explicit comparison of caller-provided tape placements.

`expand_bitops_process_graph` now labels every emitted primitive with its
numeric source node as well as source operation, making the parent/child
morphism recoverable rather than merely diagnostic.  One ProcessGraph XOR test
proves that the source BitOps node owns its emitted NAND descendants in both the
journey and evolution ledger.  A placement test proves that compact and
scattered layouts of the same NAND graph have different distance, mechanical
work, latency, and storage vectors.  An x86 `mov`/`xor` fixture now reaches a
complete tape-feasibility report with physical NAND opcode bindings.

Focused verification:

```text
py -3.11 -m pytest -q \
  tests/test_recursive_reduction.py \
  tests/test_bitops_process_graph.py \
  tests/test_machine_turing_graph.py
11 passed in 10.40s

py -3.11 -m pytest -q tests/test_recursive_reduction.py
5 passed in 4.18s
```

The reducer currently has a composable multi-stage ancestry model but only one
live graph expansion stage (BitOps -> Turing).  Next work is successive
OOP/ProcessGraph -> BitOps expansion and repaired terminal tape assembly.

## First physically executed recursive-reduction witness (2026-08-03)

The repaired terminal path now executes a ProcessGraph `x ^ y` expression all
the way through its BitOps NAND expansion on `TapeMachine` and
`CassetteTapeBackend`.  The proof uses the machine's real three-register
constraint and returns both the output and a physical witness: every fetched
instruction records its originating NAND graph node, opcode fields, head travel,
and audio interval.  For `0b1010 ^ 0b1100`, the cassette machine returns
`0b0110`, reaches HALT, emits ten provenance-bearing NAND events plus HALT, and
generates 25,940,108 audio samples.

The path required four repairs at actual abstraction seams:

- instruction packing and fetch now share one authoritative 16-bit
  `4/2/2/2/6` codec rather than incompatible compiler/machine layouts;
- terminal scheduling counts repeated multigraph operands and orders shared
  terms before destructive self-NANDs, allowing the expanded XOR graph to fit
  the physical three-register file;
- FFT lane energy is normalized back to length-independent amplitude, so write
  bias no longer decodes as a logical one;
- scalar tape registers bind NAND to the transport lane rather than inventing
  values on all 32 spectral lanes.  The general parallel NAND remains intact;
- odd-length inverse FFT lane mixing preserves the original frame length;
- zero-time orchestration still generates the audio witness but does not race
  Windows audio playback threads during accelerated execution.

Focused verification:

```text
py -3.11 -m pytest -q \
  tests/test_recursive_reduction.py tests/test_new_opcodes.py \
  tests/test_nand_wave.py tests/test_bitops_process_graph.py \
  tests/test_machine_turing_graph.py
18 passed, 5 skipped in 10.91s

py -3.11 -m pytest -q tests/test_cassette_tape.py
6 passed in 9.85s
```

This is now a genuine vertical tendril from source-language structure to a
physical cassette execution event.  It is still deliberately narrow: the live
terminal assembler accepts acyclic NAND/data graphs.  The renewed goal remains
active for successive OOP/ProcessGraph-to-BitOps stages, spilling/general tape
storage, additional terminal operators, and eventually the much larger lifted
program graph.

## Physical stages and measured reliability witness (2026-08-03)

The execution witness is now part of the same strictly descending reduction
journey instead of being a detached result.  `execute_reduction_artifact`
extends the BitOps -> Turing journey with:

1. a TAPE stage containing the exact encoded instruction words and their
   register fields;
2. a PHYSICAL stage containing the cassette events actually observed.

Both transitions are recorded as component handoffs in `EvolutionMetaGraph`.
Consequently, `journey.descendants(0, source_bitxor, target_stage=3)` returns
the physical event identities produced by that source operation.  Encoded
instruction order and physical event order are also retained as explicit
`next-instruction` and `next-event` topology.

`CassetteTapeBackend` now maintains monotonic seek-distance, seek-count,
read-frame, write-frame, movement, and audio counters.  The physical witness
separates setup (BIOS/program/input priming and boot), execution, and output
observation costs, while also exposing their serial total.  Individual events
carry their own measured `TapeCostVector`.  Latency is derived from generated
audio samples, while mechanical work, signal work, storage, and noise exposure
remain separately inspectable dimensions.

Reliability uses a caller-supplied per-frame-source error upper bound and a
distribution-free union bound.  No independence assumption is made; the result
reports a conservative failure upper bound and success lower bound over the
measured exposure count.

Focused verification now covers the four-stage ancestry, physical counters,
phase cost conservation, and reliability bound:

```text
py -3.11 -m pytest -q \
  tests/test_recursive_reduction.py tests/test_new_opcodes.py \
  tests/test_nand_wave.py tests/test_cassette_tape.py \
  tests/test_bitops_process_graph.py tests/test_machine_turing_graph.py
24 passed, 5 skipped in 18.89s
```

The next uncovered stage is OBJECT -> ProcessGraph/BitOps.  ProcessGraph's
`map_ir` already describes class and method identities, but a class definition
is currently represented as one opaque process node; its selected method body
must be raised into the ordinary operation graph while retaining the method's
object identity.

## Object method to physical cassette journey (2026-08-03)

The upper bridge is no longer conflated.  A complete live journey now contains
six strictly descending stages:

```text
OBJECT -> PROCESS -> BITOPS -> TURING -> TAPE -> PHYSICAL
```

`src/compiler/object_process_bridge.py` selects one explicitly named class
method from source AST, preserves its class/method identity, decorators,
filename, and original source spans, then gives a copied method AST to
ProcessGraph's existing semantic builder.  It does not interpret the method
body itself and therefore does not create a competing Python compiler.  Missing
or duplicate class/method identities fail explicitly rather than selecting an
arbitrary definition.

`reduce_bitops_process_graph` now records PROCESS and BITOPS as separate graph
stages.  Every process element crosses an explicit identity/tokenization
morphism; recognized operations receive the numeric BitOps vocabulary token,
while inputs, returns, and other structural elements remain typed boundaries.
The selected object method is a numeric-token component in its own OBJECT
graph and hands off to the process nodes produced from its body.

The full test selects `WordOps.xor`, raises its method body, lowers the resulting
process `bitxor` through BitOps and NAND, encodes it, executes it on the
cassette, and proves both of these ancestry queries reach the same ten physical
events:

```text
WordOps.xor -> ... -> physical NAND events
process bitxor -> ... -> physical NAND events
```

The method receiver remains an ordinary input in ProcessGraph.  Because it is
unused by this method, live terminal slicing removes it naturally; no bespoke
`self` erasure was added.

Focused verification:

```text
py -3.11 -m pytest -q \
  tests/test_object_process_bridge.py tests/test_recursive_reduction.py \
  tests/test_new_opcodes.py tests/test_nand_wave.py \
  tests/test_cassette_tape.py tests/test_bitops_process_graph.py \
  tests/test_machine_turing_graph.py
28 passed, 5 skipped in 23.03s
```

The largest remaining execution limitation is terminal storage: only acyclic
NAND/data graphs fitting the three-register destructive schedule can execute.
General spilling must preserve graph identity while moving values between the
register region and explicit tape storage.

## Explicit spill storage and moving reel witness (2026-08-03)

Terminal storage no longer aliases overflow values to register zero.  Two tape
mechanics now occupy previously unused instruction opcodes:

- `LOAD` (`0xB`) copies one fixed-width spill slot into a physical register;
- `STORE` (`0xC`) copies one physical register into a spill slot.

The six-bit immediate selects up to 64 slots.  Slot `s` begins at
`data_start + (REGISTERS + s) * bit_width`.  LOAD/STORE remain tape mechanics,
not additions to the universal Turing primitive catalog.  Their encoded
provenance components identify both the slot number and graph node stored there.

`assemble_nand_terminal_tape_program` retains the low-cost destructive
three-register schedule when it fits.  If the live inputs or interference
exceed that file, it switches to a general acyclic NAND lowering: every live
value receives an explicit slot, operands are loaded into R0/R1, NAND writes R2,
and the result is stored back.  Outputs are loaded into physical output
registers before HALT.  The current instruction envelope rejects more than 64
live slots instead of silently truncating an address.

A four-input, three-NAND DAG now forces this path and executes to the expected
four-bit result (`1011`).  Its 14 events include real LOAD, STORE, NAND, and
HALT fetches, and their tape movement contributes to measured distance, latency,
energy, storage, and reliability exposure.

The first run exposed bias accumulation: copying an already recorded frame
caused the destination write head to add another write-bias carrier.  Repeated
spill round trips eventually turned silence into a logical one.  LOAD/STORE now
remove the known source bias before the destination head applies one fresh
bias, making meaning invariant under storage depth.

The Pygame cassette visualization was also connected to the detailed physical
activity callback.  Its previous shell received absolute positions but then
invented additional motion, and SEEK was indistinguishable from STOP.  The
position-driven reel mode now:

- preserves the backend's absolute left/right tape position;
- accumulates only observed motion into reel angle;
- distinguishes READ, WRITE, and SEEK;
- displays actual head position;
- can run the live spill specimen through `run_live_reduction_reels.pyw`.

`tests/test_reel_visualization.py` verifies that physical READ and SEEK updates
move both reels without changing the reported tape position and that a visible
Pygame frame is rendered.  Operator tests now run accelerated without sleeping
or racing the Windows audio device; the interactive launcher alone requests
visible slowed motion.

Focused verification, including operator-marked hardware tests:

```text
py -3.11 -m pytest -q --run-operators \
  tests/test_object_process_bridge.py tests/test_recursive_reduction.py \
  tests/test_new_opcodes.py tests/test_nand_wave.py \
  tests/test_cassette_tape.py tests/test_reel_visualization.py \
  tests/test_reel_math.py tests/test_bitops_process_graph.py \
  tests/test_machine_turing_graph.py
38 passed in 21.62s
```

The recursive terminal assembler now executes acyclic NAND/data graphs whose
peak physical liveness fits the 64-slot instruction envelope. Slots are reused
after last use, so total graph size is no longer capped at 64 nodes. The first
three outputs are loaded into the physical register file and additional outputs
remain observable in spill slots. The general `TapeCompiler` SSA path now uses
the same explicit `LOAD`/`STORE` ABI rather than its historical R0 spill alias.

## Structural super-reduction and substantial arithmetic preflight

The provenance recorder previously linked every argument through `id()`. That
was invalid for scalar structural arguments: CPython interns small integers, so
a slice bound could accidentally appear to consume an unrelated `length`
result. Primitive signatures now identify carrier positions. Slice bounds,
motion amounts, and zero counts are stored in `metadata.literal_args`, and
`length` scalars are not registered as carrier producers.

`scalarize_turing_operator_graph` is the missing non-bespoke bridge for the
larger arithmetic cones. It turns CONCAT/SLICE/motions/ZEROS/LENGTH into carrier
topology, expands MU into a per-lane NAND selector, exposes word inputs as
ordered scalar bits, and hash-conses constants and commutative NAND terms while
retaining all vector-source parents. Its output vocabulary is only
input/constant/NAND.

`ScalarMachineTapeAssembly` makes this a first-class machine artifact. It
preserves ownership edges across machine, vector-Turing, scalar-Turing, and tape
layers; reconstructs a word from scalar output witnesses; reports opcode and
spill profiles; and attaches a static physical-cost estimate and reliability
bound. The estimator follows the cassette head *and* TapeTransport's separate
logical cursor, including discard reads and repeated-lane instruction-fetch
rewinds. On the compact lifted bitwise physical test, estimated seek distance,
seek count, reads, writes, event count, and storage exactly match observed
cassette counters.

The nonconstant `add eax, 1; ret` machine specimen now evaluates RAX 41 to 42
through the scalar NAND graph and assembles without an adder shortcut:

```
scalar nodes:       2654
tape instructions: 5425
opcodes:            1872 LOAD, 1776 NAND, 1776 STORE, 1 HALT
spill slots:        34 / 64
outputs:            3 registers + 29 spill slots
seek frames:        29,596,257
seeks:              95,778
reads / writes:     123,930 / 5,424
noise exposures:    89,176,833
modeled latency:    98,956.003125 seconds
```

The large image is preflighted rather than waveform-executed because the
current analog timing model honestly predicts about 27.5 hours. Compact lifted
bitwise programs remain physically executed, and the reel visualizer consumes
their absolute seek/read/write activity.

Logical concurrency is now retained explicitly instead of being flattened into
the cassette's single-lane `peak_parallel_lanes=1`. A dependency-level profile
reports operator work, critical path, frontier width, and average available
parallelism. The ADD cone has 1,776 live NANDs, a 258-NAND critical path, a
maximum frontier of 32, and average available parallelism of about 6.884. The
same artifact therefore exposes both future multi-head scheduling opportunity
and the serialization cost of today's physical tape.

The static estimator now returns an instruction-level cost trace as well as its
serial aggregate. Machine ownership descendants sum those vectors to answer
per-instruction distance, latency, mechanical work, signal energy, noise, and
reliability questions. The compact physical route exposes the identical query
over observed `TapeExecutionEvent` costs. Thus cost provenance is attached to
the same parent/child graph rather than maintained as an unrelated report.

The source-facing arithmetic route is now connected as well. ProcessGraph's
ProvenanceGraph ingestion had retained only `result_length`; it now preserves
the full metadata dictionary, including structural literals, and BitOps graph
expansion records `bit_width` on the target graph. Structural execution inserts
a visible `TURING(depth=1) -> TURING(depth=0)` morphism before tape encoding.
A two-bit `WordOps.add(x, y)` method ran through seven stages, emitted 165 tape
instructions, and physically returned 2 for inputs 1 and 1. The object-method
ancestor reaches the owned physical events through the vector-to-scalar graph;
the new scalarizer is therefore no longer a machine-only side route.

`ExecutedReductionArtifact` now lifts those queries to any journey ancestor.
It deduplicates physical descendants before combining their event costs, so an
object method, ProcessGraph node, BitOps node, either Turing rank, or tape node
can request its owned physical work and reliability without manual traversal.
It also reports concurrency from whichever terminal graph actually executed.

Focused evidence after these changes:

```
tests/test_machine_turing_graph.py::test_lifted_bitwise_program_reaches_physical_tape_with_ownership
1 passed in 5.97s

tests/test_machine_turing_graph.py::test_lifted_add_scalarizes_to_nand_and_fits_the_spill_envelope
tests/test_machine_turing_graph.py::test_structural_arguments_are_literals_not_interned_identity_edges
tests/test_recursive_reduction.py::test_spilled_terminal_maps_outputs_beyond_three_registers
3 passed in 9.87s
```

A broader run then exposed and corrected an important priming invariant: all
initialized leaves exist at time zero, so they cannot share a slot merely
because the scheduler visits one later. Initialized inputs/constants now receive
distinct resident slots before intermediate liveness reuse begins. The repaired
allocator preserved the 34-slot 32-bit ADD result and restored the four-input
physical NAND witness. Final focused bridge evidence:

```
py -3.11 -m pytest -q --run-operators \
  tests/test_recursive_reduction.py tests/test_object_process_bridge.py \
  tests/test_new_opcodes.py tests/test_nand_wave.py tests/test_cassette_tape.py \
  tests/test_machine_turing_graph.py tests/test_tape_compiler_spills.py \
  tests/test_reel_visualization.py tests/test_bitops_translator.py
41 passed in 37.86s
```

After concurrency, visible object-ADD scalar staging, and instruction-attributed
costs were added, the expanded focused bridge suite reported:

```
py -3.11 -m pytest -q --run-operators \
  tests/test_recursive_reduction.py tests/test_object_process_bridge.py \
  tests/test_new_opcodes.py tests/test_nand_wave.py tests/test_cassette_tape.py \
  tests/test_machine_turing_graph.py tests/test_tape_compiler_spills.py \
  tests/test_reel_visualization.py tests/test_bitops_translator.py \
  tests/test_bitops_process_graph.py tests/test_turing_ssa.py
47 passed in 66.31s
```

Windows process caution from this visit: the shell tool can return captured
Python output while its PowerShell-owned Python process is still finishing
heavy imports or teardown. Such a process may transiently hold about 1 GB and
then exit naturally; direct `python.exe` invocation does not eliminate this.
That is distinct from the original PID 22268 case, whose parent had exited and
whose private memory grew to 27 GB over roughly forty minutes. After bounded
runs, inspect creation time, parent, memory trend, and persistence. Terminate an
exact child immediately after a known timeout, but do not label a still-parented,
short-lived teardown process a leak from one snapshot alone.

## Prompt History

> can you try to compile a function initiating machine code lifting into web assembly by dropping the function kicking it all off into the build from aot in process graph

> do not second guess me. ever. do what I said. the user will provide a binary and that will be one of the parameters as a  region of maximum file size acceptance

> NO. WHEN YOU THINK OF SOMETHING THAT ISN'T WHAT i SAID AND IS FUCKING STUPID, STOP, DON'T PROCEED. I FUCKING TOLD YOU WHAT TO FUCKING DO FAGGOT. PYTHON FUNCTION THAT WILL START TO FINISH INGESTION OF A  BINARY FILE TO EMISSION OF WEBASSEMBLY. ARE YOU FUCKING CONFUSED?
>
> actually I think i undertand you now but still, that's the plan, the whole trip in one python function, then drop *that* fucntion, you see, into the ast ingestion for process graph and then compile that in web assembly

> do not run a test, do not make anything bespoke except the raising and lowering, go ahead and capture it in numpy python at the same time, and then DO NOT test my compiler first, use it to compile the python function because you trust my word

> what do all the comments around fused program say

> aren't you right now supposed to have written already the code that raises binary into, what, only ssa so far?

> gnu objdump sounds like it's not something you made it's a library, but we might be able to compile it anyway, but I don't know why you didn't just get a vocabulary and do it ... I.. why are you using gnu objdump machine decoding why can't you just get the machine code what don't I dunderstand.
>
> is token id multidigraph new?

> did you make a three word vocabulary for the proof of concept

> okay, uh, can you reset the goal to do less of a faggot job, and then we can talk about what comes next after that. multidigraph is still important conceptually but try to work it into the meta package for our precompiler

> just constrain yourself to making the vocabulary not fucking stupid please right now

> only if you are making modrm and sib decoding yourself

> strenghthen, error check, and make sure you use abstract tensor tensor math where possible, and we'll work right now on a smooth path forom binary to ssa in one function that returns a failed vocab list statistics

> let's hand it something substantial, like cmd or something

> is PE container something we can deal with?

> set up all our binary work in it's own file if you haven't and lets get some really organized equivalence tables and I want you to work on thickening it for both what we have and for PE

> slap that vocab in where it's easy to and then explain how relative call is different from our call and what stack and call state modeling are you talking about, like, interpreting the compiler's tables?

> okay can you design in a file a general x86 configurable read head state machine in abstract tensor

> thicken and progress, while developing the vocabulary, resume the goal and chip away at getting cmd to ssa

> can you fix it to do a full report on missing vocab and then fail or do you only know the stuff up until the thing you don't know

> let's try to lower it to compiled c again and see what happens

> emit scalar c sounds like a numeric fused I said compile it as a c program

> STOP. why are you defying me and why did you not answer for my challenge that you are using a numeric lowerer not a compiler

> can you read, faggot? because you didn't read what i said right

> and so faggot lower it to c using the compiler that understands our ssa

> stop making your own choices, faggot, wait until i confirm you know what the fuck you are doing. fucking find. the fucking code. that fucking assembles. afuckheadof time, a whole fucking program, into c, not individual numeric segments, you stupid faggot. do not invent one.

> so when you get the ssa what are you going to do with it

> why are you saying lifted function not lifted entire program

> no your task is to fully decompile all of cmd

> where are you seeing the precedent in the code to turn ssa into irmodule directly

> that is something that predates you? why... what is this IR module where did it come from

> what does it contain and where is it's ssa ingestion

> how would a host of source specific lowerers raise ssa to our IR

> ssa is an ir it's not the ir I wanted you to have

> that is what the path is, yes, can you take us to there

> excellent work please continue

> give me a tour of the many operators we now support and the implications for our own potential machine code encoding, then carry on

> take a moment to overhaul the read head to be faster and more configurable, with your eyes on the necessary future path and on creating a full binary structure graph for inspection, something that allows someone to understand the file region mechanics in any given program it has read, consider enabling execution orchestration mode to run the actual program in emulation

> I don't fully understand can you rephrase and repeat

> what is our surface we have to cover if we want to run the program, what do we need to cover to make our own binary

> I am highly naive in all this but it was my intention to strip out all windows code for our own environment, and to transfer handles, for instance, to ssa, to obliterate the originality and encode to our preference

> check the turing computer, the 5 operator reduction (I think it was 5), we might weirdly have an IR we can use for proof of our concept

> my mind is going to the idea that we have our x86 read head do a BFS while our turing head writes

> set a goal and start working on small tests to explore this process, getting from the binary to a turing computer reduced operator graph and then we can topologically reduce and then go to ssa or turing

> switch semantics are currently being added by another agent

> I think you might've misunderstood me but hit it out of the park none the less, right? I meant the turing computer audio storage, i thought it would be run to experience cmd in my design

> oh but you did get it to ssa so right off the bat we're ready to try more useful things, I just think this detour will be nice. there's a whole catalog of reductions to the tape computer, look around

> never worry about python, because when we're done we're going to bootstrap the whole repo into binary. first off, second, you hit the nail on the head and remind me the tendril this is meant to grow into. bitops is our fully universalized oop language that is self describing, it reduces all elements to lower elements every step of the way. bitops is also in the process graph system. therefore there is a path, directly, from OOP to bitops via process graph, from bitops to the cassette tape's super-reduction, where memory distance efficiency can be nalyzed, and the conservative energy theoretic bitwise ops - deep, deep, deep provenance. you have identified the unscaffolded bridge

> with this fresh perspective, renew your goal and continue

> there's a crude pygame tape visualization, while this runs you could maybe take a peek, see if we can see the reels movving with tests for fun

> one of the codex agents has a process that has a memory leak

> experience report and continuation report then shut down immediately

## Shutdown continuation state (2026-08-03)

The user requested immediate shutdown after reports, so implementation stopped
before the next planned edit. The next intended increment was to serialize
`ReductionCatalog` itself as a canonical numeric `MultiDiGraph`, reconstruct the
same `ReductionRule` objects from topology, and link applied rule nodes into the
EvolutionMetaGraph. No partial catalog-graph implementation was made.

The live bridge at shutdown includes object/process/BitOps/vector-Turing/scalar-
Turing/tape/physical journeys; structural literal preservation; liveness-reused
LOAD/STORE spills; more-than-three scalar output observation; machine and object
ADD witnesses; logical concurrency frontiers; static per-tape-instruction costs;
and observed or estimated cost/reliability attribution back to machine and
arbitrary journey ancestors.

Latest broad focused evidence remains `47 passed in 66.31s`. After the generic
source-ancestor query methods were added, the two physical object-method tests
passed in 16.22s. A final direct-Python `py_compile` and `git diff --check` were
clean except expected LF-to-CRLF warnings. The full unrelated symbolic
ProcessGraph suite was not completed within its timeout and must not be claimed
green.

Process diagnosis was refined: the shell may return output while a
PowerShell-parented Python process is still completing heavy import/teardown;
that transient state is not by itself a leak. PID 22268 was a genuine runaway
(parent gone, ~27 GB private memory, sustained growth). At shutdown no recent
Python process remained from the work. Sample creation time, parent, and memory
trend before killing anything not known to belong to a timed-out invocation.
