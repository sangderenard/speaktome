# Turing: extend bounded dispatch work to source/SSA ownership units

2026-10-11 N4 EMISSION FOLLOW-UP (patch34 sequence_phi_span_cast_extents.patch):
Observed N4 first-five negative list: four replacement helper i64 Phis fed
by i32 zero constants, plus a planned-region span Cast without runtime extent
origin. Tiny exact replacement helper reproduced the width refusal before
editing; lower_sequence_replace now mints zero/one as int64, matching its
existing index/backedge/length cells. Status remains its existing int contract.
Native C and LLVM replacement tests include alias/nonalias, repeated calls,
empty/excess/negative length, corrupt source capacity and preserved failure contents.

Separate tiny nested-call span Cast reproduced the exact missing-extents
message. LLVM's existing origin walk now follows the exact same-rank numeric
Cast SSA operand edge, preserving precision-specific origins. Cast-only regions
join existing %extents propagation. No fabricated size or single-element fallback.
Native tests cover rank1/2/3, nested call depth1/2 and positive/negative2**40;
an actually unbound extent still refuses. 18 targeted cases passed34.92s;
combined UI/telemetry/sequence/merge/span/adjacent LLVM gate135 passed41.20s,
zero failures/skips. Cffi deprecation remains unsuppressed.

Untouched C:/dev/tg-base comparison: same sequence index/seed width mismatch
exists, but its old pointer-Phi emitter never validates pointee widths and
reports no replacement shortfall. Current gate exposes that latent SSA contract
defect. The separate span Cast repro DOES refuse on untouched baseline too.
This is not a claim the entire user's N4 program now emits successfully:
no saved SSA survived in profile_rules_4_llvm_3kkhydfy and no agent N4 rerun.
Next user-run should add --save-ssa; exception lists only first5 shortfalls.
Actual N4 graph-work receipt remains worker_threads0/admission_limit1 and
source_and_ssa_pagedFalse;2,203,705 book cells RAM/0 pages spilled. Those
architectural concerns are not solved by the display or these native repairs.

Both fixes installed in canonical live, benchmark and repair copies;1975
Python/contract files compared, only preserved repair N4 diagnostic hooks differ.
Storage-effects assets refreshed/load-only verified for final source:
live e18005a989ae4399ef836f1ede43c0e3, benchmark d786ca7bc17cd9f79de4a614ea340d51,
repair fc187adb43eeb801700c4eff2c855f06;1690 rows/188 functions/0 structural
findings and0 unposted IDs,1116 unsourced facts OPEN in each.


2026-10-11 ONE-OWNER TERMINAL DISPLAY (patch33 compiler_live_display.patch):
Auto uses installed Rich14.3.3 Live; optional progress_renderer=auto/rich/tqdm
or TURING_COMPILE_RENDERER. No dependency installed; no Rich import headless.
One redraw owner composes timing table, stage status and measured work bars;
logs/stdout/stderr scroll above it. Existing heartbeat, no second refresh
thread. Resize collapses rows/compact view; active stage survives20x6. All
timings/counters remain in the existing summary even when rows are hidden.
Ctrl+C and drawing faults restore streams; failures degrade to plain logs.
N4 probe heartbeat now uses the active renderer instead of bypassing through
sys.__stdout__. Installed in canonical live and both existing copied compilers.
Baseline60 passed2.88s; final UI/telemetry/fixed-point gate73 passed3.67s;
broader UI+pythonic-control+scalar gate127 passed254.44s. Cffi warning retained.
VT hide/clear/restore and resize tested explicitly (command harness TERM=dumb
otherwise disables cursor control). No interactive Windows screen capture.
Matching storage-effects assets refreshed/load-only verified: live066b79899d6a71495dd9a34c5f59669a,
benchmark0930a4b9fc7e9cf642990f12012aefb1, repair2e13a919a0842dbc23b89e3fdda6c667.
Each audit1690 rows/188 functions/0 structural findings;1116 unsourced facts OPEN.
User N4 failure subsequently received: combined source lowering350.81s, full
probe658.981s, peak6.11GB; emission refuses four sequence-replace Phi widths
and a span Cast without extents. Error prints FIRST FIVE shortfalls only;
not proof there are only five. No saved SSA in the named build directory.
That native failure is separate from successful display/telemetry verification.


LOCAL GIT INTEGRATION COMPLETE: Turing main compiler commit a4548f89 contains
the preserved compiler foundations,32 repair/probe patches, tests and profiling.
Root coordination map committed9fc96c9; shared guestbook/todo initial receipt
committed9f424ba. No remote push. Unrelated pre-existing edits/artifacts remain
unstaged. New Turing worktrees must use its own main, not the parent's branch.
Matching native storage-effects assets also refreshed/load-only verified in
benchmark2bffc361c6871181b735265c159e5a7e and repair9df7d0019cc11278dabb98ecc246b46e;
all three have the same1690/188 audit and1116 unsourced facts OPEN.

FINAL VERIFICATION (2026-10-11): combined affected+telemetry gate653 passed
361.18s, zero failures/skips/exclusions; final profiling-only gate60 passed
3.77s (includes two tests added after combined collection). Third-party cffi
imp deprecation warning remains unsuppressed. Native storage-effects unit
refreshed/load-only verified: dcf60218adf76c0b6f38b40d6682c3f7,1690 rows,
188 functions,0 structural findings/0 unposted IDs;1116 unsourced FACTS OPEN.
32 probe_job_observation.patch removes a stale hardcoded concurrent-main claim:
null means not measured, not a claim that any main job exists or is absent.
1973 Python/contract files compared against both copies: compiler sources,
tests and contracts agree. Only repair probe_four_rule_timing.py differs with
its preserved --capture-ssa exact598 diagnostics and --piece-cache option.
No changes were dropped to force those diagnostic probes byte-identical.
The pending combined-gate sentence below records an earlier point in this turn.

2026-10-11 CANONICAL INTEGRATION / PROFILING UPDATE:

Canonical compiler: C:/dev/Powershell/turing (its own main repository).
All29 saved repair patches,30 four_rule_copier_options.patch, and31
compiler_profile_summary.patch are installed here and in the benchmark copy
C:/Users/alber/.codex/worktrees/native-fallback-link/Powershell/turing
and its build/n4-repair/turing copy. Those copies are ignored children of a
root-only coordination worktree, NOT Turing Git branches. Parent merging
cannot transfer them. Existing compiler foundation edits were preserved.
Prior claims below that live source is untouched/unapplied are historical.

Before telemetry changes: live affected gate595 passed335.79s without exclusions.
Profiler/progress/fixed-point gate60 passed3.77s. Combined gate rerun pending
at this receipt; latest completion will be appended before handoff.
Seven live concordance audits:0 structural findings/0 unposted minted IDs;
unsourced fact latch remains OPEN392/759/272/642/533/20/230.

Public lower_ast_source_to_ssa now retains module.metadata['compile_profile']
by default even headless; compile_profile=False disables headless collection.
The TTY display has scrolling logs, persistent stage wall/RSS table and work
counters below. Stage order is not a percent, fixed-point safety limits are
not work totals, and iterator counters advance after processing. Known totals
show completed work fractions; unknown totals have counts, no invented ETA.
Timings are source-to-SSA only, NOT emission/build. Partitioned compilation
keeps its existing manifest unit timings. Planner aggregates retain existing
phase durations and slowest function context without per-dispatch graph copies.
Use src.compiler.shell_telemetry.compile_profile_dataframe(module) afterwards;
TelemetryChannel.profile_dataframe() and DeploymentProfiler.summary_dataframe()
are lazy pandas views. Nested/inclusive timing rows are not additive walltime.
No new dependencies, no tracemalloc, no agent N4/main run, no killed processes.

Recovery: build/integration_recovery_20261011_29/manifest.json and original
affected files; combined29 applicability artifact build/compiler_integration_all.patch.
Remaining: user fresh N4 LLVM run; no whole-main speedup/parallel residency
claim, and the OPEN provenance writers remain work. Native whole copier is
optional; the original Python copier and faster serial route remain available.


## Oct11 negative follow-up green; integrate into the correct repository

595 affected tests pass with no exclusions;352 post-installation checks pass.
All four formerly excluded dispatcher tests are included. Updated obsolete
fixtures/API assertions retain intended coverage; genuine compiler defects and
new native wrong-return/empty-generator failures fixed at graph/book writers.
Seven audits0 structural findings/0 unposted minted ids; UNSOURCED FACT LATCH
OPEN remains392/759/272/642/533/20/230, not a completed provenance migration.
Patch27 Fortran/Phi repairs and28 negative_contract_followup.patch installed in
benchmark copy;29 rule_probe_cli.patch preserves the installed N/all CLI.
Storage-effects native asset refreshed/load-only verified underf8aec357b935ef27baf0ffc3ac7b1840.

NEXT integration is into live turing's OWN dirty main repo, preserving its edits.
Benchmark turing is merely an ignored copied child of the root coordination
worktree, not a compiler branch. Do not merge parent expecting compiler changes
or force-add turing there. Live compiler source untouched; no merge performed.
Then user-controlled fresh N4 LLVM verification (no trace/deadline, save SSA on
emission failure). No agent long lowering. Historical full-repository failures
and unsourced writer migrations remain open beyond the595-test affected gate;
main source/SSA parallelization/residency tiling also remains unfinished.
Use exact full receipt at turing/TEST_BASELINE_AND_HAZARDS.md; older active-job
statements below are historical, not current process evidence.

## Oct11 source torture repaired — use the actual updated benchmark worktree

Seven programs all failed the untouched baseline; now execute via our C and
LLVM backends,60 invocations/14 tests. Independent guards and empty/1/2/17 rows
catch flattened/duplicated loops; literal birth, payload reads and tuple splices
cover ordinary storage/control interactions. Interval-capacity proofs retain
unknown backedges; no arbitrary arena sizing. Existing list effects/row loads
and shared allocation requirements reused. Final focused repair gate127green;
ordinary concordance counts unchanged, restore-type finding/latch OPEN retained.
Patch26 pythonic_control_torture.patch preserved; core changes applied to actual
native-fallback-link benchmark worktree, not live checkout. Prior25 remain.

NEXT: complete post-application checks; refresh accelerator assets through OUR
compiler's existing builders, then user-requested fresh N4 verification. Do not
describe tiny native proofs as full-program parity or a timing improvement.
Retain original copier route; main native run remains user-terminated. Extend
this SMALL deterministic source corpus with lexical re-creation in repeated
loops, indexed mutations/aliases, continue/early-return combinations and valid
backend contracts; do not replace existing concordance or create a new runtime.
Main source/SSA residency tiling and parallelization remain separate unfinished
goals. Historical receipts below are not current process status.

## Oct10 N4 original-copier comparator fails; native run left active

User explicitly requested both concurrent four-law probes, no trace/deadline.
Same improved snapshot compiler, original copier with isolated fresh law cache:
exit1 /561.4117003s. Four law builds175.6213412s versus native copier1229.3s.
Combined C emission fails %t598 dynamic extent origin, then two unavailable
loop operands whose exact producer Calls(%t598) remain in SSA. Do not guess a
storage origin or repair by id/name; observe original storage/extent identity,
its concordance facts and every writer before proposing a fix. Determine
whether the two operand failures are downstream of refused extent emission.
No complete artifact/parity conclusion; current four law function counts and
opcode totals match both routes, not proof of semantics. Historical dropped
content remains unproven. Original copier stays available with all fixes kept.
Native worker14620/session52547 remains running; final counters pending, no
intervention. Full exact receipt opens live TEST_BASELINE_AND_HAZARDS.md.

## Oct10 whole original copier is native; performance remains next

Patch25 native_whole_metadata_copy.patch UNAPPLIED; concatenated25 check0.
Exact original50-line copier executes all container branches, sharing and cycles
through OUR compiler/LLVM, no item callback. Original serial route retained.
Build/register/load-only APIs in native_compiler_accelerators use existing
registry/checkpoints/fingerprint/library digests; explicitly loaded extraction
uses one native batch/shared memo for graph AND node metadata. Source-path
concordance fixes guard ordering; no coordinate-based repair. Connected15green,
ordinary audits unchanged; full receipts live turing TEST_BASELINE_AND_HAZARDS.

Native is NOT faster: solo64rows .1114ms eager/2.7408ms including boundaries.
Do not launch N4 to rediscover that. Next: warmed boundary/kernel/export timings;
reduce needless rehydration of native-only temporaries under existing graph
observability/ownership proof; improve linear member/memo lookup using the
incumbent object identities and storage contract, not a second object registry.
Retain admission refusal for builtin subclasses and propagate native faults.
Context allocations live until explicit release; no peak-RAM reduction claim.
Main parallelism/whole-source+SSA tiling remain independent unfinished goals.
Nothing reverted or abandoned; no new main/N4/process kill this turn.
Final all-changes gate132pass/4 known exclusions125.67s; fresh-process native
provider active during compiler/extraction cases12pass6.63s, without compiling
the provider at load. Guestbook dry-run0; concatenated25apply-check0.


## Oct10 paired object representations now read the shared book

Patch24 native_paired_object_concordance.patch UNAPPLIED after23;24ordered
apply-check0, live source unchanged. Observed paired tuple/component/target
facts were present, but admission declined and SSA invented row columns.
Composer flattened away the original producer; span-based placement hoisted
tuple birth outside a span-less comprehension. Representation and publication
now live on native_object_source, DERIVED from incumbent graph identity cells.
Control carries Ref; SSA/backend read latest facts and attached-book aliases.
Actual fields/projections/bindings checked; stale/missing/copy corruption refuses.
Explicit LOWERING->EMISSION lifecycle retains normal cold spilling.

Native normal/reversed/nested pairs and generator->dict update with linked
calls execute0/1/3/64entries, shared refs/backedges/order/fresh births/release,
one outer invocation, no callbacks. New7+lifecycle1=8pass10.85s; prior focused25
and broad74 green; final75pass77.19s after all changes. Audits unchanged/latch OPEN.

NEXT: full copier now stops at Boolean collection98, not pair97. Graph26 is
new is old, iterable70 zip(isolated,value), loop79, generator80. Consume the
existing zip/all identity program as a reduction, not a bool collection arena.
Also observe why tiny recursive dict-copy variants leak unowned formal6/8,
unknown dtype/rank0: exact losing writer UNKNOWN. Keep recursive guard. General
iterator suspension, filtered/nested generator control, arbitrary custom items,
Python equality/hash, linear-scan indexing and lookup complexity remain outside
the proven path. No whole copier/native recursion or main speedup claim; no
main/N4, timeout or kill. Native table get/set are still linear scans.

## Oct10 connected native contract batch now executes recursive memo work

Patch23 native_object_contract_batch.patch UNAPPLIED after22;23combined
apply-check0. Default object context ABI preserved; optional builtin type bits
share its existing identity table/heap. Boundary stamps once, native births
stamp matching bits, source isinstance lowers from actual intrinsic/class
receipts. Call-graph-derived payload needs reject opaque container subclasses
without member rows. Concordance/backend validate type and requirement receipts.

Exact alias40->memo6 now binds every captured descriptor member. Keyed Input
writes remain lexical. Deepest composed graph owner places recursive calls.
Observed early argument pruning removed caller-owned memo column, later restored
as one stack i64 and crashed. Existing ProgramABI owner now prevents that early
drop; native fixtures assert all descriptor members are public buffers and run
depth0/1/7/32, both tuple type/length guards. Final17pass20.66s checks exact memo
keys/exported [] values and one outer invocation; broad92pass75.09s preceded
test-only strengthening. Ordinary audits unchanged/latch OPEN.

NEXT connected batch: production copier paired generator collection97 remains
unknown (tuple leaves14/15, loop75, materializer76, iterable19). Admit exact
pair-row ownership/projection and paired iteration through existing contracts,
then dict update/extend, set/frozenset materializers, zip/all and memo.get.
Restore static tuple-of-types receipt at its observed reduction writer; do not
infer targets from names or accept empty evidence. Keep recursive allocation
guard. Full50-line copier refuses2.311162s, no native artifact/execution or
main/N4 speedup claim. Hash/direct indexing, bounded whole-memory tiling and
main parallelism remain unfinished. Live source unchanged, no process killed.

## Oct10 region-less loop ownership restored in existing composer

Patch22 regionless_loop_ownership.patch UNAPPLIED;22combined apply-check0.
Observed declared If11->Loop13 edge was ignored for a zero-region loop, emitted
beside its owner. Overlay now uses that edge and exact source arm/statement
identity, with incumbent dispatch nodes for placement. Existing placement book
reused; missing/cloned identity or cross-slot region refuses. No new ABI/registry.
Native body/orelse generator tests execute0/1/17/257 members, both outcomes,
one outer invocation and no unselected births. New9+prior12=21pass8.71s;
broad75pass70.22s; pre22 public-source target formal fails4.29s. Audits unchanged.

Tiny recursive tuple reaches unresolved kind predicate10, not private target0.
Full copier still refuses call14 on memo sequence6/member1,4.570708s. Trapped
accounting names metadata_memo, span/rank1 and linked sequence provenance;
exact writer losing shared keyed ownership remains UNKNOWN. NEXT trap that
binding writer and repair the existing frame contract, then admit runtime type
checks without confusing closed-domain KIND with Python MRO/metaclass behavior.
Keep recursive allocation guard. No native full copier, speedup, tiling or main
parallelism claim. Live compiler unchanged; patches remain unapplied.

## Oct10 preserve linked reference facts before structural folding

Patch21 linked_reference_kind_fact.patch UNAPPLIED;21combined apply-check0.
Structural folding ignored linked reference ABI, rebuilt scalar/opaque_ref,
proved isinstance False and deleted live collection value/control producers.
Observed writer chain, fixed Input evaluator to use incumbent contract. Unknown
dynamic kind remains runtime; no new contract or physical handle reinterpretation.
New12pass2.64s; broad58pass58.55s; audits unchanged. Native id micro executes;
recursive graph preservation test is NOT a recursive native execution pass.

Earlier source-gate success was not semantic proof: branches were erased.
Exact copier now refuses opaque recursive scratch (callsite14),1.876627s.
Tiny graph proves composed Loop13 and If11 are siblings: call3 is in If11,
member Load0 in Loop13. NEXT: repair source graph/control ownership in existing
composer, then admit needed runtime type tests through existing object contract.
Do not treat closed vocabulary kind membership as general Python isinstance.
Keep recursive allocation guard; no opaque scratch workaround. Hash/shared find,
bounded batches, whole-memory tiling and main parallelism remain unfinished.
Live compiler unchanged; speculative call-selector change was fully removed.

## Oct10 memo membership/capture fixed; native loop contracts next

Patch20 native_memo_membership.patch UNAPPLIED; twenty combined apply-check0.
Exact declared keyed inputs now support native in/not-in, int64 induction,
boolean Phis and complete captured caller storage. Existing stable-handle id
and integer-key/opaque-value ABI reused; no key-reference substitution. New
concordance receipts check exact descriptor/query bindings. Final47pass52.38s;
broader old StopIteration reproduced on pre20; seven audits unchanged.

Exact copier clears source gate (predicate37 fixed), still fails LLVM: five
loop width conflicts and four opaque-input tensor extent Calls,2.154057s.
NEXT: micro captured-object iteration repro, observe why collection ownership
is absent, restore existing object iteration/collection contract at its writer.
Then retry same source. Do not treat opaque handles as dereferenceable tensors.
Native whole copier NOT running; integer hash/direct index/shared find still
pending; linear growing-memo work can remain quadratic. Index validity/growth
must be explicit under existing arena and mutation contracts. No main speedup,
memory tiling or parallelism claim. Leave live compiler and all patches as-is.

## Oct10 native generator collection seam delivered; memo predicate next

Patch19 native_object_collection.patch UNAPPLIED; nineteen combined apply-check0.
Exact graph-owned dense tuple/list collection, real runtime extent, native body
and allocation-safe direct rows now execute in ONE native invocation. Existing
ABI fixed point types native births/results before private frame declarations.
Live Ret replaces stale structural return binding as alias authority; int64
loads retain SSA snapshots rather than rereading changed cells. No parallel
claim for shared mutable contexts, per-item ctypes or generic lazy generator.
Filtered/nested collection control is not newly supported; preparation still
scans global member rows.15new tests19.06s; broad125pass98.65s. Extra old symbol/
Phi test failures reproduced against pre19/clean baseline; audits unchanged.

NEXT: unchanged copier passes collection93 and recursive storage guards, now
refuses ONLY undefined predicate37, identity in metadata_memo (observed actual
node, parents36lhs/6rhs). Bind memo membership/lookup from existing graph/query
and identity contracts; numeric identity keys are not reference-identity keys.
Retry exact source after a tiny memo repro, no rewritten copier. Full copier
execution, dictionary equality/hash, indexed adjacency, main speedup and whole
compiler memory tiling/parallelism remain unfinished. Live source unchanged.

## Oct10 bulk native materialization/ctypes reuse; generator collection next

Patch18 native_object_materialization.patch UNAPPLIED; eighteen apply-check0.
Public source tuple/list materialization executes natively against existing
object handles, shallow identities and source-edge contracts. Bulk stamp has
no per-item Python/ctypes or member lookup/store. Prepared native invocation
reuses the existing pointer ABI and leaves rehydration explicit. Validation
capability no longer searches full IR twice per dispatch; actual native errors
remain checked. Final micro31.7249us prepare-each versus0.6329us reuse; stamping
4097items1call0.0312ms, export6.0894ms. Boundary-only, NOT main speedup.
New11tests pass; final broad110pass78.45s. Gates/audits and intermediates in live baseline.

NEXT: exact copier's collection93, source29/control58, materializer60, iterable56
(tuple recursive generator). Ordinary tuple(value)56 is now handled. Link the
existing loop's opaque item/result/extent to native collection stamping inside
the same native pocket. Do not dispatch per item through ctypes. Member passes
still scan global rows; per-object adjacency/indexing is not delivered. Dict
equality/hash, leaf operations, native copier execution and whole-main tiling/
parallelism remain unfinished. Live compiler source remains unchanged.

## Oct10 source object births/access execute; iteration/materialization next

Patch17 source_object_context.patch UNAPPLIED; seventeen combined apply-check0.
Ordinary {} and [] birth as handles under accepted SSAObjectContext through
our public compiler. Direct/forwarded/recursive/loop/conditional native tests
and reference-key dict set/get/len pass;16 source tests, two66-pass regression
batches39.72s/53.26s. Shared context exact call bindings, control-edge handle
propagation and live receipt audits prevent unknown private arenas. Seven
ordinary audits unchanged; latch OPEN; live compiler untouched.

NEXT: unchanged50-line copier now refuses members=tuple(value), source11 to
materialization56, private_formal56/call14 (final retry2.0368875s). Integrate
native context iteration/materialization at the existing source producer, then
generator update/extend and real memo/access/type predicates. Do not rewrite
the copier or force unknown rows to guessed dtypes. Dict native keys still
compare reference identity; Python equality/hash obligations remain explicit.
Break-bearing loops are not admitted by new graph handle-propagation rule.
No native copier execution, speedup, main parallelism or whole-memory tiling
claim. All intermediate faults and exact commands in live baseline receipt.

## Oct10 object conversion ABI verified; source object producer next

Patch16 native_object_boundary.patch UNAPPLIED; all sixteen apply-check0.
Existing codec/context entry/exit graph conversion preserves imported Python
identities, aliases, cycles and leaves. New native representations use the same
codec owner, including stable repeated exports. Imported immutable changes
require new handles. Native source loops1/17/257 execute in one invocation,
no per-item callback; structural0, latch OPEN. Full100pass62.89s; final17/4.36s.
Typed raw Load preservation fixed at observed ir_indexing writer with book
receipts; exact aggregate-return edges preserve oscillator projection rules.
Seven ordinary audits unchanged. Explicit native release/lease close; native
member equality remains reference identity, Python equal-key collapse refuses
at export. Exact subclasses remain opaque. Full boundary graph stays resident.

NEXT: {} is known dict kind and initial extent0 even when contents are unknown.
Publish source constructor's object handle/accepted context BEFORE recursive
workspace leasing; integrate actual source inspection/access with existing
graph identities, preserving real dict equality/hash and copier memo cycles.
Copier retry1.7746843s still unknownworkspace12/call14; no compiled copier or
live install. Do not confuse unknown content with unknown allocation layout,
ship a restricted alternate copier, or claim whole-main parallelism/speedup.

## Oct10 explicit native context executes; automatic source object birth next

Patch15 native_object_context.patch UNAPPLIED; fifteen combined apply-check0.
SSAObjectContext is an explicit ABI view over existing typed dynamic object
and member tables, shared heap and handle counter. Native constructor/access/
release operations execute; public source native links preserve that context
through forwarding. Whole1/17/257-object loops run in one native invocation,
repeat with exact results, pointer cells remain caller-owned, structural0.
Fixed observed ptr-to-element dtype settlement, caller buffer ABI losing heap
cell/owner contracts, canonical opaque_ref/int64 accounting disagreement and
bool-to-int64 zero-extension. Monotonic reverse-call propagation uses exact
accepted input IDs and audited edges, not names. Reference-identity member
keys are NOT general Python dict equality/hash. Initialization is for empty
storage; same-context mutation is exclusive. Provenance latch remains OPEN.
Final66 pass45.33s; broader168/5 old80.77s; seven ordinary audits unchanged.
Live compiler unchanged; no main/N4, installation or speedup claim.

NEXT: the original copier still rejects unknown {} workspace12/call14
(retry1.6916952s). Connect admitted source constructors, kind/member operations
and the ordinary Python-object boundary to this explicit context, without
substituting a restricted copier or silently changing key/leaf semantics.
Preserve recursive shared memo and caller-owned lifetime. The earlier missing
early-return recursive call remains separate. Do not claim automatic object
construction, compiled production copier, main parallelism or memory tiling.

## Oct10 allocator/replacement/mutation path works; source constructor bridge next

User: "okay let's clean everything around the dynamic allocation contract and
see if we can make that the last rivit". Patch14 dynamic_owned_sequence_storage.patch
staged UNAPPLIED; fourteen combined apply-check0. Existing typed heap descriptor
now supports checked geometric reserve, empty init, insert, dictionary store and
extend. Shared wrapper uses incumbent fixed mutation bodies; validate before scan,
grow only on full, preserve duplicate/update statuses. Entire helper closure and
exact owner/row formal IDs carried through extend. Native byte-exact memmove ABI,
no Python callback or handle-as-address. Strict descriptor/cross-call layout,
counter alias and copy-contract audit checks. Unsupported shaped/live/child pools
remain named refusals. Retired arenas live until owner release; interior addresses
must be refreshed after growth. Unique-key scan still linear, not a hash speedup.

Native single-call whole loops1/257/4097 for insert and store, int64/opaque_ref;
five layout byte-copy reserve tests, zero/noop/invalid extents, overlap, borrowed
257-row extend twice, owner release. These are SSA allocation-contract proofs,
NOT automatic source construction or attached-book emission. Final163 pass/5 old
fail73.98s; seven audits unchanged, provenance OPEN. Live compiler unchanged.

NEXT: exact50-line copier still refuses dictionary12/call14 (final retry2.2399622s).
Actual source {} has two unknown columns, no heap owner/accepted native object
domain. Publish the source constructor's real representation/runtime capacity/
shared-owner descriptor BEFORE recursive leasing and schedule its allocation;
do not blindly relabel unknown as opaque_ref or cast a handle to a pointer.
Existing CONSTRUCT interface is not a generic object implementation. Early-return
missing-call defect also remains. Not last rivet yet; no installed copier/main
speedup, main/N4 launch, tiling, parallelism, package or process-kill claim.

## Oct10 declared common-return recursion now executes; dynamic copier remains

User: "okay, let's proceed". Patch13 conditional_result_phi.patch staged UNAPPLIED;
thirteen combined apply-check0. Observed graph Phi18(test4/body17/orelse9), no
initial version, skipped by planner's initial-value check; lower_conditional
had no aliases despite existing len17->sequence14. Planner now consumes fresh
result Phis through incumbent result lowering. Separate bare-predicate Phi4
retains its exact owner conditional even without numerical regions/calls.
No name-history synthesis, new SSA operation, backend repair or guard weakening.

Native common-return length recursion: depths0/1/7/32 times three objects,
lengths2/0/1, exact handle above2**60; all twelve outputs correct, caller buffers
unchanged, shared declared context, only invocation-local int status scratch.
Four source assignment programs/nine native branch cases also pass. Fortran
text complete; C module explicitly lacks zeros spelling, not claimed complete.
Final148 pass/5 old failures69.15s; seven audits unchanged, provenance OPEN.

NEXT: observe losing writer in early-return kind variant (recursive call absent,
RetdtypeNone, reconfirmed without native execution). Separately exact50-line
copier2.1632375s still refuses dynamic dictionary12/call14; supply its real
constructor element/extent/owner contract using incumbent systems. Successful
declared recursion is not generic native dictionary/object-domain construction.
No live install, main/N4, package, process kill, speedup/tiling/parallelism claim.

## Oct10 recursive borrowed keyed storage settled; next is control-result producer

User: "how is our gap now", "proceed". Patch12
recursive_keyed_storage_binding.patch staged unapplied; twelve apply-check0.
Existing keyed binding closure reused before lease discovery and after linking.
Early stage settles storage only, preserves query types until their producers
link. Exact descriptor/member derivations retained, guards unchanged. Recursive
call shares declared owner keys/values/length and member pools/lengths/stride;
only int lookup-status scratch is invocation-local. This is storage proof, NOT
native recursion. Final128 pass/5 known fail56.18s; seven audits unchanged/OPEN.

NEXT: trace deterministic producer losing base-case length result17 and Phi18
(test4/body17/orelse9). Current rejected SSA Ret reads recursive Call9 alone,
dtypeNone; no length Load or merge. Original losing writer not yet observed.
Full-native gate remains enforced; never execute malformed witness. Previously
recorded early-return missing-call defect also remains. Copier not rerun; last
production dictionary12/call14 refusal unchanged as evidence, not reconfirmed.
No live compiler install, main/N4, package, kill, speedup/tiling/parallelism claim.

## Oct10 nonrecursive source context forwarding closed; recursive producer remains

Patch11 object_table_source_context.patch staged unapplied; eleven apply-check0.
Native source kind/length/get/set works through one/two forwarding levels with
existing caller table columns/lengths/stride, not SSA-only callers. Shared exact
owner/field materialization and real record claims; pooled contexts never bind as
selected row pointers. Unread keyed ghost input removed at its observed producer,
old duplicate finding reproduced on saved pre-change compiler. C/Fortran text
complete with public outputs; LLVM execution exact above2**60. Focused74 pass43.83s;
separate54 pass/5 old failures7.04s; seven audits unchanged, provenance OPEN.

NEXT: typed declared keyed view must bind from existing owner/decomposition BEFORE
recursive_frame_lease, not through a late descriptor claim. Common-return recursive
micro refuses authored book.objects GetAttr6 as unknown workspace at call9; same
pre-change. Sequence-member edges exist, owner-record claim absent. Late claim
attempt was removed because it retained unknown key typing and failed recursion.
Current refusal regression protects guard, not a completed recursion capability.
Also retain early-return recursive kind micro: loses recursive Call, RetdtypeNone
on current and pre-change. Losing writer unobserved; never run it as native parity.
Both exact source repros recorded atop TEST_BASELINE_AND_HAZARDS.md.

Original copier1.6806848s still refuses dictionary12/call14. Bare object-domain
binding, builtin dispatch, recursive result representation and constructor dynamic
extent/owner lifetime remain. No installed copier, speedup, main/N4 run, tiling or
parallelism claim. Live source and running processes untouched.

## Oct10 object SSA/backend layer works; source binding remains

Patch10 opaque_object_operations.patch staged unapplied; all ten apply-check0.
Existing opaque_ref now has semantic SSAOpaqueObject view and exact-formal
native helper operations, checked at LLVM/C/Fortran and concordance boundaries.
Native identity/kind/membership/length/get/set execute against existing declared
Book/Object columns; C/Fortran text complete. Focused66 pass37.47s; extended119
pass/5 previously baselined failures43.90s plus detector1 pass2.04s. Audits remain
0,1,0,1,5,0,0, unsourced latch OPEN. Live source/copier/processes unchanged.

Next exact gap: source root->inspect retains matching metadata.Object identity
but caller materialize_declared_row_columns only publishes kind span, deferring
members table. Carry complete existing field context through authored call edges.
SSA-built caller is NOT automatic source dispatch. Bare Python entry still needs
an accepted incumbent descriptor-domain binding, not another object registry.
Then builtin runtime dispatch, recursive result constraints, constructor capacity
and shared owner lifetime. Construct/release contracts alone are not a generic
allocator implementation. Bodyless imports not admitted by new helper API yet.
Original copier3.0152095s still refuses unknown workspace12 at recursive call14.
No copier installed, N4 speedup, memory-tiling or main-parallelism claim.

## Oct10 ordinary opaque-input boundary investigated, not implemented

No compiler edits this follow-on. Public-entry full-native id(value) produces
complete LLVM0.541498s; isinstance(value,dict) refuses0.366311s at structural
Call2/arg0=value0, call-result-unavailable. Existing token_vocabulary is scalar
int64 only; SSAReferenceDescriptor does not associate runtime objects with native
record/sequence payloads. Existing CPython thunks invoke Python callables, not
native descriptor operations. Do not reinterpret handles as tags or pointers.

Production copier accepts ordinary Python containers, unlike explicit descriptor
micros. Resolve whether a one-time entry/exit conversion through incumbent native
record/sequence storage is acceptable before inventing that changed boundary.
Preserve aliases, cycles, original leaf identities and shared memo lifetime.
No patch10, copier installation, main/N4 launch or new speedup claim.

## Oct10 steps1-3 partial: explicit descriptor inspection now native

Patch9 staged unapplied, all nine apply-check0. Native length/both indexed
reference columns/traversal/repeated-field views now use exact declared keyed
record + table ABI. Empty-dict row constraints consume all known runtime stores,
not just literals. Focused45 pass29.47s; extended97 pass/5 old failures37.33s.
New native cases zero structural findings; seven audits0,1,0,1,5,0,0; latch OPEN.

Continue1-3: bind a genuinely opaque input to an ACCEPTED existing runtime
descriptor domain (the new tests still supply explicit Book/Object columns);
lower builtin runtime kind/container operations over that binding; carry native
reference result constraints across the recursive copier's SCC. Original empty
dictionary12 still unknown at callsite14, latest2.1708728s. Do not declare that
object payload unknown means reference slot unknown; also do not fabricate a
reference ABI where no accepted binding supplies it. Native memo domain remains
distinct from ordinary copied dictionaries' equality/hash semantics.

Step4 remains source constructor capacity + owned allocation/lifetime. New empty
dictionary test proves row layout only, NOT executed allocation. Preserve one
outer memo/results owner. Additional unresolved micros: integer carried seed is
converted by loop_composer float(literal), planned int64 accumulator body returns
float64, direct carried reference assignment externalizes unnamed output. Ages
not established this turn. Earlier indexed-loop/pair Phi failures remain red.
No copier installed; main7196/live compiler untouched; no N4 speed claim.

## Oct10 boundary semantics green; allocation-only finish not reached

Patch8 native_reference_boundary.patch staged unapplied; all eight apply-check0.
Snapshot native source memo insert/lookup/update uses exact declared row layout
and borrowed caller ownership. Handles above2**60 preserved. Existing keyed record
kind access executes natively with caller descriptor columns. Focused35 pass in
18.37s; extended89 pass/5 old failures26.96s. Audits0,1,0,1,5,0,0, latch OPEN.

Next deterministic gaps: original copier's empty dict12 unknown row layout/extent
at callsite14 (latest2.0398794s refusal); bare object input lacks supplied native
descriptor columns; nested members length/index needs exact field-sequence
binding (len(item.members) structural Call16/arg6 refusal); generic construction,
iteration, immutable publication and memo growth not closed. Use existing graph
and reference/record/sequence contracts, not fabricated type facts or callbacks.
Keep shared memo/results outside per-invocation scratch; existing owned heap ABI
supports that lifetime. Leave placement/capacity policy separate, but do not claim
it is the only unresolved work. Live compiler/main7196 untouched.

## Oct10 owned heap ABI exists; source copier wiring remains

Snapshot has actual runtime-sized owned heap Alloca/Call storage, native release,
and existing sequence insert/lookup using caller pointer cells. Arena lifetime
survives helper return; native fixed-capacity table chain works at1,7,4097 rows.
Concordance tracks owner identity and malformed allocation contracts. Internal
SSA execution is not source-level copier completion. Patch7 staged unapplied,
all seven patches apply-check0. Main7196/live source untouched.

Next: resolve ordinary empty dictionary12 element layout/runtime extent from
actual existing graph/reference/sequence facts, connect source constructor and
recursive storage lease to the owned heap ABI, then runtime type/container and
memo growth. Exact copier still rejects2.015371s. Do not assign unknown to a
handle or PyObject pointer merely to bypass the guard. Unsupported shaped arena
initialization/child pools/live flags/growth remain explicit, not truncation.

Also keep OPEN the inherited pre-heap native loop Phi i32/i64 regression:
simple indexed loop1 fail and pair-update fixture3 errors on saved pre-heap
files. Two numerical alias tests pass on clean tg-base. Current heap changes
did not introduce it, but investment is not fully green until it is fixed.
Extended heap/sequence/source gate83 pass/5 fail26.77s, all five clean-baseline
failures; audits remain0,1,0,1,5,0,0 with unsourced latch OPEN.
Focused allocation + public-source gate29 pass/16.51s; guestbook dry-run0.

## Oct10 follow-on: native micros green, generic copier arena contract still missing

Snapshot now supports positively proven builtin reference type predicates and
typed fixed-size invocation-local SCC leases via existing SSA/frame allocation.
Generic opaque/dynamic arenas still refuse. Mutable native parent/child repro
returns triangular sums0,1,10,78; actual state survives recursive call. Fixes are
at deterministic writers: Store AST must not be evaluated as a read; mutable
initializer must not become separate region constants; root pruning must keep
live descriptor members using the already existing callee-liveness rule.
Final focused16 pass /14.85s; root-prune units3 pass /2.02s. Final cross-module
gate59 pass /73.63s,86 unrelated deselected; audits retain0,1,0,1,5,0,0.
Combined six-patch apply-check0, guestbook dry-run0. No audit exclusions.

Unchanged copier retried1.940700s: callsite14/private arena12 unknown dtype/no
extent. Next work must resolve actual generic object/sequence layout and lifetime
through existing reference/sequence ABI facts, then memo/type/container operations;
static positive predicates do NOT supply these runtime operations. Keep one
caller-owned shared memo, not recursive scratch. No PyObject-pointer guess,
Python-per-item callback, integer-only substitute copier or native gate bypass.
Cumulative patch6 staged only; original production copier/main7196 untouched.

## Oct10 structural/recursive cleanup: real native micros, full copier still blocked

Snapshot repairs public/private id producer and parameter concordance; id's int64
result is now known during graph planning so native helper forwarding retains
2**60+3 exactly. Recorded SCC edges now become PlanCalls; iterative owner visits
and dependency walking retain direct, nested and mutual native recursion without
unfolding bodies. Five native micro-cases execute exact; three rejection cases
cover unsupported object operations and recursive private scratch.

IMPORTANT: recovered copier calls exposed scratch-signature feedback. Callee
private arenas were cloned into caller.args inside the same SCC, recursively
becoming more callee formals. Writer now refuses before lease/mutation, using the
existing admitted SCC edge. This stops invalid expansion; it does NOT implement
invocation-local storage. Exact copier currently refuses callsite14/formal12 in
1.918846s. Next required work is that native per-invocation frame ownership, plus
dynamic opaque-reference type/container operations on an accepted ABI. Shared memo
must remain caller-owned, separate from recursive scratch. Do not alias live
invocations, invent an object layout, bypass the full-native gate, or call Python
per item. Integer recursive return type settlement remains another observed gap.

All six promotion patches apply-check together and remain UNAPPLIED live under
running main. native_structural_producers.patch contains this cleanup. Targeted
tests8 pass; extended gate/audit final receipts in TEST_BASELINE_AND_HAZARDS.md.
Earlier statement that id native lowering did not exist is explicitly corrected.

## Oct10 requested N4 and fully native metadata copier — blocker checkpoint

N4 snapshot completed507.330979 s:8.58% below previous554.948946,4.31% above
older486.379790, single runs/full-main concurrency caveat. Combined SSA300.786486
s, whole334.614719; native analysis17 calls/.622484 s.6157 MiB max stageRSS,
2203575 resident book cells/0 spills,38.56% more cells vs previous, cause unknown.
Final detector itself failed on Unresolved.value_id; no clean audit or orbital
execution claim. Preserve all source paths/artifacts; no abandonment/reversion.

Fully compiling515987-call recursive metadata copier is NOT achieved. Exact
incumbent source/public-entry/no-callback probe exposes genuine control ownership
bug: comprehension computing If.test was assigned to If.body. Existing arm-slot
rule now reused in snapshot;24 focused tests pass, original test fails live.
Next refusal is19 undefined recursive/result operands; minimal dynamic-reference
id(value) and isinstance(value,dict) have no accepted runtime result. Need native
object identity/type/container/recursive-result lowerings over an accepted exact
object ABI, not integer-only walk with Python packing/callbacks left inside.
Exact repro and fix staged metadata_copy_control_ownership.patch; no native
replacement installed, no alternate compiler/package, live main untouched.
New source fingerprint invalidates previous snapshot storage DLL until rebuilt.
Final169-test gate passes /41.74 s with DLL connected, same4 exclusions; seven
audit counts remain0,1,0,1,5,0,0. These gates do NOT certify a native copier.

## Oct10 dependency cone and original-failure audit

Reusable exact-identity cone staged in isolated snapshot; two descriptor-cone
consumers reuse one authoritative parent-record index, with original sweep/FIFO
ordering. Reverse64 chain2016 ->126 reads;32/64/128/256 ->62/126/254/510.
Final gate145 pass /44.72 s, same4 exclusions. Cold DT LLVM matches original;
one timing pair is not a speedup proof. Global165 folds remain unchanged.
dependency_cone.patch and prior three patches pass combined apply-check;
none promoted under live full-main. Optional wave publication batching removed.
Snapshot native storage rebuilt/load-verified under726431cbf86656cc6973da89e78cf980
via OUR compiler. Native DT actual0.9/.25 exact; seven audits retain0,1,0,1,5,0,0.

Four exclusions diagnosed on live: scalar-comparison expectation predates
explicit Sept30 deferral removal; other3 handbuilt graphs lack ingestion scopes.
With only the existing scoped-fixture declarations supplied in diagnostic memory,
all3 original boundary assertions pass. Compiler behavior/guards/tests unchanged
for the audit. Next test-maintenance pass:proper scoped fixtures and current
scalar numerical-work expectation; remove exclusions only after genuine pass.
Do not restore deferral or weaken identity refusal to satisfy stale tests.

## Oct10 follow-on: compilation-owned ABI receipt, not repeated schema copies

Snapshot public lowerer now attaches the same declarative program_abi receipt
to catalogue functions before/after reduction; proper regions share it through
existing metadata ownership. Mutable parameter ABI views and whole-callsite
specialization isolation remain. Schema-copy multiplication eliminated at these
boundaries, not across all compiler metadata. Fail-first2 cases pass;97-test
gate green with same4 known exclusions. Tiny DT metadata visits575688 ->516168;
60 regions'59460 ABI visits ->0. Cold LLVM hash unchanged, actual native outputs
0.9/.25 exact. Wall samples remain variable: no time or four-law gain proved.

shared_program_abi_receipt.patch and updated serial_graph_scaling.patch, plus
native_fallback_link_boundary.patch, pass a combined apply-check. All still
staged; live full-main remains untouched. Snapshot native storage unit rebuilt
and load-verified under718d64f3130504f1fc1db1746f5016ba. Future live promotion
requires its own ordinary fingerprint rebuild. No packages/foreign compilers.
Next measured candidates:165 repeated structural folds/13428 descriptor reads
in tiny DT, remaining receipt copies, region spill/reload churn. Tiny profiler
also exposed variable I/O delay; distinguish it from algorithmic work counts.
No new long benchmark this turn. Last N4 remains554.95 s, not a win.

## Oct10 raw-speed continuation — N4 completed, slower

Two remaining quadratic receipt callers fixed using their existing phase scopes:
settlement publishes once at its original first read; immutable dispatch index
does likewise. Queries stay fresh and shape-book writes stay enabled. Fail-first
97/49 checks become1/1 on48-add repro.95 targeted tests pass; same four known
dispatcher failures excluded. At128/256/512 operations publication calls stay45;
summed graph sizes14731/29323/58507 (approximately linear). LLVM hashes unchanged.
Whole compiler O(N), fold restart improvement and minutes saved NOT established.

Snapshot native storage asset rebuilt with existing fingerprint gate. Authorized
N4 ran there with original external examples,1 graph worker/8 resident
regions, no trace/deadline; root build/profile_memory_n4_native_storage_mo1powht.
Full-main remains concurrent and untouched. Latest patch apply-check passes;
do not promote under live source pursuit. Completed554.948946 s, vs486.379790 s:
68.569156 s /14.10% SLOWER, not a four-law speed win.17 native calls/143043 facts
cost0.646459 s including transfers. Combined source->SSA355.558314 s vs296.812716;
combined whole389.868437 s vs326.696119.996 functions, stage RSS6460 MiB, no
memory gain. C hash differs (value numbering/storage/call ordering); orbital
execution/parity not established. Snapshot includes both staged link/scaling
changes and full-main remains concurrent: attribution needs tiny focused repros.
Next raw-speed target is combined planning/instantiation/SSA, not this already
sub-second native unit. Fold global restart work remains an unmeasured candidate.

## Raw serial speed: graph-operator asymptotic cuts — staged

Current-source snapshot native-fallback-link/Powershell/turing now has:
exact-parent dirty-round ABI identity publication; member-key token projection;
one ABI publication during read-only plan freeze; mutation-invalidated publication
batches during structural fold. No descriptor answer cache survives a query,
no provenance disabled. Reverse identity chain O(N^2) ->O(N); region-token
projection O(regions*function-size) ->O(total region members). These are scoped
operator bounds, NOT a claim that whole lowering is linear/parallel/memory-tiled.

94 targeted tests pass; known unrelated dispatcher/hierarchy/formal-history
failures reproduced against untouched live sources. Real512-add public lowering
8.141574 ->6.570869 s (single overlapping runs);12019/23923/47736 receipt calls
at128/256/512 become429/813/1582. All three emitted LLVM texts match exactly.
DT native0.9/.25 still exact; cold DT timing effectively unchanged. Detailed
metrics and repeated-DT emission nondeterminism caveat are in TEST_BASELINE.
Patch turing/tools/compiler_probes/serial_graph_scaling.patch passes apply-check.
Live compiler remains UNCHANGED while no-deadline full-main source pursuit runs.
Do not claim four-law minutes saved yet. Native-fallback-link patch also staged.

Next raw-speed priorities: remaining unbatched publication checks (new arithmetic
probe still has quadratic summed receipt work); mutable-fold global restart
scans; repeated source-class AST copies/inference in closure. Use existing exact
parent/control/book identities; preserve current serial route. Promote staged
patches after full-main exits, regenerate fingerprinted native storage asset,
then benchmark four laws without trace/deadline. No fresh architectural machinery
or additional compiler/package is required for these operator changes.

## User-authorized N4 native-analysis timing — completed; RAW SPEED priority

tools/compiler_probes/probe_four_rule_timing.py --graph-workers 1, no trace or
deadline, serial source route, resident regions 8, fresh output root
build/profile_memory_n4_native_storage_xl7u1you. Full-main lowering runs alongside;
native-link patch remains staged. Matching storage-effects DLL loaded without
compilation; first actual native call594 facts /1.4493 ms with transfers.
Completed exit0,486.379790 s (8m06.38), four stale pieces rebuilt,996 functions.
Latest599.448 s ->486.379790 s (113.068210 s/18.8620% lower wall), but older
464.540 s remains21.839790 s faster. Old599.448 s used4 graph workers.
17 actual native-analysis calls/143043 facts cost0.510050 s with transfers.
Combined begin->SSA296.812716 s; combined whole326.696119 s. Max printed stage
RSS6437 MiB vs prior5606; no memory gain. Zero book spills. C SHA
561cc7e1061aa4f454206e5d0db3308cd910d346ef5c7ae37abf9ae59aae910b.
Compilation did not execute the orbital system or establish numerical parity.

User: "we can't accomplish our architectural goals, we're just looking for raw speedup now"
Next priority is measured whole serial planning/refinement/SSA hot paths, with
small exact eager/native checks and reusable compiled units. Do not make new
parallelism or memory-tiling machinery a prerequisite. Capture wall-time gains
without falsely attributing changed worker settings/logging/concurrency to one DLL.

Initial 4-worker setup FAILED in authored native dependency_levels bootstrap:
carried-name-arm-missing for exact arm63/initial21 and34/initial11. Preserve this
regression and fix with a tiny real-kernel repro in the isolated compiler snapshot,
not by broad fallback/reverting unrelated changes or stopping full-main. N4 retry
uses explicitly selected original 1-worker graph lane, not a hidden recovery.
Benchmark output printing is reduced vs old per-node default; timings are not
an isolated or controlled accelerator-only comparison.

## Native dependency links with optional authored Python — staged, not live

Current compiler snapshot in C:/Users/alber/.codex/worktrees/native-fallback-link/
Powershell/turing implements the verified boundary fix; main job keeps its old
sources/assets. Live promotion patch is turing/tools/compiler_probes/
native_fallback_link_boundary.patch; git apply --check passes. Do not apply
until the no-deadline main self-lowering finishes or user explicitly stops it.
After promotion regenerate the storage-effects asset normally: source fingerprint
changes must invalidate old products, never weaken the existing acceptance gate.

Existing procedure proved: same authored Python -> OUR public lowerer + LLVM
piece -> existing authored-fallback installer -> canonical deployed piece identity
-> bodyless declared ABI + native link. Source authority remains available for
eager execution and explicit rebuilding of the selected root. No new graph,
native flag heuristic, ABI reconstruction, or per-node Python callback.

Fixed source-read refusal, external-slot signature recompilation and retained
mapping/sequence wrapper-vs-piece identity loss. Tests prove exact native entry
address and value/buffer IDs, no fallback source reads, only two linked functions,
one bodyless callee, native outputs [5,1,13.25], zero concordance findings.
14 dedicated cases pass; final combined existing/new gate 59 passed /46.25 s.
.55/.54 s direct/wrapped
caller lowerings are single-sample checks, not a whole-compiler speedup.

Still separate: installed compiler runtime adapters taking whole Python graphs
are NOT raw piece ABI callables. Their graph-to-column/native-product interface
must be exported through existing accepted SSA/API link records; a verified
runtime DLL alone does not license skipping its Python adapter. Next large
target remains the original structural refinement pass and its exact existing
graph/descriptor schemas, not another replacement graph engine.

## Main entry self-lowering attempt — latest observation

User explicitly requested no timeout. The repro now runs without a deadline
or repeating faulthandler timeout-labelled dumps; the full main was relaunched.
Do not stop it on an inferred time budget. Below are the earlier attempts.

User requested the ENTIRE main compile submitted to lower and its blocker
reported. probe_main_compiler_lowering.py passes unchanged45,000-line
fortran_c_shell.py with public lower_ast_source_to_ssa selected, actual module
bindings, existing extraction contract + full-native/no-callback overlay.
45/90-second diagnostic bounds expired in source dependency closure before
SSA; NOT a semantic refusal. Repeated25/50-second stacks show cached AST
deepcopy at graph_express2.py:788 in class-field/constructor resolution;
75-second sample traverses AST children. Real main impl and ProcessGraph
dependencies were discovered; no inner helper substituted. Compiler unchanged,
no native main product or full-main success claimed. First observed obstacle
is front-end scale; first unsupported language/ABI blocker remains UNKNOWN.
Repro: python -u tools/compiler_probes/probe_main_compiler_lowering.py

## Latest: fused effects AND pointer inference compiled and installed

ONE authored native invocation owns both complete analyses, preserving the two
different alias relations and exact original inference/conflict/accounting laws.
CSR/queue/edge scratch is reused; direct native zeros replace duplicate Python
zero-list scratch. 62 exact module cases / 61 calls; real serial compile reuses
the persisted component in 3 rounds / 2090 facts, binary unchanged .9/.25.
Gates: 4 native/audit pass /32.59 s; 17 branch/adapter pass /17.66 s.
Current default product: turing/build/compiler_foundations/storage_effects/
products/e2727d33f52be72e48858609d2d2a5ff/. Same fingerprinted LOAD ONLY route.

SOLO combined-section timing (packing/unpacking/publication included):
25 values .059400/.134200 ms Python/native; 648 values 10.843000/1.061175 ms;
5400 values 471.711250/8.369925 ms (56.358x). Largest original pointer inference
alone 14.593350 ms, read-only alone454.328100 ms. Combined-section speedup,
NOT incremental pointer-only or main-compile/N4 speedup. Cold load .070959 s.
Repro: turing/tools/compiler_probes/probe_storage_effect_timing.py.
Exact CFG dominance + existing name_binding facts fix trapped arm/shared-source
miscompilations, including late conditional continuation. Preservation receipt
is audited through declared copy-on-read scopes. Unsourced latch remains OPEN.

NEXT main native target is the substantial structural/dataflow refinement
driver, not another scalar helper. The earlier short-slice profile attributes
~1.9 s cumulative to folding and ~1.4 s to descriptor queries (overlapping,
historical, NOT a fresh N4 profile). Main refinement/planning remains Python.
Main parallel compile and whole-source/final-SSA tiling remain unfinished.
Both serial/unit routes retained. N4 has NOT been launched in this checkpoint.

## Historical: whole-module effect analysis before fusion

`read_only_storage_effects` is a substantive complete native compiler program,
not a per-value decision: ~120 authored lines, 892 SSA rows/99 lowered functions
including regions. It constructs dependency edges and reverse CSR, seeds exact
definitions/writes, closes aliases AND recursive calls, owns the work queue and
writes the entire final effect table. No Python callback inside its convergence.
Python marshals existing SSA occurrences/accepted call facts and consumes the
result; no new IR, numerical physics or DT runtime. The normal physical-call
adaptation round loads this installed component and queries its whole table.
Fallback remains the existing eager analyzer if the installation is absent/stale.

Deployment uses existing ArtifactCheckpointStore, matching compiler/authored/
host-source/platform fingerprint and DLL digest. Assets are immutable content-
addressed products. Load never invokes source lowering/emission/toolchain. Default
installed directory: turing/build/compiler_foundations/storage_effects; explicit
build/load functions: compile_and_register_storage_effects_accelerator and
load_storage_effects_accelerator in native_compiler_accelerators.py. Fingerprinted
source changes require explicit regeneration; no hidden compile during a job.

Checks: 35 SSA modules/34 native calls (one empty), exact ALL value answers vs
old recursive analyzer; pure recursive SCCs and recursive writes, alias-result
definitions vs base writes, Store roles/index and imported contracts included.
Fresh process loads the saved component without compiling it. A real serial
public compile makes 3 whole-module native calls producing 2090 value facts;
resulting binary returns exact .9/.25. The DT source is only an integration
fixture, NOT the compiler program compiled natively. Gate 2 pass/26.41 s.

Load-only SECTION timing, no DT program/no source compilation, five medians,
packing/unpacking included, original Python's actual formal-query usage:
21 values/4 queries: .03475 ms Python, .09485 native (tiny loses).
584 values/64 queries: 8.78095 ms / .95705 ms (9.175x).
5016 values/384 queries, 4656 instructions: 434.70860 / 7.904625 ms (54.994x).
Native computes all value results in ONE call. Asset lookup/DLL loading is
separate, .068812 s once per process. Synthetic SSA call chains, not a whole
compiler/N4 improvement. Repro: turing/tools/compiler_probes/probe_storage_effect_timing.py.

The larger program exposed a trapped conditional-arm version leak. The child
merge rebound loop arm 215/216, and parent false edges reused it. Existing
branch snapshots now retain already-bound arm identities as well as source
initials; unchanged program compiles. Baseline refused the whole input earlier
at overlapping control regions, so the defect's baseline age is UNKNOWN.

NEXT: more branch/loop regression coverage and normal-program timing before
claiming end-to-end savings; improve native scratch allocation without duplicate
Python zero lists; preserve build-once/load-only and both serial/unit routes.
Full ingress/final-SSA memory tiling and main lowering parallelism remain the
previous, unfinished architectural follow-ups. Earlier small shape and dead-
metadata checkpoints below are historical; do not conflate their boundaries.

## Latest native-unit checkpoint — 2026-10-09

The descriptor's existing binary broadcast law is now an authored Python
unit in `turing/src/compiler/graph_refinement_kernels.py`. Eager and native
execute the SAME `broadcast_shape_extents` body; `broadcast_shapes` is its
independent-row batch entry. The existing native registry compiles it through
`lower_ast_source_to_ssa` and OUR LLVM lane, with no foreign source compiler
or packages. `compile_and_register_broadcast_shapes_accelerator` is explicit;
imports do not compile/install it. The ordinary serial descriptor consults
the provider only AFTER its existing shape/conflict/unknown guards. Dtype,
precision and provenance publication remain in their original owners.

Exact cases: 407 broadcast rows in two overlapping native invocations,
including rank zero, zero extents, incompatibility and 2**60 extents. Each
invocation owns all feeds/scratch; no global validation state. Inputs outside
the exact int64 ABI stay with the same eager law BEFORE native execution;
2**63, 2**64-1, 2**100 and non-integer cases have explicit parity gates.
No native fault is caught/retried. Unknown tensor calls do not enter the
kernel; the original shape concordance receives the proven result.

Real SERIAL DT proposal: Python/native/Python all return exactly 0.9 and 0.25,
zero structural findings, 53 admitted broadcast calls in each run. Their
Python decision time was 0.170-0.175 ms TOTAL, versus 4.326 ms native. Public
lowering: 3.765 / 3.812 / 3.907 s; complete emit/link/two-child checks:
11.131 / 10.970 / 11.104 s. These differences are noise, NOT a compiler gain.
The broad descriptor profile was not a measurement of just extent arithmetic.

After removing redundant Python per-extent conversions, final SOLO end-to-end
batch measurements (packing/ABI/unpacking included): 4096 rows rank 8,
7.504 ms eager / 6.753 ms native; rank 32, 19.926 / 17.303 ms. One rank-2 row:
1.435 us eager / 36.367 us native; 256 rank-2 rows: .271 / .305 ms. Other runs
varied, including losses under concurrent compilation. This provider is
available, not an automatic performance policy or established main-compile
speedup. Preserve the eager/default serial route; do NOT launch N4 on this.

Final gate: 16 passed, 1 deselected in 12.58 s. The deselected projection-
history test also fails on untouched C:/dev/tg-base (0.60 s); no workaround.
Both serial/unit route selectors and the earlier native topology build pass.
Concordance's unsourced latch remains OPEN; zero findings is not full closure.

Repro, 55-second parent bound, inline output:
`python -u C:/dev/Powershell/turing/tools/compiler_probes/probe_native_shape_transfer.py C:/dev/Powershell/turing/build/native-shape-transfer-recheck`
Add `--benchmark-only` for the six/seven-second kernel/ABI checks. Receipts:
`turing/build/native-shape-transfer-checkpoint`, final solo kernel in
`turing/build/native-shape-transfer-solo`.

NEXT: select a larger existing structural/dataflow island, not more individual
extent comparisons. The fresh bounded serial CPU observation is 20,292,830
calls / 7.850 s profiled. Structural fold: 165 calls, 1.901 s cumulative but
only .083 s self; descriptor queries: 18,627 / 1.447 s, .309 s self. Shape
publication .314 s, ABI graph identity publication .199 s and general book
post .903 s overlap those costs. Do not attribute them to native arithmetic.
Inspect the fold's actual callees and exact graph/book operands before the
next conversion. Keep the existing finite operator vocabulary, stable value
IDs and original graph/book publisher; reject an island that costs more to
marshal than it removes. Full tiling remains the preserved follow-up below.

## Current direction — 2026-10-09: preserve both routes; accelerate serial

The user has redirected work from further tiling expansion to identifying
parts of the reliable serial compiler that OUR compiler can compile natively.
Keep the investment below; do not revert it or replace the serial lane.

Both routes are now explicit in `turing/examples/llvm_dt_system.py`:

- `lowered_system(..., source_compilation="serial")`: ordinary public
  whole-program source lowering, without source-unit workers.
- `lowered_system(..., source_compilation="units")`: existing source-unit
  products, native dependency headers and bounded worker admission. Its
  previously enabled default is preserved, as explicitly requested earlier.
- At `lower_ast_source_to_ssa`, omitting `partitioned_directory` is serial;
  supplying it selects units. Inline/shader behavior is unchanged.

### Preserve this checkpoint

The real unchanged DT proposal slice now compiles as four source units
(84/24/135/10 source nodes), peak two workers. Native outputs are exactly
0.9 and 0.25, also reproduced on the untouched serial compiler. Latest
unit/native probe: 25.943 s; clean serial/native control: 10.801 s. The tiny
unit route is SLOWER, not an established performance improvement. Archive:
`turing/build/dt-proposal-native-owned-q7cwmwj3`.

Existing planner specialization runs before source cuts, preserving dx,
distribution=None and dt_current. Native argument joins consult surviving
parameter identities, not old argument positions. Literal-formal and three
record-handle prunes no longer rewrite an accepted imported ABI: absence of
the dependency's body is not a liveness proof. Each writer was trapped.

Control lowering's canonical expanded/planning instruction snapshots now
share a BookSpill-backed cache (default 8), with policy and audited receipts.
The existing graph region-size partitioner reads the work contract (default
256). A two-chain conditional native test uses graph limit 8/cache limit 1:
two main workers, four regions per chain, exact 64701152 and -4; 19.48 s.
Variadic-min, carried-alias and bare-carried-while regressions also pass under
8/1. This bounds SNAPSHOT residency, not final SSA bodies, source ingress or
RSS. Oversized source SCCs still refuse. No complete conversion or N4 gain.

### Native-unit roadmap — use measured serial costs

A bounded CPU-only profile of the actual serial DT proposal (no malloc trace,
no orbital job) completed twice. Quiet run: 20,293,257 calls, 7.694 s profiled;
instrumented timings are NOT normal compile benchmarks and overlap below.
Repro observer: `turing/build/profile_serial_dt_slice.py` (35 s outer bound).

| Priority | Existing owner and observed cost | Native candidate / boundary |
| --- | --- | --- |
| 1 | `glsl_deployment_strategy._tensor_descriptor`: 18,627 calls, 1.378 s cumulative; `_tensor_descriptor_rule`: 0.497 s | Batch the existing integer shape/rank transfer laws (broadcast/contraction/reduction/view geometry) over exact admitted operands. Keep source/callsite conflict guards, dtype/precision authority and book publication in their current owners. These whole Python functions are NOT directly native-ready. |
| 2 | `_fold_callsite_structural_values`: 165 calls, 1.852 s cumulative | Its finite structural decision/dataflow portion, not AST traversal, retained Python calls, runtime tensor evaluation or control deletion. Reuse the current operator vocabulary and exact dependency edges. |
| 3 | `tensor_ssa_lowering.propagate_repository_ssa_call_metadata`: 21 calls, 0.439 s; `settle_call_result_types`: 0.098 s | Existing typed call/projection/alias propagation over explicit SSA occurrences and bindings; preserve TransformationLedger priorities and source cells. Profile a larger bounded SSA slice before selecting this over shape propagation. |

Do not start with another SCC/scheduling implementation just because it is
easy to compile. `_strong_components` took 0.000079 s, `_read_only_feed`
0.005353 s, dependency levels 0.017685 s in this slice. Existing authored
topological ordering and dependency-level native providers already live in
`native_compiler_accelerators.py` / `graph_refinement_kernels.py`.

Metadata isolation is also significant: 575,688 recursive calls / 0.879 s
cumulative inside `extract_clean_process_subgraph`. Generic Python-container
copying preserves AST/leaf/backreference identities and is NOT an immediate
integer-span native kernel. Likewise book.post (0.889 s cumulative), file I/O
and GC cannot be counted as native speedups from compiling a shape loop.

### Next implementation gate

1. Measure the exact first-priority numerical/decision portion, including
   packing, publication and repeated-query cost. The micro ranking is not N4.
2. Use the existing accelerator registry and public source compiler. Factor
   only that existing logic into authored Python with the compiler's existing
   typed ABI; run that SAME function eagerly and compile it through OUR LLVM
   lane. Reuse an accepted native provider when present; missing native code
   must retain the ordinary Python route. No package or alternate compiler.
3. Keep source identity, scopes, unknown-vs-scalar, optional None, lexical
   occurrences, dtype/precision, effects and conflict vetoes exact. Native
   proposals carry the existing source indices; the original pass publishes.
4. Tiny parity gates: scalar/unknown/zero extent, broadcasting incompatibility,
   dot vs matmul, reductions, view storage, callsite conflict and carried alias.
   Check concordance and exact native values, never a tolerance-based rescue.
5. Benchmark eager/native kernel and full SERIAL lowering separately, including
   ABI overhead. Register/use the provider only after real correctness gates.
   No long four-law compile until a measured serial speed improvement lands.

At the earlier selection checkpoint no new native candidate was implemented.
The latest implementation/measurement above supersedes that status. Further source
ingress/final-SSA paging and within-SCC lowering parallelism are retained
architectural follow-ups, not the current work item. Earlier entries below
are historical, superseded where their frontier/status differs.

LATEST: exact record-parameter reachability + existing SCC contraction now
preserve specialization components instead of emitting incompatible generic
headers. Native 16->59 silent miscompile is fixed; two independent scalar
branches still use two workers. A later caller imports the contracted header
and returns 59 too. Record-only variant symbol identity and a source-reference
writer shadowing fault are fixed at producers; imports join exact refs.
Source-node caps and semantic recursion claims are retained. See latest hazards
and report; this is NOT the requested full architecture conversion.

NEXT concrete frontier: real DT proposal source call 8, distribution formal 0,
loses a compile-time None binding across the generic native interface. Preserve
that existing callsite specialization at a legal graph frontier; do not invent
a None scalar representation or weaken the partial-product rejection. Full
source ingress and fixed tensor-basis residency remain unbounded. The automatic
DT source-unit route is still not established compatible with orbital/N4.
Do not launch a long benchmark or claim memory/parallel scaling yet.

CURRENT CHECKPOINT (supersedes next historical paragraphs): normal linked
`lowered_system` requests the public source-unit runner with >=2 jobs. Record
descriptor transport, shared flat/nested call projections and accepted prelinked
law reuse pass native micros. Actual small DT helper slice: four units, two
processes, exact 0.0, zero findings, 20.464 s including linking/execution.
Missing source Ref cells were a genuine incomplete boundary: exact acceptance
predecessor subgraphs now travel with cuts, preserving keys/stamps/mint/edge
order/unsourced receipts. Full diagnostic source book remains stowed separately.

PRIMARY NEXT: prove the complete ordinary DT/orbital entry under this route on
a bounded real slice before any long benchmark. Sequence/reference and nested
record descriptors, runtime extents/slots/text sink and oversized SCC regions
still refuse. Transport the existing descriptors/graphs; never load dependency
bodies as a hidden fallback. Restrict each worker's fixed tensor basis using
the existing dependency closure. Initial ingestion still holds the source
closure; no end-to-end bounded-RAM/multicore N4 claim is justified yet.
See the latest TEST_BASELINE_AND_HAZARDS checkpoint and experience report.

NEW CHECKPOINT: canonical public source entry accepts opt-in
`partitioned_directory` / `partitioned_options`. Separate-process SCC workers
now import bodyless native interfaces and compiled objects, not transitive
callee graphs/SSA. Scalar and static shaped-span caller/callee + three-level
native micros are real; C/LLVM and the existing standalone C host link objects.
Concordance checks interface ABI/effects; header book creation/loading is
isolated; source/unit histories stay archived. See the latest top section of
TEST_BASELINE_AND_HAZARDS.md. No orbital speed/memory improvement is claimed.

PRIMARY NEXT: exact record/sequence/reference descriptor headers, multi-result
projection planning and runtime extent transport. Then wire the actual orbital
caller through that lane. Ingestion remains whole-source; oversized SCCs still
need proven smaller cuts. Do not bypass a refusal by loading dependency bodies.
The following paragraphs are the preceding historical checkpoint.

Latest: real N4 regressed to 697.227 s with unchanged ~5.5 GiB printed RSS.
The requested architecture is NOT delivered. The existing source-unit PRODUCT
runner now uses small per-SCC/dependency archives, not whole-project inputs in
each worker; source planning is released before worker launch, and its concordance
is stowed intact outside worker heaps. Explicit source-node bounds refuse giant
closures before serialization/admission. Three real source units ran under a
two-process admission cap and passed exact native output/structural concordance;
73 focused tests pass. See TEST_BASELINE_AND_HAZARDS.md for exact scope/receipts.

PRIMARY NEXT STEP: freeze and publish type/shape/storage-alias/effect/precision
interfaces, then compile bodies against summaries instead of recursively
expanding transitive authored dependencies. Preserve the current compiler's
lowering and optimized binaries. Treat recursion as SCC-owned; oversized SCCs
need a proven smaller region cut. Bound body residency and final SSA/object
assembly as well as worker admission. Current unit-product mode is NOT wired
into the ordinary orbital single-module return; whole-source ingestion and
interface-only/streamed linking remain unfinished. Do not claim otherwise.

Continue the user-requested cohesive conversion in `C:/dev/Powershell/turing`.
The checkpoint is
`speaktome/AGENTS/experience_reports/1791554599_DOC_Turing_Self_Hosted_Graph_Mechanics.md`
and `turing/TEST_BASELINE_AND_HAZARDS.md`.

The next checkpoint is
`speaktome/AGENTS/experience_reports/1791561850_DOC_Turing_Bounded_Graph_Work.md`.
Dispatch topology now has a compile-wide bounded cache, pre-carve reservations,
BookSpill-backed eviction, exact AST/container identity and generation-checked
write leases. Worker proposals use dependency-only snapshots; lexical ordering
and ASAP/ALAP equations compile through our compiler. Admission includes pending
results, publication is deterministic on the coordinator book, and new receipts
have a concordance detector/lifecycle. Region projections are lazy. Do not redo
that completed integration or claim the entire compiler is converted.

Source graphs, metadata/leaves, catalogue and module SSA remain resident; most
refinement remains serial. Extend the SAME existing ownership/provenance systems
to these units, then move expensive coordinator preparation/refinement into
proven proposals. Fixed-residency 4,096-node micro units show little improvement
from 4 to 8 worker capacity: coordinator packing/validation/spilling dominates.
Use `probe_graph_work.py` and its phase/counter receipts to target that cost.
Do not share mutable books across writer threads, add an unbounded job queue,
invent identity keys, or use another source compiler. Canonical entry:
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. No long orbital compile
without the user launching it. This probe's call counts are NOT orbital N.

Keep a separate failing micro for returning a mutable arena after conditional
control: the first levels-kernel return exposed a dominance finding. The final
worker kernel correctly publishes through its declared output span and returns
the processed count; that interface does not repair the general compiler bug.

Preserve all current uncommitted work and user dirty artifacts. Clean comparison
checkout `C:/dev/tg-base` remains at `54c67fbd`. Native micros must remain
crash-isolated and time-bounded. Eight combined loop-composer failures and the
separate `call-argument-carried` native timeout were reproduced on clean HEAD.
The current LLVM/layout/dispatch group has six unchanged clean-baseline failures,
including the two legacy reshape/tobytes AOT expectations; see the receipt.
