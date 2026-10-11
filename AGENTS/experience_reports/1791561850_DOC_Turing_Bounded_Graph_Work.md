# Bounded graph work integrated with the real compiler

2026-10-11 FRESH N4 SOURCE COMPILE VERIFIED:
User requested: "Nothing is useful to us if you fix it in ir, that is never okay, so please verify the source compile works"
Ran canonical source38c2da32 through unchanged public source compiler:
python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --save-ssa
TraceFalse, serial source lane, O3; no timeout, job termination, saved-IR
input, diagnostic transplantation, or post-lowering repair. All four stale
law pieces rebuilt against these compiler sources. Process34060 exited0.
PROBE_ELAPSED458.310s (7m38.3s including imports/profiling/stale leaf builds).
Combined source stage times total260.172s (4m20.2s); compile boundaries
178.7->443.5s (264.8s). Peak progress-sampled6143MiB; observed OS peak6.67GiB.
Source-generated output:
C:/dev/Powershell/turing/build/profile_rules_4_llvm_d96c6ueg
repository-ssa.pkl190157673 bytes; emitted LLVM4152280 bytes,
native step_3__dt_system_over.dll282624 bytes (plus import lib/debug symbols).
LLVM sha2560dcc132c36881f6e9d53bb927c82d462abb8fc6c29efbb085afbf529e72b3e5e;
996 functions. The native build refused no shortfalls.
Read-only audit of this FRESH public-lowerer result:14725 rows/996 functions/
0 structural findings,11.719s;60383 unsourced facts/468 unposted identities
still OPEN. No compiler editing or IR changes were used for this audit.
This supersedes the earlier "fresh N4 remains unverified" receipt below.
Four SOURCE-TO-NATIVE BUILD now passes. Numerical dt execution/parity was
not run; eight/all35 have not been tested on this source state.
Source-and-SSA paging and ordinary-main worker parallelism remain unfinished.

2026-10-11 N4 COMPLETE ERROR CENSUS / SOURCE FIXES (patch35):
Authorized N4 run at aa18da80 still failed:496.725s including four stale
leaf rebuilds; source-to-SSA completed. Saved190,213,415-byte repository-ssa.pkl
at build/profile_rules_4_llvm_wfdofixz. Complete emission census SIX:
four scalar Casts wrongly promoted to rank1 at pointer-kernel call settlement,
and two child-copy append int64 Phis seeded with default int32 constants.
Prior replacement Phis are absent from the complete census.
Source fixes keep shared kernel pointer transport distinct from actual
semantic rank; child-copy indices are int64, status int and live flags bool.
Also fixed authored local bool-to-double pointer input conversion (False
previously read garbage in both C/LLVM, reproduced on untouched tg-base).
All native conversions still mint/post through existing concordance causes.

Saved-module audit first crashed on Unresolved bindings. It now reports
unresolved captures without crashing; the binding-removal writer tombstones
owned capture claims with derived provenance. N4's five structural findings
were followed up: two cross-frame result ids misrecorded as local aliases,
two duplicate sequence-descriptor findings caused by live private extent
reads not following a proven handle alias, and one retired binding capture.
Existing retirement operators now use exact return edges/descriptor roles;
no name-based suppression. Native LLVM tests prove live extent rebinding.
Adjacent keyed tensor tests no longer select hardcoded region0 and require
exact row-base/shape/rank feeds; two-axis Indexed ports publish their rank
demand on the existing lookup shape concordance (real lowering gap fixed).

In-memory DIAGNOSTIC replay on the unchanged saved module:
one descriptor and two aliases retired; structural audit5->0 on14727 rows/
996 functions. Replayed prevention of four bad rank posts and regenerated
two child-copy helpers with the corrected source builder: LLVM shortfalls0,
complete native build succeeded. Final replay23.485s, artifact directory
build/n4_saved_ssa_repair_8_ww2w6k. This is NOT a fresh source compile or a
full dt execution/physics test. Eight/all35 were NOT started. Next fresh
user N4 command: python -u tools/compiler_probes/profile_memory_n2.py 4 --backend llvm --save-ssa
Use canonical repo; no trace, timeout, or job termination. Do not claim4 passed
end-to-end until that source run completes. Main parallelism/paging remain open.
Saved-book migration remains60507 unsourced facts/468 unposted minted ids,
latch OPEN; these were not hidden or relabeled as structural success.

Focused gate172 passed84.43s, zero failures/skips; final native live-extent
regression additionally1 passed1.47s. Cffi deprecation unsuppressed.
Canonical and both copied compilers/tests synchronized; unrelated files untouched.
Matching storage-effects products rebuilt and load-only verified:
live ecd4a6b2c54ed8b047a70f0fa79dbd3a,
benchmark70daf59e20ba7b1b94d27fcabfbacfdb,
repairdd182144082e22902246e76c5075b096.
Each1690 rows/188 functions/0 structural findings/0 unposted ids,
1116 unsourced facts OPEN.

Prompt history: "All errors need to be fixed"

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


**Date:** 2026-10-09
**Title:** Coordinated dispatch residency, native workers and publication

## Overview

Latest user evidence: pasted N4 log at
C:/Users/alber/.codex/attachments/4684b2ab-2ada-4af4-b09e-9c9b2d562005/Pasted text.txt.
Its final error begins (verbatim):

> ValueError: artifact has shortfalls: step_3__step_with_dt_control_used__specialized_b1717615e68e__ssa_sequence_586_replace_65: Phi: phi 2305843010213713288 physical element i64 disagrees with incoming ((2305843010213713279, 'i32'),);

Terminal-display prompt (verbatim):

> can we get something like curses or whatever to give us better control over scrolling text made to seem stationary and negotiating between the progress bar and timing chart

Current prompt history (verbatim):

> can you merge/integrate/etc so we're all up to date everywhere for any agent

> can we put in profiler analysis summary into the compiler, maybe even wrap the profiling that exists in a pandas dataframe after it's accumulated? maybe a live table, like we get scrolling messages up top, then an always present table of timings, and then the icmq or whatever progress, which, it must be said, is NOT USEING METRICS OF PROGRESS

### Oct11 negative marks followed through real native execution

The complete affected regression gate passes595 tests/325.72s without
exclusions; post-installation benchmark-copy gate352/127.52s. No failing
assertion was removed merely to make the gate green. Old synthetic graph/book,
mutation-occurrence and deprecated API expectations were modernized. Genuine
compiler faults were repaired at observed producers/readers: implicit AugAssign
loads, unary-plus return identity, loop-field effect causality, static aggregate
proof, generator materializer routing, recursive catalogue availability and
lexical class identity. Native generator execution additionally exposed wrong
unrolled constructor arguments and deduplicated ordered binding histories.
Four exact variants now execute on C/LLVM/Fortran,12 checks/repeated invocations.

All seven concordance audits now have0 structural findings and0 unposted minted
identities; unsourced fact latch OPEN remains explicit. Existing audit root
contracts were dropping declared helper values and using an old4-argument
proposal API. Raw address/cast/late-projection mints now use the existing
concordance transform posts. No interpreter oracle, tolerance or new runtime.

Verified patch28 negative_contract_followup.patch installed into benchmark copy
after27; patch29 rule_probe_cli.patch preserves the installed N/all LLVM CLI and
optional pre-emission SSA capture. Compiler source in live turing untouched.
Matching native storage-effects asset refreshed/load-only verified under
f8aec357b935ef27baf0ffc3ac7b1840. No agent N4/main launch; no four-law success or
main parallelism/memory-tiling claim. Full receipts and remaining provenance
counts are at the opening of turing/TEST_BASELINE_AND_HAZARDS.md.

Merge readiness: tested repair set ready to integrate. The benchmark turing
directory is a copied child of the parent coordination worktree, NOT a turing
Git branch; merging that parent would omit the ignored compiler directory.
Live turing is its own dirty main repository. Preserve its edits and integrate
the ordered patches there; no merge/commit/stash/reset was performed.

Prompt history:

> All negative marks need following up on. Eliminate tests that are no longer concordant with the api, otherwise fix the failing

> I think chances are we're ready to merge, we're passing tests and the things we fixed had to be fixed, no?

### Oct11 deterministic Pythonic torture exposes and closes ordinary failures

Seven source cases fail on untouched tg-base in24.38s. Repaired C AND LLVM
execute every case,60 repeated native invocations/14tests/58.96s. Exact expected
cardinality/effects, no interpreter oracle or fallback. Independent guards,
empty/1/2/17 rows, enumerate, tuple splice, nested while/if/for, break, list
literal birth and element contents. Added4 bound tests including million-trip
nested-loop multiplication WITHOUT expansion and refusal of unknown backedges
and self-mutated extents. Tests are not suppressed or marked expected failure.

Observed writers: redundant containment edges newly consumed by patch22;
ordinary nonempty List admitted as zero-length storage; loop capacity omitted;
enumerate wrongly escaping; column-zero row load absent after numeric plan
keeps only its capture; LLVM private capacity uninitialized. Fixed through
existing graph transitive reduction, lexical sequence effects, control interval
transfer, builtin identity, row-load lowering and allocation requirements.
No new object registry/opcode/runtime or alternate Python compiler. Empty-list
extra clear initially broke old sequence-truth test and was removed, not hidden.
Before/after ordinary audits retain0 view/1 controller finding and OPEN latch.

Final repair gate127pass160.26s. Verified changes then applied to original
benchmark worktree, with post-application torture/bound tests running. Preserved
live source and original25 patches; patch26 pythonic_control_torture.patch
includes core changes plus4 test files and is reverse-checkable. No main/N4
launched; previous diagnostic finished FAILED286.275331s with cached laws and
saved190086864-byte SSA. Native/old main terminated at prior user request.
Refresh matching compiler accelerator assets before next four-law run; main
tiling/parallelism and whole-program parity remain unfinished.

Prompt history:

> Okay so we badly damaged the compiler with our recent changes. Find out what happened

> So are we getting more of the program we might've ignored before?

> Troubling but also positive, we found hidden errors. Do we have a pythonic silliness torture test for the compiler? Something like that should've cought this a long time ago

> Try to make a pythonic test that breaks our compiler and then iterate against it

### Oct10 user-authorized iterative repair and process cleanup

Exact process inventory preserved newest diagnostic; killed main33064/7196 and
native N4 23236/14620 at explicit user request. Native outcome USER-TERMINATED,
not compiler refusal/success; counters unavailable. No files removed. Iteration
copy is build/n4-repair/turing inside prior snapshot, with identical sources
and copied matching storage unit. Prior25 patches preserved and unapplied.
Diagnostic graph checkpoint attempts failed on local callback and write-only
stdio serialization; fixed observer to print exact source identities and never
abort compiler for final checkpoint failure. Session78998 active; no trace or
deadline. Four prebuilt law assets reused, no implementation changes yet.

> Find out what the problem is, keep iterating on failures of any kind, trying to get 4 laws to swirl flawlessly, if you reach that point, start iterating on wall the laws, thirty something for this set i think?

> Kill all python except the active newest run

### Oct10 N4 comparator exits; native copier run remains active

Original-copier session9750 exit1 /561.4117003s. All4 laws rebuilt in175.6213412s;
combined385.7903545s; native storage17calls/143271facts/.6692912s boundary included.
C emission refuses %t598 dynamic extent without public storage origin (twice),
then two unavailable loop operands whose EXACT producer Calls(%t598) remain in
if_true.15 / if_merge.15. No evidence of missing SSA producers; missing extent
emission plausibly explains unbound results, but writer chain not observed.
No complete artifact/C hash/peak report, no compiler changes or rerun.

Read-only current four-law LLVM comparison: function counts6/13/71/38 match both
routes and opcode totals agree for every law. Properties hash identical; other3
differ, inverse sampled diff includes SSA numbering/placement. Neither matching
counts nor differences prove semantic parity/content loss. Historical507.330979s
successful N4 cannot be directly ranked against today's failed561.4117003s.
Original copier unchanged, all generic native-enabling fixes retained. Today's
newly reported extent gap may be exposed old behavior or introduced regression;
historical concordance replay remains necessary to decide. Native laws milestone
1229.3s vs175.6s original is observed, not isolated timing/kernel attribution.
Native14620/session52547 still active, CPU rising, final counters pending.
Old main left untouched. No additional compile launched.

### Oct10 user-requested N4 launch receipt

Second N4 original-copier comparator explicitly requested, launched22:28:25
America/Chicago. Parent11844/worker32076/session9750. SAME compiler fixes and
native storage component; --graph-workers1 --fresh-pieces, no native copier.
Independent empty law_cache under profile_memory_n4_native_storage_tixr18_u
rebuilds all4 and prevents concurrent cache interference. Native run remains
parent23236/worker14620/session52547; main also untouched. Initial combined
graphs BOTH18829nodes/21599edges. Original reaches combined SSA+412.3s; native
still specialization+1554.1s. Native result remains PENDING; original outcome
and limited retention interpretation recorded above.

Additional prompt history:

> Don't necessarily worry about the outcome yet just observe

> Afterwards i want you to reason about whether we previously might've dropped content that now is caught and consider using the non native copier, the original, but now with the benefit of the changes we had to make for compiling the copier

> Whatever old main compile is presumed to be still active from forever ago, that wouldn't even be the same compiler at this point and I'm sure it is frozen whatever the case, we can leave this attempt with the native copier and we have enough threads and ram to run the other instance with the optional copier but with the compiler improvements without them being in competition with each other

User: "so can we do a 4 rule test". Snapshot existing probe launched22:09:37
America/Chicago: --graph-workers1 --native-copier. Parent23236/worker14620,
execution session52547. No trace/deadline/target execution, existing main alone.
Storage unit refreshed through OUR compiler before timing (d5d7dc1bdd73dace2db368d4e04801f6);
copier loaded0ac1ded2afe6cb7887bf3b2eb1fb44b2 without recompiling. One graph
worker/eight resident regions/preserved serial source lane, prior N4 config.
Fresh build/profile_memory_n4_native_storage_l7gvdp8m. New probe option counts
native batches vs pre-execution subclass fallback, boundary-inclusive seconds.

First law rebuilt,1847root first batch1.6493831s native/no fallback. Inverse
source-catalogue7406nodes/8894edges extraction took308.985s, finishing+353.2s;
inverse1494node graph at6/8published regions+431.1s. Final N4 NOT complete;
do not infer final memory, total time or native-specific attribution yet.
Current exact run receipt opens live turing TEST_BASELINE_AND_HAZARDS.md.


### Oct10 whole ORIGINAL metadata copier executes; speed is not won

Patch25 native_whole_metadata_copy.patch UNAPPLIED after24; concatenated25
apply-check0. Live source and existing main33064/7196 untouched. No main/N4,
timeout, kill, package or alternative compiler. Exact50 production authored
lines now lower/emit AND execute natively through our sanctioned source entry.
All container branches, shared references, leaves, mutable cycle and tuple/list
cycle pass; one outer invocation, all recursive/memo/container work native.

The missing self-backedge ABI settlement, early return placement, sibling
spanless AST nesting, node0 memo.get, stale numerical effect markers, initial
memo guard ordering and recovered Cast36/contains37 placement were observed at
their actual writers. Initial memo guard was present but AFTER copy branches:
shared handle3 copied8/11. SOURCE_SPAN identity paths carry body ordinals2-7;
conditional sibling runs now consume these existing book rows, not coordinates.
New scalar identity-all operation consumes exact existing zip/Is/generator
edges and accepted native_object_source cells, not a Boolean arena or general
generator runtime. List extend/set update/frozen construction use incumbent
context helpers. A4097 bulk-copy regression introduced in this batch was caught
and corrected with a separate accepted frozen helper; guard/test unchanged.

Existing accelerator registry/checkpoint deployment now builds the original
copier once and loads matching native code without compile calls. Actual
extract_clean_process_subgraph uses one batch across graph and node metadata
when the provider is explicitly installed/loaded. Serial original route stays.
Subclass payload admission chooses serial BEFORE native execution; native
failures never retry silently. Installation test traps compilation on reload,
Python fallback on an admitted cyclic graph, and altered DLL digest. Shared
memo is bounded by imported identity count; owner/release is existing contract.

67test gate green before deployment;78green with the same4 old dispatcher
exclusions before final source-path change. Final connected15green18.73s; seven
ordinary audits unchanged493/24/0,360/36/1,488/22/0,522/27/1,525/27/5,24/2/0,
331/22/0 after that change. Whole copier structural audit0, unsourced latch OPEN.
Context release helper still emits without attached book; do not claim closed
provenance. Complete final gate/timing receipt is in live turing hazards file.
FINAL after all source changes:132pass/4 SAME previously known exclusions in
125.67s. Fresh-process LOAD ONLY with native provider active during whole-copier
compilation/execution and original extraction/compartment cases:12pass6.63s.
Guestbook dry-run0, no --apply; ordered concatenated25 apply-check0.

No lost working capability was established, and no rollback occurred. There
ARE remaining performance problems: solo64rows .111400ms Python/2.740800ms
native INCLUDING boundaries,24.60x slower. Cold profile includes6ms exporter
work in12ms, not a warm attribution. Linear native member/memo scans and
rehydration of all temporary births still need work. This is a real native
copier/deployment milestone, not main speedup, parallelism or whole-memory tiling.

Next steps are in todo/turing_graph_residency_workers.stub.md. Retain existing
graph/contract/concordance authority; profile warm boundary versus kernel before
choosing the next span. Do not expand this into replicating Python semantics.

Prompt history:

> Did I make us lose something we didn't need to lose by my impatience while this was accomplished? carry on and have a commitment to finishing the workd, to using the resources necessary to broadly solve this problem


### Oct10 paired generator/tuple/update batch uses the live concordance

Patch24 native_paired_object_concordance.patch UNAPPLIED after23;24ordered
apply-check0 on live dirty checkout. Live source unchanged; no main/N4 launch,
timeout, kill, package, alternate compiler or replacement production copier.

Observed tiny graph retains tuple3 leaves1/0, items6 receiver2, loop8 targets1/0
and result11. Collector declined paired targets/view; SSA invented unnamed row
column inputs. Composer had flattened3 to1/0 before native admission and lost
the original publication identity. Span-based mutation placer hoisted tuple3's
constructor because comprehension8 has no span. Watched both writers. Later
linked-call case also ordered publication12 before producer calls5/6. The fix
preserves that transition on the existing book and consumes exact causal edges.

native_object_source is representation acceptance DERIVED from current source,
component, receiver and loop cells. Ref travels through Control; consumers read
latest book facts, never a guessed pair of IDs or a parallel schema registry.
Emitter reads the attached book and concordant planning aliases; actual tuple
fields, positional casts, projections, collection/update operands are verified.
Missing/superseded cells and corrupted copied projections refuse. New page has
an explicit LOWERING->EMISSION lifecycle and uses incumbent spill machinery.

Native paired items iteration, tuple births, nested tuple fields and finite
generator dictionary update reuse the object context and collect/get/set/heap
helpers. One native UPDATE call consumes the complete pair collection; no host
item callbacks. Stable handles, not payload pointers, survive reserve relocation.
Keys remain admitted reference-identity keys, not general Python equality/hash
or custom dynamic methods. Existing recursive storage guards remain intact.
Stable topological unit placement uses the incumbent dependency-cone helper;
source coordinates are only the stable preference among independent units.

New7+lifecycle1:8pass10.85s; earlier focused25pass31.68s and broad74pass73.50s;
Final broad75pass77.19s after all changes; guestbook dry-run0, no --apply.
Native0/1/3/64entries checks normal,
reversed and nested pairs; shared identities/backedges; linked identity calls;
dictionary ordering/fresh output; one outer invocation and explicit release.
Pre24 nine-module public-source replay refuses unnamed columns,0.913166s.
Before/after seven ordinary audit counts unchanged; existing findings/latch OPEN.
Syntax10files0, added whitespace0, ordered24apply-check0.

Production50-line copier clears97, still refuses98boolcollection,1.808068s.
Actual graph26=new is old,70=zip(isolated,value),80=generator,79=owning loop;
98 should participate in the existing all/zip intrinsic reduction, not require
private bool scratch. Simple recursive dictionary-copy variants remain refused
on unowned formal6/8 with unknown dtype; writer UNKNOWN. NEXT observe that
concordance chain and complete zip/all; no native recursive pair-copy or full
copier execution/speedup claim. Native table find/store are still linear scans,
not an O reduction; no main tiling/parallelism claim.

### Oct10 connected native storage, type evidence and lexical ownership batch

Patch23 native_object_contract_batch.patch UNAPPLIED after22;23ordered combined
apply-check0 on live dirty checkout. Live compiler unchanged. No main/N4 launch,
deadline, kill, package, alternate Python compiler or replacement copier.

Full copier capture40 already aliases memo6 in the existing concordance. Frame
binder now consumes settled exact bindings/aliases for the whole descriptor.
Keyed Input mutations use the existing lexical mapping effect path, rather than
hoisting a memo write before its branch-local object birth. Actual composed
control depth selects the deepest declared loop/conditional owner for child
calls. These reuse current graph evidence, not a parallel registry or name guess.

First native recursive fixture crashed. Observed Function.args slice writer
pruned a ProgramABI memo value column before call expansion; late restoration
stripped its owner, causing LLVM to allocate one stack element for the entire
arena. Early pruning now preserves caller-owned members. Native tests assert
all memo columns/extent/capacity/status/live-flags are public buffers. Allocation
guard stays intact; no opaque pointer cast or private scratch workaround.

Opt-in type evidence is a third int64 column on the SAME SSAObjectContext object
identity table. Default two-column contract unchanged. Explicit finite builtin
guards stamp at the outer boundary and native object birth. Source isinstance
uses incumbent intrinsic/static class identities and operand edges, lowering to
ordinary native helper/And/Ne SSA, no per-item callbacks. Storage kind and type
evidence remain distinct. Opaque tuple subclasses may answer a type predicate
but cannot supply member rows: existing call-graph closure derives payload
requirements; boundary, backend and concordance verify/refuse accordingly.
Existing shared heap ownership/release covers the additional column.

Final focused17pass20.66s, exact memo keys/exported [] values and one outer
invocation checked. Broad92pass75.09s preceded the final graph-receipt unit and
stronger recursive checks. Syntax10sources0; patch whitespace0/new test match;
23ordered check0; guestbook dry-run0, no --apply. Native depth0/1/7/32 succeeds
with both tuple type and length guards;
eight builtin predicates across13input kinds, native birth evidence, opaque
payload rejection, and four stale-contract negatives also pass. Source modules
have zero structural findings, not a closed provenance latch. Seven ordinary
audits unchanged493/24/0;360/36/1;488/22/0;522/27/1;525/27/5;24/2/0;331/22/0.
Pre23 public-source replay refuses the memo column at call5,0.885788s.

Production50-line copier still refuses call14/collection97,unknown scalar row,
2.311162s. Trapped graph identifies paired tuple leaves14/15 in loop75,
materializer76/iterable19, not memo6. Next batch is paired iteration/row ownership,
dict update, extend, set/frozenset construction, zip/all and optional memo.get.
Whole-source tuple-of-type guards also lose existing static class evidence
upstream; helper declines empty/missing evidence. No native full copier artifact,
main/N4 speedup, asymptotic lookup, full memory tiling or parallelism claim.

### Oct10 declared region-less loop ownership now executes natively

Patch22 regionless_loop_ownership.patch UNAPPLIED;22ordered apply-check0.
Observed composer input already declares catalogue2(If11)->0(Loop13), with
exact AST13 in If11.body. Overlay ignored zero-region loop children and emitted
them at root, separating recursive call3 from its target Load0. Overlay now
consumes that edge in the exact authored arm/statement slot, using the existing
graph and dispatch-node view. No coordinate/type guess; missing/cloned source
identity or a region crossing the slot refuses. Existing placement book reused.

New9 plus linked-reference12:21pass8.71s. Real native conditional generators
execute both arms and outcomes at0/1/17/257 members with one outer invocation,
correct alias identity, zero unselected-arm births and explicit native release.
Two modules have zero findings. Pre22 public-source test refuses unaccounted
target1,4.29s. Broad75pass70.22s; seven audits unchanged, provenance latch OPEN.
Tiny recursive tuple now stops at unresolved isinstance10, not target0 storage.
Full50-line copier still refuses at call14,4.570708s. Exception frame shows a
metadata_memo sequence6 column1 incorrectly reaching private storage leasing;
its linked accounting says span/rank1. Writer losing the keyed/shared binding
not yet observed. NEXT trace that binding; runtime type admission also pending.
Allocation guards intact; no whole-copier execution/speedup/parallelism claim.
Live source unchanged; no main/N4, timeout, kill, package or alternate compiler.

### Oct10 linked reference contract prevents false type folds

Patch21 linked_reference_kind_fact.patch UNAPPLIED;21ordered apply-check0.
Observed tiny recursive collection17 initially owns value3/control13. The
structural Input evaluator discarded linked reference ABI and returned
scalar/opaque_ref, so isinstance(item,tuple) folded False and branch pruning
deleted those producers. Watched parents write and evaluator returns establish
the chain. Input folding now consumes the SAME linked reference fact before
the tensor fallback. No new contract/schema/registry/opcode. Unknown object
kind remains runtime; physical scalar handle layout is not a Python type.
New12pass2.64s, including native forwarded id and preserved recursive graph
edges (NOT recursive execution). Broad58pass58.55s; old hierarchical concat
unpack failure reproduced pre21. Before/after seven audits unchanged, latch OPEN.

Patch20 source gate pass was not semantic proof: live container branches had
been incorrectly removed. Correctly retaining them reveals prior control and
inspection gaps. Exact50-line probe1.876627s refuses opaque recursive scratch
at callsite14; tiny tuple probe refuses target0. Observed composed Loop13/If11
are siblings, call3 in If11 versus Load0 in Loop13. A call-selection experiment
was removed because it did not repair composition. Native object kind matching
is closed-domain membership, not Python MRO/metaclass support. NEXT: restore
graph-owned conditional/loop nesting at composition and admit needed runtime
type predicates through existing contract. Never weaken recursive guards or
claim the whole copier runs. Live unchanged; no indexing/batching/speedup claim.

### Oct10 integer memo membership and captured storage fixed

Patch20 native_memo_membership.patch UNAPPLIED; twenty ordered apply-check0.
Native membership consumes the incumbent parameter table and stable integer
identity keys. Observed structural recovery incorrectly required record fields;
its scan mixed int32/int64 and bool/int32 Phis. Exact parameter arenas/uses now
survive late recovery, counters/constants have their true widths, and the
existing ABI fixed point carries keyed layout/ownership across captured inputs.
No replacement registry, object-reference key cast or source copier rewrite.
New receipts are checked by concordance, including corrupt binding tests.
Ten new cases; final47pass52.38s across membership/reference/collection/source
context. Broader gate61passes then existing tensor-region StopIteration;
exact pre20 source-loader replay SAME2.76s. Seven audits unchanged, latch OPEN.

Unchanged50-line copier now clears the source full-native gate. Final2.154057s
lowering+LLVM attempt still LLVM_COMPLETE=False: five loop i32/i64 Phi conflicts
and four extent Calls treating opaque iterables as tensors. Observed collection
receipt empty; need an exact captured-object iteration repro and the existing
graph-owned iteration/collection producer contracts, not backend pointer casts.
Whole copier native execution remains unproven. Contains/lookup/store are still
linear; shared find/index/growth/mutation-validity not delivered. No speedup,
hash, general dict equality, memory tiling or main parallelism claim. Live source
unchanged. Detailed command/results and promotion order in turing baseline.

### Oct10 graph-owned native generator collection seam fixed

User: "Is this the need to make a native generator for irreduceable generators"
User: "Let's proceed to fix it"

Snapshot patch19 native_object_collection.patch UNAPPLIED; all nineteen ordered
apply-check0, live compiler unchanged. Dense tuple/list generator materialization
now uses its actual LoopResult/materializer/iterable/target/member edges, native
context, runtime extent and direct dense member rows. Shared materializer emits
fresh COLLECT helper; normal native body calls publish results in ONE invocation,
not per-item ctypes/get/set/search. Row addresses reload existing pointer cells
after heap growth. No suspended generator runtime, second registry or new opcode.
Shared mutable contexts are not claimed parallel; filtered/nested collection
control is refused, not silently densely stamped. Preparation still scans global
member rows; this is not main tiling/parallelism or whole-compiler speedup.

Observed fixes: stop1/unknown collection; premature private call-result columns
before native ABI settlement; int64 Load aliasing a changed length cell and
computing extent0; stale callee source binding0 despite live Ret2, rewriting
caller publication2 to input0. Instr.args observer caught the actual linker
write. Existing bounded ABI settlement and live Ret identity are the authority;
concordance/emission validate real row/kind/member/induction producers. Genuine
identity calls use their incumbent exact structural identity receipt separately.

New15tests19.06s; final broad125pass98.65s. Tuple/list direct/identity/native fresh
allocation/native child-copy, N0/1/17/257 and identity4097, heap growth and stale
contracts. Isolated pre-edit LLVM int64 load returns0 instead of12; fixed test12.
Additional backend20pass/3oldskip then old tuple symbol KeyError; pre19 replay
AND untouched tg-base reproduce same. Existing i32/i64 Phi failure reproduced
with pre19 sources; clean base later fails its old IR-symbol assertion. Root
frame1pass. Seven ordinary audits unchanged; provenance OPEN. Full diagnostics,
commands and preserved limits are in live TEST_BASELINE_AND_HAZARDS.md.

Exact unchanged50-line copier clears collection93/private recursive formals;
next full-native guard refuses ONE undefined predicate37: observed source is
identity in metadata_memo, parents36(lhs)/6(rhs). Normal retry2.353s. Next is
memo membership/lookup with actual identity/key semantics through existing
table/query contracts, not reference-key casting or a substitute copier.
No main/N4/package/kill/live installation; full copier execution not delivered.

### Oct10 native bulk stamping and measured dispatch reduction

User: "Move forward and recognize the delays from dispatch stack up extremely quickly right on boundaries we're trying to use native pockets to fix the speed of, so in the ideal world this may involve making a special wrapper object to sit on the option side and use ctypes to have fast object stamping"

Snapshot patch18 native_object_materialization.patch UNAPPLIED; eighteen combined
apply-check0, live source unchanged. Our source compiler's tuple/list materializer
uses the existing object context, shallow handle edges, two native table passes,
one reserve and direct stamping. No per-member host callback or lookup/store
scan. Public source tuple(value), list(value), composition execute; same exact
tuple retains identity, dict iteration yields keys. NativeObjectInvocation leases
the existing LLVMExecution/ctypes ABI with one-time preparation, explicit export,
shared addresses and closed-lease refusal. No additional registry/PyObject runtime.

Observer found20 validation-text searches for10 calls through71,996-byte IR.
Capability cache preserves validation and invalidates on exact text/flag change;
real native error7 still raises twice. Final dispatch micro1000calls31.7249us
prepare-each versus0.6329us reuse,50.126x boundary-only.4097-element native stamping
onecall0.0312ms versus export6.0894ms. This is NOT whole-compiler/eager speedup.
Global member scanning remains O(total rows); per-object indexing is still next.
Earlier107pass77.22s; post-cache59pass/3oldskip24.26s; new11pass9.50s. Final broad
110pass78.45s. Seven ordinary audits unchanged; provenance OPEN.

Exact unchanged copier passes tuple(value)56 and next refuses native generator
collection93/call14 (2.1322359s). Trace: LoopResult collection, source29/control58,
materializer60, iterable56. Next is the existing graph generator loop's opaque
item/result/extent linkage with native stamping, not rewritten copier or per-item
ctypes. Full equality/hash, generic leaves and copier execution unfinished.
Intermediates, commands, bounds and lifetime limits in TEST_BASELINE_AND_HAZARDS.
No main/N4/timeout/kill/package/alternative compiler/live installation.

### Oct10 ordinary source objects connected to native context

User: "right, we're not preserving pythonic behavior we're just preserving an aggreement of temporary memory form so when it's rehydrated in python it's exactly what it would be if whatever had been done in python to the python object it just only did what the compiler says things do, right?"

User: "proceed"

Snapshot patch17 source_object_context.patch UNAPPLIED; all seventeen combined
apply-check0. Our public source entry accepts the existing native context.
Ordinary {} and [] now produce scalar handles at original control positions;
exact context ABI cells cross forwarding/recursive call frames. Native set/get/
length on admitted reference-key dicts executes; original codec export preserves
Python leaf identities. Exact graph Phi/loop initial/update edges propagate
handle representation without creating unknown private column arenas. Sixteen
source tests cover branches, loops, recursion0/1/7/32, forwarding and stale receipt
refusal. Two regression batches66pass39.72s and66pass53.26s; ordinary seven audits
unchanged, provenance OPEN. Live compiler unchanged; no speedup/main tiling claim.

Observed losing writers/intermediates are recorded in TEST_BASELINE_AND_HAZARDS:
formal owner/counter ledger, field-slot declaration producer, reduced [] receipt,
unclaimed IndexedStore raw GEP, finish's descriptor-only length materialization,
and control-merge aggregate declarations. Guards were not waived. Emission and
concordance validate new source receipts against live context/helpers; per-function
definition/helper inventories avoid birth-times-body rescans.

The unchanged50-line production copier advances from {} workspace12 to56:
members=tuple(value), a known builtin conversion still turned into an unknown
private sequence arena. Observer recorded exact source11->materialization56 and
recursive lease refusal at call14; final retry2.0368875s. Next is native object
iteration/materialization at that source producer. Reference identity is NOT
Python equality/hash; generic tuple/generator/opaque input semantics remain
unfinished. No rewritten copier, package, alternative compiler, main/N4 launch,
timeout, process kill or native copier installation.

### Oct10 Python/native object conversion boundary verified

User: "oh this is real, I thought this was fiddly but this is just the correlary to always wanting to let things convert piecemeal into python, so they have to be able to pass pythonic objects and it all needs a conversion api/abi, carry on"

User: "of course {} is both an object and an unknown recursive workspace, that's the nature of the syntax as it interacts with the user"

Snapshot-only patch16 native_object_boundary.patch staged UNAPPLIED; all sixteen
combined apply-check0. Existing shell codec now leases its handle domain to
the existing native context; entry/exit conversion handles builtin container
graphs without recursive traversal, preserving aliases, cycles and leaf
identity. Imported mutable handles retain their Python objects and synchronize
at validated export. New native handles become fresh objects registered with
the SAME codec, preserving repeated-export identity. Imported immutable member
changes refuse and need new handles. Native heap release/retention close is
explicit; no callbacks during native loops and no new registry or PyObject ABI.
Native dictionary lookup is still reference identity, not equality/hash.
Python equality collapse in invalid native table output refuses before publish.
Subclasses remain opaque, no inferred user layouts. Full imported graph remains
resident; this is boundary conversion, not whole-compiler memory tiling.

PUBLIC whole-source loop linked native constructor/store; single invocations
create1/17/257 fresh containers with backedges to an imported list. Native
growth, borrowed initial arrays, original Python identities and repeated export
are verified. Both native/context source modules have structural0; latch OPEN.
2500-deep conversion, sets/frozensets, key aliases, invalid cycles/extents and
release tested. Final broad100pass62.89s; final boundary/shell17pass4.36s.
Syntax/diff checks0. Seven ordinary audits unchanged after exact call-result
edge distinction (initial broader raw-Load fix temporarily broke oscillator,
then corrected; not hidden). Observer trapped native status Load int->int64
at ir_indexing dtype settlement, leaving i32 Phi. Declared raw memory widths
now preserved with explicit book reasons; abstract call projections keep their
existing rule. Clean tg-base confirms former widening and None release fault.
Details, commands and failed intermediates in live TEST_BASELINE_AND_HAZARDS.md.

Original copier still refuses workspace12/call14 in1.7746843s. Important user
correction: unknown dictionary CONTENTS do not imply unknown native allocation
layout. {} has known dictionary kind/initial extent0; publish its handle and
context at its original producer before recursive workspace leasing. Next is
actual source constructor/inspection integration, not forcing unknown columns
to guessed types or requiring a static schema. No live install/main/N4/package/
process kill/alternative compiler/copier speedup/main parallelism claim.

### Oct10 explicit native object context implemented; source links execute

User: "am I understanding we need a completely bare bones context object at the backend?"

User: "okay, this sounds like a subtle thing but like a backbone architectural object that would help us to do it right, please proceed"

Implemented in the native-fallback-link snapshot; live source unchanged.
Patch15 native_object_context.patch staged UNAPPLIED; combined fifteen-patch
apply-check0. SSAObjectContext groups existing dynamic object-kind and member
sequence tables, shared heap ownership and a counter. Real native initialize,
construct, kind/get/set/length and release functions use the existing helpers.
No new Python registry, opcode, pointer-as-handle or generic Python dict model.
Explicit kind vocabulary and reference-identity member keys are limitations,
not inferred Python equality/hash behavior. Initialize requires empty supplied
storage, and mutation of one context remains exclusive. Separate function
SSA occurrences preserve frame histories; immutable descriptors are shared.

Observed source-link faults: final descriptor dtype sweep converted ptr cells
to element types; the caller's raw ptr buffer ABI became double and copied
addresses away (a real intermediate second-call access violation); storage
publication spelled opaque_ref where Alloca canonicalized int64; Cast(i1,i64)
omitted zero-extension. Fixed at the existing writers. Exact source-call
bindings carry pointee/owner contracts through a monotonic reverse-call
worklist, with audited live-edge receipts. Heap-free modules avoid an extra
instruction inventory. Integral-key auditing reads the shared layout table.
No mismatch guards were suppressed or unrelated fields inferred by name.

Native API tests verify growth,17 objects, exact >2**60 refs, overwrite,
self/backedges, missing/invalid/collision/exhaustion and release/reinitialize.
PUBLIC source programs link real native constructor/access helpers through
faithful signatures. Whole1/17/257-object loops execute in ONE native call,
directly and through an extra relay, on repeatedly reused context storage.
Pointer ABI and shared buffers asserted before executing; exact kind*count
results and zero structural findings. These are explicit-context source
programs, NOT automatic source-born {} or a rewritten substitute copier.
Provenance latch remains OPEN, including native helper IDs born outside the
source book. Broad168 pass/5 old failures80.77s; final context/heap/operations
gate66 pass45.33s. Seven compact audits unchanged in rows/findings/unsourced
counts. Original copier still refuses unknown {} workspace12/call14 after
1.6916952s. No live install, main/N4, package, process kill, production speedup
or parallel-main claim. Full commands and receipts: live turing baseline file.

NEXT: bind admitted source container operations and the actual Python-object
boundary to the context while preserving existing dictionary semantics,
leaf identity, recursive memo/backedges and caller-owned result lifetime.

### Oct10 source-constructor bridge: boundary decision required

User: "do you need any design decisions"; next user: "proceed".
The earlier answer that no decision was needed was premature. Read-only
inspection found no automatic native Python-container implementation behind
the new opaque-object helper interface. No compiler source changed and no
new compile/test process was launched during this inspection.

Observed implementation boundaries in the native-fallback-link snapshot:
`precompile_to_ssa._sequence_descriptor` creates frame arena arguments and
FIXED descriptors. It does not create owned-heap producers. Exact copier
source12 is already recorded above as an unknown/unknown dictionary descriptor.
`ssa_object_operations.declare_object_helper` validates an implementation
body that must already exist; CONSTRUCT is an interface operation, not an
implemented Python-container constructor. Object-operation tests explicitly
declare metadata.Book/Object tables and supply physical columns. The exact
copier probe instead supplies only builtins.object/builtins.dict reference
contracts, without that record/table context. The existing
`shell_external_references.ExternalReferenceValueCodec` retains Python
objects in a Python dictionary and its host invokes Python functions; using
it does not make container operations native. Heap allocation explicitly
returns native addresses, not opaque object handles.

Therefore source object birth is not merely a missing allocator call.
The unresolved user-facing boundary is whether this native replacement may
consume existing typed record/sequence storage with Python-object conversion
confined to entry/exit, or must keep direct ordinary-Python-object inputs and
outputs at the existing copier interface. Neither representation conversion
nor a new object registry was silently introduced. Ask this precise boundary
question before implementation; preserve all fourteen unapplied patches.

### Oct10 typed dynamic owned arenas execute; source object birth not closed

2026-10-10 typed owned dynamic storage path completed; source birth remains — SNAPSHOT ONLY:

User: "okay let's clean everything around the dynamic allocation contract and
see if we can make that the last rivit". Patch14 dynamic_owned_sequence_storage.patch
staged UNAPPLIED; all fourteen ordered patches pass combined git apply --check0.
Snapshot changes: ir_sequence_tables.py, ssa_heap_allocation.py, ssa_llvm_backend.py,
identity_concordance.py, sequence_storage_binding.py, transmogrifier/ssa.py,
test_ssa_owned_heap.py. No live compiler/copier install, main/N4, process kill,
package or alternative Python compiler. Native LLVM emission uses our incumbent lane.

PRE-EDIT OBSERVATIONS:
Exact50-line copier rejects sequence12, actual {} producer, columns(12, secondID),
column_dtypes unknown/unknown, scalar rows, FIXED policy, heap_owner=None.
Graph retains Dict/aggregate_kind=dict and two-column writable/key0 shape, but
provides no native object-domain/owned allocation descriptor. Existing allocator
already implements runtime size + caller-owned lifetime. Existing typed owned
DYNAMIC descriptor still gives dynamic-growth-unavailable from insert; initializer
also refused DYNAMIC policy. Separately, heap length/capacity aliases were accepted
and int64 heap columns bound to float64 heap columns through ptr ABI unchanged.
All three gaps were observed before their respective changes, not guessed.

IMPLEMENTED in the EXISTING descriptor/storage path:
- Empty typed owned columns can initialize with FIXED or DYNAMIC capacity policy.
- lower_sequence_reserve validates nonnegative live length/required count,
  length<=capacity and allocator byte-overflow bounds BEFORE copying. Chooses
  max(required, checked double capacity), capped at the shared safe row maximum.
- Allocates/zeroes all new columns, byte-copies live rows, then publishes exact
  pointer cells and capacity together. Length unchanged. Same-cell column aliases
  reuse one allocation; incompatible aliased element layouts are refused.
- Existing fixed insert/table-store bodies remain the mutation authority. A shared
  dynamic wrapper validates extents before scanning, tries the incumbent operation,
  reserves ONLY on full, then retries. Duplicate keys and existing-key updates do
  not grow. Existing statuses retained, no Python callback.
- Extend includes exact destination owner, binds row formals by their declared
  IDs and retains the ENTIRE insert/fixed/reserve helper closure. Heap element
  contracts match; shaped rows/liveness/child pools without supported replacement
  are still named shortfalls, not silent truncation.
- Native typed element transfer is ordinary Call to LLVM memmove through the
  existing heap ABI. Integer element counts checked before byte multiplication;
  scalar count width limited to64bits (no truncating a wider count). No opaque
  handle becomes an address. Byte-copy preserves float32 bits without numeric
  widening/narrowing and supports overlap. Allocation operation itself validated.
- Owned descriptors reject overlapping length/capacity/pointer cells, owner/status
  collision and pointee-layout ambiguity. Cross-call heap binding checks exact
  layouts and normalized row shapes, not merely ptr ABI equality.
- Concordance audits heap-copy-contract beside existing heap-allocation-contract.
  No new object registry, datatype, source opcode or allocation-guard relaxation.

LIFETIME / COMPLEXITY:
One caller-owned heap cell retains current AND retired arenas until explicit
release. Pointer-CELL identities stay stable; interior addresses are arena-version
views and must be reacquired after growth. No claim that an old interior view
magically follows replacement. Geometric growth amortizes row-copy work; retired
payload is geometrically bounded for monotonic insertion. Owner release frees
the chain and clears its cell. Unique-key scanning itself remains linear, NOT
a new hash table, and no main compile speedup/parallelism is claimed.

NATIVE VERIFICATION (SSA allocation-contract tests, not automatic source binding):
Five layouts int64/opaque_ref/float64/float32/bool reserve through1/2/7/4097 rows,
preserve byte patterns/retired views and zero new capacity; no-op reserve and
invalid counts preserve state. Native insert AND table-store loops run whole
1/257/4097-row jobs in ONE native call, for int64 and opaque_ref, exact values
above2**60; re-inserting keys preserves capacity/owner/columns. Native extend
copies257 borrowed rows twice without modifying source or duplicating rows.
Invalid extents reject before touching null columns. Overlap copy preserves
memmove semantics. Explicit release resets owners. Valid dynamic module audit
observed124 rows/6functions/0 findings; standalone SSA module has no attached
book, so this is not proof of source-born emission provenance.

Final targeted gate163 pass/5 PREVIOUSLY BASELINED failures73.98s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_control_branch_compartments.py tests/test_native_conditional_call_result.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two real int32/int64 exact-region
feed conflicts, no exclusions. Final seven audits unchanged0,1,0,1,5,0,0 with
identical unsourced counts, latch OPEN. Guestbook validation dry-run only.

NOT THE LAST RIVET YET:
Final production exact copier retry2.2399622s still refuses unknown dictionary workspace12
at recursive call14/SCC0. New code does not pretend that {} has a native object
handle/domain, element layout or shared allocation owner just because malloc exists.
Remaining SOURCE bridge: constructor materialization must publish its accepted
object-domain representation, runtime capacity and shared-owner descriptor before
recursive leasing, and schedule the real allocation producer. Existing opaque
construct helper interface alone is not a generic object implementation.
Previously observed early-return missing recursive Call remains unresolved.
No compiled/installed production copier, source-constructor success, N4 gain,
whole-main tiling or parallelism claim.


### Oct10 result-Phi ownership fixed; common-return native recursion verified

2026-10-10 reducer-owned result Phis retained; declared recursion executes — SNAPSHOT ONLY:

User: "okay, let's proceed". Patch13 conditional_result_phi.patch staged UNAPPLIED;
all thirteen ordered patches pass combined git apply --check (exit0). Changes
are snapshot glsl_deployment_strategy.py and test_ssa_object_operations.py only.
No live compiler/copier install, main/N4 launch, process kill, packages or new
compilation route. Existing public source entry and native LLVM lane used.

OBSERVED CAUSAL CHAIN, before editing:
Graph Phi18 has test4/body17/orelse9, binding result, source_conditional_id5,
and NO initial_value_id. _ordinary_conditional_control_programs reaches its
initial-value check and skips this Phi. lower_conditional receives result_aliases
empty while sequence_length_values is already {17:14}, descriptors14/6 present.
No one requests len-result17, so no Load or Phi is emitted; Ret reads call9.
The graph knew the complete merge; this was planner loss, not missing type data.

Fix: reducer Phis without an entered version use the EXISTING conditional-result
path with their exact arm IDs. No invented merge from names, no new SSA opcode,
no backend repair or allocation-guard relaxation. Existing control lowering
requests len17 on its true edge, publishes Phi18 and the concordance derivations;
existing linked result-type propagation settles recursive result to int64.

A second minimal case ("if index: result=23; else: result=41") exposed the earlier
retention guard: no numerical predicate/arm regions, calls or field-state merge,
but graph Phi4(test0/body2/orelse3) owns source conditional1. Observed guard drops
that conditional despite this exact Phi. Retention now admits reducer-owned
body/orelse merges as well as the prior field-state merges. No-Phi no-invention
regression remains green.

Actual source common-return recursive table-length program now lowers under the
unchanged full-native gate, has zero concordance findings, and executes LLVM
native at depths0/1/7/32 for three objects: lengths2/0/1, one handle above2**60.
All twelve executions return the base-case length; all caller columns remain
unchanged. Recursive feeds share exact declared keys/values/length and member
column pools/lengths/stride; only int lookup-status scratch is invocation-local.
Fortran text emission complete. C module text still explicitly reports missing
zeros spelling for the two invocation-local status producers; NOT C completeness.

Four additional source assignment programs execute nine branch cases: bare
predicate, comparison predicate, nested merge, and an entered carried version.
Five focused tests pass6.31s. Saved pre-edit strategy replay gives three failures/
one pass7.26s for the comparison/nested/recursive/entered set; the bare-predicate
failure was observed separately before changing its retention guard. Thus both
losses are deterministic and reproduced, not inferred from a post-edit success.

Final targeted gate148 pass/5 PREVIOUSLY BASELINED failures69.15s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_control_branch_compartments.py tests/test_native_conditional_call_result.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two real int32/int64 exact-region
feed conflicts, no exclusions. Earlier reference/control subset93 pass62.94s
before adding bare-predicate regression. Seven final audits unchanged0,1,0,1,5,0,0
and identical unsourced counts; provenance latch OPEN. Guestbook dry-run only.

Production exact50-line copier retry2.1632375s still refuses dynamic dictionary
workspace12 at recursive call14/SCC0. Borrowed declared contexts and ordinary
fresh branch results are now demonstrated, NOT generic native dictionary
construction, Python object-domain binding or installed copier acceleration.
Separate early-return kind variant rechecked: recursive Call remains absent,
exported Ret dtypeNone; losing writer still unobserved, never executed as native.
Remaining frontiers: that call-retention defect, and the dynamic constructor's
actual element/extent/owner contract. No main speedup, tiling or parallelism claim.


### Oct10 recursive borrowed storage ordering fixed; control merge still refused

2026-10-10 recursive declared keyed storage settled before leasing — SNAPSHOT ONLY:

User: "how is our gap now", then "proceed".
Patch12 recursive_keyed_storage_binding.patch staged UNAPPLIED; all twelve
ordered patches pass combined git apply --check (exit0). Only snapshot
fortran_c_shell.py and test_ssa_object_operations.py changed. Live compiler,
copier and processes unchanged. No main/N4, package, timeout or process kill.

Pre-edit observation at record-ABI completion: authored GetAttr6 book.objects
still has anonymous unknown keys/value arenas and descriptor, but the lookup
already names its owner and the exact declared length/keys/values formals exist.
The incumbent late keyed binding pass is now a single reused closure. It runs
after record ABI materialization BEFORE source-call discovery/recursive leasing,
then again at its original post-link stage. Its existing remap retires anonymous
storage and derives descriptor revisions from old descriptors and physical cells.
No new logical sequence-field claim, allocation fallback or guard relaxation.

First relocation also typed opaque queries too early: six native reference cases
hit opaque_ref/int64 physical input disagreements. The early invocation now
settles STORAGE only; query producers/formals retain their incumbent types until
the original post-link query binding. Saved pre-edit compiler replay confirms
those six pass (5.73s), revised current reference/object batch45 pass29.69s.

Common-return recursive source now passes the storage guard. Diagnostic witness
asserts recursive feeds share exactly the same declared keys/values/length and
pooled columns/lengths/stride identities; descriptor keys/values are int64 and
capacity=length. Only one int lookup-status scratch is invocation-local, no
borrowed table arena. Regression1 pass2.27s is NOT recursive native success.

Next observed blocker: full-native structural_outputs reports missing Phi18
(test4, body17, orelse9). Rejected SSA has base-case selected table length address
but no Load/result17 or Phi18; if_merge Ret reads recursive Call9 alone with
dtype=None. Recursive call DOES survive (unlike the previously recorded
early-return variant). This malformed module was inspected from the refusal
traceback, never emitted/executed; full-native refusal remains enforced.
The original writer losing the length-result/control merge is NOT yet traced.
Next repair must observe that producer/control reduction, not patch the backend.

Final targeted gate128 pass/5 PREVIOUSLY BASELINED failures56.18s:
python -m pytest -q --tb=short --show-capture=no tests/test_ssa_object_operations.py tests/test_native_reference_inspection.py tests/test_native_reference_boundary.py tests/test_native_structural_producers.py tests/test_ssa_owned_heap.py tests/test_ir_sequence_tables.py tests/test_sequence_contract_concordance.py
Same three stale ID/helper-name expectations and two genuine int32/int64
exact-region feed disagreements, no exclusions. Existing NODUS_TENSOR_CORE used.
Seven audits unchanged0,1,0,1,5,0,0, same unsourced counts, latch OPEN.

Original50-line copier was NOT rerun this step; prior measured dictionary12/
call14 refusal remains the last production evidence. No installed native copier,
main speedup, automatic object binding, dynamic constructor completion, main
tiling or parallelism claim. Storage ordering gap closed; recursive control and
the broader copier/native object gaps remain.


### Oct10 declared object table context now crosses real source calls

Patch11 object_table_source_context.patch staged unapplied; all eleven apply-check0.
Shared row-table materialization publishes existing column pools/lengths/stride
as record fields on caller AND callee. Selected-row pointers remain separate;
existing record_field_decomposition carries capacity/stride only. No new object
registry, backend repair or interpreter callback. Eight native source cases
kind/length/get/set through one/two forwarding levels execute exact refs above
2**60 and caller-buffer mutations; C/Fortran text complete with public outputs.
Earlier tests only built SSA callers; this closes that nonrecursive source gap.

Observed additional old defect: unread keyed field fallback fabricated a mapping
input beside physical length/keys/values. Root leased relay's unclaimed ghost
despite owning the field. Saved pre-change compiler reproduced duplicate-storage
finding. Producer no longer fabricates unread keyed input; existing exact book
decomposition also supplies reuse. No identity matching by names/dtypes/positions.

Final focused74 pass43.83s; separate sequence/concordance54 pass/5 known failures
7.04s. Earlier combined127 pass/5 fail51.58s before added refusal regression.
Seven audits unchanged0,1,0,1,5,0,0 and unsourced counts; latch OPEN. Live compiler,
copier, installation/processes untouched; no main/N4 or package/timeout added.

Recursive common-return micro still refuses unknown authored GetAttr workspace6
at call9: sequence-member edges exist but owner-record->sequence claim absent
before recursive lease. Same refusal on saved pre-change compiler. A late claim
attempt disturbed key typing and was removed completely. Refusal test is NOT
recursive native success. Producer-side keyed binding must precede lease selection.
Early-return variant additionally loses recursive Call and returns dtypeNone in
current AND saved pre-change compiler; no native execution, losing writer unknown.
Full original copier1.6806848s still blocked unknown dictionary12/call14. Bare
object-domain binding, builtin dispatch, recursive result/constructor extent and
shared lifetime remain. Detailed repro/limits: turing/TEST_BASELINE_AND_HAZARDS.md.

Prompt history:

> how do the limitations of our opaque ref wiring meet the incomplete handling of type and extent in the copier

> what is the industry standard way to close this gap

> continue

### Oct10 opaque object SSA/backend layer realized in isolated snapshot

SSAOpaqueObject wraps existing scalar opaque_ref + admitted storage identity;
no new registry/dtype. Exact-ID native helper contracts and ordinary Calls now
cover kind/length/get/set/construct/release shapes, validated by all three native
backends and concordance. Actual native kind, length, member read/write, identity,
equality and closed-vocabulary membership execute; C/Fortran text emits all four
descriptor helper programs. Helper bodies use our public source entry; SSA test
callers do not prove automatic source binding. Construct/release have no generic
runtime implementation yet; bodyless imports remain refused by this helper API.

Observed fixes at original writers: known literal type lost despite aggregate
receipt; two-level column Store lost its row operation and wrote a temporary;
borrowed table classified local/reset; opaque identity mislabeled tensor op;
keyed rebind overwrote book-backed rows with dict; pruning left record.members
claim on absent sequence6. Existing table rows now advance with derived receipts,
and dependent record field claims retire at pruning, not a backend repair.

Focused66 pass37.47s; extended119 pass/5 prior failures43.90s, then one added
detector test1 pass2.04s (included focused). Seven audits unchanged0,1,0,1,5,0,0;
unsourced latch OPEN. Patch10 opaque_object_operations.patch staged unapplied;
all ten combined apply-check0. Live compiler/copier/processes untouched.

Source caller still projects incomplete metadata.Object row (kind span only,
table leaf deferred), so nested helper members forwarding refuses. Corrects
earlier preliminary claim: not a Book/Object identity mismatch. Bare Python
object entry remains unbound to native descriptor domain. Exact original copier
3.0152095s still blocks unknown recursive dictionary workspace12/callsite14.
No accelerator install/main/N4 launch or speedup claim. Complete source context
forwarding, builtin dispatch, recursive storage/results and constructor/lifetime
remain. Detailed receipts: turing/TEST_BASELINE_AND_HAZARDS.md.

Prompt history:

> can we just make this auto resolve the object if it is found to be one

> what would some other fundamental and performanced focused linearizing or graph based compilers do about this opaque handle issue

> what it sounds like to me is we want an opaque object type and helpers for ssa and the backends

> can you realize this

### Oct10 follow-on: ordinary reference entry is not a descriptor-domain binding

Read-only boundary investigation in the isolated native-fallback-link snapshot.
Tiny public-entry programs under the full-native overlay, with NODUS_TENSOR_CORE
pointing at the existing live DLL: id(value) emitted complete LLVM in0.541498s;
isinstance(value,dict) refused in0.366311s, structural Call2 with arg0=value0,
call-result-unavailable. No compiler edits, no main/N4 launch, no process killed.
An earlier diagnostic omitted NODUS_TENSOR_CORE; the figures above are the repeat
with that existing connection configured, not that initial diagnostic.

Observed: ProgramABI reference normalizes to opaque_ref, but rejects record on
reference storage; token_vocabulary requires scalar int64, not reference. The
existing nested-record vocabulary predicate lowering compares a declared scalar
column with vocabulary positions; it does not resolve an opaque object. Current
SSAReferenceDescriptor has handle/identity/kind/host_resident and no dynamic
payload/type/member relation; its publishing site consumes StaticRef facts.
Existing CPython runtime keeps a separate one-based retained-object handle table
and invokes PyObject_Call through its thunks; it is not the native container
descriptor path. Production copier receives Python containers, whereas patch9
tests receive explicit caller-owned Book/Object columns. This missing boundary
must not be disguised as completed generic object introspection.

Next decision: whether one-time input/output conversion using the existing native
record/sequence representation is admissible. This changes the production object
boundary and entails alias/cycle preservation plus original Python leaf identity;
it cannot be silently invented as a second object registry or replacement copier.
No staged promotion patch10 exists. Existing nine patches remain unapplied.

Prompt history:

> lets carefully march forward, what's the look of this next step

> okay proceed

### Oct10 start steps1-3: native descriptor inspection and empty-row constraints

New derived schema view consumes already-published exact graph record identities,
not field names or a new object registry. Keyed-record member tables now declare
their columns before lowering; required control-owned lengths materialize from
their existing sequence binding. Direct resident views join the incumbent nested
table column/length/stride ABI, not only destructured-loop projections. Native
length/both reference columns/traversal/combined repeated reads execute through
our public source compiler. Repeated GetAttrs retain canonical handle producers
and publish exact planning aliases before retiring duplicate descriptors; audits
caught both missing handles and shared-member intermediate mistakes and both
were corrected. New native cases assert zero structural findings, latch OPEN.

Empty dict rows can derive (int64,opaque_ref) from exact runtime IndexedStore
operands, using all stores and rejecting inference from unknown/conflicting rows.
Source row layout proved; local-dict capacity/allocation NOT executed or solved.
Range-counter width follows int64 resident extents with explicit seed casts.
Floating traversal counts pass; integer accumulation and direct reference carried
assignment still expose separate unresolved result/binding failures. Existing
simple indexed-loop/pair-update Phi gate remains1 fail/3 errors3.19s.

Focused45 pass29.47s; extended97 pass/5 old failures37.33s before two added unit
tests. Before/after seven audits0,1,0,1,5,0,0, full provenance latch OPEN.
Patch9 staged unapplied; all nine combined apply-check0. Live source/main7196
untouched. Exact incumbent copier2.1708728s still refuses unknown dictionary12
at callsite14: bare opaque binding/runtime kind dispatch and recursive result
constraints remain. This starts1-3; it does NOT finish all three or install a
copier accelerator. Full receipt: turing/TEST_BASELINE_AND_HAZARDS.md.

Prompt history:

> Outline all the remaining steps to finish this portion of the compiler

> Go ahead and start on 1-3

### Oct10 separate native storage semantics from arena placement

Snapshot native keyed memo insert/lookup/update now consumes incumbent ProgramABI
or dict[int,object] row declarations, preserving int64 keys and opaque_ref slots.
Public source compilation executes exact handles above2**60 using caller buffers.
Observed missing aggregate_kind caused flat pointer indexing despite declared
rows; fix consumes the existing receipt at _field_slot_ops. Observed descriptor
members were private frame cells instead of borrowed caller columns; fix publishes
ownership at _sequence_descriptor for explicit known row layouts only. An overly
broad intermediate ownership change broke keyed tensor lookup by promoting a
provisional text-key view; trapped writer and restricted to explicit contracts.
Matching int64 table-store index/latch arithmetic and owner-member binding finish
this boundary. No unrelated live compiler changes or substitute object registry.

Existing keyed value_record columns support native handle-to-record kind lookup.
They are supplied by caller, not inferred from an opaque Python reference.
Focused35 pass/18.37s; extended89 pass/5 previously baselined failures/26.96s.
Seven audits retain0,1,0,1,5,0,0 findings; unsourced latch OPEN. Patch8 staged
unapplied with all eight combined apply-check0. Main7196 remains untouched.

Original50-line copier still refuses2.0398794s: private dictionary12 at recursive
callsite14 has unknown element layout/extent. Nested descriptor len(item.members)
also remains a structural-call lowering gap. Generic input descriptor binding,
container construction/iteration/immutable publication, memo growth remain.
Only-arena-allocation-remaining is NOT achieved. Existing caller-owned heap owner
expresses surviving recursion; shared memo/results cannot be invocation scratch.
Allocation placement/capacity policy deliberately not invented. Full receipt and
repro commands: turing/TEST_BASELINE_AND_HAZARDS.md.

Prompt history for this checkpoint:

> Okay so the issue right now is our user type system lacks introspective details

> Can we take care of everything and try to leave the where and how do we allocate an arena ideally

### Oct10 requested heap escape: actual owned native storage, copier still blocked

Implemented only in isolated native-fallback-link snapshot. Reuses Alloca/Call
and dtype_layout, with explicit caller-owned pointer cell, native allocation
chain, checked runtime count and explicit release. No opaque-handle-to-pointer
guess, new opcode, Python callback, new compiler or package. Typed allocations
survive helper returns and later allocations; native caller reads before release.
Existing sequence descriptor/book gains exact optional owner identity, allocation
helper initializes runtime-sized fixed-capacity columns, existing insertion and
lookup operate on them without host materialization. Concordance sees malformed
allocation contracts. Internal SSA tests are NOT claimed as public source success.

Native table chain at1,7,4097 capacity returns exact int64 value above2**60.
It exposed a real pre-existing length-cell error: untouched tg-base stores
double1.0 into declared int64 length (4607182418800017408). Deriving the operand
from the same formal fixes that writer. Insert/lookup use matching int64 loop
contracts. Heap loads snapshot before freeing; actual alignments are honored.

Extended gate83 pass/5 fail in26.77s; every failure reproduced on untouched
tg-base. Wider alias gate also has1 fail/3 fixture errors, reproduced identically
by loading the exact four saved pre-heap source files without modifying checkout.
That earlier staged-patch native loop regression remains real and unresolved;
two numerical alias tests pass on clean baseline. No misleading green gate.
Final focused heap/public-source gate29 pass/16.51s; guestbook dry-run0.
Seven audit structural counts unchanged0,1,0,1,5,0,0, provenance latch OPEN.

Exact incumbent copier retried2.015371s, still refuses untyped empty dictionary
arena12 at recursive callsite14. Allocation cannot infer runtime object types or
capacity. Source constructor/frame wiring, generic runtime operations and memo
growth remain. No compiled copier installed, no measured main speedup. Main7196
left alone. Patch7 owned_heap_allocation.patch staged live but unapplied; all
seven patches combined apply-check0. Full receipts in turing hazards document.

Prompt history for this checkpoint:

> can we escape this by giving a generic dynamic heap allocation or is that not smart

> can you please endeavor to do that

### Oct10 native predicate/scratch micros and deterministic state-loss repair

Snapshot only. Positive builtin isinstance facts now fold using the existing
ProgramABI identity, independently of opaque handle storage. Unknown dynamic
subtypes/custom same-basename types remain unresolved, not assumed false.
Typed fixed-size private SCC leases become existing local zeros SSA producers,
with existing mint/lease provenance and per-function backend frame ownership;
no caller-signature feedback. Resident descriptor/accounting must prove workspace.
Dynamic/unknown object arenas still refuse before allocation or signature mutation.

Mutable tiny repro found a genuine state-loss writer: structural evaluator treated
IndexedStore's ast.Subscript(ctx=Store) as a read of its initial [0.0], replaced it
with Const0.0 and dropped authored n. After keeping the effect, the initializer
fold still reconstructed different arenas in different numerical regions. Exact
store base edges now extend the incumbent mutation protection. Untouched live
nonrecursive slice has0 Stores/no n parameter. Snapshot actual native write/read
returns0,1,4,12; recursive parent/child case returns0,1,10,78 for depths0,1,4,12.
The added concordance assertion caught a third writer: root private cleanup
removed live descriptor capacity/status members. Hook observed removal IDs
2305843010213693953/3954 at class-surface cleanup. It now consults the same exact
live-sequence member rule as callee cleanup. No diagnostic weakened.

Final focused16 pass /14.85s; existing root-prune units3 pass /2.02s. Intermediate
broad gate125 pass/1fail /136.49s caught descriptor loss; no exclusion or masking.
Final cross-module gate59 pass/86 unrelated deselected /73.63s; seven final audits
retain0,1,0,1,5,0,0. Combined six-patch apply-check0; guestbook dry-run0, no --apply.
Unchanged whole copier retried after small success:1.940700s refusal at
isolate_metadata/callsite14/callee_ref1/SCC0/private arena12, dtype unknown,
shape(), declared_workspace=True. Positive static type proofs are not runtime
container operations; dynamic object layout/arena lifetime/memo operations remain.
No installed native copier, callback dispatcher or claimed N4 speedup. Cumulative
patch6 staged only; native fingerprint invalidates normally. Main7196 remains
untouched, observed52648.88CPU s/1157386240B RSS; no N4/main launch this turn.

### Oct10 structural producers and recursive call cleanup (snapshot, not native copier)

Prior diagnosis that native identity was absent was wrong. Existing id Cast
materialized after parameter pages closed. Private id feeds bypassed it. A shared
producer now promotes the exact declared input from its binding/value/call cells;
unused parameters stay unused. Graph planning now publishes id's int64 descriptor
early, preventing unannotated native helpers from rounding handles above2**53.
Actual OUR compiler ->LLVM native results exact1,4660,2**60+3, both public/private.
No claim of CPython object-address parity; identity audit0 gating findings.

SCC activation edge existed but hierarchy freezing required a callsite clone.
Recursive PlanCalls now use catalogue signatures without body unfolding. Existing
runtime dependency walk admits those targets by recorded activation identity.
An iterative worklist visits each SCC owner once and freezes completed plans.
Direct/nested native recursion exact0,1,4,12; mutual exact0,1,4,11. Same full-native
no-callback contract/runtime_closure_only=True. Original four positive shapes
fail on untouched live compiler. No production-native copier installed.

Full exact copier then exposed a second call-frame defect: each recursive site
cloned callee scratch into caller.args, making those clones new callee formals for
the next site. Thousands of arguments accrued. Empty unresolved_frame_value_ids
did NOT mean complete frame bindings. Now reject this invalid ABI propagation at
the lease writer, before minting. Exact copier refuses in1.918846s with caller
isolate_metadata/callsite14/callee_ref1/SCC0/private_formal12. Per-invocation native
storage is required; a guard is NOT its implementation. Dynamic opaque_ref type
tests/membership remain capability gaps. Integer recursive return also showed an
int64/double LLVM Phi disagreement, still open. No guard weakened or callback shim.

New targeted tests8 pass /7.53s (5 native cases,3 capability refusals). Extended
pre-safety gate154 pass,1skip,2 documented hierarchy exclusions /129.24s. Final
source gate155 pass,1skip,2 deselected /132.22s. Native-loop execution20s timeout
reproduced BOTH live/snapshot23.68/23.67s, not caused by this work. Seven final
audit counts unchanged0,1,0,1,5,0,0. Guestbook dry-run0, no --apply cleanup.
Six patches combined apply-check0,
unapplied live; newest native_structural_producers.patch. Native asset fingerprint
invalidated normally, not forced stale. Main7196 left alone; no N4/main launch.

### Oct10 requested N4 and entire metadata copier native attempt

N4 ran in snapshot using matching726431cbf86656cc6973da89e78cf980 unit,1 graph
worker/8 residents/no trace/deadline;all4 laws rebuilt. Completed507.330979 s,
8.5806% faster than last554.948946 but4.3076% slower than older486.379790.
Combined begin172.714214/begin->SSA300.786486/whole334.614719 s;native17calls,
143043facts,.622484 s including transfers.996functions,6157MiB max stageRSS,
2203575 resident book cells/0spills (+38.56%, not causally attributed). Root
build/profile_memory_n4_native_storage_rmf6odxo/profile_memory_n2;C SHA
d73885274cde3e219393c8bdb73b7f3c5a94a611732cc7e9b6d9cd499fe1ef0c.
Detector itself reports Unresolved.value_id exception; no clean audit/numeric
orbital parity claim. Single wall samples/full-main concurrent, not isolated.
Main7196/33064 createdOct9 18:25:20 America/Chicago, under exact earlier request
to lower entire main and remove timeout;not restarted. Four Python processes
were two parent/worker pairs, not four compiler jobs. N4 pair finished.

User asked readonly cost investigation, then explicitly requested entire~500k
recursive-call dispatcher compiled. Tiny unchanged-source profile shows165
folds/2.048s cumulative,18625descriptor calls/1.251s,515987metadata copier calls/
.860s,24372book.post/1.042s;overlapping costs,9.198s instrumented profile adds
metadata counter~.452s. Not N4 call-level causal attribution or native win.
Copied EXACT production nested isolate_metadata AST into root(value,sharedmemo),
50authored lines,no algorithm substitution,full-native/no-callback contract.
First~1s refusal:full AST.walk assigned comprehension inside If.test to If arm;
loop region(0) absent from parent arm(4). Existing ordinary conditional helper
already uses body/orelse slots. Applied SAME rule to heterogeneous nesting in
snapshot. Predicate fail-first fails live too;arm-body case passes.24 focused
ownership/control tests now pass /1.58s. Exact full copier now reachesSSA but
gate reports19undefined operands and recursive call/type/membership unavailable
results. Minimal id(value) on reference rejects unaccounted formal;dynamic
isinstance(value,dict) result unavailable. Minimal tuple/list-items LLVM emission
complete, but not runtime object parity. No native copier installed or Python
callback disguised as native. Need actual native object-operation/recursive
result lowering over accepted exact object ABI next;do not patch gate away.
Existing copier unchanged,other investments preserved. New exact repro script
probe_native_metadata_copy.py. Fix/test/repro promotion patch passes combined
apply-check with previous4;all remain staged. Snapshot source edit intentionally
invalidates old native-storage fingerprint;refresh before future timing.
Final focused gate169 pass /41.74s with connected DLL, same4 stale exclusions;
seven audit findings unchanged0,1,0,1,5,0,0 after the ownership fix. No native
copier victory claimed. Guestbook validator dry-run exit0, no --apply cleanup.

### Oct10 original four dispatcher failures: stale tests, observed on live

Live rerun4 failed /3.77 s. Scalar accessor verdict remains COORDINATOR_ACCESSOR;
comparison is NUMERICAL_WORK under explicit Sept30 removal of unrecorded scalar
deferral (existing decision document). Old test asks for the removed behavior.
Existing public-entry annotated-scalar probe rerun:3 forms pass. Other3 tests
raise at output Store identity lookup before their boundary assertions because
raw _add fixtures supply no ingestion scopes/rows. Diagnostic memory-only wrapper
uses exact declarations from existing scoped cut fixture:three scopes plus
INGESTION_VALUE NodeFacts; no guard bypass or different cutter. All original
assertions pass: lookup outputs(1,3)/(3,),identity route(1,2),external-use(1,2)
and2 Stores. No compiler/test changes for this audit; test maintenance is next,
not a compiler fix. This does not prove whole compiler/GPU performance.

### Oct10 reusable cone: two existing quadratic consumers, not global scheduler

Exact-identity iter_dependency_cone reads incumbent relation; FIFO O(Vcone+Econe)
or incumbent ordered sweeps, preserving first-cause/event order. Phase-local
parent_record_dependents indexes actual operand records, not weaker NX edges;
no new topology, descriptor cache or book facts. Existing descriptor invalidator
and call-result descendant re-settlement now reuse it.64 reverse-chain fail-first
2016 reads ->126;32/64/128/256 ->62/126/254/510.40 seeded mixed-graph exact
oracle cases compare final dictionaries/withdrawal causes/event order, FIFO
oracle retains old order. Final gate145 pass /44.72 s (no DLL env; warning).
Cold hash-seed0 DT old3.9488375/new3.7688789 s, same730 publications/24106 sizes,
40 functions/0 findings, both baseline LLVM SHA570695ad11b99133402dc4ffe833b8f658a147584dc611d74538c39e621e7b34.
One pair is not a wall-speed proof.165 global folds unchanged. Optional wave
publication batching removed; per-query publication policy unchanged.
New dependency_cone.patch plus updated serial probe/previous ABI/native-link
patches pass combined apply-check. None promoted live; full-main7196 untouched
at46907.62 CPU seconds/1193.6 MiB RSS. No new long orbital run.
Snapshot native storage rebuilt/load-verified via OUR public lowerer/LLVM under
726431cbf86656cc6973da89e78cf980;0 findings,1306 unsourced facts/93 identities,
OPEN. Actual native DT outputs0.9/.25 exact /11.7404743 s total probe;0 findings,
888 unsourced facts/0 identities, OPEN. Seven-case audit findings unchanged
0,1,0,1,5,0,0. Guestbook validation dry-run exit0; never --apply.

### Oct10 follow-on: shared declarative ABI through existing ownership boundaries

Latest "Proceed" follows slower four-law handoff. No further long lowering.
Small real DT proposal profile exposes93 graph extractions,575688 metadata
copier calls,165 folds and13428 descriptor reads. Counter attributes59460 visits
to full program_abi in60 proper regions. Public source lowerer also deep-copied
the complete receipt per function before/after reduction. All receipt writers
install whole declarations; readers create own mutable parameter-record views.
Changed only those existing ownership boundaries: shared compilation receipt
on catalogue functions; added program_abi to proper compartment shared metadata.
Whole-callsite copies still isolate every mutable container. No disabled book,
new graph/record/ABI, alternate compiler, package or tolerance.

Two fail-first proofs failed before core edits and pass after: real public-entry
catalogue shares root receipt;32 singleton regions over128-record schema do0
schema traversal; whole-copy nested edit cannot affect source.97-test gate
41.32 s, same4 known dispatcher failures excluded. New cases separated into
test_shared_program_abi_receipt.py and rerun2 pass /1.38 s. Exact small LLVM
cold hash unchanged570695ad11b99133402dc4ffe833b8f658a147584dc611d74538c39e621e7b34.
OUR source lowerer ->existing LLVM native execution yields exact0.9/.25,
0 findings,888 unsourced facts/0 identities; latch stays OPEN.
Seven-case audit rerun: unchanged findings0,1,0,1,5,0,0. No new findings.

Post metadata profile516168 calls,59520 fewer (~10.34%); region ABI work now0,
whole specialization copies still32703 visits. Work bound is specific schema
assignment O(F*S) ->O(F+S), not whole lowering linear or bounded. Before tiny DT
times3.258604/3.147716/3.400177; after with observer3.072286/3.346539/3.699394
overlapped gate/full-main. NO wall speedup established. Profile I/O varies by
seconds: io.open exclusive.776/1.970/.358, one string-table write1.584 s;
avoid blaming graph changes for those waits. Existing pool still92 segment
writes/220 reloads in this small DT. No spill redesign/policy relaxation.

Promotion shared_program_abi_receipt.patch holds independent core hunks/new
tests. Updated serial_graph_scaling.patch includes tiny profiling/copy counters;
three staged patches together pass git apply --check, unapplied live. Snapshot
native storage rebuilt and load-verified under718d64f3130504f1fc1db1746f5016ba;
source audit0 findings, same1306 unsourced facts/93 identities, latch OPEN.
Live full-main active at45512.125 CPU seconds/1273.5 MiB, unchanged. Further
work targets repeated fold/descriptor reads/remaining owned receipt copies
with small exact repros, not another speculative whole-program run. No N4
benefit for this new change measured. Existing554.948946 s receipt remains.
Guestbook validator dry-run exit0; filenames conform. No cleanup --apply.

### Oct10: remaining receipt callers, refreshed asset, N4 completed SLOWER

Proceed followed explicit next-step plan for exact receipt-work reduction,
small scaling checks, refreshed native asset and four-law benchmark. Fail-first
48-add public-entry test counted97 shape-settlement and49 dispatch-carving ABI
checks per frozen graph. Both now publish once at original first read, through
existing settlement scope/immutable carving index; every descriptor query remains
fresh. Delayed publication preserves before-proof reads and region-copy timing.
No alternate compiler, graph representation, answer cache or provenance bypass.
95 focused tests pass /40.29 s, same four known dispatcher cases deselected.
Arithmetic128/256/512 calls45/45/45; size work14731/29323/58507 vs previous partial
113547/423563/1634445. Existing LLVM hashes unchanged; audit findings0 each.
Single times1.584164/3.011236/6.743024 s include caller observer and concurrent
main; no incremental wall gain proved vs previous partial5126.570869 s.

Existing storage-effects asset rebuilt through OUR public entry and LLVM under
fingerprint619c4b55579500dfada7e845e9fab6bf. Source audit0 findings, latch OPEN,
1306 unsourced facts /93 identities,1690 rows/188 functions. No clean-book claim.
N4 ran from snapshot with this asset,1 graph worker/8 regions, no trace or
deadline, compile only; root build/profile_memory_n4_native_storage_mo1powht.
Probe's external-example option uses original laws while explicit compiler path
confirms snapshot. Unmodified full-main PID7196 still running concurrently,
last read22716.53 CPU seconds/1262.6 MiB. No main SSA result or N4 result yet.
Refreshed serial_graph_scaling.patch passes apply-check, NOT applied live.
Mutable-fold restarts/global scans remain future work, not silently claimed fixed.

Completed N4 exit0,554.948945900018 s (9m14.95), prior486.379789700004 s:
68.569156 s /14.097863% SLOWER. Identical1-worker/8-region settings, no trace,
no deadline, four stale-law rebuilds in both. Snapshot also includes staged
native-link fix plus all graph cuts, and live main overlaps: no isolated causal
claim.17 native calls/143043 value facts cost0.646459 s including transfers.
Combined begin165.077906, begin->SSA355.558314, combined whole389.868437 s.
Old begin->SSA296.812716 and combined326.696119: most extra time is in the
combined source->SSA phase.996 functions, max printed stage6460 MiB vs6437,
different progress-sampled peak6180 MiB; book1590383 resident cells,0 spills.
No four-law wall or memory gain. C SHAea56fd8317c754affc0b502870d15262ad67a2ea209ea58c4ae00b7f924b5bb7
differs: raw diff4790 lines replaced, many value-number changes but storage/call
ordering also differs. Identifier normalization is NOT a parity proof; no orbital
execution. Keep exact arithmetic LLVM proof separate from this large target.
No live promotion. Next tiny probes must attribute combined planning/instantiation
and SSA work before another long run, not optimize the sub-second native unit.
Full-main final read23235 CPU seconds/1260.4 MiB, still active; session poll has
no new output. Existing no-deadline job and its sources/assets remain untouched.
Guestbook validator dry-run exit0, filenames conform; no --apply cleanup.

### Serial graph-operator asymptotic reductions (staged; raw-speed priority)

Worked in existing current dirty-source snapshot while main source pursuit stays
untouched. No packages/other source compilers. Four existing mechanisms changed:
dirty-round propagation over exact parent records and loop-target consumer edges;
keyed region-member token projection; first-read-only ABI publication in plan
freezing; publication batches invalidated by literal/operand removal/alias writes
and every structural-fold sweep. Descriptor queries remain individually fresh;
actual shape-book publication remains enabled. Transfer and round order preserved.
Synthetic reverse chain O(N^2) ->O(N), with exact original transfer-code oracle.
Token projection O(regions*all tokens) ->O(total members), with identity tests.
Real48-add public entry observed388 ABI receipt checks between0 operand writes;
after batching <=4 per fold. Frozen plan49 ->1. No whole-compiler O(N) claim.

94 tests passed /44.29 s; four dispatcher failures deselected ONLY after all four
reproduced live before these edits. Two hierarchy failures and one formal-history
failure likewise reproduce live. Original seven-case audit findings0,1,0,1,5,0,0;
unsourced latch remains OPEN. Worklist48 cases include40 mixed-rule graphs,
loop-target side writes and exact final span shape-book facts.

Real arithmetic128/256/512 lowerings old/new seconds:
1.994243/1.911884;3.314352/2.910634;8.141574/6.570869.
Single runs, full-main job concurrently active, baseline includes an extra cached
batch-first receipt; NOT controlled four-law timing. At512 receipt calls47736 ->
1582, summed graph sizes60064147 ->1634445, publication1.256325 ->.090277 s.
All three corresponding LLVM emissions byte-identical; zero audit findings.
DT exact native outputs0.9/.25 pass; cold lowering3.806376/3.780554 s is essentially
unchanged, although receipt calls15922 ->1372 and size-work1780234 ->54288.
Cold DT LLVM hash agrees; repeated same-process second emission differs old/new
while third agrees, cause UNKNOWN. A double descriptor-read test also exposed
loss of the first returned python_type after publishing its shape, identically
under original fresh-publication policy; do not mistake non-idempotent observation
for proof of batching regression or relax the descriptor rules to pass that test.
Exact fresh-query parity assertions remain on settled plan reads; mutable-fold
checks use operand-write bounds, existing descriptor-rule gate, emitted LLVM and
actual native behavior instead. No universal numerical-parity claim.

Promotion patch turing/tools/compiler_probes/serial_graph_scaling.patch passes
git apply --check; independent of prior native-fallback-link patch. Live compiler
and installed native asset UNCHANGED while main job resolves source closure.
Current worker PID7196 ~1052 MiB/4406 CPU seconds; no main SSA result yet.
Before benchmarking N4, promote after main exits and regenerate native storage
asset under existing toolchain fingerprint. Do not kill/change source underneath
the running job. Further raw-speed work: remaining quadratic receipt checks,
fold restart scans and repeated class AST/source inference, not another scheduler.

### User-authorized N4 native-analysis timing (COMPLETED; raw-speed priority)

After read-only applicability check, user said "let's try it out". Installed
fused storage-effect/pointer unit really loads without rebuilding and the normal
physical-call adaptation selects it. N4 binds direct pieces already, so the
staged authored-fallback link fix is NOT claimed as this benchmark's gain.
Added only diagnostic tools/compiler_probes/probe_four_rule_timing.py: reuses
existing profile_memory_n2.main with four laws, explicit serial source route,
fresh TEMP/TMP, no trace/deadline, native-call counts/timing, inline stage prints.
No compiler/runtime/source edits, staged patch remains unapplied.

First run preserved historical graph_workers=4/resident_regions=8 but FAILED
before first law completed. Exact stack ensure_graph_foundation -> native
dependency_levels builder -> public lowerer -> _class_surface_ssa_program;
recorded name_arm_version_missing for arm63/initial21 and34/initial11.
This is a known new graph-foundation bootstrap regression, not a benchmark
completion; no semantic repair or silent fallback performed.

Explicit retry command:
python -u tools/compiler_probes/probe_four_rule_timing.py --graph-workers 1
Output root build/profile_memory_n4_native_storage_xl7u1you/profile_memory_n2.
Existing full-main self-lowering still runs concurrently. No isolated or direct
accelerator-only comparison with old 599.448 s (4 workers) is justified. Per-AST
default prints are reduced by an in-process public-entry progress observer;
real lowering and existing profiler histogram/book/RSS work are unchanged.
First actual native-analysis reuse:594 value facts /1.4493 ms with transfers.
Finished exit0 in486.37978970000404 s, all four stale pieces rebuilt.
17 actual native-analysis calls,143043 value facts,0.5100498000101652 s total
INCLUDING transfers. Combined begin159.68168889998924 s; combined begin->SSA
296.81271570001263 s; combined whole326.6961186999979 s. Module996 functions.
Latest599.448 s ->486.379790 s =113.068210 s/18.8620% lower observed wall.
Still21.839790 s/4.7014% SLOWER than older464.540 s. Not an isolated or causal
accelerator-only comparison: graph workers1 vs4, main job concurrent, reduced
per-node logging, while old and new both rebuilt all four laws. No timeout/trace.
Printed-stage max6437 MiB (old5606), progress-sampled peak6255 MiB (different
sampling); book1623974 resident cells/0 spill pages. No memory gain. C SHA
561cc7e1061aa4f454206e5d0db3308cd910d346ef5c7ae37abf9ae59aae910b.
Compiled system NOT executed; no numerical-parity claim.

During the run user explicitly replaced architectural prerequisites with RAW
SPEED priority. The next native conversion should target measured substantive
serial planning/refinement/SSA passes, not wait for parallelism/memory tiling.
Full-main source self-lowering remains untouched and running; staged native-link
patch still unapplied. Record its eventual outcome separately.

### Native-reference / authored-fallback identity boundary (isolated snapshot)

User asks that accepted native references link rather than ingest their optional
Python fallback, using existing introspective graph/shape systems. No-deadline
full main job remains untouched. Created an attached root-repo worktree and
copied current dirty turing src/contracts/tests/tools into its opaque turing/
child (no force-add, no root ignore change). This is a CURRENT snapshot, not a
clean baseline. Path: C:/Users/alber/.codex/worktrees/native-fallback-link/
Powershell/turing. Main authored source normalized hash still
ef4ba61f2107de15b3ec51a9aa7a0f6fdf4b035fc70cf98577bf4b24f7d74036.

Fail-first: existing installer retains exact LLVMPiece via deployed metadata,
but llvm_piece_of(wrapper) returned None. Actual public lowering trapped at
graph_express2._expand_unresolved_ast_parents -> ExtractionContract.decide ->
subject -> inspect.getsource(installed). Detector now follows only existing
__turing_deployed_callable__ references, rejects cycles/unrelated decorators,
and applies unchanged artifact/ABI validation. No graph copy or native-flag guess.

Follow-on fail-first: external declaration ignored canonical piece and called
the public compiler to generate a fresh signature. Retained mapping/sequence
factory gates read wrapper.module and published zero piece bindings. Consumers
now use canonical native identities; eager external implementation still owns
the optional source path. Native slots resolve exact piece entry address without
Python callback; IDs, buffer order and outputs unchanged.

Explicit rebuilding regression observed auto-link writer tagging submitted
_authored_leaf AST with old bodyless module. Deployment returned None and tail
failed on module.metadata. Existing selected/whole-source election now removes
that root's native binding from dependency auto-linking; other dependencies still
link. Authored rebuilding and targeted eager source authority both tested.

14 new regressions pass /12.30 s; tiny caller direct/wrapped lowerings
.549430/.540208 s, two functions each (one bodyless), zero fallback source reads,
zero concordance findings, native outputs [5,1,13.25] exactly. These are boundary
timings, NOT whole-compiler acceleration. Retained-container checks stop at the
source graph boundary, not full record execution. Existing retention suite 6/15.73s.
Final combined native-boundary/source-realization/bootstrap/extraction/retention
gate: 59 passed /46.25 s. Command in snapshot:
python -u -m pytest tests/test_native_fallback_link_boundary.py tests/test_source_realization.py tests/test_compiler_bootstrap_runtime.py tests/test_extraction_contract.py tests/test_llvm_piece_runtime_payload.py -q --tb=short
Promotion patch saved in live turing/tools/compiler_probes/
native_fallback_link_boundary.patch; read-only git apply --check passes.
Apply only after full main finishes/stops; then regenerate native storage-effects
asset normally because compiler fingerprint changes. No stale acceptance bypass.

Generic installed verified runtime callables do not carry a raw piece ABI;
especially the storage-effects provider takes Python SSA objects and marshals
columns. This fix MUST NOT label those wrappers native link leaves. Reuse
existing accepted native-product SSA/API interfaces for that later bridge.
The larger next target remains original structural refinement/descriptor work.
Initial isolated tensor import automatically built the existing CFFI bridge;
not an authored test-program compilation or new compiler/package selection.
Later gates set NODUS_TENSOR_CORE to the existing live native library and reuse
cached bridge state. All test programs compile through OUR public source entry.

### Entire public main compiler submitted to its own lowerer

User then explicitly rejected the timeout. Removed subprocess timeout AND
the repeating faulthandler timeout-labelled dumps, and relaunched the same
complete source/public entry with no deadline. Do not classify the prior
timeouts as semantic blockers or stop this run on an inferred time budget.

User: "put the entire main compile in to lower and report on the blocker".
Added diagnostic tools/compiler_probes/probe_main_compiler_lowering.py only;
no compiler changes. It submits unchanged full45,000-line fortran_c_shell.py,
entry lower_ast_source_to_ssa, actual module globals, existing program_extraction
contract + full native/no-Python-callback execution overlay. Hash
ef4ba61f2107de15b3ec51a9aa7a0f6fdf4b035fc70cf98577bf4b24f7d74036.
Neither an extracted helper nor a rewritten entry. Target program never runs.

45-second initial and90-second observed children hit diagnostic deadlines in
source closure before SSA. No semantic refusal was reached; language/ABI
blocker still UNKNOWN. The90-second run discovered main impl at work_item33,
ProcessGraph at438 and link_process_graph_functions at443. At25/50 seconds,
both sampled stacks reached graph_express2._source_ast_definition:788
deepcopy(cached) beneath class-field/constructor inference and dependency
pursuit (_expand_unresolved_ast_parents), public wrapper44277 / impl42848.
At75 seconds it was traversing AST children. Observed performance obstacle,
NOT proof of an infinite loop, total cost attribution or inability to compile.
Probe progress uses sys.__stdout__ to escape discovery's internal stdout
capture; stacks emitted inline every25 seconds, no allocation tracing.
No main SSA/native artifact, no main audit success claim. Parent terminates
its own diagnostic child only. Existing installed native analyses untouched.

### Fused native effects AND pointer inference — latest

On "let's proceed", fused original pointer-storage inference into the whole
module native effects program. Same authored ~230-line Python body compiles
through OUR public entry to LLVM/DLL and executes eagerly for exact parity.
Both full fixed points, CSR construction, conflict joins, queues and final
tables are native; no callback/per-value native call. Host marshals existing
SSA facts and publishes final accounting. Pointer closure excludes View/call
edges; exact occurrence Load/Store dtype evidence, incumbents and imported
header semantics preserved. Scratch reused, duplicate Python zero lists removed.
Installed immutable matching product under turing/build/compiler_foundations/
storage_effects/products/e2727d33f52be72e48858609d2d2a5ff/, same checkpoint,
fingerprint/digest load-only route. No other source compiler/packages.

Gates: 4 native/audit pass /32.59 s; 17 branch/adapter pass /17.66 s.
62 exact SSA module cases /61 native calls; full formal accounting exact on27
pointer modules. Fresh-process real serial lowering reuses3 whole-module
invocations/2090 facts, binary unchanged .9/.25. No N4 launched.
SOLO combined-section medians include packing/unpacking/publication:
25 values .059400/.134200 ms Python/native (tiny loses); 648 values
10.843000/1.061175 ms; 5400 values/384 queries/5040 instructions/9970 operands
471.711250/8.369925 ms (56.358x), saves463.341325 ms. Separate original Python
stages: pointer inference14.593350 ms, read-only454.328100 ms. 56x is COMBINED
SECTION gain, not incremental pointer-only or whole compiler. Cold load .070959 s.
Repro tools/compiler_probes/probe_storage_effect_timing.py; 35-second child
deadline, public compilation forbidden during timing, no DT source in timing.

Trapped before fixes: true-only store-version462 bound Phi319 thenPhi321;
parent false edge emitted Phi322(321,321). Existing exact SSA/CFG dominance now
gates carried-arm residency; concorded storage aliases still select snapshot.
name_binding assigns source366 to incumbent/proposed, but merge374 only proposed.
lower_conditional->_rebind_view->_bind republished366 as374, making the change
predicate compare374 with itself. Existing binding facts preserve other names'
sources. Tiny native nested-branch regression trapped the later continuation
writer doing the same0->Phi18 substitution AFTER correct lowering; it honors
the same preservation receipt. Audit checks against the exact declared
scope_rows, including copy-on-read scope_origin projections (rows() omits those).
Observers build/observe_storage_phi.py and build/observe_shared_conditional.py.
0 structural findings; unsourced latch OPEN. Deprecated whole-object conditional
test also fails unchanged onC:/dev/tg-base in2.66 s; excluded from final green
gate, not hidden. Full larger-source baseline refuses earlier, so Phi age unknown.

User then asks whether the really long MAIN compile can receive native treatment.
It can at substantive whole graph-pass boundaries; the current DLL is only the
two analyses, not native main refinement/planning. Prior short-slice folding/
descriptor CPU profile is historical/overlapping and does not establish fresh
N4 attribution. Main parallelism and full memory tiling remain unfinished.

### Substantial whole-module native compiler program — historical before fusion

The user rejected the Python publication boundary of dead-metadata work and
asked for a wide compiled piece immediately. Implemented an entire storage-effect
analysis program in graph_refinement_kernels.read_only_storage_effects: edge
construction, reverse CSR, root-definition vs address-write states, recursive
call/alias closure, work queue and final table writes are ALL native, no Python
callbacks/per-value native calls. Existing SSA input roles and accepted physical
call contracts are marshalled, not inferred. The physical-call adaptation round
consumes one complete native table. Original eager _read_only_feed remains.

OUR public compiler accepted the unchanged ~120-line program after a trapped
conditional binding fault was fixed; 892 SSA rows/99 functions incl regions,
0 structural findings. Native correctness: 35 SSA modules, 34 calls (one empty),
every value exactly agrees with old recursive analysis. Pure/impure recursive
SCCs, alias definition vs base effects, Store value/index, imported effect facts
and exact callee formal identities included. Latest gate 2 pass/26.41 s.
Fresh-process load-only test then exercises actual serial lowering: three
whole-module invocations/2090 facts; emitted program exact .9/.25. DT code is
only an integration input, NOT what was compiled into the compiler component.

Explicit builder persists through existing ArtifactCheckpointStore; compiler,
authored/host source and platform fingerprint select immutable product folders,
DLL digest validates before loading. Default asset installed under
turing/build/compiler_foundations/storage_effects/products/6d37c031b07e725a640a6ef274295df1/.
Normal adaptation loads it, not builds it. Missing/stale assets use existing
Python. Fingerprinted edits require explicit regeneration. No packages or other
source compiler. Structural audit unsourced latch remains OPEN, not fully closed.

Section-only load benchmark: public lowering forbidden, no DT source; fresh
lookup/DLL load .068812 s separately. Five-sample medians, packing/full result
unpacking included. Original Python timed on actual formal-query usage; native
returns ALL value facts in ONE invocation. 1 function/4 queries/21 values:
.03475/.09485 ms Python/native (tiny loss). 8 functions/64 queries/584 values:
8.78095/.95705 ms, 9.175x. 24 functions/384 queries/5016 values, 4656 instructions,
9586 operands: 434.70860/7.904625 ms, 54.994x, saves 426.80398 ms. Synthetic SSA
call chains, NOT end-to-end/N4 results. Repro: tools/compiler_probes/probe_storage_effect_timing.py,
35-second parent, finished in 8.97 s. No process left running.

Observed compiler writer: nested child lower_conditional->_rebind_view->_bind
changed arm 215/216 from loop-entry Phis to child Phi145/Phi169; parent false
edge then consumed child result. Three nested guards in the real authored
program exposed LLVM dominance errors. Existing snapshots covered source
initials 37/38, not already-bound loop arm versions. Branch-entry snapshots now
include all already-bound carried arm IDs via the existing snapshot pages/writer;
future local definitions are not materialized. Diagnostic observer:
turing/build/observe_storage_phi.py, prints exact old/new bindings/stacks and
Phi arguments. Clean baseline refuses this larger source earlier at overlapping
control regions (build/baseline_storage_effect_compile.py), so did NOT prove the
Phi defect's age. No claim of a baseline-green reproduction.

Earlier dead-metadata extension remains saved, not the active answer: one native
worklist call per graph, 81 cases/80 calls and 26 real energy passes match exact
graphs/book rows in 25.24 s. Its final graph/book mutation remained Python; user
rejected that boundary. The storage-effect stage has no such Python continuation
of its analysis worklist. Scratch currently includes redundant Python zero-list
allocation before NumPy conversion; next improve that plus branch/loop gates,
then normal compilation timings. Full memory tiling/main parallel lowering remain
unfinished and must not be represented as achieved by this native section.

### First serial native unit implemented and measured — 2026-10-09

On "Proceed", extracted only the descriptor's exact binary broadcast integer
law into graph_refinement_kernels.broadcast_shape_extents, plus its independent
row batch entry broadcast_shapes. SAME authored body runs eagerly and through
OUR public Python-to-SSA-to-LLVM compiler. Existing native_compiler_accelerators
registry owns explicit compilation/registration. Serial _tensor_descriptor_rule
now consults this provider after all existing admission/unknown/conflict guards;
dtype/Precision, source identity and record_shape_transformation stay with the
original graph owner. No package, alternate source compiler or DT/physics change.

Native child, bounded 15 seconds, forced two overlapping LLVMExecution calls:
407 exact rows agree with the independent NumPy oracle/explicit wide-integer
cases. Scalar, zero vs one, different ranks and incompatibility included. Feeds
and scratch are per invocation. Inadmissible integer ABI inputs use SAME eager
law before execution: non-integer, 2**63, 2**64-1, 2**100. No native exception
is suppressed. Descriptor unknown-call guard and original publication passed.

Actual serial DT proposal probe (55-second outer, existing 10-second native
children): Python/native/Python each return exactly 0.9, 0.25 and 0 findings.
Each serial lowering makes 53 admitted broadcast decisions. Python decision
TOTAL .175/.170 ms vs native 4.326 ms. Public lowering 3.765/3.812/3.907 s;
whole emit/link/two-child check 11.131/10.970/11.104 s. No end-to-end speedup.
The earlier 1.378 s descriptor profile does NOT mean its extent comparisons
cost 1.378 s. This observation sharply narrows the expected benefit.

Packing was measured, not excluded: removed redundant Python int conversions
on input and per-NumPy-scalar output conversion. Final SOLO batch: 4096 rows
rank8 7.504 ms eager/6.753 ms native, rank32 19.926/17.303 ms. Single rank2
1.435 us/36.367 us; 256 rank2 .271/.305 ms. Other runs vary; one benchmark
under concurrent compilation was deliberately not treated as a clean timing.
Provider remains explicit/optional; ordinary serial stays eager unless
registered. No automatic native build, main compile gain or N4 justification.

Final selection: 16 pass, 1 deselected in 12.58 s, including new native gates,
callsite shapes, earlier topology compilation and both DT route selectors.
Deselected existing_call_projection_is_enriched_by_later_exact_descriptor
fails unchanged on clean C:/dev/tg-base: page history () instead of two rows,
0.60 s. No unrelated fix. Structural concordance findings zero, unsourced
latch still OPEN (kernel: 179 facts/8 IDs; real serial slice: 888 facts/0 IDs).
Syntax/whitespace checks passed. No long four-law run or process left running.

Fresh CPU-only observer: 20,292,830 calls/7.850 s instrumented. Fold 165 calls
1.901 s cumulative/.083 self; descriptor 18,627 calls1.447/.309; shape fact
publication .314 s, ABI graph identity publication .199 s, book post .903 s.
Times overlap. Next is an actually substantial existing structural/dataflow
island with exact graph/book inputs, not another tiny scalar law. Detailed
roadmap/repro is todo/turing_graph_residency_workers.stub.md. Probe source:
turing/tools/compiler_probes/probe_native_shape_transfer.py; real artifacts:
turing/build/native-shape-transfer-checkpoint; final kernel/ABI check:
turing/build/native-shape-transfer-solo. Further architectural tiling parked.

### Current direction: preserve both routes; identify native serial units

The user redirected work after this checkpoint. Both routes remain available;
DT's lowered_system now has source_compilation="serial" or "units", leaving
the previously enabled unit default unchanged. Serial bypasses source-worker
options, not the public compiler. No broad revert or replacement compiler.
The active roadmap is todo/turing_graph_residency_workers.stub.md; earlier
sections below are historical and do not override this change in direction.

New architectural checkpoint is limited but executable: canonical region
expansions and their planning copies share one existing BookSpill-backed cache,
default 8. They reload the SAME identities rather than rerunning a minting
lowerer. The graph partitioner reads graph_region_node_limit, default 256.
Native two-chain conditional micro: two main lowering processes, four regions
per chain at graph bound 8, instruction cache bound 1, exact 64701152 and -4,
19.48 s. This does not release final SSA bodies, tile source ingress, or
subdivide oversized source SCCs; those follow-ups are parked, not completed.

Actual unchanged DT proposal now executes through four source units with
accepted dependency headers: 84/24/135/10 nodes, two workers, exact 0.9 and
0.25. Clean serial compiler reproduces both. Unit/native 25.943 s vs clean
serial/native 10.801 s: no performance claim. Artifact receipts are in
turing/build/dt-proposal-native-owned-q7cwmwj3 and the clean control in
turing/build/dt-proposal-native-baseline-24cl0x44. Zero structural findings
is not a closed unsourced latch; the latter remains open.

Observed writer chain: existing planner specializations were delayed until
after cutting away literal callers. That existing operator now runs before
source-unit planning. Imported _scalar's argument was removed at the literal
formal prune (~29690); imported _propose_dt_pen record formal 3 was removed
at three separate body-liveness prunes (~26613/~30046/~37391). Every writer
was trapped with stack/old/new arguments. Their bodyless-import cases now
respect the accepted ABI. Native source argument roles bind surviving exact
formals after specialization, never the former positional signature.

Checks: 7 cache/literal/specialization tests (28.31 s), bounded two-chain native
test (19.48 s), 10 cache/audit/while/count/carried tests under 8/1 (16.45 s),
3 record/default/nested return regressions (48.43 s). Final route/dispatch/
hierarchy gate: 33 pass, 4 fail (4.81 s). All four failures also reproduce
on untouched C:/dev/tg-base (3.12 s), including three hand-built graph tests
missing ingestion scope. Do not weaken source identity to make them pass.

Two bounded CPU-only serial observations informed native target selection;
quiet run was 20,293,257 calls / 7.694 s instrumented. Structural folding:
165 calls / 1.852 s cumulative; tensor descriptor queries: 18,627 / 1.378 s;
call metadata: 0.439 s; result type solver: 0.098 s. Costs overlap. Generic
metadata isolation (575,688 / 0.879 s), book.post (0.889 s), I/O and GC cannot
be attributed to a prospective native shape kernel. SCC (0.000079 s), effect
walk (0.005353 s), levels (0.017685 s) are not first priorities on this slice.
At that earlier selection checkpoint no new native conversion was implemented.
The latest implementation and narrowed cost evidence are above. Begin with bulk existing integer
descriptor/structural transfer decisions, preserving Python graph/book owners,
through the existing registry and OUR public compiler; eager is the same code.
Repro observer turing/build/profile_serial_dt_slice.py is 35-second bounded.

Final isolated route gate passed all 3 in 1.49 s. Syntax, whitespace and
guestbook filename validation passed (dry run); no Python probe remains.

Prompt History:

> "pick something wide and compile it now"

> "okay what is the time different between using the ordinary python vs using that precompiled native section instead"

> "Proceed"

> 1. I don't care about that there are no other people using this code and the compiler has to be able to run more efficiently, full stop. leave it enabled because we have to fix it
> 2. yes that's kind of what I thought you were going to do
> 3. similarly
> 4. yes

> let's leave a roadmap, leave what we've done, while keeping what we had, the faster serial route, we keep both options around and we switch our attention now to identifying parts of the serial reliable process that can be converted to native compiled units

### Latest checkpoint: deterministic graph cuts and exact identities

Observed a silent native miscompile: root(Actual)->relay(Declared)->leaf(Declared)
returned 16, omitting the record contribution; clean baseline native returned
59. The unit planner now closes identity-changing record bindings over exact
forwarding parameter edges and supplies indivisible edges to the existing SCC
operator. Reverse planning edges cause deterministic convex contraction;
semantic call edges, recursion and source identities are unchanged. Two scalar
branches remain separate lowerings under the two-process scheduler. Oversized
components still refuse before serialization/admission, not expand into hidden
dependency closures. This is NOT a replacement graph/compiler system.

Two producer faults became visible at that legal seam. Record-only variants
were omitted from the existing specialization symbol key; their existing
record ABI/variant scope now supplies it. The bytes-join child loop overwrote
the enclosing source function_reference before SSA metadata publication. That
writer now uses a separate child variable. Native component imports join on
the accepted function reference, not export position or parsed names. Generic
entries are headers; interior specialized bodies stay in the native object.
Exact native 59 now passes both directly and through a later importing caller;
structural concordance is checked, not interpreted as a fully closed unsourced
latch. The later-caller proof completed in 20.83 s before final regression.

The bounded real DT proposal now cuts 84/24/135/10 source nodes. The record
forwarding error no longer occurs. Last probe remains partial at source call 8
and untyped distribution formal 0. Do not call this full DT/native correctness,
full out-of-core compilation or an N4 improvement. No long benchmark launched.
The latest serial real DT proposal still lowers in 3.16 s vs baseline 2.77 s;
these one-off lowering times are not a speedup proof. The existing default DT
unit route remains premature/unverified; no broad revert was performed.

Two old compilation-unit tests fail with book lifetime/function-address
disagreements on the clean baseline too; the new convex-component constraint
test passes. One combined regression previously exceeded an outer 80 s limit;
it was not reported as passing. Final focused receipts: 3 serial/record/import
cases passed in 45.52 s; 4 tuple/tensor/law cases in 58.68 s; 5
record/default/convexity/node-cap/acceptance checks in 28.80 s. All 12 passed;
syntax/whitespace checks and guestbook filename validator passed (dry run).

Prompt History:

> okay your choices were invalid if you did not choose to act in a way that USED GRAPH THEORY, -SLUR- TO MAKE DETERMINISTIC ADJUSTMENTS. TO ACHIEVE KNOWN COMPILER THEORY OUTCOMES. OF WHICH WE HAVE SOME MYSTERIOUS "PROBLEMS" WHICH MEAN YOU DIDN'T DO THAT

### Latest checkpoint: actual lowering boundaries, not self-hosting

The user explicitly prioritized removing serial main compilation over more
self-hosted kernels. The normal linked C/LLVM `lowered_system` invocation now
requests the existing bounded source-unit scheduler (fresh directory, at least
two jobs); its actual `dt_system_over` routing has a plumbing test. Full N4 has
NOT been rerun or established compatible with all remaining boundary types.

Existing record descriptor rows now cross the interface; their field demands
feed the existing forwarding graph. Before that, a record caller allocated
uninitialized private scratch and returned garbage. Fixed at the demand writer,
not with a backend workaround. Exact linked native record output is 13. Shared
authored/native result projection handles flat/nested tuples (105/189 exact).
Accepted prelinked laws retain their native implementation (42 exact); empty
declarations are not recompiled as source bodies.

An actual small DT helper compile revealed the earlier cuts were incomplete:
an authored ReturnBlock kept an ingestion Ref whose cell was not in the worker.
The user correctly called this a serious architectural boundary bug. Now the
book's existing Ref/edge graph is restricted to that unit's required source
cells and predecessor closure, preserving facts, stamps, mint/edge ordering and
unsourced receipts. The complete diagnostic book remains stowed separately.
Persisted source archives also retain an acceptance-only view for later cuts;
no new identity/schema or alternative compiler was introduced.

The bounded actual DT slice (`probe_dt_unit_interfaces.py`) compiles unchanged
`_energy_time_limit`, `_no_exchange_observed`, `_scalar` plus a small root:
four units, two concurrent processes, selected node counts 84/24/34/16,
acceptance cells 15/15/10/5, no dependency bodies. Linked native output is exactly
0.0 under the one-participant extent contract; zero concordance findings;
20.464 s including link and isolated execution. This only covers the resulting
early returns, not all DT behavior. Full ingress remains whole-graph; fixed
tensor basis is still loaded by each worker; unsupported aggregate/runtime/SCC
cuts refuse. Do not describe the full architecture as finished or claim speedup.

Regression receipt: 119 passed in 140.62 s. A final tensor-element boundary
micro caught native vector indexing being excluded as tuple projection; the
accepted return kind/arity now distinguishes ownership. Six final main boundary
checks passed in 72.41 s, including exact scalar 43 from a vector element.
Syntax/whitespace checks pass; guestbook filename validator passes (no --apply).

Prompt history for this checkpoint:

> The primary task is unfuck the serial nature of the compiling, the first goal is not self hosting the first rule is unfuck the architecture

> that sounds like a MASSIVE bug that means you concretely did not follow an industry practice but maybe laid some early stabs at what you're supposed to be doing

### Latest checkpoint: applicable N4 change, measured partial improvement

After verifying that the opt-in source-unit tiling lane did not reach the
four-law benchmark, added an explicit compact import for already compiled
LLVMPiece links at the existing source-SSA import seam. Link mode now carries
one copied root declaration and accepted ABI/effect receipt, not each law's
private SSA functions. Inline/default source-compiler behavior retains bodies;
the original full archives remain intact. The concordance audits the new
linked-piece interface schema. LLVM arithmetic and C direct buffer publication
micros pass; a separate C unpacked-result-addition publication failure was
confirmed unchanged on clean 54c67fbd. Focused retention/accelerator tests:
13 passed in 35.75 s.

User-authorized untraced N4 completed in 599.448 s, versus 697.227 s previously;
four pieces rebuilt in both. Combined begin-to-SSA 404.3 vs 433.3 s; functions
996 vs 1066. Printed stage RSS maximum 5606 vs 5632 MiB, effectively unchanged.
Book: 1,828,426 resident cells, zero spilled pages; CPU approximately one core.
Earlier 464.540 s remains faster. This is a partial measured improvement, NOT
the requested bounded-memory/multicore orbital architecture. Compilation does
not establish orbital numerical parity. Artifact path and SHA are recorded in
turing/TEST_BASELINE_AND_HAZARDS.md. No compiler source was changed during N4.

While N4 compiled, built a bounded eager/native experiment around the existing
authored Python dependency_levels function. It can execute eagerly, reuse a
registered/explicit trusted native artifact, or compile through the existing
public source compiler API. New builds save the existing native-worker payload
format with source/payload/library digest receipts. Two workers have owned CSR
and scratch spans, with no shared graph/book. Tests include disconnected sinks,
ALAP slack, exact results, optional compilation/reuse and stale-source refusal:
7 passed in 12.78 s. At 4096 nodes/8186 edges, 16 repeats per preference, kernel
sum 138.34 ms eager vs 0.8374 ms native; wall 179.90 vs 12.53 ms. Two-thread
wall 13.52 ms, not faster at this small size. This is a usable same-authored-code
test case, not a new compiler implementation or production-wide speed claim.
Production eager registry fallback still uses ILPScheduler, not the probe's
direct authored-kernel call. No packages or alternate source compilers.

Prompt history for this checkpoint:

> can we now profile the 4 law time if and only if we made speed related changes that are expected to have landed, otherwise focus to that goal to land some improvements and begin profiling, and while that compiles we would focus our efforts entirely on finding an ideal test case that we don't need to make perfect, we just have the opportunity to use a native version of something if it's available, compile if that's desired, or run eager in python, the same python that gets compiled

> Can you do anything at all on task in any way, is it mathematically impossible for you to do anything I've asked you to do?

Work in `C:/dev/Powershell/turing`, dirty main `54c67fbd`, continues the prior
self-hosted graph-kernel checkpoint. The user requested a larger connected
feature inclusion before detailed debugging, then asked to increase worker
capacity and test larger meaningful workloads. No subagents, packages, alternate
source implementations or long orbital compilations were used.

## Steps Taken

Integrated the real dispatch-carving path with pre-carve residency reservations,
a compile-wide bounded topology cache, existing BookSpill segment storage,
dependency-only snapshots, bounded worker admission and ordered coordinator
publication. Both ordering and the existing ILPScheduler equations now have
authored integer-span kernels compiled through `lower_ast_source_to_ssa` and
the existing LLVM artifact route. Worker pools persist across a compile's shells;
foundation build directories cannot collide across compiler processes.

ProcessGraph/NetworkX remain the structures consumers read. All three region
collections share lightweight headers, and deep compiler/ephemeral constructors
are deferred until needed. Read containers cannot write old payloads; mutable
access requires a generation-checked write lease. Spill/reload preserves AST
identity, nested container cycles and current-generation aliases. Eager wrapping
was removed from regions that the source-SSA path never rereads.

New `graph_work_publication` rows derive from exact region, resource policy,
carved read-scope and scope-origin cells. The concordance checks their schema,
method/policy agreement and provenance. Lifecycle is declared through repairs.
Work/residency/spill/worker/timing counters are attached to the resulting module.

## Observed Behaviour

Public-entry 4/8/16-call programs passed exact native output under 2/4/8 worker
caps. A 32-call/8-cap probe lowered in 4.0298 seconds, generated 35 functions
and 34 publications, spilled 26 topologies, and never exceeded 8 cached plus
in-flight topology units. Three worker threads participated. A later
16-call/4-cap probe lowered in 1.7316 seconds. Neither is an orbital benchmark.

Sixteen 4,096-node/11,904-edge DAG units, with the SAME 8 resident slots and
8 spills at each point, matched the existing ASAP/ALAP scheduler exactly:
1-inline 2.790 s; 2-cap 2.726 s; 4-cap 2.724 s; 8-cap 2.661 s. Four threads
participated in both the 4-cap and 8-cap runs. These are single-sample smoke
timings, not evidence of broad speedup. The cost is mostly coordinator packing,
validation and spilling. The corresponding 1,024-node units took
0.763/0.692/0.626/0.618 seconds. More thread capacity alone is not the big win.

Tests cover exact emitted LLVM alpha-equivalence between independent books,
native kernel concurrency, bounded completed-result admission, stale proposals,
generation/alias/cycle/set safety, lazy constructors and negative concordance.
The current focused receipts and exact commands are in
`turing/TEST_BASELINE_AND_HAZARDS.md`. The final graph-work/spill/memory batch
passed 71 in 17.69 seconds, including 21 graph-work tests. Broader LLVM/layout/dispatch tests retain
the same six failures as the clean short baseline (`C:/dev/tg-base`); the
working checkout has five additional passing regression tests. Existing dirty
orbital probes, helper artifacts and deleted build README were preserved.

## Lessons Learned

The source-SSA lane reads region headers, not every carved topology. Forcing
attribute wrappers or rereads there adds unnecessary work. A bounded queue must
include completed results and in-flight full graphs, not just worker count.
An old mutable view must be rejected even if a newer generation has an active
lease. Same-generation container aliases must be retained, not copied again.
Thread count is a cap, and participating threads are measured separately.

This is still NOT the full requested conversion: source graphs, metadata/leaves,
catalogue and whole-module SSA remain resident, and expensive refinement is
mostly serial. The transient topology store is not a restartable checkpoint.
The first levels kernel returned a conditionally used arena and hit a dominance
finding; the final ABI publishes its output span and returns processed count.
The general arena-return compiler issue remains open, with no weakened detector.

## Next Steps

Extend existing graph/provenance ownership to source and SSA units, and address
the measured coordinator costs rather than repeatedly increasing thread caps.
Keep the conditional arena-return issue separate. Continue exact bounded micros;
the user launches long orbital tests. Updated stub:
`speaktome/todo/turing_graph_residency_workers.stub.md`.

## Prompt History

User: "finding lost things was previously done by making everything use the concordance, shouldn't be abandoned"

User: "do you have a plan to fix this?"

User: "paired generator row? so like we can do generators but we can't do them with arbitrary nesting?"

User: "proceed"

User: "this feels like picking up grains of sand, can we pile a few more projected problems onto a heap of changes"

User: "so do we have a limited set of unhandled cases? purely unresolveds?"

User: "can you make that change, test it on a small repro to iterate to success, then try the copier lowering again"

User: "slipping out of state is a sign of lack of concordance observance, not sure if that's relevant, but native compiled compiler portions are still responsible to communicate. again not sure if it's relevant. keep a sharp eye out for anything deterministically in our way"

User: "Can you please be more specific on the failure and how to fix it those feel like fundamental compiler errors"

User: "All of these sounds like fundamental failures can we clean them up to work professionally"

User: "let's do a 4 rule test"

User: "full-main can't be running unless you started a new one. i do see four instances of python, and one of them could be main, but who told you to do main and when"

User: "without abandoning or changing anything investigate how performance got worse and whether we can swallow up that badness with a native wrapper around whatever loop or recursion is costing so much"

User: "can we PLEASE make whatever is dispatching the 500k calls to be wrapped in native compilation fully"

User: "it wouldn't hurt to make the graph dependency cone a general helper, proceed"

User: "can you look into those 4 original failures and see if the tests are out of date or if they represent real shortfalls of the compiler"

User (follow-on after slower N4 result): "Proceed"

User: "Proceed"

User: "What are the next steps"

User: "proceed"

User: "we need to be cutting down the minutes long processes, we need to be making powers of O reductions"

User: "we can't accomplish our architectural goals, we're just looking for raw speedup now"

User: "do we have enough in place that will actually help the 4 rule compile run faster"

User: "let's try it out"

User: "while this runs, if it is still running, let's plan the next largest section we feel we can compile for if the whole thing doesn't work and can't immediately be fixed"

User: "let's make sure where the compiler is using native references but optionally python that the compiler is just accepting the link and not trying to compile the python, maybe? we have such a pattern elsewhere. let's see what we can do to move forward because regardless of whether we win big and the whole thing gets through compiling or not, we're going to need to get it to go faster, and faster, and refine more and more the ability to ocmpile in pieces, learn the procedure, and move through the pattern we wish to use for software development over and over again until we understand it completely, but without reinventing where we have substantial grpah based deeply introspective data and established shape patterns that follow best practices and first principles"

### Fused pointer-storage/native main-compile discussion

> put the entire main compile in to lower and report on the blocker

> can you not use that stupid fucking timeout

> is there any other segment, big segment, we can do like this before we try the 4 rule benchmark

> let's proceed

> can we talk about the really really long processes? the main compile? can that be put through the native treatment?

### Native interface/body boundary checkpoint

The canonical source compiler now has an opt-in partitioned-directory lane.
Planning is still canonical and whole-source; post-planning workers receive
only their SCC bodies. Native unit declarations preserve actual SSA identities,
buffer/extent ABI and read/write summaries. Caller planning now creates real
PlanCalls from those declarations without a callee shell, and shape discovery
reads accepted result descriptors without a callee graph. Empty declarations
must not have their formals pruned or be assumed pure. Final C/LLVM and the
existing standalone C host link compiled objects rather than reassembling
all tile SSA bodies. No source compiler substitution or package installation.
Final focused compiler/product/native-call regression: 88 passed in 76.87 s;
native child runs are 10-second bounded and tile worker runs are 30-second bounded.
Standalone executables also return the exact scalar/four-element outputs.

Tiny scalar and shaped-span programs prove actual caller/callee and three-level
calls with a two-process cap and no imported SSA blocks. Micro-tested failures
found two important hidden couplings: tensor outputs became scalar when shape
discovery required the missing graph; header unpickling accumulated table rows
onto the caller's active book. The native descriptor query and isolated header
book scopes address those at their owning seams. Complete source/unit books
remain stowed for diagnostics; native replacement installation stays verified-only.

This is NOT automatically wired into the orbital benchmark. Whole ingestion,
record/sequence layout interfaces, multi-result calls, dynamic extents and
oversized SCC subdivision remain. Unsupported interfaces refuse rather than
expand authored dependencies or silently approximate them. Detailed receipts
and current limits are at the top of TEST_BASELINE_AND_HAZARDS.md.

> can you please make the biggest most sweeping steps we can make to get there to that goal where things actually start to change

### Later checkpoint: actual scoped source-unit execution

The real N4 benchmark subsequently took 697.227 s versus 464.540 s, with
unchanged ~5.5 GiB peak printed RSS. It did not establish the requested full
tiling/parallel architecture. The user correctly challenged that result.

The existing resolved-unit runner was then observed loading all project graph
bodies in every worker. Its input is now an indexed, digest-checked SCC/dependency
cut with stable addresses, exact selected nodes/edges/scopes and no catalogue
topology or unrelated bodies. Source planning closes before worker launch;
weak-reference tests prove its wrappers are released. The source concordance is
stowed intact as a separate digest-matched archive, not silently disabled or
duplicated into each worker. Actual source units run in separate processes;
three small functions reached peak admission 2, passed structural concordance,
and produced exact native outputs. Focused tests: 73 passed in 23.30 s. The two
planner address-disagreement failures are unchanged on clean `C:/dev/tg-base`.

The new source-unit mode has an explicit node-work bound, refusing to serialize
or launch over-bound closures and publishing the existing subdivision records.
This controls admitted source work, not arbitrary process byte growth. It is
still the UNIT-PRODUCT lane, not the ordinary orbital single-module path. Whole
ingestion and transitive authored dependencies remain; interface-only tiles and
streamed final linking are unfinished. Do not present this as the complete fix.
Receipts and CLI are in `turing/TEST_BASELINE_AND_HAZARDS.md`.

> please do something that will split the job in emmory and execution scope now

> the shared compiler book, the concordance, is for provenance and diagnostics that cut through noise, it's a requirement we can eventually turn off or stow at any time it isn't needed to accept a transformation,what we're talking about here is work sharing and not a complete fuundamental change, I would thin. are you going in a professional and harmoneous process

> we have a particular compiler, it seems to produce good binaries that run competitively fast. it just has a probem that it's architecture uses all memory it needs all at once, and it's not parallelized at all. I asked you to use pure math and compiler theory to tile that job so we can control it. We have made many changes. *Nothing* has *in any way* established the primary concern of a known professional architectural elements promoting industry pure math best practices

> unless I'm wrong, compilers usually use more than one thread, a lot of binaries end up technically segmented into regions, what are we doing that we're doing wrong that if we just switch up that one technique this falls into place

> let's proceed, making bold swift sweeping inclusion of features before being worried about debugging initially, so our debugged state represents more advance than this last round re the total overhaul

> if we have a two thread process can we amp things up and still meaningfully test, the faster this can go the more we can do

> I want you to put tight iteration of meaningful micro-tests to overhaul our system to these ends in a coordinated, thoughtful, all in one conversion

> UNTIL WE BREAK THE NEED TO HAVE EVERYTHING IN RAM AND UNTIL WE BREAK THE NEED TO DO EVERYTHING ON ONE THREAD WE WILL NOT SEE IMPROVEMENT

> DO NOT USE OTHER COMPILERS, DON'T PUT A BUNCH OF PACKAGES IN THAT FUCK AROUND PRETENDING TO BE NATIVE

> WHEN I SAY COMPILE I MEAN OUR COMPILER
