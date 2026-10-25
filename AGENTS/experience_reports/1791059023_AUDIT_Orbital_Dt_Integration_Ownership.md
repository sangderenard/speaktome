# Orbital dt integration ownership — active work, 2026-10-03

## Scope and state

Work is in the root repository's `engine_toy/` and the independent `turing/`
repository, both on `wip/2026-10-03-inflight`. This report is a progress receipt,
not acceptance of inherited WIP. Existing unrelated speaktome files were left
alone.

Read the continuation and orbital status reports, local agent guidance,
compiler identity rules, testing hazards, and relevant earlier guestbook
reports. Initial flight used real dt outer-window coordination and managed
column rollback, but bespoke leapfrog/Euler/Cayley updates. The later entries
record their replacement by the actual library RK4, verified adaptive growth,
the completed native craft command-change gate and remaining acceptance work.
Only the fixed
central gravity sources act; game craft/stations do not mutually gravitate.

Existing native wheel verification: 3 failures, 1 pass. Closed-system angular
momentum differs by 0.080483321 N m s after the first 2-second substep even
though sampled compiled wheel and angular-rate outputs exactly match their
authored SymPy expressions. Full observations and commands are in
`turing/docs/concordance_census/ORBITAL_DT_METRICS_2026-10-03.md`.

## Integration connection

The actual library RK4 algorithm is generic stage algebra. The chosen
correction invokes `RK4Integrator.step` to construct the coupled symbolic
update and compiles through the existing `equation_piece` route. A direct
inline one-wheel/13-state symbolic construction was stopped after exceeding
90 seconds before compilation. The next construction manifests the actual
library callback evaluations as existing dt pieces and commits physical
state after the stages. Existing PieceState owns stage columns, physical
columns, and rollback. No new table, snapshot implementation, or runner.

The generic dt channel ABI permits explicitly declared appended error
channels. The LLVM-piece bridge hardcoded the shared 15-channel layout.
Extending the existing layout consistently is in progress so conservation
error can be limited independently of stored energy and exchange power.

Workers: Astra extra-high owns coupled integration and energy; Sol high owns
the declared channel bridge; Luna high supplies bounded history and source
evidence. One native build at a time. Tracker substep prediction removal
waits for the corrected actuator/impulse seam. Rendering and full game
acceptance remain downstream.

## Separately verified changes

- Root commit `672eb19`: reverse-row cache uses existing PieceCompilerRecord
  validation and concordance staleness/build receipts. Three focused cache
  tests passed. This does not claim that the long native row cache rebuild
  or a collocation flight was run.
- Turing commit `e99bffd7`: regression for cached source-definition stamps
  reused across identity books. One focused test passed; no production
  compiler change was made for this case.
- Tracker wheel-command clearing/resume and game planner adapter edits
  remain uncommitted. The existing frame/cutoff regression still fails;
  its tolerances were not changed. Three game adapter tests passed but use
  a stub planner, so do not establish rendezvous.

## Subsequent native evidence and active work

Turing commits `3202f8ba` and `8ab64810` separately verify the declared
appended-channel bridge and participant-energy authority correction. Real
native laws publish the extra errors and exchangeable energy; rejection
restores the registered span before retry. Seven sidechain checks pass for
BIND/HOLD/DILATE/SUBCYCLE and the legacy aggregate fallback. Native audits
retain zero structural findings but still have explicitly open unsourced
facts; no complete concordance closure is claimed.

The actual library RK4 one-wheel native slice passed: its first 2-second
candidate was rejected, both requested windows landed, and inertial angular
momentum error was `4.8713e-10 N m s`. Its initial accumulated Gram metric
caused 3,047 attempts; the correction publishes the exact incremental Gram
defect. That rerun and full craft acceptance remain pending. Production
source now assembles coupled library stages with stage-dependent forces,
mass properties, tanks, and work. Source changes remain unverified WIP.

Root removed the tracker's predicted substep list. A native exact throttle
ramp-area observable prices commands; actual flight progress still reads
integrated impulse. The native nine-lane ramp case passed. Reusing the same
law/book for one lane exposed compiler shape/materialization, formal-call
occurrence, and output-extent defects. Sol is tracing those precise seams;
five adjacent test failures reproduced identically on the pristine baseline.
Luna is fixing output-only `_next` registration in the existing PieceState
ownership path, with a focused native rollback test. Astra owns full craft
integration and physical initialization. Native builds remain serialized.

The corrected incremental Gram native rotor gate passes in 56.92 s with
140 attempts (formerly 3,047); final inertial momentum drift is
`2.30317515257e-9 N m s`. The original actuator gate now passes all three
tests in 11.70 s after the targeted batch-shape compiler correction.
The final broader compiler regression gate is still pending.

The output-only column registration gate now passes both tests in 7.60 s.
Its real native piece writes an output-only force readout, the first attempt
is rejected, and an observer confirms that readout is restored to its explicit
initial value before the smaller retry. The accepted window lands and keeps
the same span. The minimal bridge change is ready for a separate commit.

Tracker review also found frame-dependent burn accounting from midpoint
thrust directions and mean mass. Both craft builders now declare the applied
delta-v integral through the same library stages, and the tracker reads its
endpoint difference. This new seam awaits native flight verification.

Concurrent changes to `symbolic_process_graph.py` and its test appeared
outside this agent team. They are preserved without attribution or staging.

## Production build ceiling and compiler gate follow-up

The complete c0 craft build finished its first 19 native pieces but exceeded
the continuation report's roughly 6 GB process ceiling at the combined
endpoint. The last recorded working set was 7,164,207,104 bytes; the endpoint
had 4,516 graph nodes and 8,015 edges at the structural-value folding pass.
Root stopped the build after the integration worker initially missed the
over-limit condition. Runtime physics remains unverified. Finished caches
are retained, and the endpoint's existing equation-piece boundaries are
under read-only review before further compilation.

The compiler worker then ran the final focused broader gate: 34 passed,
5 failed in 6.33 seconds. All five failing names and categories match the
previously measured pristine-baseline failures. Turing commit `31285882`
contains only the four owned shape and concordance files; the concurrent
symbolic-process-graph edits remain untouched. Root-workspace commit
`aadda11` separately records the native throttle-ramp area and three passing
tests. The coupled craft and tracker remain uncommitted until their physical
gates pass.

Read-only cache accounting found 714,433,717 bytes on disk across those 19
completed pieces. Their individually loaded reachable objects sum to
6,006,139,090 bytes, of which 5,911,052,738 are identity books. This explains
a substantial retained-memory component without claiming an exact process
allocation profile. The runtime uses artifact and ABI, but its current
LLVMPiece loader retains source/module/book, and the artifact's emission
record also retains the book. Existing training caches demonstrate lighter
artifact execution, but no equivalent LLVMPiece API was found that retains
the exact persisted provenance and compiler stamp. No book was discarded.

The endpoint is now partitioned on inspected causal boundaries: precommit
metrics, precommit diagnostics, one physical commit, postcommit mass
properties, postcommit inverse, then the existing energy publications.
All entries are distinct participant occurrences; no candidate-state or
snapshot convention was added. Source checks show 27 entries and no
precommit output intersecting the physical commit's required input columns.
Native verification of those changed pieces remains pending.

## Infinity literal ownership correction

The actual stage-zero instantaneous-actuator law produced zero instead of
one for command one, old state zero, and infinite slew. The authored SymPy
expression gave one. The chain was traced to a numeric Const carrying SymPy
`oo`; the SSA-to-Python materializer used its repr, emitting a free name
`oo` instead of a literal. In the actual production artifact it became SSA
formal 10 without a producer. Native ABI preparation initialized its
nonfeed buffer to zero, so comparison with infinity failed and comparison
with zero succeeded. The same defect reproduced for negative infinity.

Turing commit `147dd2be` normalizes the existing SymPy Number ownership path and
uses Python's existing `ast.Constant`/`ast.unparse` literal spelling,
`1e309` and `-1e309`. Native equality and direct infinity publication pass
for both signs. The broader adjacent gate reports 55 passed and six
failures; all six reproduce identically in untouched e068ce59. The old
batch-shape tests remain passing. Native watchdog peak was 1,645,694,976
bytes. Audits remain OPEN: both equality cases have zero structural findings;
unsourced facts changed from 1,256/1,263 to 1,254/1,252. No physics threshold,
controller, or external symbolic-process-graph edit was changed.

The first production retest still returned zero because the isolated
fixture omitted base `thruster_columns` bounds and supplied maximum throttle
zero. The compiler's rebuilt source contained the correct `1e309` literal.
This was a fixture error, not another compiler failure. The separate tiny
infinity reproductions remain valid; the isolated fixture is being corrected
to use both existing base and machine column initializers before rerunning.

## Production laws and physical refinement measurements

The corrected actual stage-zero fixture passes in 21.45 seconds. Three
actual production energy pieces pass with stored energy, power, and infinity
publications intact. The actual derivative law preserves mass/inertia flux
and separates electrical input, copper loss, and absolute exchange: the
opposing-wheel case publishes 2.104 W, 0.104 W, and 10.04 W respectively.

The base pure-torque run measured the genuine legacy orthogonality failure
before obsolete substep-count assertions: at the new default local Gram
limit 1e-9, ten seconds gave drift 1.17688078171696e-8 (86 attempts), failing
the unchanged 1e-14 physical bound. Configuring only the existing Targets
channel to 1e-17 on the same cached bank yields 2.3314683517128287e-15 drift,
phase 2.399999999999658 versus 2.4, and determinant 0.9999999999999976.
There were 7,003 attempts and 5,594 rejections; 13.672 seconds including
load, sampled peak 1,395,056,640 bytes. Root approved adopting this measured
new default instead of modifying the test's physical accuracy requirement.
No projection or alternative integration/rollback was added. Full craft,
frame-partition, and game acceptance remain pending.

## Prompt History

> turing\docs\concordance_census\CONTINUATION_SESSION_2026-10-03.md
> turing\docs\ORBITAL_GAME_STATUS_2026-10-03.md
>
> I want you to assume ownership of the branch doing this work, all this work, prioritize the orbital game and necessary architectural pieces of the compiler that are non controversial when it comes to their impact,
>
> I want you to feel free to send out many sub-agents provided they are of adequately inexpensive model configuration, though we are testing new models and we have a credit reset for the weekly budget, so *shrug*

> the hd you're working on is a cacheless sata ssd that is being encrypted for its total 1tb mostly full size by bitlocker, delays are normal, don't let this notice change your work, it's just fyi

> the energy of thrusters or reaction wheels should be signaling to a well defined configuration of dt system's energy limits and metrics, that the dt during maneuvering, really any burn change, should be done as finely as necessary to pass, that is the beuty of that system

> if the game is not using the turing dt system, if it is a bespoke system, it is wrong and must be replaced with the dt system

> are all bodies exerting gravity?

> it saying you need your integrator to accomodate different sized frames means you're not doing things right, you're not using the real dt system, it has integrators, you use whatever substep but the outer coordinator has a fixed window for the span requested. I ask again are you using dt system

> can you make an action plan paired with a model recommendation and subagent model recommendations

The subsequent user instruction adopted these exact model and priority statements:

> I recommend **GPT-6 Astra at extra-high reasoning to lead the integration and energy-accounting correction**, then **GPT-6.1 Sol at high reasoning for implementation and coordination** once that boundary is established. Use **Luna for narrow, well-defined tasks**. This fits the roles described in [OpenAI’s model-selection guidance](https://developers.openai.com/api/docs/guides/model-selection); the assignments below are my recommendation for this repository.
>
> The immediate priority is correct dt-managed flight. Collocation, rendering, and performance follow that.

The user's action table:

| Order | Action | Acceptance gate | Model |
| ----- | ------ | --------------- | ----- |
| 1 | Trace the existing dt integrator, registration, managed-column, and native compilation paths—including relevant history and worktrees. Establish the actual supported connection before editing. | A concrete source-to-execution trace identifying who owns integration, windows, state, metrics, and rollback. | **Astra · extra-high** |
| 2 | Replace bespoke orbital integration with the existing dt-system integration mechanism. Register craft and station state through its established ownership contract. Remove tracker code that independently predicts integration substeps. | The dt system owns substeps and completes the requested outer window; save/restore and rejection operate on the registered state. | **Astra · extra-high** |
| 3 | Connect thruster, wheel, and rotational energy exchanges to declared dt publications and limits. Distinguish stored energy, exchangeable energy, power, and conservation error. | Ignition, cutoff, reversal, and wheel desaturation visibly affect dt through the configured metrics; rejected attempts restore state correctly. | **Astra · extra-high**, with **Sol · high** implementation |
| 4 | Fix only compiler defects that block those real paths. Preserve exact identities and leave concordance edges at every decision. | Small native reproductions pass; concordance audits show no regression; compiled laws retain their numerical meaning. | **Sol · high** |
| 5 | Verify flight correctness: closed-system angular momentum, energy accounting, fuel cutoff, and different frame windows. | Existing accuracy requirements pass without loosening tolerances; frame partitioning does not materially change the trip. | **Sol · high** |
| 6 | Finish collocation integration and rebuild its compiler-validated row cache. Test clicks, disturbances, and subsequent re-planning. | Actual game flights rendezvous; ON/OFF PLAN and re-plan behavior are observed. Adapter tests alone do not satisfy this gate. | **Sol · high** |
| 7 | Re-render the game, measure frame cost, and optimize measured bottlenecks. Then address further gravity/gameplay/web scope. | Correct phased burns and arrival are visible; performance changes preserve the preceding gates. | **Sol · high**, **Luna · medium** support |

> Run **at most three workers alongside the coordinator**, with disjoint file ownership and **one native build at a time**. Each worker must report what finished, what failed, and what remains unknown. Verified changes land separately; inherited WIP stays unverified until its gates pass.
>
> The first deliverable should be a working dt integration and energy contract—not another frame-boundary workaround.

Applicable source guidance includes the following exact rule:

> The dt system (`turing/src/common/dt_system`) is used AS IS. Engines
> register their columns as parameters; the dt system saves and restores
> them. No runner, table, snapshot, or rebind convention of your own.

Later steering, preserved verbatim:

> what's the problem?

> oh can we elect to not save them

> we could also dump to file instead of ram? like we could map a file maybe?

The user selected a memory-lifetime correction rather than approving a higher
process ceiling. The implementation keeps the existing full `.piece` archive
on disk and adds a runtime-only piece index for the game. Compilation still
creates its concordance book; completed pieces can release their SSA/source
and emission-book references while retaining exact ABI, LLVM text, and
compiler source stamps. Ordinary runtime loads use the small index. Mapping
the existing pickle would not avoid reconstructing its Python object graph;
no new file-backed identity-book implementation is being introduced.

> maybe we can serialize and compress instead of using pickle? is pickle making this any more complicated or is it the sheer mass of data

Compression can reduce archive bytes but does not remove the expanded Python
graph after loading. The independent runtime cache avoids that load. Native
verification passes: four compiler payload tests in 2.84 s and two real cache
tests in 9.03 s; peaks 1,626,808,320 and 1,630,035,968 bytes. They cover original
book preservation, source stamps, constant outputs, repeated native execution
into registered spans, legacy archive loading, separate runtime/full cache
paths and explicit refusal of source linking with discarded compiler metadata.
The compiler audit remains unchanged (zero structural findings, 1,128
unsourced facts, OPEN). Verified changes were committed separately as Turing
`c73c2a26` and root `c4b4fd2`, with inherited physics/game WIP left unstaged.
The read-only 19-piece comparison completed: 714,433,717 full archive bytes
versus 7,879,555 compact runtime bytes, sampled RSS peak 766,705,664 and ending
435,572,736 bytes. All 19 retain compiler stamps and LLVM text and omit all
four compiler/book-owning fields. No stale DLL was executed or cache changed.
The read-only comparison used Turing's venv Python 3.11.7, with Windows RSS
sampled every 50 ms; native test builds used system Python 3.11. The separate
no-write zlib level-1 check compressed the largest original stage13 archive
from 61,223,411 to 10,470,417 bytes in 0.229 s. Compression is still only a
measured storage option; no new archive codec was introduced.

The runtime-payload postcommit properties/inverse gates subsequently passed
(2 tests, 113.95 s, peak 1,441,718,272 bytes) and the updated copper rotor
gate passed (48.94 s, peak 1,006,317,568 bytes, 140 attempts, H drift
2.30317515257e-9 Nms, copper 0.013 J and energy balance below 1e-12 J).
No physical tolerances were loosened. The sole native slot next belongs to
the compiler worker for the measured no-growth retry-cap regression and
minimal correction; full craft regeneration waits for that source freeze.

> if you found a no growth setting you may have made it and not used the system properly or it may not be fully finished, there isn't a need to do no grow dt that's harmfull to everything everywhere and isn't intelligent

This overrides the proposed no-growth cap correction. Root interrupted that
work; no cap edit had been applied and no new no-growth tests remain. Root's
mistake was treating the inherited setting as an intended requirement. The
false plan default traces to `aeb6b7c8` (2025-08-16, commit message
"intigrating dynamic dt controller, bugs remain"). More directly, the actual
LLVM-piece bridge omitted the graph's growth flag entirely from PieceState,
its ABI and the run_superstep call. The current work enables existing adaptive
growth defaults and carries the graph's actual choice through that bridge,
preserving declared physical limits, exact windows, rollback and explicitly
pinned operation. OrbitalJumper now explicitly requests growth in its plan;
bridge/native verification is pending. No new step-size formula or blanket cap
is being introduced.

> something important i neglected to mention was the planner should run independently and announce plans when they arrive

> and then regarding the speed of rejection we are using power of two feel free to work with a nonlinear of your choosing

Root assigned an existing-path read-only planner lifecycle inventory and an
existing-controller retry-policy trace. Independent plan computation must
return a visible ready result while game frames continue. Any retry sizing
change remains inside the existing dt owner and must preserve the measured
physical acceptance, rollback, and exact outer-window completion.

The adaptive-default/LLVM-bridge correction passes 31 focused tests in
20.31 s, sampled peak max(working set, private bytes) 1,980,862,464 bytes.
Actual native growth/rejection/rollback and exact landing pass; an authored
write using the typed growth bool passes source-to-SSA-to-C execution.
Full coordinator compilation remains unverified. The two audits retain
0 structural findings / 0 unsourced ids, with 1,472 and 1,680 unsourced facts
OPEN. A broader interrupted batch has no summary and is not counted green.
Astra now has the sole native slot for the torque and full production gates.
Sol is inspecting independent planning using the existing SolutionService
lifecycle and isolated native row execution arenas. Compiler/dt source is
frozen during flight compilation. No no-growth cap was implemented.

The adaptive correction is committed as Turing `78c4f502`. The original
forty-window torque gate then passed against it (113.96 s including the
nine-piece rebuild; peak max(working set, private bytes) 2,187,321,344).
Its 7,003 attempts / 5,594 metric rejections / 1,409 within-limit attempts
and physical errors are unchanged. Growth transport did not reduce rejection
cost. The full 27-piece craft build is now in progress below the 6 GB cap;
its runtime assertions have not yet executed.

The existing SolutionService now supports optional captured requests and
completed result/error publication, while preserving its firing API. Its
four focused lifecycle tests pass in 1.27 s (`python -u -m pytest -q
--tb=short tests/test_solution_service.py`, from engine_toy). These verify
supersession, one publication, failure/recovery, stop/late-result handling,
and firing API compatibility. Stop cannot interrupt a solve already running;
it prevents a duplicate worker and suppresses a late result. Game, tracker,
and collocation arena-ownership integration remain WIP and are not validated
by this service-only gate.

The full 27-piece c0 craft bank has now compiled below the 6 GB ceiling:
sampled max(working set, private bytes) peak 3,041,787,904 bytes. Its first
runtime was stopped after compilation to add bounded test-only progress
output; archives were preserved. The cached command-change gate passes in
211.83 s at a 1,644,638,208-byte peak. Wheel ignition/reversal, main
ignition/cutoff and exact no-burn coast preserve the declared energy,
angular-momentum, mass/impulse, applied-delta-v and registered-span assertions.
Final Gram defect is 3.33066907388e-15. Ten simulated seconds nevertheless
cost 27,556 attempts / 23,969 error-limit failures, primarily incremental
Gram error. The default proposal aims back toward loose CFL after accepted
steps because its error-ratio penalty is floored at one. Growth stays enabled;
proposal/distribution mechanisms are being inspected read-only while Astra
continues long-duration, growth-after-stop and cutoff gates. No performance
acceptance or new timestep formula is claimed.

Root commit `6c22bdf` contains only the verified SolutionService extension
and its four tests. Game/tracker/collocation source remains WIP; 21 lifecycle
checks pass in 7.71 s using declared test seams, not full native flights.
The planner now calls the actual library RK4 for its seven-state dynamics.
Tiny native arena-isolation and seven-row/Jacobian tests are prepared,
including finite-propellant changing stage mass, but await the sole native
slot. Actual collocation row regeneration and rendezvous are still pending.

Subsequent pre-feedback physical results: 150-second wheel momentum and
60-second desaturation assertions passed; near-empty cutoff separately passed
in 28.80 s. Growth-after-stop recovers full-window steps. The 20-second
untrimmed-burn run was stopped while spending tens of thousands of retries;
it has no completed batch receipt. Native row-arena isolation and delayed
Hohmann assumption checking passed, without claiming physical game flight.

The user-authorized nonlinear proposal is now implemented inside the existing
dt controller. It uses attempted_dt/sqrt(largest declared error ratio), bounded
by existing CFL/energy limits, and feeds both acceptance and rejection through
the existing PI update. Error headroom permits growth; failed uninformative
proposals retain the old halving fallback. Rejected metrics are consumed
before existing registered-state restoration. No alternate integrator,
checkpoint, or no-growth cap was added. Twenty-one focused tests passed in
26.56 s (peak 1,938,812,928 bytes), followed by three strict native helper cases
in 14.39 s (peak 1,756,524,544 bytes); exact representable outputs and one-ULP
sqrt parity pass. Toy power-six/eight errors verify growth, command changes,
rollback and exact windows; production flight remeasurement remains open.

One adjacent source-order failure reproduces identically on untouched HEAD
0d101e3c at C:/dev/Powershell/.dt-feedback-baseline: test_item_capture_depends_on_its_operand_producer,
1 failed in 9.48 s, two use-not-dominated findings in _energy_time_limit.
No source was altered in the baseline worktree. Two older persistent-failure
tests remain excluded and unresolved because their expected raises conflict
with existing floor-retention/default-unresolved behavior. No acceptance
policy was changed to satisfy them.

Root initially called a missing native sqrt a compiler issue too early and
corrected that statement. The generic extraction contract permits Python-host
boundaries; direct whole-native helper tests now use the existing full-native
dt contract and the requested source-root export. Aggregate partial-boundary
accounting still omits the sqrt boundary, so that separate debt remains OPEN.
The compiler and dt source are frozen for normal current-stamp rebuilds.

The first error-feedback implementation was committed as Turing `cf78f24f`.
Its unchanged torque gate passed but reduced attempts only 7,003 -> 6,828.
A cache-only observation found a concrete zero-error singularity: loose CFL
produced roughly 5e35 s, immediately saturating the existing PI accumulator.
Pure comparisons through the existing controller rejected two unit-response
alternatives because they approached the acceptance boundary from above for
hundreds of retries. The selected response is 1.8*h/(1+sqrt(R)), bounded by the
existing CFL/energy limit. It retains repeated growth at zero error and gives
proposal headroom R=.64; physical acceptance remains R<=1.

Turing `6a879db3` contains that verified correction. All 23 focused nodes pass
across the initial selection and corrected test-only rerun. Native q6/q8
errors cover .05/1e-6 openers, command changes, four zero-error windows and
restart, with registered-span rollback and exact .5 s windows. The unchanged
40-window torque test passes in 98.07 s including rebuilding nine pieces,
peak 2,187,173,888 bytes. A guarded normal-cache invocation repeats the same
assertions and emits 1,081 attempts / one rejection (84.56% fewer attempts
than the original), Gram6.66133814775094e-15 below1e-14, phase2.39999999999929
versus2.4 and angular speed.48. No wall-time speedup is inferred from the
differently instrumented/cold runs. Full craft remeasurement remains open.

The planner's current-stamp tiny RK4 native gate fails only two Jacobian
zero cells, position_y/throttle and position_z/throttle (residuals around
1e-18); every forward value is exact and every nonzero derivative meets
four ULP. No threshold or tolerance was relaxed. Reading its saved full SSA
book exposed a separate pre-existing ArgumentBindingFact tuple-pickle defect.
Turing `7d984b78` repairs reconstruction at that declaration, preserving exact
fields and shared source aliases, including original legacy archives. Nine
archive tests and the formerly failing native child-record return test pass.
The real92MB book now round-trips with identical audit/argument/output IDs.
Its raw unsourced debt stays OPEN; no stale native executable was used.

Sol is now using the compiler's existing observed_outputs seam to expose
the actual mass/reciprocal and reverse accumulation values for those two
native failing rows. Current IDs are selected through source and adjoint
identity links; observation does not invent an interpreter parity reference
or alter the four-ULP assertion. The bounded native build is running under
the existing6e9-byte ceiling. Full27 craft regeneration is held until this
trace determines whether another compiler-source edit is necessary. The
compiler/dt source is frozen; root has not touched unrelated symbolic_process_graph
and its test WIP.

The observed-output native derivative gate has completed (243.23 s,
peak4,015,517,696 bytes), reproducing exactly the two zero-cell residuals.
Its source/adjoint links identify51 throttle contributions per row, five
nonzero. Rounded contributions already sum to-3*2^-61 for position_y and
-5*2^-60 for position_z; one position_y addition adds another-2^-61.
Every traced primitive agrees with its authored AD rule. No compiler identity
defect is established, and changing final summation alone cannot fix this.
Full source, role mapping, native operand arrays and92,768,354-byte SSA/book
are saved at C:/Users/alber/AppData/Local/Temp/orbital_collocation_native_observed_gate_rows/c95ab2ca664c80067fa587db6f28a5b9.
The audit has6,637rows/332functions, zero structural findings and72,929
unsourced facts/10ids, OPEN;442extra facts are observation declarations.
No numerical threshold was relaxed. Sol is investigating exact stage-level
algebra at the existing law owner, retaining actual RK4Integrator.step.
Astra has resumed the full27 craft build/validation against6a879db3 with
compiler/dt source frozen and the unchanged6e9-byte process ceiling.

Root and Luna traced a separate game arrival bug. The game captured a fixed
frame endpoint before fly(), but an accepted asynchronous replacement could
move arrival-plus-settle earlier during that call. A real tracker-loop and
service regression with analytic craft seams reproduced 600 s versus the
required 325 s. The tracker now accepts a callable stopping time alongside
its existing float endpoint and re-reads it after replacement adoption before
requesting a craft window. The game supplies min(frame end, current arrival).
A replan whose entire arrival window elapsed during solving is recaptured.
Eighteen service/tracker tests pass in 7.77 s; the added earlier/later arrival
parameter cases pass in 8.86 s. No native flight is claimed by these tests.
The dt library still completes each requested window; no timestep or
integration convention was introduced. Source review is pending and the
surrounding game/tracker integration remains uncommitted WIP.

Final arrival-fix check: 19 service/tracker lifecycle tests pass in 7.63 s.
Luna's source-only review found no finite static-deadline regression. Root
retained the original less-than comparison predicate so a NaN static target
also retains its previous no-advance behavior. Game/tracker ownership is
returned to Sol for eventual actual flight acceptance; source stays WIP.

Sol's cancellation correction is now source-frozen in orbital_collocation.py.
It collects the actual library callback forces as temporary symbols, cancels
rational coefficients in the affine endpoint relations, and restores the
callback force laws in dependency order. No stage weights or integrator are
reimplemented. Five source-only tests pass in 29.34 s: all seven generic
relations preserve exact binary coefficients and finite flow; original versus
reduced laws agree with nonzero gravity on each axis. The exact coefficient
6 * binary(1/6) remains 18014398509481983/18014398509481984 in source, not one.
Native representation is still to be measured. The native gate compares the
reduced execution against an independent copy of the original RK4 laws and
Jacobian, retaining four ULP and exact zero. No compiler source changed.
It awaits Astra's full c0 command gate; at the latest report 20/27 pieces
had rebuilt with current memory about 2.32 GB. After the c0 bank completes,
Astra will run guarded no-rebuild physical tests while Sol owns the sole
native build slot for the numerical gate.

Full current-stamp c0 bank plus unchanged command gate PASSED: 1 test in
1478.85 s including1442.193 s construction; peak2,969,436,160 bytes.
Five-phase runtime30.123 s versus186.260 s previously. Attempts/rejections
by phase: ignition273/81, reversal271/90, main1360/356, cutoff1086/188,
coast901/205; total3891/920,2971accepted. All original energy, H, fuel,
impulse, applied-delta-v and registered-span assertions pass. The current
receipt is committed in Turing e39c6fb0. Warm physical gates use compiler
entrance guards: near-empty exhaustion and60sdesaturation pass;150sH passes
with52733attempts and6.33707055476e-13Nms worst error. Full/empty supply
has a failure whose details await batch completion; ordinary20sburn runs.
No rejection count is inferred from those tests' attempt-only counters.

The reduced planner's native gate FAILS its unchanged acceptance gate:
134.29 s pytest, peak3,818,995,712 bytes. Original transverse throttle zero
cells now return exactly zero, and all state values pass, but momentum
rows0/1/2 versus propellant_mass differ by348/6/7ULP. Maximum absolute
Jacobian discrepancy is4.829470157119431e-15. Audit5297rows/260functions,
zero structural findings,57544unsourced facts/10ids OPEN. Sol's following
diagnostic forbids emission/build and uses the normal validated current
cache. No numerical acceptance or production row bank is claimed. Existing
AbstractTensor Precision wiring is being traced read-only; no compiler or
dt source edit is justified yet. The native slot is back with Astra after
its current physical batch and supply-failure diagnosis; c1 is not started.

User steering, verbatim: "abstract tensor stage can be enriched with extra limbs for difficult precision issues".
Existing symbolic_abstract_tensor_source(policy) promotes section operands
with Precision.of before the public source compiler. The reverse-motion
entry currently has no policy/source option; its planned root is Call/GEP/Load.
Materializing that entire module explicitly refuses nineteen helper bodies,
so no incomplete source is used. Sol is tracing the earlier AbstractTensor
seam, with Luna searching prior usage. Existing Precision.constant(like,parts)
is also relevant to constants that lose bits before promotion: the current
native coefficient rounds i64(2^54-1) to double(2^54), despite exact source
algebra. Neither route is yet a verified planner correction.

A guarded cache-only supply rerun isolates the failed assertion before the
empty-tank case: full supply is0.9999999999999999 versus exact1.0. The readout
divides separately integrated supplied propellant by unrestricted demand;
Astra is tracing its stage values and quotient without modifying the law or
tolerance. One failure in26.76s, peak1.644GB. The final ordinary20sburn is
still progressing, so compiler/dt source remains frozen and c1 held.

Existing earlier reverse source found: vehicle_inverse_compilation._motion_source
renders the actual fused ProcessGraph and canonical BACKWARD_RULES before
prepare_vehicle_inverse_adam_python promotes inputs using Precision.of. Its
vehicle tests execute eagerly, so native preservation still needs proof.
This supersedes the narrow earlier inventory's failure to find a seam; no
replacement source interpreter is needed. The actual orbital motion needs
multi-loss labels/seeds and nary/min/max coverage at this existing owner.

Supported generated-section native proof PASSES: existing precision policy
emits three Precision.of inputs, multiply/add/collapse; result
0x1.434a98a0d86b4p-51 equals exact-binary Fraction reference. Public ABI stays
scalar-width, no LLVM shortfalls; guarded7.16s/peak1,608,740,864bytes. Archive
TEMP/orbital_generated_precision_probe_vhmzq7vv has source/module/book/audit.
Audit154rows3functions0structural,759facts134ids OPEN. Generic authored
Precision.constant still has no source reducer boundary; its direct tiny
probe is refused before LLVM, not evidence against the supported of path.

Warm c0 batch COMPLETE:4passed/1failed1778.19s, peak2,507,046,912bytes.
H150s,desat60s,nearEmpty andordinary20sburn+bothcutoff/coast windows pass.
Ordinaryburn t20 has152315attempts; finaltotal notprinted. Full supply is the
only batchfailure. Allwarm/diagnosticprocesses ended; sourcefreeze released,
SolownsnextTuringprecisionedit/build,c1held. Exactstage magnitudes match full
demand; final diagnosticquotient introduces the oneULP supply discrepancy.
Empty impulse/tanks/velocity pass exactly; initialmass741.8800000000001 vs
native741.8799999999999 traced tocharged-node hostreduction vsnative dry+tanks.
Astra will aligninitial dependentcolumns at existing owner and fix proven
diagnosticalgebra; neither correction is yet verified. Source-only reverse
materializer checks3fail/1pass15.99s:multi-losslabels short,Min/Maxunsupported,
actualnaryingestionalreadybinaryfolded andpasses,so no narypatchjustified.

Tiny actual fused reverse, (x*y+z)^2 withcanonicalmul/addrules andseed.375,
refuses two collapse calls beforeLLVM(12.45s,peak1.770GB). Exact sourcegraph
trace:call12 authoredtuple leaves14/18 is incorrectlytaggedPrecision2 aswhole;
Indexed14/18 lack numericdescriptors, collapse16/20remainunsupported. Existing
returned-facts dedup atsourcepropagation assignsone wholecall descriptor;
RETURN_SITE_SLOT refs andcallerprojectionpaths provide exactslotidentities.
Sol ownsfixatthatboundary; no latecollapse/tagging workaround. Fullsource
TEMP/orbital_fused_precision_probe_6ligvual/source.py, rawgraphreceipts
TEMP/orbital_fused_precision_capture. No native reverse acceptance yet.

Astra authoredcanonicalmassordered-add initializationand exactclamped quotient
cancellation. Symbolicclampdomains allzero differences;26/27equationhashes
unchanged, onlydiagnosticschanged. Sourceemptyinitialmass matchesarchivednative
exactly. No sourceintegrationstage/conservation/commit/storechanges and no
nativeverificationyet. Astraidlewithnoactiveprocess, waitingstablecompilerstamp
fornarrowverification; c1/frame/game stillpending.

User steering, verbatim: "if this continues for longer, we must downshift the intensity of the models used, and recognize whether we are making steady progress for the time put in thereafter".
At2026-10-04 03:22UTC rootstoppedSol/Astraactivework;no nativeprocessesremain.
Luna is soleactiveworker onone15-minute selected-return investigation.
Verifiedprogress:longphysicalbatch4pass/1fail;plainprecisionnativepass;full
fusedreverseprecisionstillhasno native result. Uncommittedprojectionfix
resolveslossbutnotgradientcollapse; sourcegate4pass/1literalint.shape failure.
Current exactchain:unbroadcast0+52Precision2,62reshapeunknown→bw_mul5/8unknown
→actual_motion26unknown→tick18unknown. ExistingGLSL identityspecialization
appendsreshapeunconditionallydespiteauthor'ssame-shape returnG;Lunatracingthat
owner. Anotherbroadinventorywillnotcountasprogress. No productionrowbank.

Lower-costpass locatedselectedreturnrecords but no complete scalarclass
transfer/nativefusedresult. Rootstoppedthatlineearly, archivedexactownedhunks
andnewtestat turing/docs/concordance_census/precision_experiment_2026-10-03/,
checkedthenreverse-appliedonlythetwosession-ownedsourcefilediffs. No stash,
no pathcheckout,no unrelatedsymbolicgraphWIPtouched. Compilercontentsrestored
topre-experiment; archivedpatchclearlyunverified. NextworkusesLunamediumfor
targetedphysicalfixvalidation,thenremaininggrowth/frame/gamegates.

Lower-cost checkpoint, 2026-10-04 03:47 UTC: one Luna worker at medium
reasoning validated all 27 c0/b1 runtime pieces with their stored per-module
source records. Fresh aggregate compiler digests vary with import closure;
root corrected an erroneous aggregate-digest guard. The original supply test
compiled only new diagnostic key 7f378bc5f01feaafa33ed1b0, then FAILED in
99.10 seconds: full supply still 0.9999999999999999 instead of exactly one.
The cancellation-only edit has not fixed the gate. Its empty-tank assertions
were not reached, so canonical mass initialization still awaits native
acceptance. A live memory sample was 2,059,247,616 bytes, but the worker lost
the outer command-session handle and did not recover the final sampled peak;
no exact peak is claimed. The existing 6 GB / 600-second guard completed
without terminating the test. The c0/t4 growth bank is stale in dt_controller,
participants, and concordance_declarations; no growth rebuild or c1/game
build was launched. A final five-minute read-only inspection of the emitted
diagnostic quotient is bounded to locating exact arithmetic, with no edits
or native builds. Expensive workers remain stopped and precision remains parked.

User steering, verbatim: "what if we coasted on green's function we derive using abstract tensor leplace 3d and abstract tensor linalg solve"; "Move decisively to implement it and we'll see the payoff of those bold steps"; "if solve uses basic abstract tensor operators it works with precision".
The 2026-10-04 implementation adds `engine_toy/orbital_green.py`,
`orbital_green_world.py`, and `orbital_green_demo.py`. One registered
PieceState contains Earth, Moon, craft, all stage columns, exact pairwise
Newtonian gravity, collocation residual, and energy/linear/angular conservation
publications. It uses the existing equation compiler, `instantiate_system`,
and `advance_round`; no alternate runner or timestep policy was introduced.
The first live launch exposed a duplicate `_first_occurrence` in tensor linalg;
removing the stale duplicate restored eager `AbstractTensor.linalg.solve`.
The next launch exposed two SymPy identities named `gravitational_constant`
with different assumptions; the metric law now uses the canonical positive
symbol. Exact headless execution then completed three 600 s windows. The dt
continuation was 64.1178, 1.7621, then 1.7487 s, with the third collocation
residual 4.2605e-7 m and relative conservation defects 1.2419e-23 energy,
3.3522e-22 linear momentum, 9.6329e-24 angular momentum. The demo is a
verified all-body coast slice, not yet MachineCraft thrust/attitude integration.
Compiled source pursuit of `AbstractTensor.linalg.solve` now passes prior
canonical-value reuse and bool-mask call-shape failures, but native precision
proof remains open at cumsum emission and root output dtype metadata.
