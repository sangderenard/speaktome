# Turing spectral graph report

**Date:** 2026-08-20
**Title:** User-facing spectral report over saved SSA

## Overview

Added a command-line spectral graph report for a saved lowered SSA module. It
reuses the existing `field_from_ssa` adapter and spectral analysis vocabulary,
so it does not grow a parallel IR walker or conflate topology analysis with the
existing source-colour viewer.

## Steps Taken

- Reviewed the recent compiler, GEMM/deployment, and spectral-analysis commit
  history without modifying the active GEMM files.
- Added `analyze_region_spectrum`, a public single-region API, so a caller can
  inspect any selected loop or whole-program region directly.
- Added `tools/spectral_graph_report.py`, which loads a saved
  `control_repository_ssa.pkl`, reports topology/edge roles/natural loops,
  selects FFT for circulant regions and `AT.eigh` for the remaining regions.
- Added focused tests and ran the new tool against the newest saved SSA
  artifact with `--dense-limit 1`.

## Observed Behaviour

The real artifact at `build/sfdc-condfix/control_repository_ssa.pkl` contains
2,817 nodes, 3,227 edges, and 13 natural loops. The initial topology smoke run
confirmed that the report finds all those regions and dispatches its circulant
two-node loop regions through FFT.

`python -m pytest tests/test_spectral_graph_analysis.py
tests/test_spectral_graph_report.py -q --tb=short` passed: 16 tests in 12.37
seconds.

### Trace-dye follow-up

Added `src/compiler/spectral_trace_dye.py` and
`tools/spectral_dye_trace.py`. They consume the shell's existing telemetry or
native trace-ring JSON, resolve emitted sites through the existing trace
manifest, retain measured duration/timestamp-derived cadence and phase, map
the result onto SSA influence dye and real relaxed transport paths, then write
both a machine-readable report and a vivid timeline PNG. The path intentionally
does not guess a site-to-SSA mapping when a historical trace lacks its
companion manifest.

`tools/compile_spectral_dye_trace.py` provides the complete source-file
workflow: it accepts an authored file, entrypoint, optional JSON feeds and
frame count; performs trace-enabled AOT and native-shell compilation; profiles
both compilation and execution through shell telemetry; persists manifest,
telemetry and native trace together; then renders the resolved report and
image automatically.

### Decision-tree experiment

The tool was evaluated against the existing translation decision tree rather
than treated as an independent oracle. `translation_scorecard.py` reported
17/19 end-to-end equivalences (the rebound-name and generator-`any` journeys
stop at materialization), while `tracer_stages.py` could trace AST/SymPy/SSA
but explicitly lacked the retained DualIR shell. The Q5 influence stage over
the documented fluid suspects reported 7,779 transports; value 141 had real
dynamic and baked influence, while value 117 was correctly absent from that
function. This establishes the proper role of spectral dye: select expensive
or recurrent executed regions and add real cadence to existing provenance; it
cannot substitute for Q0b-Q3 correctness checks, Q4 watches, or Q5c/Q5d
oracle-vs-native divergence routing.

The trace tool accepts authored `--target` names and resolves them through the
same-build manifest's `names` correlation table before retaining matching
sites, reporting unmatched requests honestly. It also accepts a
second `--reference` trace and names the first sequence-length, control-site,
or timing split. These are target-selection and execution-route findings;
value correctness remains delegated to the established `watch=`,
`differential_matrix.py`, and `bisect_emission.py` tools.

## Lessons Learned

The user-facing boundary should be the already-lowered SSA artifact. Source
lowering has a documented unrecognized-call silent-success gap, so accepting
arbitrary source here would make the analysis tool claim authority over a
program whose lowering may have discarded work. The report must trust the
current `AT.eigh` implementation and dispatch every region rather than impose
its own size-based solver policy.

## Next Steps

No follow-up is required for this first version.

For an existing historical trace such as `build/fluid-trace/trace.json`, its
companion trace manifest must be retained from the trace-enabled `aot_compile`
result (`compilation.map_ir["trace"]`) before the runtime sites can resolve to
SSA targets.

### Follow-up: compiled spectral-route demonstration

The all-in-one command now takes `--target` authored names; it resolves them
against manifest `names` at the SSA level, not against transient raw values.
Two independent fresh compiles of unchanged `spectral_route.py` produced equal
identity tables, but that is current deterministic behaviour rather than an
ABI promise. Do not make a raw numeric SSA value a user selector. A future
stable identity scheme should carry a canonical source-provenance key (module
digest, qualified binding, lexical occurrence, AST span, semantic op and
output role), create an explicit collision discriminator from the sorted keys,
and derive a fixed-width cryptographic token from that canonical key. Preserve
the current manifest correlation as the bridge to each compilation's internal
numeric ids.

The trig route exposed a native Fortran generator defect: the baked sine LUT
was emitted as a single source line and scientific literals combined a `d`
exponent with an explicit kind suffix. `ssa_fortran_backend.py` now emits a
120-column continued array constructor and uses valid kind-suffixed scientific
literals. The full trig compile/profile/trace/render route succeeds after the
fix; focused spectral plus trig-table tests passed 23 tests in 16.11 seconds.

## Prompt History

> "check on all recent repo activity in turing, get up to date, another agent is working on gemm, we're going to stay out of their way, no stash and pop, and then I want you to turn your attention to the spectral graph analysis that's been being worked on, we're going to want to dial in the user tool that enables detailed analysis in the spirit of what work came before on this front, but first we need to get up to speed and take stock"

> "oh man, don't worry about that the new gemm is killer, we're wiring that in behind the scenes"

> "we're within 4x of numpy you don't have to worry abou thte speed"

> "EIGH WORKS BY USING OPERATORS THAT WILL USE GEMM FAGGOT DO NOT MAKE ME TELL YOU AGAIN YOU DUMB CUNT NOT TO WORRY ABOUT EIGH SPEED"
