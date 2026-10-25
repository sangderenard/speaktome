# Self-hosted graph mechanics: correctness checkpoint, not completed overhaul

Date: 2026-10-09. Repository: `C:/dev/Powershell/turing`, dirty main at
`54c67fbd`. Untouched short baseline: `C:/dev/tg-base`.

## Observed result

The active native topology-provider build now compiles the authored
`src/compiler/graph_refinement_kernels.py` through the one public compiler
entry, then the existing LLVM artifact path. It no longer builds the historical
hand-written Fortran module. Registration stays explicit. Existing registry
tests prove the canonical reducer consults the registered implementation.

A 7.52-second bounded child test executes 100 deterministic small directed DAGs
and multigraphs on two synchronized threads, with equal-key lexical ties and
separate work arenas. Every output equals the existing NetworkX fallback.
This is concurrency isolation, not a measured compiler-wide speedup.

The heap/frontier micros drove real compiler repairs: RHS occurrence read
receipts survive indexed writes; Name captures do not re-expand their producers
at a different control site; indexed elements and call returns settle through
their existing exact edges; Store values and GEP indices are reads, not writable
aliases; int64 floor division does not round-trip through double; loop entries,
latches and break exits choose their actual binding generations; scalar Phis
capture values simultaneously before reusable output slots change. Ranked and
reference Phis retain arena identity. Incompatible physical Phi layouts fail
emission instead of compiling to unsafe reads.

The new `name_binding_capture` page records source/target binding versions and
AST-identity loop ownership, survives canonicalization/copy-on-read scopes, has
a concordance detector and cold-page lifecycle, and is indexed once per SSA
builder. An alias assigned inside a retained loop previously read its source's
pre-loop seed forever. The new public-entry test distinguishes that from an
alias assigned before the loop. No line-number proxy decides the identity.

Two-book forced-interleaving spill reads exposed the process-global unpickle
registry exchanging book objects. The registry is now a ContextVar. This does
not make shared-book append/mutation concurrent.

## Receipts and reproductions

Commands are run unbuffered with output inline. No new long orbital lowering,
trace, package installation or alternate source compiler was used. Native
crash-sensitive micros have 10/15-second subprocess bounds.

- `python -u -m pytest tests/test_graph_refinement_kernels.py -q --tb=short --show-capture=no`
- `python -u -m pytest tests/test_native_compiler_accelerators.py -q --tb=short --show-capture=no`
- `python -u tools/compiler_probes/probe_graph_region_frontiers.py 4`
- Writer observation: `python -u tools/compiler_probes/probe_graph_kernel_reads.py push`

The final LLVM/adapters/spill/graph batch passed 82 tests, skipped 3 in 63.91
seconds, including the last scope-index micro; the index and real capture proof
also passed together in 5.87 seconds. The memory/copy-on-read/loop-scope/provider
batch passed 47 tests in 17.33 seconds. The result-type/transformation/control/
operand/loop-composer batch passed 100 and failed the same eight tests as clean
HEAD (9.91 versus 9.66 seconds). Adding the five already-bounded native binding
cases gives 105 passed, the same eight failed, one historical timeout deselected
in 30.36 seconds. The 30-second `call-argument-carried` native timeout was also
reproduced separately on the clean baseline.

The four-call PUBLIC micro is not the orbital four-law benchmark. It lowers in
1.344 seconds, totals 1.683 seconds, has exact native parity, zero structural
findings, and C SHA256
`3f43760e36b1f3aec4cad2b11008b03d1a32325bc6c84359b81e13a8ea41af69`
identical to its baseline. Its 148 unsourced facts remain OPEN.

The complete, current receipt is in `turing/TEST_BASELINE_AND_HAZARDS.md`.
No commits. Existing dirty orbital probe, generated helper metadata/artifacts,
scratch memory probe and deleted build README belong to the user and were not
reverted. The old large N35 probe was previously stopped under authorization;
its caches were retained. No old trace was restarted.

## Architectural limits and next steps

This is a verified prerequisite slice, NOT the requested all-in-one conversion.
Whole source/expanded region/function collections are still resident and serial.
The provider's compact CSR/work arenas are still whole admitted operation tables.
The real remaining work is bounded region leases/admission against the existing
identity/provenance boundaries, workers on admitted immutable data, deterministic
publication of proven proposals on the one book, and work/residency progress
receipts. No shared-book writer threads, opportunistic unbounded executor queue,
or invisible fallback stand-in is licensed by these tests. Use the existing
region/scope/output receipts, not name/coordinate guesses. Keep public-entry
micros exact, child-bounded and small; the user launches long orbital lowerings.

Next-step stub: `speaktome/todo/turing_graph_residency_workers.stub.md`.

## Prompt History

> I want you to put tight iteration of meaningful micro-tests to overhaul our system to these ends in a coordinated, thoughtful, all in one conversion

> UNTIL WE BREAK THE NEED TO HAVE EVERYTHING IN RAM AND UNTIL WE BREAK THE NEED TO DO EVERYTHING ON ONE THREAD WE WILL NOT SEE IMPROVEMENT

> DO NOT USE OTHER COMPILERS, DON'T PUT A BUNCH OF PACKAGES IN THAT FUCK AROUND PRETENDING TO BE NATIVE

> WHEN I SAY COMPILE I MEAN OUR COMPILER

> we have all manner of data in our extended ir and provenance system

> The source compiler's public entry is
> `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`.
