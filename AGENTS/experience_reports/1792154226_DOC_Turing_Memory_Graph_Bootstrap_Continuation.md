# Turing memory graph takeover and provenance continuation

Date: 2026-10-16. Repository: `C:/dev/Powershell/turing`, dirty main at
`f503ba22`.

## Outcome

The canonical public source compiler now materializes its `ProcessGraph` in the
repository's `BitTensorMemoryGraph`. The existing NetworkX-shaped adapter is
independent: no inheritance from, embedded instance of, or import of
`networkx.DiGraph`. Canonical relabel, extracted function graphs, empty scoped
worker catalogues, deferred scheduling, and checkpoint rehydration preserve the
same MG/G ownership contract rather than quietly replacing storage.

NetworkX is still permitted as an explicit linked execution target when the
host supplies it. The self-hosted compiler path uses the repository graph
implementation. The memory graph can now be pickled as physical bytes plus its
exact integer allocation map and rehydrated with working callbacks and
backreferences: a concrete first form of the user's executable-state "spore"
model. Semantic attribute dictionaries are not yet encoded into the byte arena,
so this is not a claim of complete payload compression.

The initial fresh provenance blockers were repaired from their identities, not source
spellings. Nested `set[str]` fields now complete the authored record ABI and the
exact `_fork_of_scope` expression lowers two typed membership operations.
Nested list values reached through `mapping.values()` now transfer their
existing aggregate/dtype receipt along the exact mapping/values/For edges to
the loop Input; the regression uses an unrelated target name to exclude the
earlier name-collision false positive. Strict sort validation remains.

A subsequent materialized run confirmed that both earlier errors were gone and
that both units reached SSA completion, then exposed deeper finalization facts.
Unit 026's scoped source omitted nested ClassDefs even though ProgramABI retained
the full `IdentityPage.book -> IdentityBook -> Registry.private_pages` record
contract; the reducer now reconstructs class and aggregate field contracts from
that authoritative ABI. Unit 031 deleted a provisional keyed descriptor before
rebinding its resident parts to ProgramABI's declared keys/values/length parts;
the transition now uses the existing binding and provenance mechanisms before
removal. Its exact direct `self.__dict__.get("_compressed_ram")` predicate is
now an authored bool projection consumed by the public wrapper.

The run is also useful negative evidence about current efficiency. It proves
that the materialized memory graph is the canonical topology store, but topology
RSS was about 3.12 GB versus about 1.68 GB in the preceding nonmaterialized
observation. The units stage was modestly faster (about 2m28s versus 2m42s), but
that single comparison does not establish causality. Rehydration and storage
authority are real; compressed payload utility remains unproven optimization
work rather than something to claim from architecture alone.

A third fresh run confirmed the unit 026/031 repairs and advanced to four new
receipts. Two were causal shape-publication defects: a canonical glyph constant
shape was duplicated through an undeclared raw page, while UTF-8 loop rewiring
discarded exact rewire receipt references before invalidation. The proven-shape
page is now declared and causal, rewire cells reach invalidation, and scalar
identities no longer manufacture withdrawals for nonexistent shape proofs.

The remaining partials were concrete boundary identity losses. Authored
`Any`/`object` values now use the existing `opaque_ref` ABI. Resident row loads
claim their earlier provisional definitions, tuple stream publications consume
their retained aggregate member IDs, and projected Python-object columns use
the same concrete reference transport. Finalization treats `opaque_ref` as a
resolved ABI, not an unknown dtype.

This run's topology completed in 1m20s at about 2.56 GB RSS, down from the prior
materialized observation of 3.12 GB but still above the 1.68 GB nonmaterialized
observation. Compilation units completed in 2m33s at 1.58 GB RSS with a 2.93 GB
stage peak. This is evidence of improving behavior, not yet proof of compression.

A fourth fresh run confirmed those four repairs and had no partial units. Unit
038 then rejected an exact glyph vocabulary reference only because pickle
rehydration rebuilt its Page as an equivalent but non-identical Python object.
The active identity book already contained the complete fact; validation now
resolves the retained row there and compares the full canonical Ref key. Unit
045 advanced through normalization and revealed that `_replace_cells(page:
Any, ...)` had erased the receiver class needed to interpret `{page.name}`.
Its exact `IdentityPage` annotation now selects the existing resident
string-token path from the retained ProgramABI field contract.

The expanded suite exposed that `proven_shape` and fourteen lazily declared
source/glyph/constant/metadata pages lacked spill lifecycle entries. They now
carry lowering or repair lifetimes, restoring the invariant that every declared
page participates in memory release. This matters directly to the memory-graph
goal: an unclassified page is conservatively retained even when its facts are
otherwise complete.

## Verification

The prior expanded gate passed 193 tests in 26.99 seconds, covering sequence
contracts, scalar source projections, ProgramABI record propagation, causal
shape pages, loop invalidation, concrete reference boundaries, provisional row
loads, aggregate stream publication, native boundaries, dispatch, memory
allocation and graph protocol, and a scoped compilation checkpoint. No full
bootstrap was run after the latest edits. A fresh provenance run is the next
authoritative observation; the frozen unit plans are invalid after toolchain
edits.

The final gate after the fourth-run repairs passed 345 tests in 51.72 seconds,
including the complete glyph-provenance and precompile-to-SSA modules, exact
rehydration/receiver regressions, and the declared-page lifecycle audit.

See
`turing/docs/compiler_convergence/17_memory_graph_bootstrap_repairs_continuation.md`
for the exact code paths, command, architectural boundary, and next step.

## Prompt History

> here is a fresh run, networkx is prepped to be replaced, check that it is the case that memory graph has now taken over, the networkx bypass is removed.

> please dutifully as a software engineering informational authority, proceed with fixing the errors

> they were not authorized to just embed networkx in the alternative

> I personally oversaw the creation of networkx methods for an abstract tensor based graph

> is there evidence this will provide utility or evidence that the memory graph is too convoluted without purpose

> a compressed storage of a rehydrateable programmatic state ... a kind of, spore of executable instruction

> networkx has never been a compile blocker and therefore is a valid target provided if it needs use of it's own binaries as recalled necessary links, then if the system compiling satisfies that need, we could use it, elsewise having the system use our versions of everything

> with the help of agents, achieve a stable state ... so that the abstract tensor memory graph is just transparently the means of acting like networkx, and we sit with the bootstrap in provenance core chipping away at the errors in the compiler preventing it
