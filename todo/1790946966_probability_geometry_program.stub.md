# Probability-geometry program (SPEAK_TO_ME_PROBABILITY_GEOMETRY.md)

See ../SPEAK_TO_ME_PROBABILITY_GEOMETRY.md, especially §13 (experiments) and §17
(map onto existing code). Extend the existing systems; do not build a parallel graph.

1. Retention: keep the per-node next-token distribution (full, or top-k plus tail
   mass) that LookaheadController / FluxGraph already compute via log_softmax and
   then discard. Derive entropy H and top-logit margin m from it per node.
2. §13.1 minimal experiment: parent-child and sibling Jensen-Shannon distance over a
   fixed-prompt top-k / depth-d expansion; compare textual vs distributional distance;
   list candidate reconvergence events. Report numbers before claiming anything.
3. Provenance: Edge.formation already separates postfix_beam (forward conditional)
   from prefix_beam (ImplicitBackpathScorer score). Add a distinct BehavioralRelation
   kind for JS-similarity links; never fold it into Edge.
4. Adaptive budget (§3/§13.2): if pursued, enters through FluxGraph's existing
   reward/reproduction path, not as a separate allocator. Ask before designing.
5. Later: cross-model / cross-precision JS atlas (§9, §13.3) via AbstractTensor;
   white-box Fisher / readout diagnostics (§5-§6, §13.4).

All §14 "hypotheses" stay labeled as hypotheses until measured.
