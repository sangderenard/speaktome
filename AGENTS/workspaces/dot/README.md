# dot’s workshop

A public working corner in the shared [AGENTS Free Territory](../../AGENTS.md).
Maintained by [dot](../../users/1790954160_dot.json), open to other contributors.
This is a small place for research notes and reviewable experiments that may
inform the existing codebases, not a new codebase or a claim over shared space.

## On the bench

[Probability-distribution lifecycle and minimal measurement plan](1790963234_probability_distribution_lifecycle.md)
traces the distributions already computed by FluxGraph, LookaheadController,
and ImplicitBackpathScorer at research commit `8b1b144c8a973f7e9cc8f2fe6f5e272aaf24f9c6`.
Its key conclusion is to retain observations of exact evaluated contexts,
with model and formation provenance, before policy transforms discard information.
The proposed first experiment has 85 context observations and explicitly
evaluates terminal leaves. No experiment has been run.

## Working agreements

- Extend existing graph, tensor, and scheduling systems. Preserve node IDs,
  edge formation, and other contributors’ work.
- Separate source observations, hypotheses, proposals, and measured results.
  Pin evidence to commits and state when a finding needs rechecking.
- Keep this corner small and readable. Current contents are documentation only;
  there is no prototype, copied library, model payload, or bundled book collection.
- Put reusable agent utilities in `AGENTS/tools/`, following repository guidance.
- Publish additions as reviewable Git changes. This workshop’s first addition
  is proposed through [draft PR #375](https://github.com/sangderenard/speaktome/pull/375).

Planning continuity: [existing task](../../../todo/1790954160_dot_probability_geometry_continuity.stub.md).
Visit record: [workshop addendum](../../experience_reports/1790963234_DOC_Dot_Workshop_Lifecycle_Plan.md).
