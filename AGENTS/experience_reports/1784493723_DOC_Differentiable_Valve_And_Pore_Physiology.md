# Differentiable Valve And Pore Physiology

**Date:** 1784493723
**Title:** Soft traversal-manifold survival learning

## Overview

Added a differentiable physiology layer to FluxGraph. Every continuous edge
delivery/return valve, traversal subedge valve, node hull, material-specific
node pore, heart nexus valve, and seed-reservoir gate now has a stable
sigmoid-constrained PyTorch logit when physiology learning is enabled.

The learning objective does not hard-select one route. Model-derived traversal
scores are converted into a temperature-controlled softmax distribution over
the complete audited causal manifold. Each gate receives the sum of the
score-weighted survival credit of every traversal that uses it, while every
open gate pays a global resource cost. Shared useful circulation therefore
opens and weak unsupported physiology closes.

## Steps Taken

- Added optional gradient-enabled forwards to `PyTorchModelWrapper`.
- Scoped model-gradient mode around individual FluxGraph model calls so a
  shared wrapper is not left globally in the wrong mode.
- Added server configuration for learning enablement, learning rate, resource
  cost, initial opening, traversal temperature, and model-forward gradient
  tracking.
- Added stable graph-owned trainable logits for edge, traversal, node, heart,
  and reservoir physiology.
- Applied learned edge valves to pressure conductance.
- Applied learned traversal valves to bulk flow, osmotic ion flow, and heart
  distribution.
- Applied learned hull/pore gates to humidity, ambient solute, and soil uptake.
- Applied learned heart valves and reservoir gates to each beat.
- Replaced hard best-route reward with soft score diffusion across all audited
  traversals.
- Algebraically accumulated one reward coefficient per gate before building
  the tensor loss, avoiding one autograd subgraph per traversal.
- Persisted learned logits and learning telemetry across server restarts.
- Published compact physiology summaries plus per-node and per-edge openings.
- Added web controls and heart-HUD telemetry.
- Added regressions for every gate category receiving a gradient, selective
  opening/closing, persistence, and model-forward gradient tracking.

## Observed Behaviour

- A 17-node focused graph created 48 trainable physiology parameters across
  edge, traversal, node, and heart categories.
- Every parameter changed on the first soft-manifold learning step.
- Manifold-supported shared gates opened while resource-only weak gates closed.
- The objective and parameters round-trip through saved fluid state.
- Model forwards remain inference-only by default but produce real PyTorch
  autograd graphs when explicitly enabled.
- The live server resumed its existing valid graph and is running the patched
  source.

## Lessons Learned

Top-k/top-p token choice is discrete and must not be represented as a
differentiable operation. Differentiability is honest when traversal scores
serve as soft reward weights over continuous physiology. This treats the
audited graph as a sampled model manifold without claiming gradients through
token identity.

Mutable conserved inventories remain scalar state. The learning path uses
native PyTorch tensors, which are the active server backend beneath the
repository's AbstractTensor scoring interface.

## Next Steps

None.

## Prompt History

> "could you please attempt to make sure all the parts of the server use abstract tensor if that's what's used across it, or otherwise torch, and make all the valves and pores for all edges and nodes learnable parameters, and we try to catch the model call for the sake of differentiability tracking - could we please give the network a reason to live by learning every valve and pore and judging it based on the best traversal score"

> "we can use any selection method that would be safe, we're not doing hard searching, we're diffusing through the model manifolds"
