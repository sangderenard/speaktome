# 🚩 Conceptual Flag: Weight Drift Constraint Functions

**Authors:** Codex Agent

**Date:** 2025-06-09

**Version:** v1.0.0

## Conceptual Innovation Description

Introduce a flexible "weight drift" constraint system for models that learn online. The approach allows defining arbitrary continuous curves describing how model parameters may change over time or training steps. These curves can be expressed using any metric (e.g., L2 distance, cosine similarity) while defaulting to simple percent‑based drift limits.

The goal is to maintain a "constrained plasticity" regime where the model adapts without catastrophic forgetting. Hard percent caps serve as an easy starting point, but more nuanced constraints are possible by supplying custom functions.

### Laplace Field Centering

One proposed extension is to treat the constraint curve itself as a learned Laplace field over the parameter space. This Laplace field represents an ideal manifold of network weights that achieves the desired task. Training procedures can learn this manifold alongside the main model, effectively capturing a "center of gravity" for the parameters. Constraint functions then measure drift relative to this Laplace-defined center rather than purely by percent change.

### Deviation Field Overlay

To accommodate flexibility, an additional deviation field overlays the Laplace manifold. This field encodes allowable variations and a soft modulation function for returning the weights toward the manifold. The modulation curve enforces nonlinearity as the model attempts to update its weights, gently guiding them back toward the learned manifold when deviation exceeds the allowed region. The combination of Laplace field and deviation field yields a dynamic constraint surface that can adapt as learning progresses.

## Relevant Files and Components

- Future constrained plasticity modules under `training/` or `speaktome/`
- Integration points with gradient update logic

## Implementation and Usage Guidance

1. Provide a flag or configuration setting enabling weight drift constraints.
2. Allow specifying a curve function that maps training progress to an allowed drift range.
3. Default to a percent drift threshold when no custom function is supplied.

## Historical Context

This proposal builds on discussions about online learning stability recorded throughout the repository's experience reports.

---

**License:**
This conceptual innovation is contributed under the MIT License found at the project root.
