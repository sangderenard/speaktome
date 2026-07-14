# Vision Brief: Parallel Forward/Backward Diffusion Beam System

This document exists to point any agent or contributor at the intended direction
for the beam search stack in this repository, before they start "optimizing"
it into something narrower. Read this before touching `speaktome/core/beam_search.py`,
`speaktome/core/lookahead_controller.py`, or `tensors/abstraction.py`.

## The existing building blocks

The current code already gives us the primitives we need:

- `tensors/abstraction.py` — `AbstractTensor`, a backend-agnostic tensor layer
  (PyTorch / NumPy / pure Python / JAX) so beam logic is not welded to one runtime.
- `speaktome/core/beam_search.py` — `BeamSearch`, which manages a tree of active
  and retired beam candidates over a sequence.
- `speaktome/core/lookahead_controller.py` — `LookaheadController`, which runs the
  actual autoregressive next-token loop used to score and extend candidates.
- `speaktome/core/scorer.py` / `model_abstraction.py` — the model wrapper
  (currently GPT-2) that the controller calls at each step.

Today this stack runs a single, forward-only, left-to-right beam search.

## Where we're taking it

The goal is to turn this into a **forward and backward beam system**, where beams
are grown in both directions from any point in a string, and can be deployed as
independent workers — batched and threaded — that each own a different region
of the sequence at the same time. Independent does not mean uncoordinated: workers
run in parallel and their beams get reconciled where regions overlap or meet.

This is deliberately compared to diffusion, but it is **not** a denoising diffusion
model in the usual sense (no noise schedule, no learned reverse process). The
resemblance is structural: instead of committing to one path through the string
token by token, estimation is spread across the whole space of a moment —
every position can be worked on independently and in parallel, the same way a
diffusion process treats the whole canvas at once instead of one raster line
at a time.

## The core idea: "wide truth" in a model region

The point of running beams forward *and* backward, and spreading many beams
across a local region instead of just following the single highest-probability
continuation, is to characterize the actual **shape** of the model's belief at
that region — not just its argmax.

A single greedy or narrow-beam path tells you the model's best guess. It does
not tell you whether that guess sits on a sharp, confident peak or a wide, flat
plateau of near-equally likely alternatives. The "wide truth" of a region is
that shape: the local probability mass, sampled broadly enough to see whether
the model is actually certain there, or just picking a winner among many close
options. Forward and backward beams covering the same region from opposite
directions are two independent readings of that same shape, and agreement or
disagreement between them is itself signal.

## What needs to be built on top of the existing code

- A backward-running counterpart to `LookaheadController` / `BeamSearch`
  (predicting leftward instead of rightward) sharing the same `AbstractTensor`
  and `CompressedBeamTree` machinery.
- A worker/orchestration layer that can assign independent regions of a string
  to separate beam workers, run them batched and threaded, and merge/reconcile
  results where regions overlap.
- Aggregation logic that turns "many beams around a region" into an explicit
  wide-truth estimate (e.g. entropy/shape of the local distribution), not just
  a top-1 or top-k list.

## What this is not

- Not a replacement for the existing forward beam search — it's a superset.
- Not a diffusion-model reimplementation — no denoising, no noise schedule.
- Not a request to add speculative infrastructure ahead of need — build the
  backward pass and the worker layer when we're actually wiring them up, not
  as scaffolding now.
