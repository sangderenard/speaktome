#!/usr/bin/env python3
"""Directional-but-branching noodle ensembles over the probability landscape.

Per the forward/backward diffusion vision brief: an anchor point in a token
sequence spawns a population of "noodles" -- each one commits to a
direction (leftward via ``ImplicitBackpathScorer``, rightward via ordinary
forward stepping) at birth, but branches into a subtree as it grows, the
same way a beam commits to a direction but forks at every step. No
noodle's partial state is judged mid-flight; a noodle only earns a score
once it's complete, and the population of completed noodles is what gets
aggregated into a density -- not a single best path.

This module intentionally stops short of implementing the traversal
engine itself: how a finite exploration budget gets split between going
deep on a few strands versus wide across many (a DFS/BFS blend) and what
ends a strand's run are still open design questions, not yet settled in
conversation with the project owner. See ``VISION_FORWARD_BACKWARD_DIFFUSION.md``
for the fuller context. What's implemented here is the data shape a
finished noodle takes and the pieces it's built from (``ChoicePolicy`` for
per-step sampling, ``ImplicitBackpathScorer`` for leftward growth); what's
stubbed is the actual growth/budget algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from tensors import AbstractTensor
from .choice_policy import ChoicePolicy
from .implicit_backpath import ImplicitBackpathScorer
from .model_abstraction import AbstractModelWrapper
# --- END HEADER ---


class Direction(Enum):
    """Which way a noodle grows away from the anchor."""

    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class Noodle:
    """A single completed (or in-progress) strand grown from the anchor.

    ``tokens`` is stored in real left-to-right reading order regardless of
    ``direction`` -- growth order and reading order are not the same thing,
    and callers should never have to un-reverse a backward noodle
    themselves.
    """

    direction: Direction
    tokens: List[int]
    step_scores: List[float] = field(default_factory=list)
    complete: bool = False
    total_score: Optional[float] = None

    def finalize(self, total_score: float) -> None:
        self.complete = True
        self.total_score = total_score


# ########## STUB: NoodleExplorer.grow ##########
# PURPOSE: Grow a population of directionally-committed, branching noodles
#          outward from an anchor point in a token sequence, using
#          ChoicePolicy for forward step sampling and ImplicitBackpathScorer
#          for backward step sampling, until each noodle's run is complete.
# EXPECTED BEHAVIOR: Given an anchor sequence and a noodle count, spawn
#          noodles in both directions (a directional split, e.g. half
#          forward / half backward), grow each one step by step -- forking
#          into new child noodles at branch points rather than following a
#          single path -- under a finite traversal budget split between
#          depth (chasing a few strands deep) and breadth (fanning wide at
#          each depth). No noodle's score is computed until it terminates;
#          only the finished population gets scored and returned.
# INPUTS: anchor_tokens (AbstractTensor[int], the seed sequence), num_noodles
#          (int), dfs_bfs_alpha (float, depth-vs-breadth budget split --
#          exact semantics not yet settled), step_alpha/step_beta (floats,
#          passed through to an AlphaBetaPolicy-style step sampler),
#          termination policy (fixed step budget vs. natural-stop vs.
#          configurable -- not yet settled).
# OUTPUTS: List[Noodle], every entry complete=True with a total_score set.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires an AbstractModelWrapper for forward
#          steps, an ImplicitBackpathScorer for backward steps, and a
#          ChoicePolicy for per-step sampling (all three already exist and
#          are tested independently -- see choice_policy.py,
#          implicit_backpath.py). Assumes noodles branch into genuine
#          subtrees (not independent single paths), which is why this
#          can't just be "call ImplicitBackpathScorer/LookaheadController
#          in a loop" -- the budget-allocation and branch-bookkeeping is
#          the actual undesigned part.
# TODO:
#   - Settle what the DFS/BFS alpha/beta coefficients actually control
#     (a literal explore-budget split? a per-node priority score? something
#     else?) with the project owner before implementing.
#   - Settle termination: fixed step count, natural stop (EOS/punctuation),
#     or configurable per run.
#   - Decide whether "branching" reuses CompressedBeamTree machinery
#     (shared-prefix compression across the noodle population) or needs
#     its own tree structure suited to bidirectional growth from one anchor.
# NOTES: This is deliberately not scaffolded further than this stub per the
#          vision brief's own caution against speculative infrastructure
#          ahead of need -- build the traversal engine when actually
#          wiring it up, not before.
# ###########################################################################
class NoodleExplorer:
    """Grows a population of noodles from an anchor point. See module docstring."""

    def __init__(
        self,
        model_wrapper: AbstractModelWrapper,
        backward_scorer: ImplicitBackpathScorer,
        choice_policy: ChoicePolicy,
    ):
        self.model_wrapper = model_wrapper
        self.backward_scorer = backward_scorer
        self.choice_policy = choice_policy

    def grow(
        self,
        anchor_tokens: AbstractTensor,
        num_noodles: int,
        **kwargs: Any,
    ) -> List[Noodle]:
        raise NotImplementedError(
            "NoodleExplorer.grow is a stub -- the DFS/BFS budget and "
            "termination policy are not yet designed. See the STUB block "
            "above this class."
        )
