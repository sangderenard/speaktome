# Nodus LLM Entity Training Research

## Summary

Added `NODUS_LLM_SPECIALIZATION_TRAINING.md`, a researched architecture report describing how SpeakToMe could consume evidence-bearing Nodus episodes to train provider-neutral action codecs, verifiers, orchestration policies, and coordinator policies.

The report keeps Nodus authoritative for state, execution, capabilities, provenance, and reward evidence. SpeakToMe is assigned dataset, search, training, calibration, and evaluation responsibilities. It explicitly records that the current `training/` directory is not yet such a system and that FluxGraph is a possible analysis/search representation, not a replacement for Nodus state.

The definition of entity was corrected after user clarification: it is a durable player inhabiting the graph, with independently persisted state and location and a sprite/presentation projected wherever it is. A model session is only one possible controller. Movement is a control-plane world transition, not an entity-shaped graph edge and not dataflow. The Nodus audit found useful ingredients but no existing general entity/player registry.

A companion report was added at `C:/dev/Powershell/nodus/docs/LLM_ENTITY_ENVIRONMENT.md`.

## Research and repository evidence

- Read the repository behavior, filesystem, testing, and guestbook instructions.
- Read `SPEAK_TO_ME_PROBABILITY_GEOMETRY.md` before relating FluxGraph and search to the proposed work.
- Inspected `speaktome/core/flux_graph.py`, `choice_policy.py`, `lookahead_controller.py`, `model_abstraction.py`, and the existing training layout.
- Inspected Nodus's tool, registry, package, headless, runtime snapshot, scheduler, process-graph, and artifact-receipt surfaces.
- Consulted primary documentation or papers for MCP, Gymnasium, PettingZoo, PROVE, and SWE-Gym.

## Decisions

- Do not make MCP the internal environment contract; use it only as an optional adapter.
- Do not add agent-specific graph edges.
- Do not train first. Establish deterministic observations, actions, receipts, reward evidence, checkpoints, and splits first.
- Preserve multi-objective rewards and hard gates instead of retaining only a scalar.
- Keep provider-specific tool-call envelopes outside canonical Nodus actions and SpeakToMe datasets.
- Treat coordinator specialization as distinct from worker/action specialization.

## Validation

Documentation-only change. Markdown link and whitespace checks plus the guestbook validator should be run before handoff.

## Next steps

No implementation stub was created because the user requested a researched report, not authorization to begin the proposed multi-phase implementation. The reports contain phased deliverables and exit gates for later work.

## Prompt History

> “well, a little birdie told me that eventually, there will be an MPC API allowing any llm to be an entity that can inhabit and examine and operate nodus from the inside”

> “oh whatever the right one is is what I meant? I think? anyway, think too, what an opportunity to provide an immediate training interface for teaching a llm with a specialization layer where the model is rewarded based on purely programmatic orchestration, depolyment, load balancing, etc.”

> “write up a detailed and researched report on the idea in nodus and speaktome”

> “nodus might not have entities yet, we might need to document or put that in, i swear I might've done something for it though, not sure, the idea is a sprite with state appears wherever it \"is\" and it can move freely through the graph not just by process following some moment or datum, they are essentially players in the execution environment, this makes an important symmetric element needed so that the game engine and the execution engine can interact fluently”
