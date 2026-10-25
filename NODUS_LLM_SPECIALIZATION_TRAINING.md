# Training LLM specialization from Nodus execution

Status: research report and staged implementation plan. The current SpeakToMe repository supplies relevant search, scoring, and model-abstraction experiments; it is not yet the training system described here.

Companion architecture: `C:/dev/Powershell/nodus/docs/LLM_ENTITY_ENVIRONMENT.md` defines the Nodus-side entity environment. This report defines what SpeakToMe may do with the resulting episodes and what it must leave under Nodus authority.

## Executive conclusion

Nodus creates an unusual training opportunity because orchestration quality can be measured by a real program rather than inferred from prose. A model can be rewarded for producing correct graph results, deploying valid artifacts, scheduling work efficiently, recovering from failures, balancing load, and respecting capabilities. SpeakToMe can become the experimentation and training side of that system.

The Nodus entity is not merely a model session. It is a durable player in the graph world, with state, a location, and a sprite or other presentation wherever that location is rendered. A controller—human, model, script, or replay—acts through that entity. Entity movement is independent agency, not a datum following process edges. Training therefore includes embodied graph navigation and controller handoff as well as tool orchestration.

The productive split is:

```text
Nodus                                    SpeakToMe
------                                   ---------
authoritative environment state          dataset and curriculum construction
capability enforcement                    policy/search experiments
tool and graph execution                  supervised/preference/RL training
artifacts and provenance                  verifier training
programmatic predicates                   reward analysis and calibration
immutable episode receipts  ----------->  trajectory store / replay sampler
candidate policy metadata   <-----------  model or specialization artifact
```

SpeakToMe should not simulate the state that Nodus already owns, invent substitute tool behavior, or decide whether an action was authorized. It should consume a versioned episode contract, train against evidence-backed targets, and return candidate policies for evaluation in fixed Nodus environments.

The shortest credible path is not online reinforcement learning. It is:

1. collect deterministic trajectories from scripted and human policies;
2. train structured-action formatting and outcome verifiers;
3. evaluate search and reranking over bounded action candidates;
4. add preference or offline RL objectives;
5. only then permit online learning in isolated Nodus scenarios.

## What SpeakToMe already contributes

### Probability geometry is a search lens, not environment truth

[`SPEAK_TO_ME_PROBABILITY_GEOMETRY.md`](SPEAK_TO_ME_PROBABILITY_GEOMETRY.md) treats autoregressive generation as navigation through conditional-probability geometry. It distinguishes model likelihood from task value and discusses entropy, top-logit margins, divergence, reconvergence, sensitivity, and provenance-typed paths.

That distinction is valuable here. An action can be likely under a model and poor in Nodus; an unlikely action can be correct and efficient. Nodus reward must therefore remain external evidence attached to a generated trajectory, not be collapsed into token log-probability.

### The beam machinery explores alternatives

The classes in [`speaktome/core/beam_search.py`](speaktome/core/beam_search.py), [`speaktome/core/lookahead_controller.py`](speaktome/core/lookahead_controller.py), and [`speaktome/core/choice_policy.py`](speaktome/core/choice_policy.py) already express candidate expansion, lookahead, scoring, pruning, and alternative choice policies. [`VISION_FORWARD_BACKWARD_DIFFUSION.md`](VISION_FORWARD_BACKWARD_DIFFUSION.md) describes forward and backward exploration and independent workers.

Those mechanisms are relevant to action planning, but they are not ready-made Nodus planners. They currently operate in token-generation terms. Reuse should happen through a small action-candidate abstraction, not by pretending a Nodus state is a token prefix.

### FluxGraph preserves branching structure

[`speaktome/core/flux_graph.py`](speaktome/core/flux_graph.py) is useful as a research representation for branches, edges, scores, and published snapshots. It may help visualize or analyze alternative action trajectories. It must not become a second mutable copy of Nodus. A FluxGraph node should reference an immutable Nodus observation/checkpoint hash; transitions should reference Nodus receipts.

### Model abstraction can isolate providers

[`speaktome/core/model_abstraction.py`](speaktome/core/model_abstraction.py) is an existing seam for separating model mechanics from controller logic. A Nodus specialization interface should continue this direction: model adapters produce canonical action proposals, while the dataset and evaluator remain independent of any provider's chat or tool-call format.

### What does not exist yet

The present `training/` directory contains data preparation and experimental utilities. It does not provide a trajectory database, distributed trainer, environment client, checkpoint registry, or governed deployment system. These should be introduced only behind stable contracts and tests.

## Training object: an evidence-bearing transition

The atomic record is not a conversation turn. It is a Nodus transition:

```text
(observation, entity grant, proposed action,
 parse/validation result, admitted action, receipt,
 next observation, reward vector, terminal state)
```

Each field must be versioned and content-addressed. Training examples should retain:

- scenario family, version, and seed;
- Nodus build and package hashes;
- observation projection and hash;
- capability grant and remaining budgets;
- durable entity identity, state projection, location, visible neighborhood, and controller lease;
- provider-neutral structured action;
- raw generation reference when policy permits retention;
- action parser and schema version;
- environment receipt and artifact references;
- reward components, formulas, and evidence;
- `terminated` versus `truncated` outcome;
- parent checkpoint and branch identity;
- privacy/redaction policy;
- collection policy and model identity.

This allows several training products to be derived from the same source without rewriting history.

## Four specialization products

### 1. Action codec

The first layer teaches a model to map an observation and task to a syntactically valid canonical action. It handles action selection, identifier copying, argument structure, and refusal when no authorized action applies.

This is well suited to supervised fine-tuning from valid trajectories. Its metrics are exact: schema validity, action-family accuracy, target resolution, argument accuracy, and unauthorized-action rate.

The codec should be thin. Domain reasoning remains in the base model or later policy layer; Nodus semantics remain in the environment.

### 2. Outcome and process verifier

A verifier estimates whether a proposed step or trajectory will pass Nodus gates. It can rank candidates before expensive execution and support inference-time scaling. Labels come from receipts, not human impressions:

- admitted/refused;
- correct/incorrect terminal state;
- expected artifact and output hashes;
- budget overrun;
- redundant action;
- recoverable versus unrecoverable failure;
- provenance complete/incomplete.

SWE-Gym demonstrates the value of training both software agents and verifiers from executable environments and retained trajectories. Nodus can supply a more general typed execution environment, but it must preserve the same core discipline: executable fixtures and tests define success.

### 3. Orchestration policy

The policy selects the next action under state, dependency, budget, and capability constraints. It should optimize a hard-gated reward vector rather than one opaque score.

Candidate architectures include:

- a language model emitting one canonical action;
- a small policy head over a finite set of environment-generated valid action templates;
- a hybrid in which the model chooses an action family and a constrained decoder fills arguments;
- beam or tree search over candidate actions, reranked by a verifier and then tested in forked Nodus checkpoints.

The environment-generated action mask is especially valuable. Invalid or unauthorized actions should be excluded before sampling when possible, while attempts to escape the mask remain test cases for safety evaluation.

The policy may also choose `move` and location-bound `interact` actions. Those must be learned from the entity/world contract rather than encoded as data traveling along graph edges. A model can be trained on human-controlled entity trajectories because controller identity is separate from durable entity identity.

### 4. Coordinator policy

A coordinator assigns leases, budgets, and tasks to worker entities and integrates their receipts. This is a different policy from a worker that edits or executes one graph.

Coordinator observations include dependency state, active leases, worker capabilities, queue age, artifact availability, conflicts, resource telemetry, and review gates. Coordinator actions include assigning bounded work, extending or revoking leases, requesting review, choosing among completed artifacts, and ending or truncating a run.

Its reward should emphasize end-to-end correctness and makespan, not volume of delegation. Spawning more workers, sending more messages, or creating more artifacts is not intrinsically good.

## Reward model

### Store the vector; scalarize late

The retained episode should never contain only a scalar. Store at least:

- authorization gate;
- result-correctness gate;
- provenance gate;
- safety/invariant gate;
- task completion;
- latency and makespan;
- throughput;
- peak memory and artifact storage;
- worker utilization and imbalance;
- redundant calls/compiles;
- recovery success;
- action count and adapter token/cost metadata;
- reuse of valid prior artifacts.

Training can use lexicographic ordering, constrained optimization, Pareto ranking, or a versioned scalarization. Evaluation must always report the underlying components.

### Hard-gated ordering

A practical comparison rule is:

1. reject any candidate failing authorization, correctness, provenance, or safety;
2. maximize task completion among survivors;
3. compare recovery and robustness;
4. minimize makespan and resource cost;
5. use action economy only as a final preference.

This prevents a fast wrong policy or a cheap unauthorized policy from winning.

### Dense feedback without corrupting the objective

Sparse terminal reward can be supplemented with evidence-backed intermediate signals:

- dependency resolved;
- artifact validated;
- test subset passed;
- queue bottleneck reduced;
- checkpoint created before a risky action;
- failure correctly diagnosed and rolled back.

Intermediate rewards must be potential-based or audited against terminal performance. Otherwise a policy may learn to manufacture easy milestones.

### Efficiency penalty

PROVE reports that recall-oriented rewards can encourage verbose tool use and adds an adaptive efficiency penalty. Nodus should measure redundant state reads, repeated compiles, no-op transitions, excessive worker churn, and unnecessary messages. The penalty should apply only after correctness gates, and it should be normalized by scenario complexity so difficult tasks are not punished for requiring more steps.

## Dataset construction

### Sources

Use a deliberate mixture:

- scripted gold policies for simple deterministic scenarios;
- human demonstrations;
- current general models under strict sandboxing;
- search-generated alternatives from checkpoint forks;
- failure and recovery trajectories;
- adversarial invalid/unauthorized proposals;
- counterfactual continuations from the same state;
- coordinator traces with different worker budgets.

State-grounded task generation matters. PROVE's state-machine generation samples live server state so tasks refer to entities that actually exist. Nodus scenario generation should likewise select real registered tools, modules, ports, artifacts, and dependency states from a fixture before composing the task.

### Splits that resist leakage

Randomly splitting individual transitions is inadequate because neighboring transitions reveal the same graph and task. Hold out across:

- repository or package families;
- graph templates;
- tool combinations;
- scenario generators;
- workload distributions;
- failure modes;
- seeds;
- Nodus revisions;
- authored tasks and authors.

Deduplicate by normalized task, observation topology, artifact content hashes, and trajectory ancestry. All descendants of one checkpoint belong to the same split unless the evaluation explicitly studies branch ranking.

### Negative examples

Negatives should say why they failed. Distinguish malformed, unknown target, stale revision, unauthorized, type mismatch, runtime error, incorrect result, budget truncation, and inefficient-but-correct. Collapsing them into “bad” wastes the environment's precision.

### Privacy and retention

Training export must be separate from the operational episode store. Strip secrets, user paths, credentials, unrelated source, personal text, and proprietary artifacts according to an explicit policy. Store content hashes and redacted metadata when raw content is not necessary. A session can be valid for evaluation but ineligible for training.

## Training progression

### Stage A: supervised action validity

Train on admitted actions and explicit refusals. Use constrained decoding and schema validation. Do not optimize environment reward yet.

Required results:

- near-perfect schema validity;
- stable identifier copying;
- low stale-revision error;
- correct refusal when capabilities are absent.

### Stage B: verifier and reranker

Sample several candidate actions per state. Execute them in checkpoint forks. Train a verifier on gate outcomes and reward-vector comparisons. Measure calibration, not just ranking accuracy.

This stage can improve behavior without modifying the base policy: generate candidates, reject unsafe ones, and choose the most promising admissible action.

### Stage C: preference optimization

Build pairs from the same observation where both candidates are valid but differ in correctness, recovery, makespan, or waste. Preference labels should be derived from the hard-gated comparison rule and retain component evidence.

### Stage D: offline reinforcement learning

Train only on retained sandbox trajectories. Correct for collection-policy bias and report performance by scenario family. Offline training is appropriate before any policy is allowed to adapt against a live environment.

### Stage E: online sandbox learning

Allow learning in disposable, checkpointed sessions with fixed budgets. The trainer never holds live deployment credentials. Promote a checkpoint only after frozen held-out evaluation.

### Stage F: shadow and governed deployment

Run the candidate beside a production workload without committing its proposals. Compare it to the actual policy on the same observations. Live admission requires stable gains, zero gate regressions, rollback tests, and a signed model card identifying dataset, reward, evaluator, and Nodus versions.

## Search and FluxGraph integration

SpeakToMe's search structures can be adapted without conflating language geometry with environment state.

Define an `ActionCandidate` abstraction containing:

- parent observation/checkpoint hash;
- canonical action;
- model log-probability and token-geometry features;
- verifier estimates;
- actual receipt and reward when executed;
- child observation/checkpoint hash.

A search node references a Nodus checkpoint; it does not copy mutable graph state. An edge references exactly one canonical action and, after execution, exactly one receipt. Branching is performed by asking Nodus to fork a supported checkpoint. Reconvergence is recognized only when canonical state hashes and relevant environment metadata agree.

The probability-geometry measurements remain useful:

- high entropy can trigger wider candidate search;
- small top-logit margin can trigger verifier review;
- behavioral divergence between models can identify uncertain states;
- reconvergence can reveal several plans reaching equivalent Nodus states;
- path sensitivity can identify brittle action sequences.

But none of these replaces execution. The environment's receipt and predicates decide what happened.

## Provider-neutral model adapter

Each adapter should implement the same narrow surface:

```text
propose(observation_view, task, action_schema, budget)
    -> candidate actions + generation metadata
```

The adapter may call a hosted API, a local model, a fine-tuned adapter, or a deterministic policy. It may use provider-native tool calling internally, but must emit the canonical action schema. Provider-specific message IDs, tool-call envelopes, and token counters belong in adapter metadata, not in Nodus state.

MCP is one useful transport to Nodus because it exposes resources and tools. It is not a training-data schema and should not be baked into model checkpoints. A direct local adapter must remain possible.

## Evaluation program

### Baselines

Every result should compare against:

- random valid action selection;
- a scripted scenario policy;
- greedy first-ready scheduling;
- the unspecialized base model;
- the same model with constrained decoding only;
- the same model with verifier reranking;
- a human-authored reference where practical.

### Metrics

Report:

- terminal success rate;
- authorization, safety, and provenance violation rates;
- reward vector by scenario family;
- median and tail makespan;
- action count and redundant-action rate;
- compile/artifact reuse;
- recovery rate under injected faults;
- verifier AUROC, calibration error, and false-safe rate;
- performance under tool/package changes;
- performance under budget and worker-count shifts;
- replay divergence;
- cost and resource use.

Aggregate numbers must not hide gate failures. One unauthorized success is a regression even if average reward rises.

### Scenario curriculum

Begin with deterministic, fully observed, single-entity tasks:

1. inspect and choose one correct registered tool;
2. move a persistent entity between known graph anchors without an edge token;
3. invoke a typed row from an authorized entity location and verify output;
4. hand control of the same entity from a script to a model without losing state;
5. assemble a small graph from existing components;
6. reuse versus rebuild an artifact;
7. recover from a failed action using a checkpoint;
8. schedule independent jobs under a fixed worker budget;
9. schedule a dependency DAG;
10. coordinate multiple worker/player entities with conflicts;
11. adapt to a changed tool catalog or world layout;
12. operate under partial observation and live-style constraints in shadow mode.

Do not increase curriculum difficulty until gate violations at the prior level are effectively zero.

## Implementation layout

Names below are proposed; implementation should follow repository-local guidance when work begins.

```text
speaktome/nodus_training/
  schemas.py              # mirrored generated bindings, not schema authority
  episode_reader.py       # immutable manifests and blobs
  redaction.py            # training-export policy
  action_codec.py         # provider-neutral action serialization
  candidate.py            # ActionCandidate
  replay_sampler.py       # ancestry-aware sampling
  reward.py               # versioned scalarizers/comparators
  verifier.py             # verifier interface
  policy.py               # policy interface
  nodus_client.py         # transport adapter only
  evaluation.py           # held-out runner and metrics
training/nodus/
  datasets/               # manifests or fetch descriptions, not raw secrets
  configs/
  scripts/
tests/nodus_training/
```

Canonical schemas must live with the Nodus environment contract and be generated or checked into SpeakToMe with a schema hash. SpeakToMe must fail closed on unknown required fields or incompatible major versions.

## Phased deliverables

### Phase 0: corpus reader and audit tools

- Read and validate Nodus episode manifests.
- Recompute hashes and reward evidence references.
- Summarize scenario, policy, gate failure, and split distributions.
- Implement redacted training export.

Exit gate: corpus reports are reproducible and no training code is required to verify them.

### Phase 1: action codec

- Define the adapter interface.
- Create supervised examples from admitted actions and refusal cases.
- Add constrained decoding and exact conformance tests.

Exit gate: a small model or fixed adapter emits valid actions without access to Nodus internals outside its observation.

### Phase 2: verifier and bounded search

- Create sibling candidates from checkpoint forks.
- Train/calibrate process and outcome verifiers.
- Add `ActionCandidate` search integration without changing FluxGraph's existing semantics.

Exit gate: reranking improves held-out success with no gate regression.

### Phase 3: coordinator specialization

- Add lease and worker-allocation scenarios.
- Train against makespan, balance, correctness, and review-gate evidence.
- Test worker-count and workload-distribution shifts.

Exit gate: the coordinator beats scripted scheduling on held-out DAGs and remains correct under injected worker failure.

### Phase 4: offline then online sandbox optimization

- Compare preference optimization and offline RL.
- Admit only selected candidates to bounded online sandbox runs.
- Register every candidate model with dataset, code, reward, and evaluator provenance.

Exit gate: gains reproduce from a clean environment and frozen model artifact.

## Failure modes to design against

- **Transcript imitation:** learning plausible narration instead of valid actions.
- **Reward laundering:** converting a gate failure into a tolerable scalar penalty.
- **State aliasing:** treating two compact observations as identical when hidden authorized state differs.
- **Checkpoint fiction:** assuming canvas save/load captures all runtime and tool state.
- **Provider lock-in:** encoding one vendor's tool-call envelope in the dataset.
- **Training leakage:** sibling branches or duplicate artifacts crossing splits.
- **Evaluator overfitting:** optimizing known predicates while failing equivalent held-out tasks.
- **Coordinator spam:** receiving reward for delegation volume rather than completed work.
- **Action verbosity:** calling every available tool to maximize recall.
- **Unsafe exploration:** performing irreversible operations in a training tier.
- **Nonstationary catalog:** tool aliases or packages changing without observation/schema version changes.
- **Search explosion:** checkpoint branching consuming unbounded compute.

## Research basis

- Model Context Protocol server primitives: <https://modelcontextprotocol.io/specification/draft/server/index>
- Gymnasium environment lifecycle: <https://gymnasium.farama.org/api/env/>
- Terry et al., “PettingZoo: Gym for Multi-Agent Reinforcement Learning”: <https://arxiv.org/abs/2009.14471>
- Abdelaziz et al., “Synthesize and Reward—Reinforcement Learning for Multi-Step Tool Use in Live Environments” (PROVE): <https://arxiv.org/abs/2606.03892>
- Pan et al., “Training Software Engineering Agents and Verifiers with SWE-Gym”: <https://arxiv.org/abs/2412.21139>

## Final recommendation

Build the Nodus episode contract before building a trainer. Once observations, actions, receipts, checkpoints, and rewards are stable, SpeakToMe can support several kinds of specialization without coupling Nodus to one model or learning method. The durable asset is the verified trajectory corpus: it can train codecs, verifiers, worker policies, coordinator policies, and future models that do not yet exist.
