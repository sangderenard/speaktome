# Speak to Me: Navigable Token-Probability Geometry

**Technical report / design rationale — 2026-10-02**

> **Status in this repository:** this is the project's research thesis. It
> sits alongside [`VISION_FORWARD_BACKWARD_DIFFUSION.md`](VISION_FORWARD_BACKWARD_DIFFUSION.md)
> (which describes the forward/backward beam *machinery*) and states what
> that machinery is *for*: sampling and navigating the conditional-probability
> geometry of an autoregressive model. Sections 1–16 are the report as
> received; mathematics was converted from a damaged Pandoc/LaTeX export to
> plain `$…$` notation with no change in meaning. Section 17 was added on
> integration and maps the report onto existing code.

## Abstract

Autoregressive language models expose, at every prefix, a probability
distribution over the next token. Conventional decoding repeatedly
collapses this distribution into one token or a small beam of strings.
*Speak to Me* instead treats recursively generated conditional
distributions as an object to explore: a graph of reachable textual
states carrying probability fields, uncertainty measures, distributional
distances, and, where available, latent/readout geometry.

This report connects that design to recent research on Jensen–Shannon
divergence for token disagreement, Fisher/KL output geometry,
uncertainty-guided exploration, readout geometry, and causal
relationships between representation and behavioral manifolds. It
explicitly separates literature-supported results from proposed Speak to
Me mechanisms: adaptive expansion, behavioral reconvergence, backward
exploration, bifurcation diagnostics, and geometry-aware graph layout.

## 1. Core object

For vocabulary $V$, an autoregressive model assigns each context $x$

$$P_x(v) = P(v \mid x), \qquad v \in V.$$

Generation usually reduces $P_x$ to a token. Speak to Me instead retains
a sampled family of contexts:

$$x \rightarrow x+t_i \rightarrow x+t_i+t_j \rightarrow \cdots$$

and associates every context with a point in the probability simplex,

$$x \mapsto P_x \in \Delta^{|V|-1}.$$

A node can minimally store

$$n_x = (x,\ \ell_x,\ P_x,\ H(P_x),\ m_x,\ \ldots),$$

where $\ell_x$ is the logit vector, $H$ entropy, and $m_x$ a
leading-token margin. An edge $x \rightarrow x+t$ has both a
transition probability

$$w_{\mathrm{transition}} = P(t \mid x)$$

and a change in predictive state

$$P_x \rightarrow P_{x+t}.$$

These answer different questions: *how likely was the token?* versus
*how much did taking it change the model's expectations?*

## 2. Jensen–Shannon distance

For distributions $P, Q$, let $M = (P+Q)/2$. Jensen–Shannon divergence is

$$D_{\mathrm{JS}}(P,Q) = \tfrac12 D_{\mathrm{KL}}(P \,\|\, M) + \tfrac12 D_{\mathrm{KL}}(Q \,\|\, M).$$

It is symmetric and finite. Its square root

$$d_{\mathrm{JS}}(P,Q) = \sqrt{D_{\mathrm{JS}}(P,Q)}$$

is a metric. This makes it useful for comparing predictive states.

Li et al. [1] introduce Divergent Token Confidence (DTC), locating
positions where two models' next-token distributions disagree strongly
under Jensen–Shannon divergence. They report that counts of such
positions correlate strongly with reasoning accuracy and improve
calibration over several token-probability confidence baselines. The
important result for Speak to Me is that the *whole next-token
distribution* contains useful structure not captured by the selected
token's probability.

DTC deliberately compresses divergence into a count. Speak to Me can
retain what that estimator discards: where divergence occurs, its
magnitude, which alternatives produce it, which branches originate
there, and whether they later separate or reconverge.

**Proposed project metric:**

$$\Delta_{\mathrm{behavior}}(x,t) = d_{\mathrm{JS}}\big(P(\cdot \mid x),\ P(\cdot \mid x+t)\big).$$

For siblings $x+t_i,\ x+t_j$,

$$d_{\mathrm{sibling}}(i,j) = d_{\mathrm{JS}}\big(P(\cdot \mid x+t_i),\ P(\cdot \mid x+t_j)\big).$$

This separates lexical fan-out from behavioral fan-out.

## 3. Entropy is not geometry

$$H(P_x) = -\sum_v P_x(v) \log P_x(v).$$

Entropy measures spread, but two distributions with equal entropy may
have entirely different organization: two opposed alternatives, many
synonyms, several semantic families, or branches that immediately
reconverge.

Recent entropy-guided reasoning work reinforces the usefulness of
interpreting uncertainty in relation to exploration rather than treating
it only as scalar confidence. For Speak to Me, entropy is best
understood as one indicator of **branching opportunity**, not a complete
local geometry.

**Project proposal:** replace fixed beam width with adaptive expansion,

$$k(x) = f\big(H(P_x),\ m_x,\ \text{local separation},\ \text{reconvergence},\ \text{user focus}\big).$$

A low-entropy region whose candidate successors induce nearly identical
distributions may deserve little graph budget. A high-entropy region
containing behaviorally separated successors deserves more. This
adaptive policy is a Speak to Me proposal, not a result claimed by the
cited papers.

## 4. Behavioral reconvergence

A decoding tree permanently separates distinct strings. Predictive
behavior need not.

For trajectories $a, b$,

$$R_t(a,b) = d_{\mathrm{JS}}\big(P(\cdot \mid x_t^{(a)}),\ P(\cdot \mid x_t^{(b)})\big).$$

If $R_{t+k} \ll R_t$, the paths have behaviorally reconverged
despite retaining distinct textual histories.

This motivates a graph rather than a tree. Nodes can remain textually
distinct while receiving a relation

$$n_i \sim_{\mathrm{behavior}} n_j$$

when their predictive distributions become sufficiently close. The
browser can then expose textual divergence with behavioral convergence,
textual similarity with behavioral divergence, predictive basins,
excursions, and returns.

Reconvergence detection is a project hypothesis requiring empirical
validation.

## 5. Local information geometry

Jensen–Shannon distance compares observed distributions. A
complementary question asks which *infinitesimal* directions at one
state change the output distribution most.

Entesari et al. [2], studying pre-logit steering, use the local KL
geometry of the token distribution. For final hidden state $h$, output
matrix $W$, and token probabilities $p$, the induced Fisher quadratic
uses

$$F = W^\top\big[\operatorname{diag}(p) - p p^\top\big] W.$$

For a small hidden displacement $u$,

$$D_{\mathrm{KL}}(p_u \,\|\, p) \approx \tfrac12\, u^\top F u.$$

Their work derives analytic relations between this local Fisher term and
sequence-level KL behavior and uses the geometry to regularize steering.

For Speak to Me, JS supplies finite observed displacement while Fisher
geometry supplies a local sensitivity tensor. A selected node could
therefore expose not only uncertainty but anisotropy: directions with
large $u^\top F u$ are behaviorally consequential at the
readout, while other hidden directions may be comparatively invisible.

## 6. Latent geometry is not readout geometry

Yuan et al. [3] test whether extracted concept subspaces align with
dominant right-singular directions of the unembedding matrix. Across
their tested models, several activation-derived concept estimators carry
little energy in the dominant readout span, while output-oriented
controls align substantially more strongly.

The methodological warning is:

$$\text{latent similarity} \not\equiv \text{output similarity}.$$

Speak to Me should therefore distinguish a latent manifold
$\mathcal{M}_{\mathrm{latent}}$ from a probability/readout
manifold $\mathcal{M}_{\mathrm{probability}}$. Where
white-box access exists, the browser can expose the map

$$h \rightarrow \ell = Wh + b \rightarrow P = \operatorname{softmax}(\ell)$$

rather than treating a direct logit-lens projection as a complete
interpretation of an intermediate state.

Useful diagnostics include unembedding singular spectra, principal
angles to readout subspaces, and comparison of latent-space versus
probability-space neighborhoods.

## 7. Representation manifolds and behavioral manifolds

Wurgaft et al. [4] provide evidence that internal representation
geometry can correspond causally to output behavior. They fit manifolds
to activation states and output probability distributions, then
intervene along different activation-space paths. Manifold-respecting
steering produces behavioral trajectories closer to the model's natural
behavior than straight Euclidean steering; conversely, optimizing
internal paths to follow desired behavioral trajectories can recover
curved activation-space paths.

This supports treating probability geometry as more than a decorative
layout. It may expose behavioral structure systematically related to
internal geometry. But Speak to Me should not assume that the
relationship is globally simple, isometric, or one-to-one. The readout
results above specifically warn against that.

A long-term research question is:

$$\text{Which geometric structures survive } \mathcal{M}_{\mathrm{latent}} \rightarrow \mathcal{M}_{\mathrm{probability}}\,?$$

## 8. Margins and bifurcations

For ordered logits $\ell_{(1)} \ge \ell_{(2)} \ge \cdots$, define

$$m(x) = \ell_{(1)} - \ell_{(2)}.$$

Small margin means a small relative perturbation can exchange the
leading tokens. Under greedy or narrow-beam decoding, that is a discrete
event:

$$\text{continuous logit perturbation} \rightarrow \text{token switch} \rightarrow \text{different autoregressive trajectory}.$$

Speak to Me can mark small-margin states as potential bifurcation
regions. Margin should remain distinct from entropy: a distribution can
have a broad tail but a clear winner, or two nearly tied leaders with
little remaining mass.

## 9. Cross-model and cross-precision comparison

The same distributional machinery naturally compares

$$P_x^{(A)} \quad\text{and}\quad P_x^{(B)}$$

for different models, checkpoints, fine-tunes, ablations, quantizations,
prompts, or arithmetic precisions.

For precision $p, q$,

$$D_{\mathrm{precision}}(x; p, q) = d_{\mathrm{JS}}\big(P_x^{(p)},\ P_x^{(q)}\big).$$

This separates ordinary numerical deviation from behaviorally meaningful
movement. Most arithmetic error may be absorbed; a small subset can
cross ranking boundaries and create new autoregressive trajectories.

This is especially useful if the runtime eventually supports
controllable extended or limb precision: identical model state can be
replayed under multiple numerical regimes and the resulting predictive
geometry compared directly.

## 10. Backward exploration

Forward generation is native:

$$x \rightarrow x+t.$$

A probability browser can also ask a complementary question: which
plausible predecessor contexts could lead into a selected state or
predictive neighborhood?

Exact inversion of an autoregressive model is generally unavailable, so
backward exploration should be described carefully. It is an inferred or
searched predecessor relation, not the literal inverse of
$P(t \mid x)$.

A backward explorer can nevertheless be useful for finding multiple
textual histories that converge toward similar predictive states.
Coupled with forward expansion, this turns the interface from a decoding
tree into a neighborhood explorer.

**Project proposal:** maintain separate provenance for:

1. exact forward conditional probabilities;
2. searched/inferred predecessor scores;
3. behavioral similarity edges.

Conflating these edge types would make the visualization mathematically
misleading.

## 11. Graph layout should not define the science

A spring layout, sphere, hyperbolic projection, or other visualization
is a coordinate choice. The measured quantities should remain
independent of display coordinates.

The graph may use visual channels such as:

- edge thickness: transition probability;
- edge length: JS distance or another behavioral metric;
- node size: expansion budget or probability mass;
- halo/radius: entropy;
- warning marker: small top-logit margin;
- clustering: behavioral reconvergence;
- optional ellipsoid/tensor glyph: local Fisher anisotropy;
- alternate layer: latent/readout alignment.

The renderer should preserve raw measurements so that changing the
layout does not change the interpretation.

## 12. Proposed node and edge schema

A practical node record might evolve toward:

```text
Node {
    context_id
    token_id
    parent_ids[]
    text_or_token_path
    logits
    probabilities
    entropy
    top_margin
    effective_support
    optional_hidden_state
    optional_fisher_summary
    optional_readout_projection
    provenance
}
```

Edges should be typed:

```text
ForwardEdge {
    parent
    child
    token
    conditional_probability
    log_probability
    behavioral_distance
}

BehavioralRelation {
    node_a
    node_b
    distribution_distance
    threshold_or_cluster_provenance
}

BackwardCandidate {
    candidate_predecessor
    target
    inference_method
    score
}
```

The distinction is important: graph topology can contain empirical model
transitions, inferred relations, and visualization-derived neighborhoods
simultaneously, but they should never silently become the same kind of
evidence.

## 13. Experimental program

### 13.1 Minimal experiment: probability geometry

For a fixed model and prompt:

1. expand top-$k$ successors to depth $d$;
2. retain logits and full or sufficiently complete probability
   distributions;
3. compute entropy and top-logit margin at every node;
4. compute parent-child and sibling JS distances;
5. compare textual tree distance with distributional distance;
6. identify candidate reconvergence events.

Questions:

- Do lexically distant paths reconverge behaviorally?
- Do near-identical prefixes sometimes separate sharply?
- Does entropy predict behavioral fan-out?
- Does margin predict unstable branch identity?
- How quickly does JS distance accumulate or contract along paths?

### 13.2 Adaptive exploration

Compare fixed-$k$ search with geometry-aware allocation under an equal
model-evaluation budget.

Measure:

- distinct behavioral regions discovered;
- redundancy among expanded children;
- coverage of probability mass;
- number of reconvergence events;
- diversity of downstream predictive distributions.

The relevant claim is not that adaptive search generates "better text."
It is that it samples the model's predictive state space more
efficiently for inspection.

### 13.3 Cross-model divergence

Run identical contexts through two models and map

$$x \mapsto d_{\mathrm{JS}}\big(P_x^{(A)},\ P_x^{(B)}\big).$$

Then expand around high-divergence nodes. This generalizes the insight
behind DTC from a scalar confidence estimator into an interactive
disagreement atlas.

### 13.4 White-box geometry

Where hidden states are available, compare:

- hidden-state Euclidean/cosine distance;
- readout-projected distance;
- JS distance between outputs;
- Fisher-local distance.

The objective is to determine when internal proximity agrees with
behavioral proximity and when the readout strongly distorts it.

## 14. What is established, and what is proposed

### Supported directly by cited research

- Whole next-token distributions contain useful uncertainty
  information beyond selected-token probabilities [1].
- Jensen–Shannon divergence is useful for detecting token positions
  where predictive distributions disagree [1].
- Local KL/Fisher geometry characterizes sensitivity of token
  distributions to pre-logit interventions [2].
- Extracted latent concept geometry need not align strongly with
  dominant output-readout directions [3].
- Manifold-respecting internal interventions can induce more natural
  behavioral trajectories than straight Euclidean steering in studied
  settings [4].

### Speak to Me hypotheses/design proposals

- Treat recursive beam expansion as sampling a probability-field
  graph.
- Use JS distance as a behavioral edge length.
- Allocate expansion budget adaptively from entropy plus geometric
  separation.
- Detect and visualize behavioral reconvergence across textually
  distinct paths.
- Treat small logit margins as candidate bifurcation regions.
- Combine forward transition edges, inferred predecessor edges, and
  behavioral-similarity relations while preserving provenance.
- Use the browser to compare models, precisions, checkpoints, and
  interventions.
- Couple latent and probability geometries without assuming they are
  equivalent.

These should remain labeled as project hypotheses until experimentally
evaluated.

## 15. Attribution and accreditation

This report intentionally separates the Speak to Me synthesis from prior
work. The mathematical definitions of KL divergence, Jensen–Shannon
divergence, Fisher information geometry, entropy, singular-vector
analysis, and probability simplices are established
mathematical/statistical tools and are not original to Speak to Me.

The following recent works directly motivate specific project
measurements (titles and author lists checked against arXiv on
2026-10-02):

**[1] Feiyang Li, Shengjing Liu, Qi Zhan, Sijie Cheng, Weiqing Wang,
Hongwen Chen, Yuxuan Yang, Wen Wang, Yile Wang, Hui Huang.**
"Probability is Not Enough: Exploring and Counting Divergent Tokens for
Reasoning Uncertainty Quantification in LLMs." arXiv:2609.38070, 2026.
<https://arxiv.org/abs/2609.38070>
Contribution used here: token-level cross-model disagreement measured
with Jensen–Shannon divergence; Divergent Token Confidence; evidence
that disagreement structure can improve reasoning-confidence
calibration.

**[2] Taha Entesari, Jingyu Zhang, Daniel Khashabi, Mahyar Fazlyab.**
"Minimally Invasive Steering of Language Models." arXiv:2609.30218,
2026.
<https://arxiv.org/abs/2609.30218>
Contribution used here: local KL/Fisher geometry of induced token
distributions; analytic Fisher quadratic for pre-logit steering;
connection between local distributional sensitivity and intervention
cost.

**[3] Aojie Yuan, Zhiyuan Julian Su, Haiyue Zhang, Zijian Su.**
"Concept Subspaces Compute Beyond the Logit Lens: A Weights-Only Test
for Locating Representations Upstream of Readout." arXiv:2609.39263,
2026.
<https://arxiv.org/abs/2609.39263>
Contribution used here: geometric diagnostic comparing extracted concept
subspaces with dominant unembedding/readout directions; evidence that
upstream concept structure and direct readout alignment are distinct.

**[4] Daniel Wurgaft, Can Rager, Matthew Kowal, Vasudev Shyam,
Sheridan Feucht, Usha Bhalla, Tal Haklay, Eric Bigelow, Raphael Sarfati,
Thomas McGrath, Owen Lewis, Jack Merullo, Noah Goodman, Thomas Fel,
Atticus Geiger, Ekdeep Singh Lubana.** "Manifold Steering Reveals the
Shared Geometry of Neural Network Representation and Behavior."
arXiv:2605.05115, 2026.
<https://arxiv.org/abs/2605.05115>
Contribution used here: causal experiments relating activation-manifold
paths to behavioral/output-probability manifold paths and contrasting
manifold-respecting with Euclidean steering.

## 16. Project thesis

Speak to Me should not be framed merely as a beam-search visualizer.

Its stronger research framing is:

> **Speak to Me is an interactive instrument for sampling and navigating
> the evolving conditional-probability geometry of an autoregressive
> language model.**

The text graph supplies reachability and provenance. Probabilities
supply transition mass. Entropy supplies one measure of local branching.
Jensen–Shannon distance supplies finite behavioral separation. Margins
expose possible decoding bifurcations. Fisher geometry can supply local
sensitivity. White-box readout analysis can relate latent structure to
emitted behavior. Backward search can add candidate predecessor
neighborhoods.

The central object is therefore not a generated sentence and not even a
beam tree. It is the recursively sampled map

$$x \longmapsto P(\cdot \mid x),$$

together with the graph of textual operations that moves the model from
one predictive state to another.

That formulation gives the project a testable technical program:
determine which structures in this sampled probability geometry are
stable, informative, computationally exploitable, and related to known
internal geometry—and which are artifacts of sampling, projection, or
visualization.

---

## 17. Where this lands in the existing code (added on integration)

This section records how the report's vocabulary maps onto what already
exists, so future work extends these systems rather than building beside
them. It describes the code as found on 2026-10-02; it does not propose
replacing anything.

| Report concept | Existing system | Current state |
|---|---|---|
| Recursive expansion $x \to x+t$ | `speaktome/core/lookahead_controller.py` (`LookaheadController`), `beam_search.py` | Computes full `log_softmax` over the vocabulary each step, then keeps only `topk`. $P_x$ exists transiently and is discarded. |
| Probability-field graph (§1, §4) | `speaktome/core/flux_graph.py` (`FluxGraph`, `FluxNode`, `Edge`) | Already a DAG, not a tree (`FluxNode.parent_ids`), growing forward and backward from an anchor. Nodes store `local_evidence` (mean log-prob) only — no $P_x$, $H$, or $m_x$. |
| Backward exploration (§10) | `speaktome/core/implicit_backpath.py` (`ImplicitBackpathScorer`) | Exactly the report's "searched/inferred predecessor score": teacher-forced suffix likelihood under each prepended candidate. Not $P(t \mid x)$ inverted. |
| Typed edge provenance (§10, §12) | `Edge.formation` = `"postfix_beam"` / `"prefix_beam"`, permanent across re-roots | Separates forward conditional edges (≈ `ForwardEdge`) from backpath-scored edges (≈ `BackwardCandidate`). There is no `BehavioralRelation` edge type yet. |
| Adaptive expansion budget (§3) | `FluxGraph` tick: highest-pressure unexpanded nodes get compute | Budget is allocated by physical pressure, which by design model score cannot manufacture. Entropy/JS-informed allocation would have to enter through that existing reward/reproduction path, not as a separate allocator. |
| "Wide truth" region shape | `VISION_FORWARD_BACKWARD_DIFFUSION.md` | The vision's "shape of the model's belief at a region" is the informal form of §3–§4 here; entropy/JS/margin are the concrete measurements it was asking for. |
| Cross-precision comparison (§9) | `tensors/abstraction.py` (`AbstractTensor`) and the workspace `Precision` substrate | The backend-agnostic tensor layer is where replay under different numerical regimes would happen. |
| Layout ≠ science (§11) | `graph_layout.py`, `graph_visualizer.py`, `flux_radar/` | Display coordinates are already separate modules; the rule here is that raw measurements must be stored on nodes/edges, not derived from layout. |

The smallest first step toward §13.1 is therefore retention, not new
machinery: keep the per-node distribution (or a top-$k$ + tail-mass
summary) that `LookaheadController` and `FluxGraph` already compute and
throw away, and derive $H$, $m_x$, and parent/sibling $d_{\mathrm{JS}}$
from it. Outstanding work is tracked in
`todo/1790946966_probability_geometry_program.stub.md`.
