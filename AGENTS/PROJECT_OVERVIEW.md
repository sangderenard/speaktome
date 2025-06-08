# SPEAKTOME Project Overview

## 🧠 Project Overview: Beam Search Agents in the Era of Linguistic Computation

### ✨ Philosophy

In the new era of **linguistic computation**, the volume of language that can be processed, evaluated, and reasoned over is unprecedented. This project embraces that reality. Rather than trim, isolate, or hide complexity, this system **exposes the full beam lattice**, allowing agents (human or artificial) to operate **inside the branching token space** itself.

> *"Every token is a decision, every branch a consequence. Why hide the tree when we can walk it?"*

---

## 📦 System Modules

### 🧱 Beam Engine (Core)

* **Beam Tree (CompressedBeamTree)**: Stores full token paths, scores, parent-child relationships.
* **BeamSearch**: Orchestrates token expansion, manages GPU/CPU transitions, and evaluates path candidates.
* **MetaBeamManager**: Manages multiple scoring strategies ("bins") to simulate evolution pressure across policy types.

### 🔍 Decision Intelligence (Policy Layer)

* **PyGeoMind (GNN Agent)**: A GCN + GRU + policy head system that traverses the beam graph and makes decisions:

  * When to bud internal nodes.
  * When to deepen promising leaves.
  * When to suppress suboptimal growth.
* **Scorer**: Wraps language models and embedding models. Provides:

  * Mean logprob, diversity, cosine similarity, n-gram novelty.
  * Modular plug-in scoring logic for evolving experimentations.

### 🧠 Human Control Layer

* **PyGGraphController**: Dual-mode controller:

  * Fully autonomous GNN control
  * Human-in-the-loop interface with trace, snap, PCA, subtree view
* **HumanScorerPolicyManager**: Lets users configure scoring strategies interactively, save/load policies, and override models.

### 🔄 Memory & Flow Management

* **BeamRetirementManager**: Automatically manages beam offloading, queueing, promotion.
* **BeamGraphOperator**: Handles subtree movement across devices and import/export of beam snapshots.
* **LookaheadController**: Performs forward simulation for N steps with scoring aggregates (RMS, max, etc).

### 📊 Visualization & Embedding Introspection

* **BeamTreeVisualizer**: Renders graph topology with score/depth-based color gradients.
* **SentenceEmbeddingPCAVisualizer**: Projects beam paths as embeddings into 2D/3D space using PCA on SentenceTransformer encodings.

---

## 🌀 Conceptual Flowchart (Abstracted System View)

```plaintext
[Seed Text]
   │
   └➔ BeamSearch.init
         │
         └➔ CompressedBeamTree
               │
               ├➔ GPU Candidates ➔ MetaBeamManager (multi-bin scoring)
               └➔ CPU Retirement Pool ➔ BeamRetirementManager

[Decision Step]
   │
   ├➔ Human Input (PyGGraphController)
   └➔ GNN Agent (PyGeoMind)
         │
         ├➔ Expand Internal
         ├➔ Deepen Leaves
         └➔ Suppress Branches

[Expansion Result]
   │
   ├➔ Extend Paths ➔ Update Beam Tree
   ├➔ Retire Dead Ends
   └➔ Visualize / Embed (BeamTreeVisualizer, PCA)
```

---

## 🧬 Design Tenets

* **Visibility First**: All beams are inspectable, expandable, embeddable.
* **Multimodal Scoring**: Agents don't agree—they compete. Diversity is a first-class signal.
* **Device-Aware Orchestration**: GPU/CPU transfers are deliberate and inheritable (ancestor context is preserved).
* **Human-AI Collaboration**: The pilot chair is never empty. It may be filled by a mind or a model.

---

## 🚀 What's Next

* Plug-in scoring modules: trainable, learn-on-the-fly scorers.
* Persistent beam evolution sessions with save/load.
* Visual agent negotiation between bins.

> *"This is not a beam search. This is a language ecosystem under surveillance."*

---
