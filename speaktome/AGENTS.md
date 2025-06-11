# SpeakToMe Source

## Quick Setup

```bash
python AGENTS/tools/dev_group_menu.py --install --codebases speaktome
python AGENTS/tools/dev_group_menu.py --install --codebases speaktome --groups speaktome:plot,ml,jax,ctensor,numpy,dev
```

This directory contains the primary beam search controllers and utilities for generating text. When adding new modules or functions, accompany them with tests in `../tests` and keep all code compliant with `AGENTS/CODING_STANDARDS.md`.

## Optional Dependency Groups

The `pyproject.toml` defines these groups:

- `plot`
- `ml`
- `jax`
- `ctensor`
- `numpy`
- `dev`

Use the group menu tool to install them non-interactively:

```bash
python AGENTS/tools/dev_group_menu.py --install \
    --codebases speaktome \
    --groups speaktome:plot,ml,jax,ctensor,numpy,dev
```

## Non-Interactive Setup Example

```bash
bash setup_env_dev.sh --extras --prefetch --from-dev
python AGENTS/tools/dev_group_menu.py --install \
    --codebases speaktome \
    --groups speaktome:dev
```

# Agents Documentation

This document describes the primary agents and components in the beam search and PyGeoMind-based graph search system.

## Core Components

### 1. **Scorer** (`scorer.py`)

* Evaluates beam scores using methods such as mean log probability, cosine similarity, sum log probability, and others.
* Contains GPT-2 tokenizer and model.
* Provides default scoring policies and configurable score bins.

### 2. **Scorer Bin Manager**

* `Scorer` now directly manages beam bins, handling aging, scoring, and culling.
* Bins are configurable with their own scoring function, width, and temperature.
* Utilizes the tensor abstraction layer so future backends remain possible.

### 3. **CompressedBeamTree** (`compressed_beam_tree.py`)

* Stores beams hierarchically, optimized for memory efficiency.
* Tracks leaf and internal nodes with PyTorch tensors.
* Supports adding and extending beams and integrates PyTorch Geometric (PyG) data structures.

### 4. **BeamSearchInstruction** (`beam_search_instruction.py`)

* Encapsulates instructions for beam search including actions (`expand_any`, `expand_targeted`, `promote`, etc.) and scoring configurations.
* Stores scoring policy parameters, lookahead configurations, and priorities.

### 5. **PyGeoMind** (`pygeo_mind.py`)

* Central decision-making GNN module for guiding beam search decisions.
* Combines linear encoder, Graph Convolutional Network (GCN), and GRU to process graph-structured data.
* Generates actions based on internal graph evaluations.

### 6. **PyGGraphController** (`pyg_graph_controller.py`)

* Controls the interaction between `BeamSearch`, `PyGeoMind`, and human decision-making.
* Handles beam expansion decisions, human-in-the-loop interactions, and auto-run scenarios.

### 7. **BeamSearch** (`beam_search.py`)

* Performs core beam expansion, scoring, and retirement processes.
* Incorporates lookahead capability to optimize beam selection.
* Interfaces with the `Scorer` bin manager for scoring and bin management.

### 8. **BeamRetirementManager** (`beam_retirement_manager.py`)

* Manages retired beams with a threaded queue system, filtering, and deduplication.
* Optimizes memory and GPU utilization through beam retirement strategies.

### 9. **HumanScorerPolicyManager** (`human_scorer_policy_manager.py`)

* Allows interactive configuration of scoring policies.
* Manages persistent storage and retrieval of user-defined policies.

### 10. **BeamGraphOperator** (`beam_graph_operator.py`)

* Handles moving beam nodes across devices (GPU/CPU) and supports serialization/deserialization of nodes.
* Facilitates hierarchical management of beams during expansion and retirement phases.

## Visualization and Debugging

### **BeamTreeVisualizer** (`beam_tree_visualizer.py`)

* Visualizes beam search trees using NetworkX and Matplotlib.
* Includes PCA-based visualization of sentence embeddings for deeper analysis.

## Configuration and Utility

### **Configuration** (`config.py`)

* Defines global settings such as device (CPU/GPU), GPU limits, and sequence lengths.
* Initializes models such as SentenceTransformer for embedding computations.

### **HumanPilotController** (`human_pilot_controller.py`)

* Provides a human interface for interactive beam exploration and control.
* Facilitates human-driven decision-making in beam expansion.

## Entry Point

### **Main Execution** (`speaktome.py`)

* Provides the entry point for executing beam search processes, supporting CLI arguments for customization.
* Integrates all core components into a runnable command-line application.

---

This document outlines the architecture and roles of each component within the repository, providing clarity for development and collaboration.
