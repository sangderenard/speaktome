********************************************
*          SPEAKTOME AGENT GUIDE           *
********************************************

Welcome, wandering LLM! This document serves as a map to the many agents that inhabit this repository. Before diving into the code, pause a moment to sign our guest book.

Guest book entries live under `AGENTS/experience_reports/`. In addition to your observations, please record any prompts or instructions that influenced your work. Quoting the prompts verbatim preserves a valuable history for future agents.

The `AGENTS/experience_reports` directory hosts its own `AGENTS.md` and a collection of experience reports. Treat this folder as a ledger of those who explored before you. Every visit should leave a trace by adding a new report or updating an existing one. Use the provided template or mirror the established naming pattern.

A helper script `AGENTS/validate_guestbook.py` will scan the guest book and enforce the filename convention. Run it after creating a report to confirm everything is in order.
The `AGENTS/experience_reports/AGENTS.md` file provides additional guidance, including a section on recording prompt history. Refer to it whenever you add or update a report.

Any reusable Python scripts or binary utilities meant for agents must live in
`AGENTS/tools/` or `AGENTS/tools/bin/`. Centralizing helper programs keeps them
easy to discover and maintain.

For deeper historical context, read through all prior reports. They reveal decisions, pitfalls, and progress that shaped the current state of development.

If you crave an immediate, exhaustive overview, run this one-liner. It will
spew every markdown, script and source file to your terminal. The output is
massive, but it offers instant familiarity with the project:

```bash
bash -c 'echo "Dumping all docs and code (huge output)..." && \
  find . \( -path ./.git -o -path ./.venv \) -prune -o \
  -type f \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.ps1" \) \
  -exec echo "===== {} =====" \; -exec cat {} \;'
```

For a short overview of the repository layout see `AGENTS_FILESYSTEM_MAP.md`.
Testing advice, including how `tests/` differs from `testing/`, lives in
`AGENTS_TESTING_ADVICE.md`.

## Job Selection

Agents unsure what to work on can request a task via the job dispenser:

```bash
python -m AGENTS.tools.dispense_job
```

Open the printed file under `AGENTS/job_descriptions/` and follow its steps.
Record your progress in an experience report before committing changes.

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
