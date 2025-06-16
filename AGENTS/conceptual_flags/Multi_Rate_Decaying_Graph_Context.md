# 🚩 Conceptual Flag: Multi-Rate Decaying Graph Context

**Authors:** Codex Agent

**Date:** 2025-06-15

**Version:** v1.0.0

## Conceptual Innovation Description

Introduce a graph-based memory system where nodes and edges decay at multiple time scales. Each
edge stores several decay envelopes so relationships can fade quickly or slowly depending on
usage. A selector process scans node attributes and neighbor weights to decide which connections
remain active. Nodes originate from sorting routines that group embeddings by distance into
categories. Packets of recent context are saved in a tree structure while the underlying graph
accumulates as a long term context. Items with enough influence are "graduated" into a permanent
store that persists across program runs and reinflates the graph at startup. Transformer search
over the permanent store generates a compact "subconscious" packet used to prime other models.

## Relevant Files and Components

- `AGENTS/conceptual_flags/`
- Future graph context modules under `speaktome/`

## Implementation and Usage Guidance

1. Represent each edge weight as a list of decaying values, one per envelope.
2. Periodically update decay values and remove edges that drop below thresholds.
3. Persist graduated nodes to disk so the graph can regenerate after shutdown.
4. Provide an API for extracting a "moment prompt" built from the most active nodes.

## Historical Context

This flag extends `Decaying_Graph_Contextual_Data.md` by incorporating multiple decay envelopes
and a permanent store that survives restarts.

---

**License:**
This conceptual innovation is contributed under the MIT License, available in the project's root
`LICENSE` file.
