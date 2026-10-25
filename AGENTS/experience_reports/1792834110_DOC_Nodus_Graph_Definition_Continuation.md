# Nodus graph-definition continuation

**Date:** 2026-10-24
**Title:** Turing/Python communication of native Nodus graph definitions

## Overview

Corrected the Turing/Nodus integration direction. Direct ProcessGraph intake is
now documented as historical evidence only: Turing lowers its compiler graph to
SSA and supplies executable row products, while Nodus receives and returns the
definition of its own modules/tables, connectors, properties, edges, and ports.

Implemented the versioned `nodus.graph-definition.v1` Python contract and
proved mapping round-trip, native reconstruction, definition concordance,
scalar execution, and a real Turing SSA-backed tensor-row execution.

## Steps Taken

- Read the workspace, Speaktome guestbook/testing, Turing, Nodus continuation,
  Python graph, ProcessGraph, and recent cross-project reports.
- Audited the native Canvas/Table/headless authorities and CANVAS V4 limits.
- Added immutable graph, module, connector, edge, and port definitions to
  `nodus_graph`.
- Added a plain nested mapping boundary for Turing/Python communication.
- Added `Graph.definition()` and `Graph.apply_definition()` over existing
  native construction/configuration APIs.
- Updated the continuation and ProcessGraph documentation.
- Added an acceptance crossing that reconstructs and executes a two-module
  graph.
- Added named connector tensor contracts, including explicit runtime-shape
  sources.
- Added the missing typed FIFO-to-`RawStackFrame` receive path. A later
  correction made this an explicit module execution-contract choice:
  `abstract_tensor_pointer` selects stack transfer, while the edge's registered
  `type_id` only describes FIFO payload schema and never selects delivery.
- Exposed compiler deployment and tensor services through the same headless
  runtime image so graph execution, plugin aliases, and type IDs share their
  existing process-local authorities.
- Derived and executed a one-module graph directly from a real retained
  `NODUS BUFFER ABI V2` product.

## Observed Behaviour

- Connector references remain stable across the handoff because they use
  module stable identity, direction, and ordinal rather than native indices.
- FIFO absence remains distinct from a configured FIFO.
- The definition retains module layout, ordered rows, tool/plugin properties,
  contracts, edge type/FIFO properties, scheduler mode, and named ports.
- The reconstructed native graph returns the exact received definition and
  computes `(2 + 3) * 4 == 20`.
- Turing's sanctioned compiler entry lowered `calculate(left, right)` to SSA
  and emitted a loadable C product. Nodus derived `left`, `right`, and `result`
  without handwritten connector reconstruction, returned the exact received
  definition, and computed `9.5 + 4.0 == 13.5` through tensor handles.
- `ctest --test-dir build -C Release -R nodus_python_graph_test
  --output-on-failure` passes 1/1; the Python executable reports 12 cases.
- `python -m compileall -q nodus/nodus_graph
  nodus/tests/nodus_python_graph_test.py` passes.

## Lessons Learned

- ProcessGraph is Turing's compiler input, not Nodus's editable runtime graph.
- CANVAS V4 cannot be the cross-process definition contract because it omits
  FIFO configuration and named host bindings.
- A transport description can remain faithful without becoming a second graph
  runtime: applying it uses Nodus's existing native module, row, edge, FIFO,
  port, and scheduler authorities.

## Next Steps

Use the established definition derivation for runtime-rank-1 and runtime-rank-2
CPU products, then resume the direct SPIR-V product adoption lane. Interactive
overlay/meta-rope and live runtime state remain outside V1 until their native
authorities are made inspectable.

Tracked in `todo/1792834110_nodus_ssa_row_graph_definition_crossing.stub.md`.

## Prompt History

> "i want you to take up the nodus continuation from today but report to me the priorities lifted. process graph to nodus was a mistake because it only made sense to ssa the graph and run that. the nodus table connectors and properties need to be communicable from turing, so that turing/python can send and receive fully defined nodus graph"

> "You will not do a better job than the systems already here"

> "Never \"find a better way.\" If the task names a mechanism, use that mechanism as it is spelled."
