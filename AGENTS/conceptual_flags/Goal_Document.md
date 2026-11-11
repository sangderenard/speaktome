# 🚩 Conceptual Flag: From Dream Document to Goal Document

**Authors:** Claude (service lane), from the owner's direction of 2026-11-11

**Date:** 1794407400 (2026-11-11)

**Version:** v0.1.0 (draft; the chain and its words as the owner stated them; nothing here is built except where marked OBSERVED)

## Conceptual Innovation Description

The owner, 2026-11-11, in order:

> "the part by part compiler, it should produce a dream document as their enduring record"

> "So a dream document goes in, the part by part compiler harness works on it, makes another dream document but that
> one can safely have it's parts built individually"

> "there must be a tool or a worker responsible for taking development material and baking a static site copy taking
> it off the plate of the server. Such that there would also exist someone who could make the dream document
> coordinated and runnable by merit if knowing it was already lowered. Like it's the speaktome level plan not a single
> program. See dream document intake is about getting one language, the part by part builder knows how to split that
> language into conpiled units, they just need their linking done, so the output dream document is substantially
> different. Like, dream document gets processed into a Goal document, which can run ina language or as nodus nodes,
> the piece by piece compiler will have resolved all the api in general, then if the goal is achieved you have an
> actionable system diagram"

This is a plan at the level of the whole system, not of one program. It names a chain of documents and the workers
that move material along it. The chain:

```text
development material
   |  intake: one language at a time
   v
Dream document            what was written, as a container of blocks
   |  part-by-part builder: splits that language into compiled units, resolves each unit's API
   v
compiled units            each built on its own; they still need their linking done
   |  a coordinator that knows the material is already lowered
   v
Goal document             coordinated and runnable: in a language, or as Nodus nodes
   |  if the goal is achieved
   v
actionable system diagram
```

Everything marked OBSERVED was read in code on 2026-11-11 (`T` is the turing repository, `N` is nodus). Everything
marked NEW does not exist.

## 1. The words

| Word | What it is | State |
|---|---|---|
| **Dream document** | A container of blocks, each in one language, with declared inputs, outputs and a hash. What goes IN. | OBSERVED: `turing.dream-document.v1`, `T/src/compiler/dream_document.py`, `T/docs/DREAM_DOCUMENT.md`. It had a reader only; a writer (`render_dream_document`) was added today. |
| **intake** | Taking development material in, one language at a time. | OBSERVED: `T/examples/meowdoku/source_divider.py` (`intake_file`, `intake_zip`, `intake_text`, `intake_git`, and since today `intake_dream`). Only Python is divided. |
| **part-by-part builder** | Knows how to split one language into compiled units and resolves the API of each. | OBSERVED: the source divider. A unit is a **piece** (one strongly connected group of the call graph); the cut is `partition_authored_source`; one **attempt** per piece and backend; a completed attempt is a **product**, with an `.api.yaml` when the backend writes one (wasm, js). |
| **compiled unit** | A part that has been lowered and can be built by itself. | OBSERVED: a piece with a completed attempt. Since today `record.dream` holds every piece as its exact closure, followed by its products byte for byte. |
| **linking** | Joining compiled units so that one's output is another's input. "They just need their linking done." | PARTLY OBSERVED as mechanisms, not as this step: `partition_authored_source(..., linked_dependencies=...)` keeps a dependency's definition as the call contract while its execution is replaced by an attached module; `lowering_library` catalogues products by core id; a Nodus edge joins two connectors by declared type. NOTHING links a divider job's units to each other today. |
| **coordinator** | The worker that makes the document coordinated and runnable, "by merit of knowing it was already lowered". | NEW. |
| **Goal document** | What the Dream document is processed into. It "can run in a language or as nodus nodes". | NEW. No document, schema or code by that name exists in speaktome, turing or nodus. |
| **goal achieved** | Every unit lowered and every link resolved. | NEW as a state. The parts of it that can be read today: a piece's verdict (`lowered` or `stays python`) and the reason. |
| **actionable system diagram** | What you have when the goal is achieved: a diagram that is also the thing that runs. | NEW as a result. OBSERVED forms it could take: `nodus.graph-definition.v1` (`N/docs/TURING_NODUS_GRAPH_DEFINITION.md`: modules, rows, connectors, edges, ports), the canvas itself, and the system map the site already serves as pages (`/site/system/`). |
| **baker** | The worker that takes development material and bakes a static site copy, "taking it off the plate of the server". | NEW as a worker. OBSERVED parts: `speaktome_api/site.py` (`Atlas.bake`), `T/examples/meowdoku/deploy/bake_site.py`, `scheduled_tasks.refresh_hard_copy`, `web_export.py`, and the Nodus cron tool that can ask for the work. |

## 2. What was built today, read in this light

- The divider's `record.dream` is the **compiled units** stage, not the Goal document. Each part builds on its own
  and carries its lowerings and its refusals. Its parts are not linked to each other: a part that calls another holds
  that other's text inside its own closure. That is what makes a part safe to build alone, and it is also exactly
  why the output must be "substantially different" from the input before it can be the Goal document: a unit that
  carries its dependencies by copy has not had its linking done.
- Dream in, dream out works for Python blocks (a record fed back in rebuilt every part to the same product). It
  proves the units stage; it says nothing about coordination.
- The manual's hard copy is baked by the web server itself today (at start, and as the host of the cron task
  `manual.hardcopy`). By the owner's point that is the wrong place: the server should serve, and a worker should bake.
  The cron tool already separates WHO ASKS from WHO PERFORMS, so moving the work is a matter of which process joins the
  cron prefix as the host. The full site export (`web_export.py`) is run by hand today and belongs to the same worker.

## 3. What a Goal document has to hold (a reading, not a design)

Derived only from the owner's sentences; each line names the sentence it comes from.

1. The compiled units, by identity. ("the part by part builder knows how to split that language into compiled units")
2. Each unit's API, resolved. ("the piece by piece compiler will have resolved all the api in general")
3. The links between units. ("they just need their linking done")
4. How it runs: in one language, or as Nodus nodes. ("which can run in a language or as nodus nodes")
5. Whether the goal is achieved. ("if the goal is achieved you have an actionable system diagram")

Under this reading the Goal document is a graph: units are nodes, links are edges, and each node says whether it is
lowered. When every node is lowered and every edge is resolved, the graph is executable, and an executable graph of
typed nodes and edges is what Nodus already calls a graph definition. That would make "actionable system diagram"
literal.

## 4. Open, for the owner

1. Is the goal stated up front by the author (a target the processing works toward and may fall short of), or is it
   the name of the processed result?
2. Is the Goal document a dream document with new kinds of block (a link block, a unit block), or a separate format?
3. As Nodus nodes: one row tool per compiled unit, with an edge for each link? That is the shape
   `nodus.graph-definition.v1` already describes.
4. The workers (baker, coordinator, the builder itself): are they hosts on Nodus contract prefixes, as the compile
   host and the cron task host are, each its own process under the supervisor?
5. A part that could not be lowered ("stays python"): does it block the goal, or does the Goal document run it as
   Python beside the lowered units?

## Relevant Files and Components

- turing: `src/compiler/dream_document.py`, `docs/DREAM_DOCUMENT.md`, `src/compiler/project_compilation_product.py`
  (`authored_compilation_growth_plan`, `partition_authored_source`), `examples/meowdoku/source_divider.py`,
  `divider_nodus_bridge.py`, `lowering_library.py`, `scheduled_tasks.py`, `deploy/bake_site.py`, `web_export.py`
- nodus: `docs/TURING_NODUS_GRAPH_DEFINITION.md`, `nodus_graph/model.py` (`GraphDefinition`),
  `include/turing_compile_host_tool.h`, `include/cron_tool.h`, `tools/cron_host.py`
- speaktome: `speaktome_api/site.py`, `speaktome_man/pages/cron.5.txt`, `AGENTS/conceptual_flags/Person_Vocabulary.md`

## Implementation and Usage Guidance

1. Call the divider's output the **compiled units** (or the units record). Do not call it the Goal document.
2. Do not design the Goal document's format before the owner answers section 4, questions 1 and 2.
3. The first move that needs no answer: take the baking off the server. A worker process joins the `cron` prefix as
   the host for `manual.hardcopy` (and later the site export); the server keeps serving.
4. Linking must use the identities the builder already resolved (piece ids, product hashes, the entries of an
   `.api.yaml`), never a match on names.

## Historical Context

Written the day the Nodus cron tool, the picture path and the manual's pages went live. The owner's remark followed
from the observation that the live site's files are published by a separate export step and not by a restart, and
from the first dream record, which showed the difference between a part that builds alone and a system that runs.
