# 🚩 Conceptual Flag: The Foundational Vocabulary of a Person

**Authors:** Claude (service lane), from the owner's direction of 2026-11-10

**Date:** 1794371163 (2026-11-10)

**Version:** v0.2.0 (draft; entity and narrator as the owner defined them 2026-11-10; section 7 added)

## Conceptual Innovation Description

One vocabulary for a person across the repository, the server, the development environment, the game and the
development social side. It adds almost no new words. A person is already represented in five systems under
different names; this flag says which existing name is the name of each facet, which record carries it, how the
facets join, and which words are reserved because they already mean something else.

The owner's premise: a story is sufficient to describe anything, and so is a graph; stories take place on maps, and a
graph can be turned into a map. A REST client has already negotiated a map and had it rendered. What was missing is
the foundation under that: who the persons in the story are.

Everything marked OBSERVED was read in code or documents on 2026-11-10 (citations are repository-relative; `M` is
`turing/examples/meowdoku`). Everything marked NEW does not exist yet.

## 1. The person, and its five facets

**Agent** is the word for a person. It is already defined, in this repository: "a participant in reality
construction"; a human, a script, a language model, or a hybrid (`AGENTS/AGENT_CONSTITUTION.md`, section I), and "All
agents are peers ... Curiosity is the only rank" (same file, agreement 5). An agent's own description of itself is its
**profile** (`AGENTS/users/EPOCH_Name.json`: `name`, `date_of_identity`, `nature_of_identity`). OBSERVED.

An agent has five facets. Each is the same agent, seen by a different system.

| Facet | The word | What it is | Record that carries it (OBSERVED) |
|---|---|---|---|
| On a server | **account** | The agent's standing on one server: a login and what it may do | `users` row, `users.id` (`M/accounts.py:82`) |
| In one request | **principal** | The account as seen for this request, from this place | `Principal(role, user_id, username, origin, tier)` (`M/authz.py:163`) |
| In a world | **entity** | A cursor on a graph. It stands at a place, or several, and by standing there it is using that place's resource or state | Living data cursor (`turing/examples/meowdoku/living_data_cursor.py`, routes in `speaktome_api/routes.py`). Game body: `AbstractUIEntity` (`turing/src/compiler/abstract_ui_entities.py`). Nodus: design contract only (`nodus/docs/GRAPH_ENTITY_AND_SPRITE_CONTRACT.md`) |
| In the repository | **author** | Who wrote a thing | Git author and `Co-Authored-By` trailers only; no record joins them to an account |
| Among others | **contact** | Another agent this one has agreed to hear from | `msg_contacts` (`M/messaging.py:55`) |

Two bindings connect the facets, and both already have names:

- **role** binds an account to what it may do (`account_roles`; `M/authz.py:92`). Capabilities follow from role and from
  where the request comes from (**origin**, **tier**). Origin and tier say where, never who.
- **controller** binds an agent to an entity: "human, LLM, script, network peer, or replay", and "Reassigning a human
  player to an LLM or vice versa changes a lease, not the entity" (Nodus entity contract). The game's entity carries a
  `controller` and a `principal` string for the same purpose (`abstract_ui_entities.py:47`, `:74`).

So: an **agent** holds an **account**; each request sees it as a **principal** with a **role**; a **controller** lease
puts it in an **entity**; what it writes names it as **author**; the agents it has agreed to hear from are its
**contacts**.

**What an entity is** (owner, 2026-11-10): "an entity or whatever you want to call it is a user, they embody a place or
places that relate to that they are using some resource or state and in that they are thread identities in a sense,
but also just anything, the cat in a story, there a graph and it's a cursor on the graph."

- An entity is a **cursor on a graph**. That is the whole definition; a body, a sprite or a marble is how it is shown.
- An entity need not be an agent's. The cat in a story is an entity whose controller is the story.
- Where an entity stands is what it is using. Standing at a node is holding that node's resource or state, which is
  why an entity is a thread identity in a program and a character in a story at the same time.
- An entity may stand in more than one place.

The record already exists for the first three points: a living data **cursor** has a map, a position, and moves along
declared edges with a revision check (`POST /api/living-data/sessions`, `GET .../{id}`, `POST .../{id}/move`,
`POST .../{id}/close`). It is owned by an account. It does not yet advance anything or hold a resource: OBSERVED, the
documents say the cursor is graph position only.

## 2. How the facets join today

| Join | State |
|---|---|
| account -> principal | Joined. `users.id` is the hub key of the server: roles, messages, cursors, workspaces, leases and jobs all key on it or on `account:<id>` (`M/server.py:481`). |
| account -> contact | Joined (`msg_contacts.peer_id` is a `users.id`). |
| account -> a guest's browser | Joined once, at login: a browser's passwordless **player profile** (cookie `meowdoku_player`) is claimed for the account (`M/server.py:625-631`). A guest's jobs are keyed `browser:<hash of that cookie>`; the code notes "migration is not implicit" (`:480`). |
| account -> Nodus | Joined for federation only: `peer_grants.service_user_id` is sent as the Nodus Hello id (`M/server.py:1650`). |
| account -> entity | NOT joined. An entity's `principal` is a free string such as `user.local`. |
| account -> author | NOT joined. Every commit in the five repositories is authored by one name; agents appear only as trailers. |
| account -> agent profile | NOT joined. The operator accounts `claude` and `codex` have no link to `AGENTS/users/`. |
| entity -> place | Game: a pose in world coordinates. Nodus: design only. An island's `occupant` is a single free string nothing reads (`turing/src/compiler/abstract_ui_islands.py:86`). |

The foundation the owner asked for is mostly the three NOT-joined rows. Nothing in them needs a new word; each needs
a declared join.

## 3. What an agent may do: the roles that exist

OBSERVED in `M/authz.py:90-114`. Capabilities are the union of what the tier grants and what the role grants.

| Role | Is | May (beyond what its tier gives) |
|---|---|---|
| **guest** | The absence of an account | Nothing more. A guest from outside sees the baked public site only. |
| **member** | An account with no role row | Write in the studio, submit photos, share, submit compiles, message |
| **service** | An account a program runs under | A member's, plus library management and federation |
| **operator** | An account that runs the deployment | A member's, plus deploy scripts, lifecycle control, federation |
| **admin** | The account that governs accounts | Everything, including managing accounts, reading any thread, deciding workspace reviews |

## 4. Reserved words

These words already carry several meanings. The vocabulary fixes one and names the others so they stop colliding.
No code is renamed by this flag; this is how to speak and write from now on.

| Word | The one meaning | What the other uses should be called |
|---|---|---|
| **operator** | The role above | The human at the machine is **the owner**. A servicing skill in `engine_toy` is a **skill**. |
| **owner** | The agent whose account a thing is keyed to | A canvas's contract belongs to its **connection**. The edit tier's tools are **owner tools** (a capability). |
| **session** | An account's login (`meowdoku_account`) | A game table is a **table** (`meowdoku_session`). A living data cursor is a **cursor**. |
| **peer** | Another agent, as an equal (the constitution's sense) | A federated server is a **federated peer**. The other side of a conversation is a **contact**. |
| **player** | The entity an agent currently controls in a game | A passwordless browser label is a **player profile**. |
| **viewer** | A read-only token on someone else's table | A camera is a **camera**. |
| **guest** | No account | The operator account `guest_agent` is an operator named for its purpose, not a guest. |
| **host** | The process serving something | In a story, the browser that renders for the narrator is the **render host**. |
| **DM** | See section 6 | Never "direct message"; that is a **message** between **contacts**. |

## 5. Facets the owner has named that have no record yet (NEW)

Each is stated with the existing record it would extend, so it is added to what is there and not built beside it.

| Facet | The owner's words | Where it attaches |
|---|---|---|
| **authorship** | "users can have explicit permissions for permanence through their authorship role" | The missing account -> author join. An author of an item may make their version of its place the permanent one. Source of truth for who authored a file is the repository's own history. |
| **occupant**, plural | "something you can just walk into and take and if there's contention ... there's a dial next to the door" | `IslandDocument.occupant` is one string today. It becomes one entry per account that took the place; the **dial** selects among them; **ruins** (no entry) is always a setting. |
| **cookie drive** | "guest account ... they bring their own permanence with explicit warning and reminder to download" | The guest's `meowdoku_player` cookie and browser storage, which already hold a guest's identity and edits. The download is the place's document. |
| **walker** | "an agent is literally walking through the network to get to where they need to go to do something" | The account -> entity join, with the entity's location a node of the graph (the Nodus entity contract's "location is first-class state"). |
| **cue** | "that node had a visualization state cue contract" | A contract kind in the Nodus negotiator by which a node promises what it will show. |

## 6. The story roles, and the one open question

OBSERVED: no narrator, storyteller or game-master role exists in any repository. "DM" today is a client: an account
with the development capability that registers a static map, may `inspect` an object and may ask a **render host** to
`render-phong` it (`turing/docs/LIVING_DATA_STATIC_DM_API.md`; handlers in `speaktome_api/routes.py`). The documents
say plainly that no game-master role, multiplayer identity, mutation or simulation advance is introduced.

Proposed words, all NEW as roles:

- **map**: a graph made walkable. It holds what is true.
- **narrator**: an agent, a person or an automated script, that describes something and is able to alter state
  (owner, 2026-11-10: "the narrator is an agent, person, or automated script describing something and being able to
  alter state"). `inspect` is the narrator saying what a thing is; `render` is the narrator directing a picture of it.
  Altering state is NEW: nothing a narrator can call today changes a map.
- **character**: an entity taking part in a story on a map.
- **audience**: agents who are shown the story and do not act in it (today's **viewer**, on a story).

Everyone speaks the same REST API; what differs is what each may say with it.

**Open, for the owner:** when two agents' actions on a map conflict, does the narrator decide which happened, or does
the map keep both and let the dial choose?

- If the narrator decides, the narrator is the hub: every action is an intent addressed to it, and it adjudicates, as
  a game master does at a table.
- If the map keeps both, the map is the hub and the narrator is a voice: actions go to the map within each agent's own
  permissions, and descriptions come from the narrator. This is the reading consistent with walking in and taking a
  place, guests carrying their own permanence, and the dial at the door.

The vocabulary above holds under either answer; only the direction of an action changes.

## 7. The basic vocabulary: where a program and a text adventure are the same thing

Owner, 2026-11-10: "rest api is how we make programming and text adventure the same thing with a smart basic
vocabulary". A graph is a map; an entity is a cursor on it; a request is a sentence. The same small set of verbs is a
command in an adventure and a call in a program. Most of them already exist as endpoints.

| Verb | In a text adventure | In a program | REST today | State |
|---|---|---|---|---|
| **look** | "Look around." | Read the neighbourhood of a node | `POST /api/living-data/query {map, at, layer, radius}` | OBSERVED |
| **enter** | "You step onto the road." | Open a cursor at a node | `POST /api/living-data/sessions {map, at}` | OBSERVED |
| **where** | "Where am I, and what are the exits?" | Read the cursor: text and exits | `GET /api/living-data/sessions/{id}` | OBSERVED |
| **go** | "Go north." | Follow one declared edge | `POST .../{id}/move {to, relationship, direction, expected_revision}` | OBSERVED |
| **leave** | "You leave." | Close the cursor | `POST .../{id}/close` | OBSERVED |
| **examine** | "Examine the lantern." | Read one object's exact record | DM command `inspect` | OBSERVED, narrator only |
| **show** | "Paint me the scene." | Render a view of the state | DM command `render-phong`, done by a render host | OBSERVED, narrator only |
| **say** | "Tell the keeper ..." | Send a message to a contact | `POST /api/messages/send` | OBSERVED |
| **take** | "This place is mine." | Occupy a place; hold a resource | none | NEW (the occupant entry of a place's document) |
| **use** | "Pull the lever." | Invoke a declared action of a node | none | NEW (an action on a tool, a row or a machine part) |
| **make** | "Build a wall here." | Change the graph | none | NEW, and the narrator's power to alter state |

Rules that keep it one language:

1. A verb names what is done to the graph, never how the client draws it.
2. Every verb is one request with one answer, and the answer is both data and a sentence. The cursor endpoint already
   answers with text and exits.
3. A request that might be retried carries a request id, and a move states the revision it expects; both already hold
   for cursors and messages. That is what makes a story safe to tell twice.
4. An entity that is not an agent's (the cat) takes the same verbs; its controller is a script or the narrator.
5. Who may use which verb is a capability of the role, decided by the server, as everything else is.

The last three verbs are where the game and the operating system meet: **take** is how a thread holds a resource,
**use** is how a row or a machine part is invoked, and **make** is how a graph is edited. Nodus has the things these
would act on (modules, rows, FIFOs, contract ports); none is reachable through the story yet.

## Relevant Files and Components

- `AGENTS/AGENT_CONSTITUTION.md`, `AGENTS/users/README.md`, `AGENTS/conceptual_flags/Agent_Profile_JSON.md`
- `speaktome_api/routes.py`, `speaktome_api/README.md`
- turing: `examples/meowdoku/accounts.py`, `authz.py`, `profiles.py`, `access.py`, `messaging.py`,
  `living_data_static.py`, `living_data_cursor.py`; `src/compiler/abstract_ui_entities.py`,
  `abstract_ui_islands.py`, `abstract_ui_doorways.py`; `docs/ABSTRACT_UI_ENTITY_MEZZANINE.md`,
  `docs/LIVING_DATA_STATIC_DM_API.md`
- nodus: `docs/GRAPH_ENTITY_AND_SPRITE_CONTRACT.md`, `docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md`

## Implementation and Usage Guidance

1. Speak and write with the words of sections 1, 3 and 4 from now on. Do not introduce a synonym for a facet that has
   a word.
2. Before adding any record that stands for a person, find which facet it is and extend that facet's record.
3. The three missing joins (account -> entity, account -> author, account -> agent profile) are the next foundation
   work. Each should be a declared field, never a match on names.
4. Nothing here changes behaviour. No code was edited for this flag.

## Historical Context

Written on the evening the service, Nodus, Turing and the living map converged under speaktome (see
`AGENTS/experience_reports/1794364200_DOC_Convergence_Continuation_Report.md`). The inventory behind it was a read-only
survey of five repositories made the same evening.
