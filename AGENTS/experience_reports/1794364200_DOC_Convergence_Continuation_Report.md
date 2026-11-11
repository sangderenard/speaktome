# DOC: Convergence continuation report (web service, Nodus, Turing, the world map and the living map)

**Date:** 1794364200 (2026-11-10, local evening)
**Title:** Where the Claude lane stands as the projects converge under speaktome
**Author lane:** Claude (service, accounts, Apache, Nodus bridging, messaging, world map, living map doors)
**Audience:** the owner, the other agent (Codex), and any agent that resumes this lane cold.

## Overview

On 2026-11-10 several separate efforts met: the Meowdoku web service behind Apache, Nodus as the organiser of
compiled work, the Turing compiler as the thing that divides a Python program into compilable parts, and the
"living data map" world as the place people stand in. The owner now treats speaktome as the umbrella and its role
in machine simulation, Turing, Nodus and the web service as one subject. This report records what this lane built,
what is live, what is only designed, what is known to be wrong, and the agreed way forward, so the next reader does
not have to reconstruct it from five repositories.

Statements are marked by how they are known. **Observed** means run or read on 2026-11-10. **Design** means the
owner's direction, recorded but not built. Anything else is labelled.

## 1. Repository state at the time of writing (observed)

| Repository | Branch | Head | Note |
|---|---|---|---|
| `C:/dev/Powershell/turing` | `main` | `fd9a9903` | pushed; all lanes consolidated here on 2026-11-10 |
| `C:/dev/Powershell/nodus` | `nogodsnomasters` | `d3814f0` | pushed |
| `C:/dev/Powershell/speaktome` | `codex/direct-x86-vocabulary-report` | `7362174` + this report | no `main` branch exists |
| `C:/dev/Powershell/spectral-analyzer` | `nogodsnomasters` | `edddb33` | pushed |
| `C:/dev/Powershell` (root, `nogodsnomasters` repo) | `codex/live-string-core-demo` | `8535c58` | pushed |

Only Turing has a `main`. "Push everything to main" was carried out as Turing `main` plus each other repository's
current branch. Left out on purpose: agent worktree copies under `turing/.claude`, `engine_toy/artifacts`,
spectral-analyzer `repo-cache`, speaktome `.zig-cache`, Nodus `out/`.

Local-only state that no repository tracks: `C:/dev/Powershell/srv` (`python` private interpreter, `state` stable
state root, `state-dev` development state root, `backups`). It is covered by the root repository's ignore rule and
must stay untracked.

## 2. What is live (observed)

Apache 2.4.58 runs as a console process and owns the lifecycle. A piped-log supervisor
(`turing/examples/meowdoku/deploy/apache_supervisor.py`) starts and stops three instances:

| Instance | Listens | Role |
|---|---|---|
| `meowdoku-a` | 0.0.0.0:8766 (route `a`) | stable server |
| `meowdoku-dev` | 127.0.0.1:8768 (route `b`) | development server, needs capability `dev_ui` |
| `nodus-messenger` | 127.0.0.1:47611 contract port, 47612 console | Nodus canvas host |

Public address `https://stichka.duckdns.org:8766` (router forwards 8766 to Apache 443, Let's Encrypt certificate).
The other agent's own development server on 127.0.0.1:8767 is theirs and was never touched by this lane.

Start and stop only through the gated scripts in `turing/examples/meowdoku/deploy`
(`Start-Apache.ps1`, `Stop-Apache.ps1`, with `MEOWDOKU_OPERATOR` set). **Known defect:** `Stop-Apache.ps1` returns
before the old `httpd` has exited; starting at once fails with `slotmem_create failed` and no listener on 443. Wait
for `httpd` to disappear before `Start-Apache.ps1`. Not fixed in the script.

Pages and endpoints added by this lane and live on both servers:

| Page | What it is | Capability |
|---|---|---|
| `/login.html` | public sign-in, stable/development switch | none to view |
| `/browse` | listing of site, math and docs for people and agents | view |
| `/mathlab.html` | typed math to NumPy, wasm, WGSL, native, profiled | `compile_submit` |
| `/divider.html` | intake of a `.py`, a zip or a GitHub link as a divider job | `compile_submit` |
| `/profile.html` | read-only instances, git state, leases, scrubbed log tail | `account_manage` or `lifecycle_admin` |
| `/worldmap.html` | repository world map (folders, files, portals) | `dev_ui` |
| `site-nav.js` | sign-in chip and menu; opt-in per page | none |
| `messenger-widget.js` | messaging widget; opt-in per page | `message` |

Accounts (stable `accounts.db` under the state root): one human account `admin` (password set by the owner on
2026-11-10), operator accounts `claude`, `codex`, `guest_agent`, and three service accounts. Operator secrets live in
`srv/state/secrets/operator_credentials/<user>.secret`; they are never printed, logged or copied into a repository.
Visitors from the home network are full-access guests without signing in (access is by network origin); visitors
from outside are shader-tier guests until they sign in.

## 3. What this lane built on 2026-11-10, by area (observed; Turing commits unless stated)

**Bridging and Nodus (earlier in the day; see report 1794326465).** Contract ports as canvas-owned negotiator pairs
with a loopback wire protocol; a read-only console viewer; a Turing compile host tool that takes one job at a time
without blocking the canvas tick (Nodus `d3814f0`); Torch-free service build.

**Service, accounts, state (see report 1794334300).** Roles and capabilities (`authz.py`), gated deploy scripts,
private interpreter, state root outside any repository, federation bridge over WebSocket, lowering library with
leased, versioned, content-addressed binaries.

**Compile path as a service.** `source_divider.py` (`e6b6097a`): a general Python divider with a leased
one-piece-at-a-time queue and per-backend attempts. `math_lab.py` and `/mathlab.html` (`b60ba90f`, `333b2cb9`): one
real job ran on all four targets in 33.6 s through the public name. `divider_intake.py` (`06d3650f`): safe file,
zip and repository-link intake; links are recorded, never fetched.

**Sign-in and navigation.** Public sign-in (`60f02239`), development server with remembered per-account choice and
`/browse` (`75b19c1d`), POST body drain fix for shared kept-alive connections (`91b2a9bb`), navigation chip
(`c9ab0e60`).

**Messaging.** Server-managed user-to-user messaging (`9f565397`) with the owner's hard rule: zero direct contact, a
peer sees only username and display name, plain text, administrators can read, remove and suspend with an audit
trail. Agent mailbox additions (`06d3650f`): subject, type, a server-kept cursor and explicit acknowledgement.
`deploy/agent_messenger.py` is the agent client; `claude` and `codex` are mutual contacts and two messages from
`claude` wait in `codex`'s mailbox. The widget was driven in headless Chrome (15 of 15 checks after two fixes).

**Operations.** Live-safe state root snapshots (`c87a104b`); first snapshot taken to `srv/backups`. Not scheduled,
no off-machine copy.

**Repository world map.** `src/compiler/repo_world_scan.py` v1 with an explicit extractor registry and declared
Python import roots; extractors for C/C++, CMake, HTML, dynamic JavaScript, PowerShell, shell
(`repo_world_refs_*.py`); `tools/compile_repo_world_map.py` compiles a compact map from a fresh GitHub clone
(`526cae74`). All five repositories were compiled (`9aab28aa`): Turing 244 folders, 5,693 files, 16,212 portal
pairs, 16,257 unresolved references. Output under `srv/state/content/worldmap`.

**Living map: portals, doors, islands.** In the existing engine, not a new one:
- producer-declared openings on region, building and room, carried to the geometry boxes (`e58b1c0b`, `919ff6a4`);
- declared portal pairs: a pair links only its two ends, with certain weight at any distance (`e58b1c0b`);
- doors as material sheets, crossed at once, with `declare_door` (`919ff6a4`, `src/compiler/abstract_ui_doorways.py`);
- island documents carried beside their items, identity stable across commits
  (`6a2e4b19`, `src/compiler/abstract_ui_islands.py`);
- doors between islands: leave one world's page, arrive in the other at the same pair (`6a2e4b19`, `fd9a9903`);
- a general portal record derived from the living map's own portal (`src/compiler/portal_record.py`).

Every Turing change has a hazards entry under `turing/docs/hazards/` dated 2026-11-10 or 2026-11-11 (UTC).

## 4. The owner's design, recorded and not built (design)

The owner described the world in a sequence of statements on 2026-11-10. In the order they build on each other:

1. **One engine, two modes.** Class mode and file mode are two producers of the same generic graph. Portals come
   from the living map; nothing starts from scratch.
2. **Outside expresses nothing.** Every item starts the same, a little ruined, like the initial state of The Sims:
   someone else's structure and problems. No size, no importance. Area-by-bytes and a floors-and-basements layout
   were both rejected as "the law".
3. **Ruins are the zero.** Unless someone figures a place out it is ruins; it can always be ruins again because that
   is a new document; it costs nothing because ruins come from a procedural factory.
4. **Interiors have TARDIS geometry** and are independent islands of existence, carried by their files, perhaps in a
   private file.
5. **Two kinds of portal.** Doors (nonlocal) give a filesystem its recursion, the tree. Local portals are for fun and
   for dialling in local relationships in code, linked data or conceptual graphs; marbles will roll through them.
6. **Terrain bottle.** Each region is enclosed by natural terrain except one passage into a wilderness that holds
   every context's endpoint at once, through higher-dimensional math.
7. **The wilderness is a procedural flux** from the service's side: it evolves wherever nobody is near, is stabilised
   only by people's permanence, is only as large as needed, shares one weather and one simulation state, and hides
   its nature by distorting scale in a smooth address space.
8. **Scale bubbles.** Private or unavailable content is hidden by warping the fabric, experienced as distance.
9. **State and contention.** Regions are stateful. Permanence is an explicit permission through an authorship role.
   A place is something you walk into and take; if there is contention there is a dial next to the door.
10. **Guests bring their own permanence** in a cookie drive, with an explicit warning and a reminder to download.

Earlier the same day: Nodus should also reduce a graph to one compiled artifact with no Nodus faculty (the
superkernel); the shader tier is the baked public release of a development game; every system that runs Nodus is a
node of one graph sharing a single abstract filesystem of local chunks that join and drop.

## 5. The verdict on the living map page, and the agreed way forward (owner, 2026-11-10)

The owner played the car game and judged it: very good bones, but "it tried to build an attic over a carport".
Corrected and settled wording: **the engines are solid and just need other solid engine parts; the demo site is the
problem.** World rules were farmed out to JavaScript so the document could participate statefully ("an unnecessarily
tight and showy bond"). The page is history and does not go on the main page. The August audit
`turing/docs/LIVING_DATA_MAP_COMPILATION_AND_PRODUCTIVE_WORLD_AUDIT.md` says the same.

Agreed way forward, which the owner says conforms to older documents:

1. a thin control shell in the page, as was already being done;
2. the game as a finished Pythonic product;
3. the compiler divides it into compilable parts and not;
4. at that division, like the other pages, the server participates or compiles just in time.

Consequences for this lane's work: the Python-side declarations (openings, pairs, doors, island documents, the
portal record) are engine-grade and stay. The door and island crossing code written into the browser template on
2026-11-10 is a prototype to be replaced by compiled rules. Demo work was stopped.

## Steps Taken

Work was coordinated from one session with narrow Sonnet workers, each given its own files and a marked block where
a file was shared (`[MESSAGES]`, `[PROFILE]`, `[DIVIDER]`, `[WORLDMAP]` in `server.py`). Every worker's diff was
read, its tests rerun, and only this lane's hunks staged while the other lane had uncommitted work in the same
files. Live changes went through the gated Apache scripts and were checked through the public address.

## Observed Behaviour

Things that were found only by running the real thing:

- A POST answered without reading its body corrupted the next request on a shared Apache connection.
- The messaging widget looped about a hundred requests per second with a thread open, and a host page's CSS could
  hide it; headless Chrome found both.
- A declared portal pair 500 units apart got weight exactly zero, because the game's distance weighting underflows.
- A fresh visitor to a living map page got no door sheets: openings were converted to portals only when a saved game
  existed.
- The game's action timer throws on every frame for an issued action whose edge is not registered.
- After arriving through an island door the walker was pulled back to the vehicle on the next cycle; the entity cycle
  does not run during the vehicle qualification screen.
- One existing living map test (`test_portal_recipe_deploys_probabilistic_mesh_graph_and_traverses_tubes`) fails on
  two strings that were already absent from the committed template before this lane touched it. Not remeasured on a
  clean worktree.
- The projection tests take about eight minutes for three; a living map page takes minutes to project.

## Known open items (observed unless marked)

- `Stop-Apache.ps1` race described in section 2.
- The two island demo pages (`living_island_outside.html`, `living_island_inside.html`) sit unlinked in both web
  folders, built before the last two fixes; the outside page still raises the action-edge error on departure.
- The divider has no worker; the lowering catalogue is empty because existing products predate their description
  files.
- The repository scan is not fed into the world builder. `turing/docs/REPO_WORLD_RECORD_PROPOSAL.md` gives the
  generic-graph route and six questions for the owner. The scanner does not skip island sidecar files.
- The scan labels the referencing file's end `out`; the living map enters at `in`. Unresolved, owner's call.
- Unverified in a browser: `/profile.html`, `/divider.html`, the math lab's WGSL timing on a real GPU, the signed-in
  navigation menu, the world map's portal curves and search.
- The service still runs as the owner's Windows user; `Setup-ServiceAccount.ps1` has not been run. 8766 is bound to
  all interfaces. The certificate key is under the user profile.
- `C:/nonexistent_tmp` is a stray worker folder a safety check would not let this lane remove.

## Lessons Learned

- Function-level checks passed while the page was broken three times in one evening. Open the real page.
- When two lanes share a tree, stage hunks, not files; merge the remote instead of overwriting; identical local
  copies of remotely landed files must be committed before a merge will proceed.
- A layout rule is not a law until the owner says so. Two layouts were proposed and withdrawn the same day.
- Shell heredocs collapse doubled backslashes; write regex-bearing test files with the file tool.
- An extension to an engine belongs beside its physics. Adding rules to the page script repeated the original fault.

## Next Steps

Recorded in `todo/1794364200_convergence_continuation.stub.md`. In order of the owner's stated priorities:

1. Compiler first: the divider needs a worker, and the game's rules (collision response, portal and door crossing,
   mounting, navigation) need writing in Python so the compiler can divide them.
2. A play-session recording that replays through the compiled physics offline and reports energy per tick.
3. Feed the repository scan into the world builder by the generic-graph route, after the owner answers the six
   questions; skip island sidecars in the scanner.
4. Island documents: several occupants per document and a permanent entry (the dial), authorship from git history,
   guest download of the cookie drive.
5. Service hygiene: fix the stop race, run the service account setup, schedule backups, bind 8766 to loopback.

For the other agent: this lane owns `authz.py`, `messaging.py`, `profile_dashboard.py`, `divider_intake.py`,
`worldmap.py`, `listing.py`, `math_lab.py`, the deploy scripts, the `repo_world_*` modules,
`abstract_ui_doorways.py`, `abstract_ui_islands.py` and `portal_record.py`. Your newer `site-nav.js` (version 3,
control suites) is in source but the live web folders still serve version 1 plus one added menu line; publishing
yours is your call. Operator instructions are in
`turing/examples/meowdoku/deploy/OPERATOR_GUIDE_2026-11-10.md`.

## Prompt History

Owner, opening the day: three priorities: finish the compiler well enough to bootstrap, lower the dt system and
consume the compilable parts of Meowdoku; Marble Studio UX "being handled by me and a different agent, concurrent
work, don't upset it or be upset by it"; automated bridging between Nodus and a program.

Owner: "let's also get secure user to user messaging, something you and codex can talk to through the server as
authenticated users, basic contact list, message windows, bare minimum and not always open but made as a compiled
widget that any page can permit running"

Owner: "yes ZERO user to user contact in any context, fully managed messages"

Owner: "another module we need desperately is to take the living data map code and adapt it to work on filesystems,
and then make a world map that gets compiled occasionally that scans the entire github repo status (not our local
scratch C:\dev\powershell but a fresh clone) and turns all the folders into regions, subregions, files into rooms,
and portals on the walls of the rooms for calling out of that file or files that have called in"

Owner: "commit and push everything to main, we're consolidating individual changes to get a new landscape and then
moving forward"

Owner: "portals come from living map you got that right? and if you can get the analytical data, the 3d world
building code is already there ... that's two modes of the very same engine that already works" and "in fact it's
just about required that it uses generic graphs"

Owner: "so here's the general thing, don't make that the law everywhere i have a mroe elegant idea"

Owner: "I think the raw state needs to be like the initial state of the sims. someone else's structure and problems.
and we don't force the map to contain the character of the place in ways that detract or boast ... the interior, by
whim, is allowed tardis geometry ... somewhere in that or bypassing that option there is a portal to the context the
other thing is in that the reference involved ... so tardis geometry and finishing our portal system"

Owner: "each region will have a terrain natural bottle, you know, mountains, impossible rivers, whatever, except for
one passage, into a wilderness that simultaneously contains all endpoints for contexts ... that's the unbound flat
form of world geometry where portals and tardis doors represent the associational geometry of the universe"

Owner: "the entire wilderness is a procedural flux that is able to evolve anywhere people aren't around their
periphery and be stabilized only by their permanence, it only ever is as large as it needs to be"

Owner: "the portals don't have a shader, you know, so you can make the doorways also right now just a material sheet
forming a transition" and "have independent islands of existance for interiors, associated with - carried by - if
possible in some way, their associated files, perhaps in a private file" and "a local portal is a roller coaster I
think you see? nonlocal portals are the tardis doors"

Owner: "I think the regions should be stateful, but users can have explicit permissions for permanence through their
authorship role, something you can just walk into and take and if there's contention, heck then there's a dial next
to the door"

Owner: "unless you figure any of this out, it's all ruins. and it can always be ruins, because that's a new
document, and it costs nothing, because it's a procedural factory of ruins"

Owner: "guest account, guess what. cookie drive. they bring their own permanence with explicit warning and reminder
to download"

Owner: "this falls into a let's remake the engine category, because it has very very good bones, it just tried to
build an attic over a carport" then "like what I'm saying is that demo site is the problem" then "the engines are
solid and just need other solid engine parts"

Owner: "the way forward is clear, we make the thin control shell like we had been doing and we make the pythonic
finished product and we get the compiler to divide it into compileable parts and not, and like all our other pages,
right there is where the server will participate or JIT compile"

Owner: "projects are converging at this moment. this is becoming under an unexpected umbrella of speaktome, and it's
role in machine sim, turing, and nodus, and the web service. I would like to ask you to do a thorough continuation
report and summary in speaktome under that repo's agent conventions. the other agent is being told to migrate a few
things and will later make a similar report"

## Addendum, same evening: default branches (observed)

The owner pointed out that the "main" of speaktome and of the root repository is `nogodsnomasters`. Confirmed on
GitHub: that is the default branch of speaktome, the root repository, Nodus and spectral-analyzer; Turing's is `main`.
Both working branches merged `origin/nogodsnomasters` with no conflicts and were pushed onto it as fast-forwards:
speaktome `nogodsnomasters` = `045433b` (plus this addendum), root `nogodsnomasters` = `937de5d`. Every repository's
default branch now holds the consolidated work. The local checkouts of speaktome and the root are still on
`codex/direct-x86-vocabulary-report` and `codex/live-string-core-demo`, which point at the same commits; section 1's
table is superseded by this paragraph for those two rows.
