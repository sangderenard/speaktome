# LOG: show with a camera, and a map as a website

**Date:** 1794404400 (2026-11-11)
**Title:** REST contract version 2: `picture_route`, `site_route`, `speaktome_api/site.py`; the `show` tool

## Scope
`speaktome_api/routes.py` (two handlers, API_VERSION 2), `speaktome_api/site.py` (new), `speaktome_mcp` (the `show`
tool; a bounded file read in `rest.py`), `speaktome_man` pages `show.1` and `speaktome.7`, three test files. The host
side is in turing: `examples/meowdoku/studio_pictures.py`, the server and supervisor wiring, `deploy/bake_site.py`;
and in nodus: `tools/add_contract_port.py`. Standard library only here.

## Environment
Windows, system Python 3.11. Tests: `python -m pytest --noconftest tests/test_speaktome_routes_show_site.py
tests/test_speaktome_site.py tests/test_speaktome_man.py tests/test_speaktome_mcp.py -q`.

## Commands Run
- The four test files above: 131 passed in 13.2 s.
- turing: `tests/test_meowdoku_studio_pictures.py` (12 passed; the opt-in real Nodus round trip passed with the
  library named), the REST dependency guard (4 passed), supervisor / Nodus service / Studio product (44 passed).
- A scratch end-to-end run: a picture request through the real Node scene builder and a real Nodus contract port on
  a scratch canvas host to the commit, about 1 s; the photo service was a recorder (no GPU).
- `deploy/bake_site.py --map manual`: 30 static pages; opened in a browser at desktop and phone width.

## Log Summary
**show.** `POST /api/show/pictures {request_id, settings?, photo?}` answers 202 with a receipt and
`GET /api/show/pictures/{id}` reads it. The host turns the request into its Studio request, carries it over a
dedicated Nodus prefix and commits it as a photo job owned by the signed-in account. The account, owner keys and the
authority recheck come from the authenticated host and are passed to the handler; nothing in the body is identity.
The lighting input is the fully resolved Exposure profile and camera coordinates are metres. The receipt ends as
committed, refused, failed or uncertain; an uncertain commit is never retried. The job's status, image and
cancellation stay on the existing photo endpoints. The `show` tool asks, reads the receipt with the exposure state,
and can attach the finished image.

The owner asked for the photo interface "as something like the action of look". The vocabulary already had `show`
for a rendered view, so it is filed there and described as look with a camera. Renaming it is one line in each of
the tool table, the page and the route path.

**A map as pages.** `site.py` turns any map into pages: a place is a page, a declared edge is a link. It uses the
host's graph indexer by injection and parses nothing itself. A page is rendered on first look and kept under a
folder named by the map's content hash, so a changed map is a new set and the old one is pruned. `bake()` writes the
whole map with the same renderer and relative links. `site_route` serves `/site/<map>/`; the host declares each
map's source, capability and refresh time. Pages carry no script and escape everything the map supplies.

**Not live.** The running service was not restarted: the gated stop/start was refused by this session's permission
system. Until a restart, `/api/show/pictures` and `/site/` do not exist on the live site. The live canvas must also
gain the Studio contract port in that stop window (a canvas host serves only the ports its file declares). The exact
order is in turing `docs/hazards/20261111T133801Z_main.md`. No GPU exposure has been made through the new path.

Other limits: guests outside the local network do not hold `view_site`, so help for them needs the baked folder
placed in the public release, which was not done. The repository world map is in another format and is not yet a
declared map. No menu link was added.

## Live, later the same day
The owner answered "You run it". The stop window ran at 07:50 local through turing `deploy/Enable-StudioPort.ps1`:
no exposure was running; the stack stopped; the canvas gained `studio.product` on port 47613 (the old canvas is kept
beside it); the stack started. The stable log's first line is "[studio] pictures go through Nodus port 47613
(studio.product)".

- `/site/manual/` and `/site/system/` answer on the live site as the account `claude`; an unknown place answers 404.
- One real picture through the `show` tool over stdio: asked (queued), preparing at 2.4 s, committed at 4.6 s,
  exposing at 9.0 s, complete at 13.2 s. Photo job `53aa9257...`, 256 pixels, 5.078 exposure seconds. `show
  picture=<id> image=true` returned a text part and an image part (169,678 bytes of PNG), which was saved and looked
  at: the default Studio subject, noisy as five seconds would be. stderr empty, exit 0.

This is the first GPU exposure through the path, and it removes the "not live" paragraph above.

## Prompt History
Owner: "it'll be in a menu somewhere, help in the form of putting the graph online in another step in the
universalization of computation into a physical system. because we have a graph, we can make a website, is the
principle there"

Owner: "a few points, we should have the ability to run this dynamic and bake it static, so we don't need files for
everything anybody ever looks at but we can cache them. the other thing is this, a need to integrate photo api into
the rest api as something like the action of look" (with the Studio lane's handoff pasted: use `StudioAgentProducts`,
route through `studio_product_nodus`, commit with `commit_studio_ray`, identity only from the authenticated host).

Owner: "Two small additions for the API handoff: the lighting input must be the fully resolved Exposure profile, not
Room-only overrides, and camera coordinates are in metres. The payload hash checks integrity; authorization still
comes from the authenticated host."
