# Nodus bridging: contract ports and web-lane API contract

## Prompt History

User: "a whole program from the compiler should become one row. the entire system works on tick scheduling. if we finish nodus entities we can have multiple messengers running concurrently so it isn't a big deal if one row in a table is a blocking complex program"

User: "the prototype connection to a thread manager instance, I think you realize, should be made into a portable canvas file, built into one somehow, and then if we made a port in the canvas, the top level ports, that was a websocket type, we could put a direct connection between the running nodus and the python server that wants to do business with it"

User: "Sounds like it was a hacky trial i trust you can do it right"

## Findings

Nodus commits d16e395, 01f7d65, ddd78c5 add contract ports to the headless canvas: canvas-owned negotiator pair, result edge as transport, 127.0.0.1 listener, persisted via EDGE_FIFO / PORT / CONTRACT_PORT trailer records, refusal of unregistered plugin rows on programmatic adoption. Turing worktree bridging/web-lane-api (0d267080, cc1b307e) makes the Meowdoku wasm/WGSL products write `<stem>.api.yaml`; the graph-forward lane is not covered.

Not done: GUI hosting (in progress), autosave trailer rewrite, Spectral/Meowdoku server still on NativeMailHost, restart durability, authentication beyond loopback.

Details: `C:/dev/Powershell/nodus/docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md`. Turing hazards entry: `docs/hazards/20261110T155027Z_bridging-web-lane-api.md` in C:/Temp/wtb-api. Proposal: `C:/dev/Powershell/NODUS_BRIDGING_PROPOSAL_2026-11-10.md`.
