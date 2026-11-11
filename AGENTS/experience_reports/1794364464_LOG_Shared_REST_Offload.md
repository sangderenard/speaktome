# Shared REST offload into SpeakToMe

Moved the existing Messenger and Living Data query, navigation and static-DM
HTTP handler bodies into speaktome_api/routes.py. Turing retains tiny host
bindings and imports this explicit source dependency. The package imports only
the standard library. No second listener, mailbox, role registry, graph store
or simulator was introduced. Existing identity, authorization, TLS, limits,
handle/idempotency and command-receipt behavior is preserved.

Work is isolated in speaktome-rest-20261111 and turing-speaktome-rest-20261111,
based on SpeakToMe 7362174 and Turing fd9a9903. Canonical checkouts, concurrent
Claude work, Apache lifecycle, federation, credentials and live state are
untouched. The two commits must travel together; SpeakToMe deploys first.
Missing or mismatched source is an explicit startup error, not a legacy fallback.

The topology/ownership README identifies Turing's existing compiler and graph
services, Nodus's contract-port/tick runtime, and nogodsnomasters engine_toy
Machine/PartPort/MachineSystem. These are real existing boundaries, not a claim
that machine simulation or game-master permissions are exposed by this change.
Static DM still supports only inspect and render-phong.

## Verification

The baseline existing messaging/query/cursor/static suite passed 45 tests.
After extraction, 103 Turing tests passed in 69.43 seconds: messaging, clients,
graph query/cursors/static DM, dependency loader and existing authz/proxy/public
login suites. One pre-existing cffi imp deprecation warning. Twelve new isolated
SpeakToMe handler-contract tests passed. Independent review compared the
extracted bodies with their AST baseline and found no behavioral changes beyond
dependency substitution. Full suites, public TLS and browser rendering were not
run. No live restart or deployment is claimed.

Remote verification/publication remains blocked. Git's bundled SSH crashed;
native OpenSSH then reported github.com port 22 Permission denied. No alternate
transport was attempted after that denial. Local commits are not remote proof.

## Prompt History

User: "start offloading the rest api into speaktome it's really that repo's and there will develope a link beetween turing compiler nodus server and such relationship now to speaktome being realizeable on the right topography"

User: "right and it relates to nogodsnomasters machine system"

## Next Steps

Coordinate the paired source release through the existing Apache service lane.
Any later backend/storage extraction must preserve existing account, schema,
world and machine identity rather than introduce parallel state. See
`todo/speaktome_rest_boundary.stub.md`; live machine commands are not implemented.
