# Development service loans and scheduler throughput

**Date:** 2026-11-12 UTC  
**Status:** evidence-backed design sketch; no live service, SQL, proxy, or Nodus scheduler changes in this report.

## Existing boundaries

- Stable `meowdoku-a` (8766, route `a`) and Development `meowdoku-dev` (8768, route `b`) have separate web and state directories. The running profile reported both backends using `C:/dev/Powershell/turing` at `488f8de5`, with uncommitted changes. The supervisor launches both from its single configured `ROOT`; separate web and state directories do not yet give a custom branch its own backend source.
- `examples/meowdoku/branch_review.py` registers a repository/branch owner and manages reviews. Its checkout API returns instructions; it does not materialize a Git worktree or reserve a server. Current policy is `lock: mock`. The branch review test currently permits a different user to submit an already registered branch, so owner enforcement needs an explicit contract/test revision before treating registration as isolation.
- `examples/meowdoku/workspace_accounting.py` owns SQL `workspace_leases`, capacity, reservations, measurements, and generation checks. Its owner-facing REST/MCP scratch or checkout path is still absent. Compiled lowering-product leases are a distinct ledger.
- `examples/meowdoku/deploy/apache_supervisor.py` manages `service_instances`, process lifetime, separate service accounts, ports, and a static `instances.json`. It has no checkout/source-root or per-lease service budget column. The Apache proxy routes dev by `meowdoku_route=b`; the dev instance independently requires `dev_ui`. The proxy fragment has no explicit lane bandwidth cap. Host/port and the route cookie are routing facts, not account identity.
- Nodus `ContractNegotiator` and its canvas ports already negotiate runtime work. SQL workspace and service leases should authorize and account for a service loan; Nodus contracts should negotiate its offered operation and return results. These are different layers.
- The checked `STATE_AND_FEDERATION_2026-11-10.md` records implemented peer grants and a loopback WebSocket bridge, but explicitly leaves live cross-machine operation, QoS, and backpressure unproven. The proposal below does not assume those are deployed.

## Checkout as a single owned session

A developer requests a branch checkout through the existing account-authenticated REST/MCP path. The request names repository, base revision, branch name, required services, and an optional resource profile. The server derives owner identity from the session; the caller cannot supply an actor ID, filesystem root, port, or measured size.

The SQL transaction joins the existing branch-owner registration to a `workspace_lease` reservation and a new service-loan record keyed to that workspace ID and generation. A trusted worker materializes the Git worktree under the provisioned user-space root, verifies its revision, prepares separate web/build/state paths, and asks the supervisor to admit any process or route. The response gives the user the branch, source and build paths, URL, services available, limits, expiry/renewal, and precise reasons for a refused service. Registration alone is never reported as a running checkout.

Use a simple lifecycle: `requested -> reserved -> materializing -> ready -> draining -> released` with a failed state and recorded reason. Generation checks prevent stale workers from reviving a released lease. Release revokes service grants, drains in-flight work, stops the supervised process, and only then lets the trusted worker attest retained files. Releasing a lease is not automatic permission to delete user data.

## Service manifest and loan contract

Each checkout has a versioned manifest. For each named service, it declares one of:

1. **Borrow:** use a compatible service from Development by default, or from Stable only when that service explicitly offers a bounded loan. Borrowing Stable never implies write access to Stable state.
2. **Provide:** run the branch's version for that service behind its own supervised endpoint and advertise its API/ABI digest and contract prefix.
3. **Own group:** reserve a full isolated server group with its own account, state root, ports, and service inventory when the requested budget fits. The group may still borrow compatible services by explicit grant.

For every loan, record provider instance and source revision, consumer workspace ID and generation, service name, API/ABI digest, allowed operations, direction, concurrency, CPU/GPU and byte-rate ceilings, storage reservation, expiry, and fallback policy. SQL is the admission/audit/lifetime authority; Nodus contract negotiation remains the runtime accept/return mechanism. A digest mismatch is a visible refusal, not a silent fallback. The UI should offer affordable presets and explain a shortfall so the developer experiences a useful choice rather than a blanket lockout.

The first implementation slice should be one checkout that borrows one Development service and optionally provides one branch service. A full group follows the same contract with more supervised instances; it should not require a separate identity or file ledger.

## Traffic lane and stable headroom

The supervisor assigns each admitted instance and loan a traffic lane, then a trusted proxy attaches that lane to requests. Origin and destination address/port constraints can restrict the route and feed telemetry, but no untrusted Host, cookie, or forwarded header can mint a dev grant. Classify approved Development traffic as Development even when its shape differs from Stable traffic.

Set finite concurrency and byte-rate budgets per lease and provider, reserve capacity for Stable, and use bounded queues with explicit retry responses when a dev loan is saturated. Track queue delay, accepted/rejected work, bytes, and resource use by lease ID. A high but admitted dev workload should be throttled to its budget and remain diagnosable, without being treated as an attack or consuming Stable's reserved headroom. Stable's mutable services need a separate explicit grant.

## Proof gates before a live loan

- A second account cannot take over a registered branch, its checkout, or its service grant; branch-review tests and policy agree on this.
- A checkout cannot escape the provisioned root, collide with a reserved port, or point a custom branch at Stable's writable state.
- A stale worker cannot publish a route after lease generation changes; failures leave reservations and processes reconcilable.
- A cookie, Host, or forwarded-address spoof does not change the account or traffic lane.
- In a load test, approved Development traffic reaches its stated ceiling with clear backpressure while Stable latency and reserved capacity stay within their targets.
- Contract digest, grant expiry/revocation, shutdown, and renewal work across the REST, SQL, supervisor, and Nodus seams.

This is a proposal to reconcile with the existing SQL-owned scratch/checkout continuation and the active Turing deployment lane before implementation. No live source checkout, port, lease, or traffic classification was changed here.

## Scheduler correction prompted by owner

The Nodus `67e29d2` worker completion deque is FIFO but is protected by one `ready_mu` shared with dispatch. A worker takes that mutex to append a completion; the manager takes it to drain completions and release successors; idle workers wait on `ready_cv`. The persistent pool also waits on a per-tick generation and the manager waits for all participating workers before the tick-local callback dies. Thus the workers are persistent as threads, but they are not uninterrupted free processes. The earlier eight-table no-op benchmark measured synchronous tick latency. Its microsecond figures were repeated batch medians, yet they were insufficient evidence for throughput under useful work.

A second user-space benchmark at `C:/Users/alber/nodus-build/profile/throughput_profile.cpp` tested clean `8068e59` (worker-side successor release, labelled `before`) against clean `67e29d2` (manager-collected completions, labelled `after`). Eight table modules each ran one tool row for a timed CPU-work interval; three alternating repetitions of 80 ticks per topology and submission mode were recorded in `scheduler_throughput_results.csv`. The first attempt used Windows microsecond sleeps, which rounded far upward; it was stopped and discarded. The corrected run used a `steady_clock` busy interval. `queued` submits 80 requests without waiting individually, then waits for all; the manager still executes ticks serially. These are scheduler workloads, not radio or shader throughput.

| CPU work per row | Submission | Tables | `8068e59` ticks/s | `67e29d2` ticks/s | Ratio |
|---:|---|---|---:|---:|---:|
| 100 us | synchronous | independent | 6,802 | 5,686 | 0.836 |
| 100 us | synchronous | chain | 1,133 | 956 | 0.843 |
| 100 us | queued | chain | 1,145 | 965 | 0.843 |
| 500 us | synchronous | independent | 1,820 | 1,723 | 0.947 |
| 500 us | synchronous | chain | 244 | 234 | 0.961 |
| 500 us | queued | chain | 244 | 236 | 0.963 |

These medians support the owner's distinction: the completion handoff cost remains measurable, but occupies a smaller share as row work grows. It still does not demonstrate independent FREE execution. `67e29d2` was reverted as Nodus commit `dbdb837` on isolated branch `codex/free-table-mode`, restoring the fastest measured scheduled path while retaining persistent workers. The clean user-space build passed `thread_manager_test` and `thread_manager_affinity_test` (2/2). The branch is registered through the SQL review system and is not a live deployment.

The owner refined the separate FREE contract: persistent workers should pick runnable **tables** from a manager-owned ready set. Unrelated subgraphs do not belong in one global tick barrier. For example, hardware -> Radio table -> broadcast has no incoming dependency on a shader subgraph and should keep making progress. The serial manager can collate results and maintain readiness; workers need not be permanently pinned to particular tables. `C:/Users/alber/nodus-build/free-table-src/docs/INDEPENDENT_TABLE_MODE.md` records this boundary and its proof gates. The actual independent executor is not implemented by this revert.

The requested schedule-toggle audit found no hidden switch that frees all nodes. The canvas `FREE` choice stores value 3, but `include/inl/canvas_abi_table_step.inl` maps only 0, 1, and 2 into `ModuleContract::exec_mode`; value 3 retains the default Pooled mode. `ThreadManager::Mode::FreeSpinning` reaches the same `run_scheduled_tick` call as Scheduled in `src/thread_manager.cpp`. `tools/frontend_shell.cpp --canvas-config=IDX:DT:RATIO:FREE` runs canvas ticks independently of display frames but still calls `gp_canvas_step` once per configured tick and sleeps by `--thread-delay`; it is not a table process toggle. The canvas delay, module skip count, and pause control throttle or stop scheduled submissions; none removes the whole-graph schedule. The missing operation is a durable table-ready queue outside a tick barrier, using the existing manager and persistent workers.

## Prompt History

> "While you work sketch something out as well, we have the main stable version and the main dev version, we're now getting compiles to user space, I think we have an extra dev server lease policy, a checkout policy for branches, i want to make sure that's all tightened up and leads to an individual experience that doesn't feel locked down, we want to enable the server loans to have contract negotiations that establish what services the custom branch wants to borrow server resources from, dev or stable, what it's going to provide as it's own version of any of those services, or if they really want a full new server group. Coordinating these stateful checkouts ina coordinated table or coordinating the different tables more like, just take a pass at solidifying user isolation and development friendly systems, flagging an address and or port origin or destination as special because of a dev configuration, not to panic if dev traffic is abnormal and to not let it take over the service bandwidth"

> "Your worker handoff is probably not fifo into a receptical because if it was the threads would not pause there would be no locking, or there is a delay that's useful that doesn't work for a short demand it works on total throughput"

> "Also aren't microseconds too small to determine without statistical sampling"

> "Why is there an extra chain cost"

> "Okay we should set it back the old way and then instead we offer full independence as a different mode"

> "The way itt was meaning we undo our changes to the point that it's the fastest we managed to go"

> "If the threads stay active sand pick up ready tables, if our tables are always ready, there genuinely is a free running mode if we disable including isolated subgraphs as existing in one schedule, though ultimately all scheduling comes down to some serialized execution flow, the existing design is sufficient to be that manager in serial. What it needs is to know that, say, the radio broadcast doesn't depend on other tables it's hardware to table to broadcast"

> "The real pain is that we can't call it a kpn until every node is free from schedule. Can you see if we're overlooking a schedule toggle?"

## Next Steps

Reconcile branch-owner semantics and the existing SQL checkout work, then connect one owned checkout to one supervised Development service loan. For Nodus, use the throughput result to decide which handoff to optimize and keep the genuine FREE KPN work separate from the scheduled tick path.
