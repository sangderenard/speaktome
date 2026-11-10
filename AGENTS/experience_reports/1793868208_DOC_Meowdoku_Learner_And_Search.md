# Meowdoku learner and chess search, 2026-11-05

The work was in the independent `C:/dev/Powershell/turing` repository. The live Meowdoku server kept playing Lichess while its learner counters froze. A measured training step inside the process-wide AbstractTensor `no_grad` scope failed with `No gradient found for input`. Lichess evaluation and the learner share the same value network, so the network now serializes evaluation and training. The learner also reports any exception in the page status. The user asked to stop the live game server while working; PID 18544 was stopped.

The chess network gained a zero-initialized residual convolution, preserving predictions from existing eight-parameter checkpoints. Search now uses legal replies and a third ply on the three leading moves. Five focused tests passed. A 25-second standalone learner run advanced the cached corpus from epoch 0.02 to 0.12 with its thread alive.

The existing graph compiler emitted a small chess forward DLL, but reverse compilation stopped in the shared backward-rule registry at an unadmitted `Raise` in `index_adjoint` before it reached the chess graph. No LLVM chess training artifact was installed. The measured compiler refusal and training results are in `turing/docs/hazards/20261105T084244Z_main.md`.
The user then requested the C backend. The reverse graph acquisition stopped at the same `Raise`, before C lowering or emission.
Later, the user asked to investigate that refusal. A clean Turing worktree reproduced it. The registry graph builder skipped the public source compiler's existing exception conversion because it built without pursuit. Applying `ingest_exception_syntax` and recording control scopes before direct graph ingestion preserved the authored `index_adjoint` guard and built all 90 backward-rule functions. The focused regression passed in both the detached and canonical checkouts. The next chess reverse stop is `conditional ownership requires ingested control activity` during training-motion lowering; this is separately recorded in `turing/docs/hazards/20261105T090043Z_main.md`.
The next request added page controls to pause Lichess at the next idle point and resume it. The server was restarted on port 8766 after the current game ended; both HTTP actions were verified. Self-play learning continued (50 games and corpus epoch 0.75 in live status). The generated training motion's missing activity declaration was posted with root provenance. The probe then crossed a redundant raw ABI polymorphism write and an untyped `detach` in `bw_tanh`; both were repaired. The next native reverse refusal is an untyped host scalar `ts` in `unbroadcast`. No compiled chess training artifact exists yet. See `turing/docs/hazards/20261105T092023Z_main.md`.

## Prompt History

> The most recent changes to the chess game lichess network etc have led to the server not seeming too keep training can you look into it

> Meowdoku it's under it's a mode in the

> Turing, i think in build

> The server is able to keep running lichess but the training stopped almost immediately

> Can you deepen our search and nn while you get the training to work right?

> There's a way to do a graph compile of an abstract network nn using Adam and a graph based inversion backwards, there's a demo of it, it has an okay chance of being able to compile into llvm, so we could maybe run it a little faster

> Also kill the server while you work it's just losing

> Try the c backend i believe we put raise in it

> Is raise in the training ina way we can just remove

> Can you run the server now

> Okay, can you look into the raise error

> Can you diagnose and fix the new error, and, provide page controls to stop lichess at next idle

## Next Steps

The `unbroadcast` host-scalar class refusal needs its source and callsite evidence traced before C or LLVM chess training can proceed. The Python learner and Lichess idle control are running in the restarted Meowdoku server.
