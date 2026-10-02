# Reversible shared-memory virtual multicore

Replace the current independent-memory execution heads with a deterministic
shared guest address space. Define a reversible scheduler, atomic transaction
ordering, per-core architectural state, memory-version ownership, barriers,
thread creation/join system-port events, and snapshot metadata sufficient to
explain which core produced every shared-memory transition. Preserve exact
rewind and branch/fork behavior without serializing display observation.
