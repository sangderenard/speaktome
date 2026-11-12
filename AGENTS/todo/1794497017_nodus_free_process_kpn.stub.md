# Nodus FREE process network

Owner objective: when all tables run FREE, persistent workers execute true
independent table processes, communicate through lossless FIFOs, and pump
per-worker result FIFOs into a greedy manager-side result pool. Keep Scheduled
mode's same-tick dependency and FIFO behavior. The persistent-worker debug
toggle is an initial repair only; see the matching experience report.

Do not touch the separate Nodus collision engine. Qualify the free mode with
producer/consumer progress without additional canvas ticks, fan-out ordering,
seeded cycles, shutdown, dynamic graph quiescence, and lossless backpressure.
