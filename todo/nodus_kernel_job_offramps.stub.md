# Nodus optical job ingress and off ramps

- Use authored `RayIntent` launch vectors and an explicit transport contract as
  the inputs to a graph-owned job interpreter. Preserve Spectral's source
  layouts and physics.
- Admit the T1–T5, forward, backward and BDPT configurations through the
  kernel library's existing shader and pipeline registrations. Schedule stage
  rows separately so a completed stage can publish its own outputs.
- Expose crossing vectors, full path records, pre-integration linear data and
  post-integration film data on typed graph edges as soon as each stage ends.
- Give every outgoing product job and generation identity, record layout and
  count, stage provenance and completion state before the generic TCP byte
  stream row sends it.
- Verify a real headless Nodus graph takes a job from the socket and returns
  at least one optical off ramp through another socket. Test concurrent jobs
  and backpressure without replaying uncertain work.
