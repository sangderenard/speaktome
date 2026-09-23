# Engine Toy electrical hardware integration

## Result

The externally authored electrical-hardware patch was treated as an intention
bundle rather than applied verbatim. Its seven cable specimens, six connector
specimens, duplex receptacle, home/military/industrial panel interiors, passive
24-port 8P8C panel and Class L service cord now live in `engine_toy` source.
The redundant patch payload was removed after integration.

Manufacturing evidence remains field-level published, derived, authored or
unknown data. Authored plug counterparts no longer inherit published
receptacle-only evidence. Installed `ElectricalService` declarations are
separate from component maximum ratings.

Supported power cables emit the existing conductor-row ABI and project into
the existing thermal path. Connector contact proxies name the role-qualified
electrical buses consumed by the current Spectral electrical registry. The
composed cord registers on those same buses. Breaker contacts, receptacle tabs
and other paths whose resistance/switch laws remain unresolved are explicit
construction declarations rather than misleading live electrical edges. The
balanced-data paths likewise remain in their signal domain.

## Verification

- Focused hardware authoring and real-workspace integration: 81 passed.
- Related engine electrical, circuit, thermal, station and machine suites:
  97 passed with one pre-existing cffi deprecation warning.
- Full exporter: seven cables, six connectors and 19 real Machine graph
  documents produced.
- All 19 exported graphs passed the existing electrical coverage audit; seven
  contain live conductor bundles.
- Generated catalogue JSON validated and was regenerated from the integrated
  Python declarations.
- A wider run including the Spectral electrical suite reached 180 passed and
  five failures in pre-existing Spectral/Turing LLVM and Metrics boundaries:
  missing span extents/`greater` LLVM emission and the current `Metrics`
  constructor rejecting legacy `pub_tau` arguments. No source in those
  projects was changed.

## Prompt History

> investigate, please, the git patch and document just added to engine\_toy

> oh, well, okay did we make things then? your task is to fully scrub it all into our game, retaining all the handy reference details and concepts, all the same items. this was made by an agent with no direct access so think of it all like an intention

## Next Steps

None required for this bounded integration. Breaker trip/contact laws, live
mating events, RF transfer laws and manufacturer-unresolved construction facts
remain deliberately unbound until their existing game interfaces and evidence
are available.
