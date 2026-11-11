# Follow-on REST boundary work

The first extraction is executable in speaktome_api/routes.py. Deployment and
remote publication are not complete. Coordinate the SpeakToMe-first/Turing-second
release through the existing Apache lifecycle and reverify against the chosen
source revisions. Do not move or initialize live state as a side effect.

Further backend migration is deliberately unimplemented: accounts, messaging
storage, StaticWorlds and graph cursor/query services still belong to the Turing
host. Nodus contract ports and nogodsnomasters Machine/PartPort/MachineSystem
remain their existing authorities. A later authorized integration must bind to
those exact objects/contracts, not add a second mailbox, world or simulator.
