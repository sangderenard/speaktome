# OpenRGB Direct mapping follow-up

Human hardware access and observation are required.

- Close ASRock Polychrome and every old OpenRGB instance.
- Start exactly one loopback-only OpenRGB SDK server with the reviewed isolated configuration.
- Inventory current zones after the Polychrome firmware update; do not reuse an assumed numeric header mapping.
- Dry-run, then explicitly approve one dim-red Direct frame with all other logical positions black.
- Record which motherboard, pump, fan, or quadrant sites respond for each tested logical position.
- Only after one-position Direct behavior is verified, dry-run and then
  explicitly apply the finite `openrgb_dt_walk.py` white-point pass. Record
  the reported dense-domain, region, system-address, and physical-site mapping.
- Stop on patterns, mirroring, failure to isolate, or unexpected controller behavior.
- Trace whether the pump and fans are connected serially, through a parallel splitter/hub, or on distinct headers before defining software segments.
