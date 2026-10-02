# Mesh field-domain substrate

Implement the general `field_domains.py` layer already specified in
`engine_toy/FARADAY_CHAMBER_CONCEPTION.md`: real mesh occupancy, DEC topology,
per-domain law/state registration, exact managed-dt rollback/publications, and
derived presentation buffers. Reuse the repaired shared Hodge/cross substrate;
do not replace the honorary equations or create a demo-specific solver.
