# Cross-bundle external-reference API

Build on `SystemPort(kind="external_reference", external_domain="bundle")` and
the external request/completion rings. Define stable bundle identities,
versions and compatibility ranges, signed/content-addressed manifests, export
tables, discovery rules, loading and caching, lifecycle/release semantics, and
failure behavior. HTML must continue to resolve only explicitly registered
Turing bundles; it must not turn this facility into arbitrary URL loading or
host-system access.
