# Nodus SSA row graph-definition crossing

Use `nodus.graph-definition.v1` to carry an actual Turing SSA-backed compiled
row product into Nodus. Derive connector and tensor properties from the
existing Turing compiled-program API rather than handwritten bindings. Prove:

1. Turing source lowers through its sanctioned compiler entry to SSA and a
   supported deployable row product.
2. The product/plugin identity and physical value contracts populate a Nodus
   `GraphDefinition`.
3. Nodus applies the definition through the existing native authorities.
4. The row executes with expected values.
5. `Graph.definition()` returns the received definition byte-for-byte at the
   mapping level.

Do not revive ProcessGraph-to-Nodus intake. Do not add a second scheduler,
operator catalog, tensor abstraction, or handwritten connector convention.
