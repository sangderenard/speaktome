# Turing CSS and interface binding graph

Continue the HTML containment frontend added in
`turing/src/compiler/html_process_graph.py` without expanding containment into
an all-purpose UI graph.

Next work:

- Define `AbstractUI`, its primitive operation vocabulary, and a backend
  registry patterned after `AbstractTensor`.
- Provide graph, browser, SDL, and Pygame/PyOpenGL backend contracts.
- Make prosaic OOP-annotation compilation and an interactive world-map
  projection mandatory reference backends in the conformance tables.
- Define the typed OOP adornment vocabulary and prove both projections preserve
  identical value/action identities and accessible routes.
- Normalize selected words from
  `turing/src/compiler/abstract_ui_vocabulary.py` into strict canonical graph
  schemas and record backend coverage/shortfalls without shrinking the
  aspirational authoring vocabulary.
- Grow `turing/src/compiler/abstract_ui_introspection.py` from one SSA
  `ClassEmission` to a correlated `ClassEmissionPlan`, retaining recursive
  inter-class tracks and honest function/field references.
- Lower the introspective world records into the prosaic HTML and planar
  world-map reference projections.
- Define neutral `InterfaceRule`, `InterfaceSelector`, and
  `InterfaceProperty` concepts.
- Parse a bounded CSS profile into selector-to-container and
  property-dependency relations.
- Define typed form-value and event/action bindings to program class fields
  and methods.
- Preserve containment as the sole ownership/scope hierarchy.
- Correlate the finished interface graph with `DualIRShell` hierarchy and
  reference tables.

See experience report
`1787602339_DOC_Turing_HTML_Interface_Containment_Graph.md`.
