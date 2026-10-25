# Symbolic spring runtime-string Fortran and shader proof

Resume the whole-function `run_symbolic_spring_image(expression_text)` compile
after the concurrent retained-loop correction finishes. Prove that
`expression_text` remains in the complete hierarchy/card public ABI, compile
the normal artifact to Fortran, and run at least two expressions without
recompiling. Finish automatic extraction of both PyOpenGL `compileShader` and
raw `glShaderSource/glCompileShader` wrapper sites, then package stages through
the existing `turing.shader-component.v1` ABI. Do not use a bespoke harness, a
baked SymPy equation, a reduced numerical trace, or a second AST control-flow
interpreter.

Continuation starts by testing the final unverified shader bundle/discovery
patch:

```text
py -3.11 -m pytest -q tests/test_shader_extractor.py
```

The wrapper extractor itself was green at 5 tests before that last patch. See
`1785934823_LOG_Turing_Symbolic_Spring_Whole_Program_Compiler_Session.md` for
the exact post-handoff fixes, real-source coverage counts, checkpoint keys,
concurrent ownership boundaries, and the required full-program proof.
