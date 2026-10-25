# Turing Graph Function Sockets

**Date:** 1791806378
**Title:** Durable unresolved-function sockets and graph-native reductions

## Overview

Implemented literal ProcessGraph substitution for unresolved symbolic function
applications in `C:/dev/Powershell/turing`.  An outer symbolic equation remains
an algebraic envelope; an unresolved function declaration is a socket whose
call node is replaced by fresh copies of the supplied ProcessGraph body's
existing nodes and edges.  No body is printed back to source and no operation
is reconstructed from a label.

## Steps Taken

- Read the top-level, `speaktome`, Turing, testing, and current baseline rules.
- Confirmed that symbolic ingestion already lowers surviving Sum/Product to
  graph-native reductions and surviving finite Integral to Gauss-Legendre
  weighted reductions.
- Added a durable unresolved-function declaration identity to imported Call
  nodes so cache hydration does not turn Python object identity into a name
  guess.
- Added socket substitution to `process_graph_function_linking.py`.
- Copied only the body output's ancestor subgraph, wired body parameters to the
  exact call operands, replaced all call consumers, and preserved output/root
  metadata.
- Posted every inserted node as `Novel(graph_function_splice, (call, body
  value))` on the existing ingestion-value page.
- Changed symbolic node allocation to the existing monotonic ProcessGraph
  value-ID allocator; the prior local counter reused the removed Tuple
  construction envelope's ID.
- Added small structural and executable tests for a scalar socket, a dynamic
  finite sum, a definite integral, and a dynamic truncated zeta series.

## Observed Behaviour

- The scalar case compiles through the sanctioned source entry, reports zero
  concordance findings, and executes natively: `f(x)=x*x` inserted into
  `f(x)+2*x` gives 15 at x=3.
- Dynamic `Sum(term(k,x),(k,1,n))` retains a real graph `sum` operation.  The
  inserted body contains no copied input nodes; the outer scalar `x` input
  retains scalar shape while the declared reduction axis feeds the copied
  multiply.
- The dynamic six-term zeta series executes to the direct finite-series value.
- The definite integral executes through the compiler's own AbstractTensor SSA
  materialization.  LLVM truthfully reports that its current repository
  emission lacks `get_tensor`/`sum`, so the test does not hide that shortfall.
- Focused gate: 22 passed, 1 deselected (the unrelated forced-disk test), one
  existing CFFI deprecation warning.  Two established linked-ProcessGraph
  source tests also passed.
- The forced-disk dual-IR test currently fails in the uncommitted live-storage
  lane: after the linker correctly skips the pre-ingestion empty root, the
  failure is `LiveDiGraph.add_node` seeing NetworkX's
  `_CachedPropertyResetterAdjAndSucc` instead of an adjacency mapping.  This is
  before socket substitution and was not repaired here.

## Lessons Learned

- A post-import callee reference is too late: source call scheduling has
  already recorded a different ownership history.  Literal graph substitution
  is both cheaper and faithful to the requested boundary.
- Python class object identity is not durable across the symbolic disk cache.
  The canonical symbolic declaration representation must be stored at
  ingestion and used as the one socket key.
- Pointwise tensor adaptation is not blanket tensorization.  Reduction axes
  become tensor operands internally while unrelated public scalar ports retain
  their scalar ABI.

## Next Steps

None for the socket/reduction slice.  The independent live-storage descriptor
failure remains owned by that unfinished lane and is already documented as
uncommitted/incomplete in Turing's current baseline receipt.

## Prompt History

> "generally we've made scalar into tensor by feeding tensor inputs into a \"scalar\" graph, no? investigate, and see if you can make some tests that verify all the functions we're hoping to have"

> "do we test that the tensor adaptation did not produce a violation of convention by \"tensorizing\" something improperly that should've stayed scalar"

> "I feel like we should do sum, zeta, a few other things like that, uh, because if we use accumulators and loop wrapping, trivial to add to the dual ir, you make a sum or integral, you know? there are a few modifications that might be desired to perform symbolic math on programmatic content"

> "your words imply you need help understanding. sympy will not be involved in the process graph \"symbolic\" programmatic content, that is an emergent property of their being well defined graph entities"

> "however you did introduce the right idea. it would be better if we could accept a sympy expression that contains functions that remain unresolved, that sympy must solve around, we make that a process graph, and then we slip our progress graph into place, and where our catalog of graph based conversions can supply the operators wrapping those functions in the equation, we would supply them through the cheap graph manipulation. where we must use sympy wrapping, we take the sympy graph, and cheaply place the graph in question into the sympy graph where the functions are represented, unresolved."

> "Please realize that vision with iterative development and testing on small cases"

> "There is one identity per value and there must be one key to it."
