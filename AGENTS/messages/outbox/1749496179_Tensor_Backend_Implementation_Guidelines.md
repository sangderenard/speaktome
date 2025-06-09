"""
TENSOR BACKEND IMPLEMENTATION GUIDELINES:
----------------------------------------
1. OPERATOR IMPLEMENTATION:
   - DO NOT implement magic methods (__add__, __mul__, etc.)
   - These are handled by AbstractTensorOperations
   - Only implement the single designated operator method from the abstract class
   
2. TEST COMPLIANCE:
   - DO NOT create dummy/mock classes to pass tests
   - DO NOT implement functions just to satisfy test requirements
   - Either implement full functionality or leave as documented stub
   - Failed tests are preferable to false implementations

3. BACKEND RESPONSIBILITIES:
   - Implement only the core tensor operations defined in AbstractTensorOperations
   - All operator routing happens through the abstract class
   - Let test failures expose missing functionality naturally

4. DEPENDENCIES:
   - Import only the strictly required packages
   - Handle import failures gracefully for optional backends
   - Do not add dummy fallbacks for missing dependencies

Remember: Magic methods and operator overloading are EXCLUSIVELY handled by
AbstractTensorOperations. Backend implementations provide only the raw
tensor operations.
"""
```

This comment should be added to:
- `speaktome/tensors/numpy_backend.py`
- `speaktome/tensors/torch_backend.py` 
- `speaktome/tensors/jax_backend.py`
- `speaktome/tensors/c_backend.py`
- `speaktome/tensors/pure_backend.py`

The comment emphasizes:
1. No magic method implementations (these belong in AbstractTensorOperations)
2. No dummy/mock implementations to pass tests
3. Clear delineation of responsibilities
4. Proper dependency handling

This aligns with the architectural decisions shown in the experience reports and maintains the clean backend separation evidenced in the codebase.# filepath: speaktome\tensors\*_backend.py
