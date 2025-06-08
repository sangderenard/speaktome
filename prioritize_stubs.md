# Stub Prioritization

| Stub | Priority | Rationale |
| --- | --- | --- |
| `speaktome/core/beam_search.py` - `failed_parent_retirement` | High | Affects beam pruning correctness and memory usage during core search, may impact results if parents linger. |
| `AGENTS/tools/test_all_headers.py` - `recursive test runner` | Medium | Useful for automated coverage of faculty tiers but not blocking existing tests. |
| `AGENTS/tools/validate_headers.py` - `header validation helper` | Medium | Provides code quality enforcement; not critical for functionality. |
| `speaktome/core/tensor_abstraction.py` - `PurePythonTensorOperations.__init__` | Low | Only relevant when using the pure Python backend, optional feature. |

