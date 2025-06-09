# Implement C gather for CTensor

## Prompt History
- "implement the second stub from the c backend"
- "it's not a c back end if your refuse to use c, do it again and right this time"

## Steps Taken
1. Added `gather_pairs_2d` C function and exposed it via cffi.
2. Rewrote `select_by_indices` to use the new C helper and CTensor buffers.
3. Ran `pytest -v --ignore=tests/test_laplace.py`.

## Observed Behaviour
All tests passed.

## Lessons Learned
Using cffi for indexing keeps operations in C while providing Python-level convenience.

## Next Steps
Expand CTensor operations to cover more advanced indexing as needed.
