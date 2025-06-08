from __future__ import annotations

"""Micro benchmark for tensor backends.

This script measures the CPU ``process_time`` for a simple operation
using whichever tensor backends are available. The goal is to provide a
quick sanity check when experimenting with alternate implementations.
"""

# --- END HEADER ---

import time
from speaktome.faculty import Faculty
from speaktome.core.tensor_abstraction import get_tensor_operations


def benchmark_sqrt(faculty: Faculty, reps: int = 1000) -> float:
    """Return seconds taken to repeatedly compute sqrt."""
    ops = get_tensor_operations(faculty)
    tensor = ops.tensor_from_list(list(range(100)), dtype=ops.float_dtype, device="cpu")
    start = time.process_time()
    for _ in range(reps):
        ops.sqrt(tensor)
    end = time.process_time()
    return end - start


def main() -> None:
    reps = 1000
    for fac in (Faculty.NUMPY, Faculty.JAX, Faculty.CTENSOR):
        try:
            t = benchmark_sqrt(fac, reps=reps)
            print(f"{fac.name:7} {t:.6f}s for {reps} sqrt calls")
        except Exception as exc:  # pragma: no cover - optional backends
            print(f"{fac.name:7} failed: {exc}")


if __name__ == "__main__":
    main()
