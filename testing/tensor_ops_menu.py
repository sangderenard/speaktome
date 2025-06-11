"""Interactive exploration program for tensor abstraction operations.

Allows selecting available backends and running small correctness tests.
"""
# --- END HEADER ---

import importlib.util
import os
import sys
from typing import Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tensors as ta


def available_backends() -> list[tuple[str, type[ta.AbstractTensorOperations]]]:
    backends: list[tuple[str, type[ta.AbstractTensorOperations]]] = [
        ("PurePython", ta.PurePythonTensorOperations)
    ]
    if importlib.util.find_spec("numpy") is not None:
        backends.append(("NumPy", ta.NumPyTensorOperations))
    if importlib.util.find_spec("torch") is not None:
        backends.append(("PyTorch", ta.PyTorchTensorOperations))
    try:
        from tensors.c_backend import CTensorOperations
        backends.append(("CTensor", CTensorOperations))
    except Exception:
        pass
    return backends


_SAMPLE_INPUT = {
    "stack": lambda ops: ops.stack([[1, 2], [3, 4]], dim=0),
    "pad": lambda ops: ops.pad([[1, 2], [3, 4]], (1, 1, 1, 1), value=0),
    "topk": lambda ops: ops.topk([1, 3, 2, 4], k=2, dim=-1),
    "sqrt": lambda ops: ops.sqrt([4.0, 9.0]),
    "view_flat": lambda ops: ops.view_flat([[1, 2], [3, 4]]),
}


def to_plain(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        np = None  # type: ignore

    if hasattr(value, "tolist"):
        return value.tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def run_test(name: str, ops: ta.AbstractTensorOperations, ref_ops: ta.AbstractTensorOperations) -> Tuple[bool, str]:
    if name not in _SAMPLE_INPUT:
        return False, "no sample defined"
    func = _SAMPLE_INPUT[name]
    try:
        expected = func(ref_ops)
        result = ops.benchmark(lambda: func(ops))
    except Exception as exc:
        return False, f"error: {exc}"

    if to_plain(result) == to_plain(expected):
        return True, ""
    return False, f"expected {to_plain(expected)!r}, got {to_plain(result)!r}"


def list_tests() -> list[str]:
    return list(_SAMPLE_INPUT.keys())


def choose_backend(backends: list[tuple[str, type[ta.AbstractTensorOperations]]]):
    print("Available backends:")
    for idx, (name, _) in enumerate(backends, start=1):
        print(f" {idx}. {name}")
    choice = input("Select backend: ")
    try:
        idx = int(choice) - 1
        _, cls = backends[idx]
        timing = input("Track operation time? [y/N]: ").lower().startswith("y")
        return cls(track_time=timing)
    except Exception:
        print("Invalid choice")
        return None


def menu() -> None:
    backends = available_backends()
    if not backends:
        print("No tensor backends available.")
        return

    ops = None
    while ops is None:
        ops = choose_backend(backends)

    ref_ops = ta.PurePythonTensorOperations()

    tests = list_tests()
    while True:
        print("\nMenu:")
        print(" 1. Run all tests")
        print(" 2. Run single test")
        print(" 3. List tests")
        print(" 4. Quit")
        choice = input("Select option: ")
        if choice == "1":
            for name in tests:
                ok, msg = run_test(name, ops, ref_ops)
                status = "PASS" if ok else f"FAIL ({msg})"
                if ops.track_time and ops.last_op_time is not None:
                    status += f" ({ops.last_op_time:.6f}s)"
                print(f"{name}: {status}")
        elif choice == "2":
            test_name = input("Test name: ")
            if test_name not in tests:
                print("Unknown test")
                continue
            ok, msg = run_test(test_name, ops, ref_ops)
            status = "PASS" if ok else f"FAIL ({msg})"
            if ops.track_time and ops.last_op_time is not None:
                status += f" ({ops.last_op_time:.6f}s)"
            print(f"{test_name}: {status}")
        elif choice == "3":
            print("Available tests:")
            for t in tests:
                print(" -", t)
        elif choice == "4":
            break
        else:
            print("Unknown option")


if __name__ == "__main__":
    menu()
