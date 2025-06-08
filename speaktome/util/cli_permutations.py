from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Sequence, Tuple
# --- END HEADER ---


class CLIArgumentMatrix:
    """Generate permutations of CLI arguments with simple exclusion logic."""

    def __init__(self) -> None:
        self.options: Dict[str, Sequence[Any]] = {}
        self.exclusions: List[Tuple[str, ...]] = []

    def add_option(self, flag: str, values: Sequence[Any] | None = None) -> None:
        if values is None:
            values = [None]
        self.options[flag] = values

    def exclude(self, *flags: str) -> None:
        self.exclusions.append(tuple(sorted(flags)))

    def _is_excluded(self, combo: Tuple[Tuple[str, Any], ...]) -> bool:
        flags = {f for f, _ in combo if _ is not None or f.startswith('-')}
        for exc in self.exclusions:
            if set(exc).issubset(flags):
                return True
        return False

    def generate(self) -> List[List[str]]:
        keys = list(self.options)
        all_values = [self.options[k] for k in keys]
        combos: List[List[str]] = []
        for values in product(*all_values):
            items = tuple(zip(keys, values))
            if self._is_excluded(items):
                continue
            args: List[str] = []
            for flag, val in items:
                if val is None:
                    args.append(flag)
                else:
                    args.extend([flag, str(val)])
            combos.append(args)
        return combos

    @staticmethod
    def test() -> None:
        """Quick sanity check for permutation generation and exclusions."""
        matrix = CLIArgumentMatrix()
        matrix.add_option('--flag', [None])
        matrix.add_option('--num', [1, 2])
        matrix.exclude('--flag', '--num')
        combos = matrix.generate()
        assert ['--flag', '--num', '1'] not in combos
        assert ['--flag', '--num', '2'] not in combos
        assert ['--flag'] in combos
        assert ['--num', '1'] in combos and ['--num', '2'] in combos

