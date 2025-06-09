# Standard library imports
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
if TYPE_CHECKING:  # pragma: no cover - type hints only
    from torch import Tensor
# Local application/library specific imports
from .scorer import Scorer # Assuming Scorer is in scorer.py
# --- END HEADER ---

class BeamSearchInstruction:
    """
    A single, self-contained “search instruction” that tells BeamSearch:
      1. Which scoring bins (function+width+temperature) to use this turn.
      2. What pre-sampling parameters (pre_temp, pre_top_k) to use.
      3. How many rounds before culling (cull_after).
      4. Which “action” to take (“expand_any”, “expand_targeted”, “promote”, “done”) 
         and, if targeted, what node_id/beam_idx.
    """

    @staticmethod
    def test() -> None:
        pass

    def __init__(
        self,
        node_id: int,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        justification: Optional[str] = None,
        priority: Optional[float] = None,
        issued_by: Optional[str] = "PyGeoMind",
        local_timestamp: Optional[float] = None,
        symbolic_id: Optional[str] = None,
        scorer: Optional['Scorer'] = None,
        score_bins: List[Tuple[Callable, int, float]] = None,
        pre_temp: float = None,
        pre_top_k: int = None,
        cull_after: int = None,
        lookahead_steps: Optional[int] = None,
        # New: Lookahead-specific scoring configuration
        lookahead_score_bins_config: Optional[Dict[str, Dict[str, Any]]] = None, # e.g. {'lh_bin1': {'fn': callable, 'params': {}}}
        lookahead_aggregate_fn: Optional[Callable[["Tensor"], "Tensor"]] = None, # Takes [num_bins, num_cands] -> [num_cands]
    ):
        self.node_id      = node_id
        self.action       = action
        self.metadata     = metadata or {}
        self.justification= justification # type: ignore
        self.priority     = priority
        self.issued_by    = issued_by
        self.local_timestamp = local_timestamp
        self.symbolic_id  = symbolic_id
        self.scorer       = scorer # Scorer might not be needed if policy details are passed directly

        _scorer_to_use_for_defaults = self.scorer
        if _scorer_to_use_for_defaults is None:
            # Create a temporary Scorer instance if none was provided, solely for fetching defaults.
            # This instance is not stored in self.scorer if self.scorer was intentionally None.
            _scorer_to_use_for_defaults = Scorer()

        # Store scoring policy details directly
        if score_bins is None:
            # Ensure access to default_score_policy and its "score_bins" key is safe
            default_bins_dict = _scorer_to_use_for_defaults.default_score_policy.get("score_bins", {})
            self.score_bins: List[Tuple[Callable, int, float]] = list(default_bins_dict.values())
        else:
            self.score_bins: List[Tuple[Callable, int, float]] = score_bins # type: ignore

        self.pre_temp: float = pre_temp if pre_temp is not None else _scorer_to_use_for_defaults.default_pre_temp
        self.pre_top_k: int = pre_top_k if pre_top_k is not None else _scorer_to_use_for_defaults.default_pre_top_k
        self.cull_after: int = cull_after if cull_after is not None else _scorer_to_use_for_defaults.default_cull_after
        self.lookahead_steps: int = lookahead_steps if lookahead_steps is not None else _scorer_to_use_for_defaults.default_score_policy.get("lookahead_steps", 1)
        
        # Store lookahead-specific scoring configurations
        self.lookahead_score_bins_config = lookahead_score_bins_config # Defaults to None if not provided
        self.lookahead_aggregate_fn = lookahead_aggregate_fn # Defaults to None if not provided

    def __repr__(self):
        base = f"{self.action.upper()} @ {self.node_id}"
        if self.priority is not None:
            base += f" [p={self.priority:.3f}]"
        if self.justification:
            base += f" -- {self.justification}" # type: ignore

        # Describe bins
        bins_desc = ", ".join(
            f"{fn.__name__}(w={width},t={temp:.2f})"
            for (fn, width, temp) in self.score_bins # Use self.score_bins directly
        )
        return (f"{base}  | bins=[{bins_desc}]  "
                f"| pre_t={self.pre_temp:.2f}, pre_k={self.pre_top_k}, cull={self.cull_after}, lookahead={self.lookahead_steps}"
                f"| lh_bins={True if self.lookahead_score_bins_config else False}"
                f", lh_agg={True if self.lookahead_aggregate_fn else False}")

    def get_lookahead_config_summary(self) -> str:
        num_lh_bins = len(self.lookahead_score_bins_config) if self.lookahead_score_bins_config else 0
        agg_fn_name = self.lookahead_aggregate_fn.__name__ if self.lookahead_aggregate_fn else "Default (RMS/Direct)"
        return f"Lookahead uses {num_lh_bins} scoring bins, Aggregate: {agg_fn_name}"
