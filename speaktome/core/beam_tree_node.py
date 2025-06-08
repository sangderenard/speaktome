# Standard library imports
from typing import List, Optional

# Third-party imports
import torch

from .tensor_abstraction import (
    AbstractTensorOperations,
    get_tensor_operations,
)
# --- END HEADER ---

class BeamTreeNode:
    def __init__(
        self,
        token: int,
        score: float,
        parent_node_idx: Optional[int],
        depth: int,
        device="cuda",
        pyg_node_id: Optional[int] = None,
        tensor_ops: AbstractTensorOperations | None = None,
    ) -> None:
        self.tensor_ops = tensor_ops or get_tensor_operations()
        self.token_tensor = self.tensor_ops.tensor_from_list(
            [token], dtype=self.tensor_ops.long_dtype, device=device
        )
        self.score_tensor = self.tensor_ops.tensor_from_list(
            [score], dtype=self.tensor_ops.float_dtype, device=device
        )
        self.parent_node_idx = parent_node_idx  # index of parent node in self.nodes list
        self.depth = depth
        self.children_node_indices: List[int] = []
        self.pyg_node_id: Optional[int] = pyg_node_id # Store the PyG node ID
