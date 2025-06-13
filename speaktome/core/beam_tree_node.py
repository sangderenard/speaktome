# Standard library imports
from typing import List, Optional

import torch

from tensors import (
    AbstractTensor,
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
        tensor_ops: AbstractTensor | None = None,
    ) -> None:
        tensor_ops = tensor_ops or get_tensor_operations()
        float_dtype = torch.float32 if torch is not None else None
        self.token_tensor = tensor_ops.tensor_from_list(
            [token], dtype=tensor_ops.long_dtype, device=device
        )
        self.score_tensor = tensor_ops.tensor_from_list(
            [score], dtype=float_dtype, device=device
        )

        self.parent_node_idx = parent_node_idx  # index of parent node in self.nodes list
        self.depth = depth
        self.children_node_indices: List[int] = []
        self.pyg_node_id: Optional[int] = pyg_node_id # Store the PyG node ID
