# Standard library imports
from typing import List, Optional

# Local imports
from .tensor_abstraction import get_tensor_operations
# --- END HEADER ---

class BeamTreeNode:
    def __init__(self, 
                 token: int, 
                 score: float, 
                 parent_node_idx: Optional[int], 
                 depth: int, 
                 device="cuda", 
                 pyg_node_id: Optional[int] = None): # Added pyg_node_id
        tensor_ops = get_tensor_operations()
        self.token_tensor = tensor_ops.tensor_from_list([token], dtype=tensor_ops.long_dtype, device=device)
        self.score_tensor = tensor_ops.tensor_from_list([float(score)], dtype=None, device=device)
        self.parent_node_idx = parent_node_idx  # index of parent node in self.nodes list
        self.depth = depth
        self.children_node_indices: List[int] = []
        self.pyg_node_id: Optional[int] = pyg_node_id # Store the PyG node ID
