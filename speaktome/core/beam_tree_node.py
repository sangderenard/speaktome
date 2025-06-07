# Standard library imports
from typing import List, Optional

# Third-party imports
import torch
# --- END HEADER ---

class BeamTreeNode:
    def __init__(self, 
                 token: int, 
                 score: float, 
                 parent_node_idx: Optional[int], 
                 depth: int, 
                 device="cuda", 
                 pyg_node_id: Optional[int] = None): # Added pyg_node_id
        self.token_tensor = torch.tensor([token], dtype=torch.long, device=device)
        self.score_tensor = torch.tensor([score], dtype=torch.float32, device=device)
        self.parent_node_idx = parent_node_idx  # index of parent node in self.nodes list
        self.depth = depth
        self.children_node_indices: List[int] = []
        self.pyg_node_id: Optional[int] = pyg_node_id # Store the PyG node ID
