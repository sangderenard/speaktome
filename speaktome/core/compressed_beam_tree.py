# Standard library imports
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import torch
from ..util.lazy_loader import lazy_install

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer  # For type hinting

if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData  # pragma: no cover

# Local application/library specific imports
from .beam_tree_node import BeamTreeNode  # Assuming BeamTreeNode is in beam_tree_node.py
from tensors import (
    AbstractTensor,
    get_tensor_operations,
)
# --- END HEADER ---

class CompressedBeamTree:
    def __init__(
        self,
        device: str = "cuda",
        tokenizer: Optional['PreTrainedTokenizer'] = None,
        operator=None,
        tensor_ops: AbstractTensor | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.operator = operator
        self.tensor_ops = tensor_ops or get_tensor_operations()
        self.nodes: List[BeamTreeNode] = []
        # beam_idx (external ID for a path/leaf) -> node_idx in self.nodes
        self.leaf_node_indices: Dict[int, int] = {}
        self.next_beam_idx = 0
        self.token_dtype = self.tensor_ops.long_dtype
        self.score_dtype = self.tensor_ops.float_dtype
        self.tokenizer = tokenizer # Needed for pad_token_id
        self.verbose = True # Or make it a parameter

        # PyG related attributes
        self.pyg_next_node_id_counter: int = 0
        # Stores attributes for PyG nodes: {pyg_id: {'token': int, 'score': float, 'depth': int, 'original_node_idx': int}}
        self.pyg_node_attributes: Dict[int, Dict[str, any]] = {}
        # Stores PyG edges as (source_pyg_id, target_pyg_id)
        self.pyg_edges: List[Tuple[int, int]] = []
        # Mapping from original node_idx in self.nodes to its pyg_id
        self.node_idx_to_pyg_id: Dict[int, int] = {}
        # Inverse mapping: pyg_id to original node_idx in self.nodes
        self.pyg_id_to_node_idx: Dict[int, int] = {}

    def add_children_to_node(
        self,
        original_node_idx: int,
        prefix_tokens: List[Any],
        prefix_scores: List[Any],
        num_children: int
    ) -> List[int]:
        """
        Under the given original_node_idx, create `num_children` new beams
        that share the same prefix (by re‐using add_root_beam on the prefix).
        Returns the list of newly created beam indices.
        """
        new_beam_indices: List[int] = []

        # Convert the prefix tensors into plain Python lists
        tokens = [int(t.item()) for t in prefix_tokens]
        scores = [float(s.item()) for s in prefix_scores]

        for _ in range(num_children):
            new_idx = self.add_root_beam(tokens, scores)
            new_beam_indices.append(new_idx)

        return new_beam_indices

    def extend_from_node(self, original_node_idx: int, num_children: int) -> List[int]:
        """
        Given an existing node index in self.nodes, generate `num_children`
        new children under that node. Return the list of newly created beam indices.
        """
        new_leaf_indices: List[int] = []

        # 1) Reconstruct the prefix of tokens+scores for that node
        prefix_tokens: List[Any] = []
        prefix_scores: List[Any] = []
        cur_idx = original_node_idx
        while cur_idx is not None:
            node = self.nodes[cur_idx]
            prefix_tokens.append(node.token_tensor)
            prefix_scores.append(node.score_tensor)
            cur_idx = node.parent_node_idx
        prefix_tokens.reverse()
        prefix_scores.reverse()

        # 2) Now call a helper that attaches `num_children` new tokens under original_node_idx
        #    (This assumes you factor out your “add root beam” logic into a method that can
        #     also add children under an arbitrary existing node. We’ll call it add_children_to_node.)
        new_leaf_indices = self.add_children_to_node(
            original_node_idx,
            prefix_tokens,
            prefix_scores,
            num_children
        )
        return new_leaf_indices
    def set_operator(self, operator):
        self.operator = operator
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    def add_root_beam(self, beam_tokens: List[int], beam_scores: List[float]) -> int:
        current_beam_idx = self.next_beam_idx
        self.next_beam_idx += 1
        parent_node_idx = None
        parent_pyg_id = None # PyG ID of the parent node
        last_node_idx_in_path = -1

        for i, (token, score) in enumerate(zip(beam_tokens, beam_scores)):
            # Create PyG node ID
            current_pyg_node_id = self.pyg_next_node_id_counter
            self.pyg_next_node_id_counter += 1

            node = BeamTreeNode(
                token,
                score,
                parent_node_idx,
                depth=i,
                device=self.device,
                pyg_node_id=current_pyg_node_id,
                tensor_ops=self.tensor_ops,
            )
            self.nodes.append(node)
            new_node_idx = len(self.nodes) - 1
            self.node_idx_to_pyg_id[new_node_idx] = current_pyg_node_id
            self.pyg_id_to_node_idx[current_pyg_node_id] = new_node_idx

            # Store attributes for PyG
            self.pyg_node_attributes[current_pyg_node_id] = {
                'token': token, 'score': score, 'depth': i, 'original_node_idx': new_node_idx
            }

            if parent_node_idx is not None:
                self.nodes[parent_node_idx].children_node_indices.append(new_node_idx)
                if parent_pyg_id is not None:
                    self.pyg_edges.append((parent_pyg_id, current_pyg_node_id))

            parent_node_idx = new_node_idx
            parent_pyg_id = current_pyg_node_id
            last_node_idx_in_path = new_node_idx
        
        if last_node_idx_in_path != -1:
            self.leaf_node_indices[current_beam_idx] = last_node_idx_in_path
        else: # Should not happen if beam_tokens is not empty
            self.next_beam_idx -=1 # Rollback
            return -1 
        return current_beam_idx

    def extend_leaves_batch(
        self,
        parent_beam_idxs: torch.LongTensor,   # [M] on GPU
        full_tokens:       torch.LongTensor,   # [M, L] on GPU
        full_scores:       torch.FloatTensor,  # [M, L] on GPU
        lengths:           torch.LongTensor    # [M] on GPU (each in [1..L])
    ) -> torch.LongTensor:
        """
        Batch‐extend multiple leaf beams in one call.

        Args:
          parent_beam_idxs: 1D LongTensor of existing beam_idx values you want to extend (shape [M], on GPU).
          full_tokens:      2D LongTensor (shape [M, L], on GPU) with each row’s token sequence.
          full_scores:      2D FloatTensor (shape [M, L], on GPU) with each row’s token scores.
          lengths:          1D LongTensor (shape [M], on GPU) where lengths[i] = valid length of full_tokens[i].

        Returns:
          1D LongTensor (shape [M], on CPU) containing the newly created beam_idx for each inserted child.
        """
        # 1) For each row i in [0..M-1], pick out the “new token” at position (lengths[i]-1):
        rows = torch.arange(parent_beam_idxs.size(0), device=full_tokens.device)   # [0,1,...,M-1] on GPU
        col_idxs = (lengths - 1).clamp(min=0, max=full_tokens.size(1)-1)           # shape [M]

        new_tokens = full_tokens[rows, col_idxs]   # shape [M], on GPU
        new_scores = full_scores[rows, col_idxs]   # shape [M], on GPU

        # 2) Stack (parent_beam_idx, new_token, new_score) into one [M,3] tensor on GPU
        stacked = torch.stack(
            [parent_beam_idxs,
             new_tokens,
             new_scores],
            dim=1   # → shape [M, 3], on GPU
        )

        # 3) Move that [M,3] tensor to CPU in one shot
        stacked_cpu = stacked.cpu()  # shape [M, 3] on CPU
        # Convert to NumPy array for efficient iteration
        stacked_numpy = stacked_cpu.numpy()

        M = stacked_cpu.size(0)
        new_beam_idxs = torch.empty(M, dtype=torch.long, device='cpu')

        # 4) One Python loop on CPU to call `extend_leaf_by_beam_idx` per child
        for i in range(M):
            p_idx, tok, scr = int(stacked_numpy[i, 0]), int(stacked_numpy[i, 1]), float(stacked_numpy[i, 2])

            # (Re‐use your existing single‐child code from compressed_beam_tree.py :contentReference[oaicite:1]{index=1})
            if p_idx not in self.leaf_node_indices:
                raise ValueError(f"Parent beam_idx {p_idx} not found as a leaf.")

            parent_node_idx = self.leaf_node_indices[p_idx]
            parent_node_obj = self.nodes[parent_node_idx]
            parent_pyg_id   = parent_node_obj.pyg_node_id
            new_depth       = parent_node_obj.depth + 1

            # Assign a new PyG node ID
            current_pyg_node_id = self.pyg_next_node_id_counter
            self.pyg_next_node_id_counter += 1

            # Create and append the new BeamTreeNode
            new_node = BeamTreeNode(
                tok,
                scr,
                parent_node_idx,
                depth=new_depth,
                device=self.device,
                pyg_node_id=current_pyg_node_id,
                tensor_ops=self.tensor_ops,
            )
            self.nodes.append(new_node)
            new_node_idx = len(self.nodes) - 1

            # Update PyG mappings
            self.node_idx_to_pyg_id[new_node_idx] = current_pyg_node_id
            self.pyg_id_to_node_idx[current_pyg_node_id] = new_node_idx
            self.pyg_node_attributes[current_pyg_node_id] = {
                'token': tok,
                'score': scr,
                'depth': new_depth,
                'original_node_idx': new_node_idx
            }

            # Link parent → child
            parent_node_obj.children_node_indices.append(new_node_idx)
            self.pyg_edges.append((parent_pyg_id, current_pyg_node_id))

            # Assign a new beam index for this child
            new_beam_idx = self.next_beam_idx
            self.next_beam_idx += 1
            self.leaf_node_indices[new_beam_idx] = new_node_idx

            new_beam_idxs[i] = new_beam_idx

        return new_beam_idxs

    def get_lengths_for_beam_indices(self, beam_indices: List[int]) -> torch.LongTensor:
        """
        Helper to get the lengths of existing beam paths in the tree.
        Length is defined as (depth of leaf node + 1).
        """
        lengths = []
        for beam_idx in beam_indices:
            if beam_idx in self.leaf_node_indices:
                node_idx = self.leaf_node_indices[beam_idx]
                # Depth is 0-indexed number of steps from root. Length is number of tokens.
                lengths.append(self.nodes[node_idx].depth + 1)
            else:
                # This should ideally not happen if beam_idx is for an active parent
                print(f"[Warning] get_lengths_for_beam_indices: beam_idx {beam_idx} not found in leaf_node_indices. Returning length 0.")
                lengths.append(0)
        return torch.tensor(lengths, dtype=torch.long, device='cpu') # Return on CPU

    def extend_leaves_batch_lookahead(
        self,
        parent_beam_idxs: torch.LongTensor,      # [M] on GPU - beam_idx of the parent leaf in the tree
        full_new_paths_tokens: torch.LongTensor, # [M, L_full] on GPU - *complete* paths from root
        full_new_paths_scores: torch.FloatTensor,# [M, L_full] on GPU - *complete* scores from root
        new_paths_lengths: torch.LongTensor,     # [M] on GPU - lengths of the full_new_paths
        initial_lengths_of_parents: torch.LongTensor # [M] on CPU/GPU - length of the parent_beam_idxs path in the tree
    ) -> torch.LongTensor:
        """
        Batch-extend multiple leaf beams with entire new path segments.
        Each row in full_new_paths_tokens represents a complete sequence from the root.
        The method will append the part of the sequence that comes *after* the parent_beam_idx's path.
        """
        M = parent_beam_idxs.size(0)
        new_leaf_beam_idxs_for_tree = torch.empty(M, dtype=torch.long, device='cpu') # Return CPU tensor

        initial_lengths_of_parents = initial_lengths_of_parents.to(parent_beam_idxs.device) # Ensure same device

        for i in range(M):
            parent_beam_idx_val = parent_beam_idxs[i].item()
            
            if parent_beam_idx_val not in self.leaf_node_indices:
                if self.verbose: print(f"[ERROR] Parent beam_idx {parent_beam_idx_val} for lookahead extension not found as a leaf.")
                new_leaf_beam_idxs_for_tree[i] = -1 
                continue

            current_parent_node_idx_in_tree = self.leaf_node_indices[parent_beam_idx_val]
            len_of_parent_path_in_tree = initial_lengths_of_parents[i].item()
            
            current_full_new_path_tokens = full_new_paths_tokens[i]
            current_full_new_path_scores = full_new_paths_scores[i]
            current_new_path_total_length = new_paths_lengths[i].item()

            start_index_of_new_segment = len_of_parent_path_in_tree
            last_added_node_idx_for_this_segment = current_parent_node_idx_in_tree

            for k in range(start_index_of_new_segment, current_new_path_total_length):
                token_to_add = current_full_new_path_tokens[k].item()
                score_for_token = current_full_new_path_scores[k].item()
                new_node_depth = self.nodes[last_added_node_idx_for_this_segment].depth + 1
                
                current_pyg_node_id = self.pyg_next_node_id_counter
                self.pyg_next_node_id_counter += 1

                new_node_obj = BeamTreeNode(
                    token_to_add,
                    score_for_token,
                    parent_node_idx=last_added_node_idx_for_this_segment,
                    depth=new_node_depth,
                    device=self.device,
                    pyg_node_id=current_pyg_node_id,
                    tensor_ops=self.tensor_ops,
                )
                self.nodes.append(new_node_obj)
                new_node_idx_in_self_nodes = len(self.nodes) - 1

                self.node_idx_to_pyg_id[new_node_idx_in_self_nodes] = current_pyg_node_id
                self.pyg_id_to_node_idx[current_pyg_node_id] = new_node_idx_in_self_nodes
                self.pyg_node_attributes[current_pyg_node_id] = {
                    'token': token_to_add, 'score': score_for_token, 'depth': new_node_depth,
                    'original_node_idx': new_node_idx_in_self_nodes
                }
                self.nodes[last_added_node_idx_for_this_segment].children_node_indices.append(new_node_idx_in_self_nodes)
                parent_pyg_id = self.nodes[last_added_node_idx_for_this_segment].pyg_node_id
                if parent_pyg_id is not None: self.pyg_edges.append((parent_pyg_id, current_pyg_node_id))
                last_added_node_idx_for_this_segment = new_node_idx_in_self_nodes
            
            final_leaf_node_for_this_path = last_added_node_idx_for_this_segment if current_new_path_total_length > start_index_of_new_segment else current_parent_node_idx_in_tree
            new_beam_idx_val = self.next_beam_idx
            self.next_beam_idx += 1
            self.leaf_node_indices[new_beam_idx_val] = final_leaf_node_for_this_path
            new_leaf_beam_idxs_for_tree[i] = new_beam_idx_val
            
        return new_leaf_beam_idxs_for_tree

    def get_pyg_data(self) -> Optional['PyGData']:
        """
        Constructs and returns a PyTorch Geometric Data object representing the current tree state.
        Node features (x) will be [token_id, score, depth].
        """
        original_node_indices_for_pyg: Dict[int, int] = {} # Must-Fix 1.1: Initialize the dictionary
        if not self.pyg_node_attributes:
            return None

        # Ensure pyg_ids are contiguous from 0 up to max_pyg_id_used
        if not self.pyg_node_attributes: # Handles empty tree
            max_pyg_id_used = -1
        else:
            max_pyg_id_used = max(self.pyg_node_attributes.keys())
        
        num_pyg_nodes = max_pyg_id_used + 1
        
        # Node features: [token_id, score, depth]
        node_features_list = []

        for pyg_id in range(num_pyg_nodes):
            attrs = self.pyg_node_attributes.get(pyg_id)
            if attrs:
                node_features_list.append([attrs['token'], attrs['score'], attrs['depth']])
                original_node_indices_for_pyg[pyg_id] = attrs['original_node_idx']
            else:
                node_features_list.append([0, 0.0, 0]) # Placeholder

        x = torch.tensor(node_features_list, dtype=torch.float, device=self.device)

        # Edge index
        if self.pyg_edges:
            edge_index_tensor = torch.tensor(self.pyg_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
        pyg_data_mod = lazy_install('torch_geometric.data', 'torch_geometric')
        PyGData = getattr(pyg_data_mod, 'Data')
        data = PyGData(x=x, edge_index=edge_index_tensor)
        
        # Add pyg_node_is_leaf and pyg_node_to_beam_idx
        pyg_node_is_leaf_list = [False] * num_pyg_nodes
        pyg_node_to_beam_idx_list = [-1] * num_pyg_nodes

        # Create reverse mapping from self.nodes index to beam_idx for leaves
        leaf_node_idx_to_beam_idx = {v: k for k, v in self.leaf_node_indices.items()}

        for pyg_id in range(num_pyg_nodes):
            original_node_idx = self.pyg_id_to_node_idx.get(pyg_id)
            if original_node_idx is not None and not self.nodes[original_node_idx].children_node_indices:
                pyg_node_is_leaf_list[pyg_id] = True
                pyg_node_to_beam_idx_list[pyg_id] = leaf_node_idx_to_beam_idx.get(original_node_idx, -1)

        data.pyg_node_is_leaf = torch.tensor(pyg_node_is_leaf_list, dtype=torch.bool, device=self.device)
        data.pyg_node_to_beam_idx = torch.tensor(pyg_node_to_beam_idx_list, dtype=torch.long, device=self.device)
        return data

    def snap_beam_path(
        self,
        tokens: List[int], 
        scores: List[float], 
        insert_under_beam_idx: Optional[int] = None,
        device_str: Optional[str] = None 
    ) -> int:
        """
        Inserts a pre-defined path (tokens and scores) into the beam tree.
        Checks for existing prefixes and extends them if found.
        Otherwise, attaches the new path under insert_under_beam_idx or as a new root.
        Returns the new beam_idx for the snapped path's leaf.
        """
        if not tokens or not scores or len(tokens) != len(scores):
            raise ValueError("Tokens and scores must be non-empty and have the same length.")

        effective_device = torch.device(device_str if device_str is not None else self.device)

        current_parent_node_idx: Optional[int] = None
        current_depth: int = 0
        token_idx_in_snap_to_process = 0 # Index of the first token in `tokens` to actually create a new node for

        if insert_under_beam_idx is not None:
            if insert_under_beam_idx not in self.leaf_node_indices:
                # This logic could be expanded to allow attaching to an internal node specified by its node_idx
                raise ValueError(f"snap_beam_path: insert_under_beam_idx {insert_under_beam_idx} not found as a leaf.")
            current_parent_node_idx = self.leaf_node_indices[insert_under_beam_idx]
            current_depth = self.nodes[current_parent_node_idx].depth + 1
        
        # Traverse existing tree to find common prefix if attaching to an existing node
        if current_parent_node_idx is not None:
            # Start from the node *after* current_parent_node_idx
            # We try to match tokens[0], tokens[1], ... as children of current_parent_node_idx
            # current_parent_node_idx itself is the end of the prefix we are attaching to.
            # The first token of the snapped path (tokens[0]) will be a child of current_parent_node_idx.
            
            # Let temp_current_node_idx be the node we are trying to find a child under
            temp_current_node_idx = current_parent_node_idx 

            for i in range(len(tokens)):
                snapped_token_to_match = tokens[i]
                found_match_for_this_token = False
                
                # Check children of temp_current_node_idx
                parent_node_obj = self.nodes[temp_current_node_idx]
                for child_node_idx in parent_node_obj.children_node_indices:
                    child_node = self.nodes[child_node_idx]
                    if child_node.token_tensor.item() == snapped_token_to_match:
                        # Matched an existing token. This child becomes the new temp_current_node_idx.
                        temp_current_node_idx = child_node_idx
                        # current_depth for the *next* token to be added will be child_node.depth + 1
                        current_depth = child_node.depth + 1 
                        token_idx_in_snap_to_process = i + 1 # Next token in `tokens` to consider
                        found_match_for_this_token = True
                        break # Found match for tokens[i], move to tokens[i+1]
                
                if not found_match_for_this_token:
                    # No child of temp_current_node_idx matches tokens[i].
                    # So, tokens[i:] will be new.
                    # The parent for tokens[i] is temp_current_node_idx.
                    current_parent_node_idx = temp_current_node_idx
                    # current_depth is already set correctly for tokens[i]
                    break 
            else: 
                # All tokens in the snapped path matched an existing path in the tree.
                # temp_current_node_idx is the leaf of this existing path.
                # Check if this existing path is already a registered leaf beam.
                for b_idx, n_idx in self.leaf_node_indices.items():
                    if n_idx == temp_current_node_idx:
                        return b_idx # Path already exists as a leaf
                
                # Path exists but is internal. Make this internal node a new leaf.
                new_beam_idx = self.next_beam_idx
                self.next_beam_idx += 1
                self.leaf_node_indices[new_beam_idx] = temp_current_node_idx
                return new_beam_idx
        else: # insert_under_beam_idx is None, so this is a new root path
            # We may already have one or more root nodes. Attempt to reuse any
            # matching prefix so we do not duplicate common segments.
            for root_idx, node in enumerate(self.nodes):
                if node.parent_node_idx is not None:
                    continue
                if node.token_tensor.item() != tokens[0]:
                    continue

                current_parent_node_idx = root_idx
                current_depth = node.depth + 1
                token_idx_in_snap_to_process = 1

                temp_current_node_idx = root_idx
                for i in range(1, len(tokens)):
                    snapped_token_to_match = tokens[i]
                    found = False
                    parent_node_obj = self.nodes[temp_current_node_idx]
                    for child_idx in parent_node_obj.children_node_indices:
                        child_node = self.nodes[child_idx]
                        if child_node.token_tensor.item() == snapped_token_to_match:
                            temp_current_node_idx = child_idx
                            current_depth = child_node.depth + 1
                            token_idx_in_snap_to_process = i + 1
                            found = True
                            break
                    if not found:
                        current_parent_node_idx = temp_current_node_idx
                        break
                else:
                    # Entire sequence already exists in the tree
                    for b_idx, n_idx in self.leaf_node_indices.items():
                        if n_idx == temp_current_node_idx:
                            return b_idx
                    new_beam_idx = self.next_beam_idx
                    self.next_beam_idx += 1
                    self.leaf_node_indices[new_beam_idx] = temp_current_node_idx
                    return new_beam_idx
                break

        # `token_idx_in_snap_to_process` is the index of the first token in `tokens` to create a new node for.
        # `current_parent_node_idx` is the tree node_idx under which the new segment should be attached.
        # `current_depth` is the depth for the first new node.

        last_added_node_idx = current_parent_node_idx 

        for i in range(token_idx_in_snap_to_process, len(tokens)):
            token_val = tokens[i]
            score_val = scores[i]
            
            current_pyg_node_id = self.pyg_next_node_id_counter
            self.pyg_next_node_id_counter += 1

            new_node = BeamTreeNode(
                token_val,
                score_val,
                parent_node_idx=last_added_node_idx,
                depth=current_depth,
                device=effective_device,
                pyg_node_id=current_pyg_node_id,
                tensor_ops=self.tensor_ops,
            )
            self.nodes.append(new_node)
            new_node_idx_in_tree = len(self.nodes) - 1
            
            self.node_idx_to_pyg_id[new_node_idx_in_tree] = current_pyg_node_id
            self.pyg_id_to_node_idx[current_pyg_node_id] = new_node_idx_in_tree
            self.pyg_node_attributes[current_pyg_node_id] = {
                'token': token_val, 'score': score_val, 'depth': current_depth, 
                'original_node_idx': new_node_idx_in_tree
            }

            if last_added_node_idx is not None:
                self.nodes[last_added_node_idx].children_node_indices.append(new_node_idx_in_tree)
                parent_pyg_id = self.nodes[last_added_node_idx].pyg_node_id
                if parent_pyg_id is not None:
                     self.pyg_edges.append((parent_pyg_id, current_pyg_node_id))
            
            last_added_node_idx = new_node_idx_in_tree
            current_depth += 1

        if last_added_node_idx is None: # Should only happen if tokens list was empty initially
            return -1 

        new_beam_idx = self.next_beam_idx
        self.next_beam_idx += 1
        self.leaf_node_indices[new_beam_idx] = last_added_node_idx
        return new_beam_idx


    def trace_beam_path(self, beam_idx: int, tokenizer: 'PreTrainedTokenizer') -> List[Tuple[int, float]]:
        """Trace back a beam path from the given ``beam_idx`` to the root.

        The method prints the decoded tokens and scores for each step and also
        returns them as ``[(token_id, score), ...]``.  If ``beam_idx`` does not
        correspond to a valid leaf, an empty list is returned.
        """
        path_ids: List[Tuple[int, float]] = []

        if beam_idx not in self.leaf_node_indices:
            if self.verbose:
                print(f"[Warning] trace_beam_path: beam_idx {beam_idx} not found.")
            return path_ids

        node_idx = self.leaf_node_indices[beam_idx]
        current_idx = node_idx
        while current_idx is not None:
            node = self.nodes[current_idx]
            token_id = int(node.token_tensor.item())
            score = float(node.score_tensor.item())
            path_ids.append((token_id, score))
            current_idx = node.parent_node_idx

        path_ids.reverse()

        print("\nTrace of beam path:")
        for i, (tok_id, score) in enumerate(path_ids):
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
            print(f"[{i:3}] {tok_str:12} score = {score:.2f}")
        print(f"\nPath length: {len(path_ids)} tokens")

        return path_ids
    def get_beam_tensors_by_beam_idx(self, beam_idx: int, max_len: int, read_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if beam_idx not in self.leaf_node_indices:
            device = torch.device("cpu") if read_only else self.device
            return (torch.zeros((0,), dtype=self.token_dtype, device=device),
                    torch.zeros((0,), dtype=self.score_dtype, device=device), 0)

        node_idx = self.leaf_node_indices[beam_idx]
        tokens_list, scores_list = [], []
        current_node_idx = node_idx

        while current_node_idx is not None and len(tokens_list) < max_len:
            node = self.nodes[current_node_idx]
            if read_only:
                # Extract as python scalar—do not risk device copy.
                tokens_list.append(int(node.token_tensor.item()))
                scores_list.append(float(node.score_tensor.item()))
            else:
                tokens_list.append(node.token_tensor)
                scores_list.append(node.score_tensor)
            current_node_idx = node.parent_node_idx

        tokens_list.reverse()
        scores_list.reverse()

        if not tokens_list:
            device = torch.device("cpu") if read_only else self.device
            return (torch.zeros((1,), dtype=self.token_dtype, device=device),
                    torch.zeros((1,), dtype=self.score_dtype, device=device), 0)

        if read_only:
            # Assemble from scalars into fresh CPU tensors
            tokens_tensor = torch.tensor(tokens_list, dtype=self.token_dtype, device="cpu")
            scores_tensor = torch.tensor(scores_list, dtype=self.score_dtype, device="cpu")
        else:
            tokens_tensor = torch.cat(tokens_list, dim=0)
            scores_tensor = torch.cat(scores_list, dim=0)

        length = tokens_tensor.size(0)
        return tokens_tensor, scores_tensor, length


    def get_batch_by_beam_indices(self, beam_indices: List[int], max_len: int, read_only: bool = False):
        batch_tokens_tensors, batch_scores_tensors, batch_lengths = [], [], []
        actual_max_len = 0
        for beam_idx in beam_indices:
            tokens, scores, length = self.get_beam_tensors_by_beam_idx(beam_idx, max_len, read_only=read_only)
            batch_tokens_tensors.append(tokens)
            batch_scores_tensors.append(scores)
            batch_lengths.append(length)
            if length > actual_max_len:
                actual_max_len = length
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else 0

        final_beams = torch.full((len(beam_indices), actual_max_len), pad_token_id, dtype=self.token_dtype, device=self.device)
        final_scores = torch.zeros((len(beam_indices), actual_max_len), dtype=self.score_dtype, device=self.device)
        final_lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long, device=self.device)

        for i in range(len(beam_indices)):
            l = batch_lengths[i]
            if l > 0: # only copy if length > 0
                final_beams[i, :l] = batch_tokens_tensors[i][:l] # Ensure slicing matches length
                final_scores[i, :l] = batch_scores_tensors[i][:l]

        return final_beams, final_scores, final_lengths_tensor

    @staticmethod
    def test() -> None:
        """Verify batch extension of leaves maintains full paths."""
        import torch

        tree = CompressedBeamTree(device="cpu")
        idx = tree.add_root_beam([1, 2], [0.1, 0.2])

        parent = torch.tensor([idx], dtype=torch.long)
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        scores = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float)
        lengths = torch.tensor([3], dtype=torch.long)

        new_ids = tree.extend_leaves_batch(parent, tokens, scores, lengths)
        new_idx = int(new_ids[0])

        tokens_tensor, _, length = tree.get_beam_tensors_by_beam_idx(new_idx, 3)
        assert length == 3
        assert tokens_tensor.tolist() == [1, 2, 3]
