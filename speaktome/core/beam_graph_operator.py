# Standard library imports
import json
from typing import List, Set, Optional, Tuple, Dict, Callable, TYPE_CHECKING, Union, Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional
    torch = None

from ..tensors import (
    AbstractTensorOperations,
    get_tensor_operations,
)

# Local application/library specific imports
from .beam_tree_node import BeamTreeNode # Assuming BeamTreeNode is in beam_tree_node.py
# --- END HEADER ---
if TYPE_CHECKING:
    from .compressed_beam_tree import CompressedBeamTree # For type hinting 'tree'

class BeamGraphOperator:
    """
    Move a set of beam tree nodes—including all necessary ancestors for each path—onto a target device.

    Hierarchical Device Dependency Principles:
    -----------------------------------------
    - **Upward Migration (To Work Device):**
      When a leaf node (or a collection of leaf nodes) is moved to a work device (e.g., GPU),
      *all* of its parent nodes (ancestral lineage up to the root, or an optional stop ancestor)
      are also recursively moved to the device. This ensures that:
        - Each subtree or beam path on a device contains all the state needed to reconstruct the entire
          generation history for that path (tokens, scores, parent links).
        - No “dangling” references: every node on the device has its complete ancestral context present.

    - **Downward Migration (Retirement / Offload):**
      When a subtree or a leaf is moved off a work device (e.g., retired to CPU),
      *only* the selected nodes are moved. Their ancestors remain on the work device, 
      unless/until they are themselves selected for migration or retirement. 
        - This minimizes data movement and avoids unnecessary shuffling of “living” ancestors.
        - If an ancestor becomes unused (has no more active descendants), it will be independently
          retired or garbage-collected in due course, but is not immediately affected by any single descendant’s move.

    Rationale:
    ----------
    This hierarchical approach provides:
      - *Data integrity:* Upward moves are “closure operations” over the ancestral path.
      - *Efficiency:* Downward moves are “localized,” minimizing memory and transfer costs.
      - *Causal Traceability:* Every live beam or subtree on a device is always fully reconstructible.

    Args:
        leaf_indices (List[int]): Indices of leaf nodes to migrate. For each, the full path to the root
                                  (or the specified stop ancestor) will be included in the move.
        device (str): The target device ('cpu', 'cuda', etc.)
        stop_ancestor_idx (Optional[int]): If provided, migration of ancestors stops at (and includes) this node.

    Returns:
        Set[int]: The set of node indices that were actually migrated.

    Example:
        # Suppose you have a beam tree like this:
        #    root
        #     |
        #    a1
        #     |
        #    a2
        #   /  \
        #  b1  b2
        #
        # move_paths_to_device([b1_idx], "cuda")
        # will move b1, a2, a1, and root to "cuda"
        #
        # move_paths_to_device([b1_idx], "cpu")
        # (when offloading/retiring) moves only b1 to "cpu"
        # Ancestors a2, a1, root remain on their current device until/unless independently retired.
    """


    def __init__(
        self,
        tree: 'CompressedBeamTree',
        tensor_ops: AbstractTensorOperations | None = None,
    ) -> None:
        """
        :param tree: The CompressedBeamTree instance to operate on.
        """
        self.tree = tree
        self.tree.set_operator(self)
        self.tensor_ops = tensor_ops or get_tensor_operations()

    # --------- UNIVERSAL NODE MIGRATION ---------
    def move_nodes_to_device(self, node_indices, device: str):
        """
        Move arbitrary nodes to the given device (CPU/GPU).
        """
        for idx in node_indices:
            node = self.tree.nodes[idx]
            node.token_tensor = self.tensor_ops.to_device(node.token_tensor, device)
            node.score_tensor = self.tensor_ops.to_device(node.score_tensor, device)

    # --------- TOPOLOGICAL SELECTORS ---------
    def select_leaves(self) -> List[int]:
        """Return all node indices that have no children (leaves)."""
        return [i for i, n in enumerate(self.tree.nodes) if not n.children_node_indices]

    def select_subtree(self, root_idx: int) -> List[int]:
        """Return all node indices under a root (including root itself)."""
        stack = [root_idx]
        result = []
        while stack:
            idx = stack.pop()
            result.append(idx)
            stack.extend(self.tree.nodes[idx].children_node_indices) # type: ignore
        return result

    def select_path_up(self, leaf_idx: int, stop_ancestor_idx: Optional[int] = None) -> List[int]:
        """Return node indices from leaf up to root or stop_ancestor_idx (inclusive)."""
        path = []
        idx = leaf_idx
        while idx is not None:
            path.append(idx)
            if idx == stop_ancestor_idx:
                break
            idx = self.tree.nodes[idx].parent_node_idx # type: ignore
        return list(reversed(path))

    def select_ancestors(self, node_idx: int) -> List[int]:
        """Return all ancestor indices up to root, starting from parent (excludes self)."""
        ancestors = []
        idx: Optional[int] = self.tree.nodes[node_idx].parent_node_idx # type: ignore
        while idx is not None:
            ancestors.append(idx)
            idx = self.tree.nodes[idx].parent_node_idx # type: ignore
        return ancestors

    def select_descendants(self, node_idx: int) -> List[int]:
        """Return all descendants below node_idx (excludes self)."""
        descendants = []
        stack: List[int] = list(self.tree.nodes[node_idx].children_node_indices) # type: ignore
        while stack:
            idx = stack.pop()
            descendants.append(idx)
            stack.extend(self.tree.nodes[idx].children_node_indices) # type: ignore
        return descendants

    # --------- BATCH HELPERS ---------
    def move_leaves_to_device(self, device: str):
        """Move all leaves to device."""
        leaves = self.select_leaves()
        self.move_nodes_to_device(leaves, device)
        return leaves

    def move_subtree_to_device(self, root_idx: int, device: str):
        """Move entire subtree to device."""
        nodes = self.select_subtree(root_idx)
        self.move_nodes_to_device(nodes, device)
        return nodes

    def pad_nodes_to_relative_tree_depth(
        self,
        nodes: List[BeamTreeNode],
        pad_token_id: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        include_unwashed_parents: bool = False
    ) -> Tuple[Any, Any, List[List[int]]]:
        """
        Args:
            nodes: list of node objects, each with .token_tensor (1D tensor), .tree_depth (int), .clean (bool), and .parent_node_idx
            pad_token_id: int, value to use for padding
            device: torch device for output
            include_unwashed_parents: bool, if True, include up the parent chain all parents on the same device with clean == False

        Returns:
            data: [num_paths, width] tensor, with tokens placed at relative tree_depth offset
            mask: [num_paths, width] bool tensor
            lineage_indices: list of lists, giving for each row the sequence of node indices (leaf first, then its dirty parents)
        """
        if not nodes:
            return (
                self.tensor_ops.zeros((0, 0), dtype=self.tensor_ops.long_dtype, device=device),
                self.tensor_ops.zeros((0, 0), dtype=self.tensor_ops.bool_dtype, device=device),
                [],
            )

        if device is None:
            device = nodes[0].token_tensor.device

        dtype = nodes[0].token_tensor.dtype

        # --- For each node, gather the lineage to first clean or device break ---
        all_lineages = []
        for node in nodes:
            this_lineage = []
            n: Optional[BeamTreeNode] = node
            # Collect leaf first, then dirty parents (still on device)
            while n is not None:
                if n.token_tensor.device != device:
                    break
                this_lineage.append(n)
                # Only go up if include_unwashed_parents is True and node is not clean
                if not include_unwashed_parents or getattr(n, "clean", False):
                    break
                parent_idx = getattr(n, "parent_node_idx", None)
                n = self.tree.nodes[parent_idx] if parent_idx is not None else None # type: ignore
            all_lineages.append(this_lineage)

        # Now: each all_lineages[i] is a list of nodes, leaf first, then up toward ancestors
        # Find depth range for padding
        all_depths = [n.depth for lineage in all_lineages for n in lineage]
        min_depth = min(all_depths)
        max_depth = max(all_depths)
        max_tok_len = max(n.token_tensor.shape[0] for lineage in all_lineages for n in lineage)

        width = (max_depth - min_depth) + max_tok_len
        num_paths = len(nodes)

        data = self.tensor_ops.full((num_paths, width), pad_token_id, dtype=dtype, device=device)
        mask = self.tensor_ops.zeros((num_paths, width), dtype=self.tensor_ops.bool_dtype, device=device)
        lineage_indices = []

        for i, lineage in enumerate(all_lineages):
            # By convention, order in lineage: leaf first, then parents upward
            # To show all, you can concatenate tokens in that order or handle as you wish;
            # Here, we concatenate them in ancestor order, so tokens at earliest depth go left
            tokens = []
            depths = []
            node_indices = []
            for n in reversed(lineage):  # Ancestor order: root->leaf
                tokens.append(self.tensor_ops.to_device(n.token_tensor, device))
                depths.append(n.depth)
                node_indices.append(self.tree.nodes.index(n)) # type: ignore
            tokens = self.tensor_ops.cat(tokens, dim=0)
            d_start = min(depths) - min_depth
            d_end = min(d_start + tokens.shape[0], width)
            tlen = d_end - d_start
            data[i, d_start:d_end] = tokens[:tlen]
            mask[i, d_start:d_end] = True
            lineage_indices.append(node_indices)

        return data, mask, lineage_indices

    def move_paths_to_device(
        self,
        leaf_indices: List[int],
        device: Union[str, torch.device],                # <-- now accepts str or torch.device
        stop_ancestor_idx: Optional[int] = None,
        force: bool = False,
        file_path: Optional[str] = None
    ) -> Set[int]:
        """
        Move paths from leaves up to (and including) a stop ancestor to a device, with agnostic input:
         - device can be: 
             * "cpu" or "cuda" (strings),
             * torch.device("cuda") / torch.device("cpu"),
             * or "text" (special case to EXPORT to HD).
         - stop_ancestor_idx: optional index to stop at.
         - force: override rules (as before).
         - file_path: for import/export.

        Behavior:
         • If device == "text": treat as EXPORT (ignore any CPU/CUDA move).
         • Else if device is a torch.device or "cpu"/"cuda" string → treat as real device migration.
         • If file_path is non‐None AND device is "cpu"/"cuda": treat as IMPORT from HD.
        """
        seen: Set[int] = set()

        # 1) Normalize device argument
        special_export = False
        if isinstance(device, torch.device):
            target_dev = device
        elif isinstance(device, str):
            if device.lower() == "text":
                special_export = True
                target_dev = torch.device("cpu") # Placeholder, not used for export
            else:
                # e.g. device = "cuda" or "cpu"
                target_dev = torch.device(device.lower())
        else:
            raise ValueError(
                "move_paths_to_device: 'device' must be 'cpu', 'cuda', torch.device, or 'text'."
            )

        # 2) If "text" → EXPORT each path and return
        if special_export:
            if not file_path:
                raise ValueError("Must provide file_path when using device='text'.")
            for leaf in leaf_indices:
                if stop_ancestor_idx is not None:
                    path = self.select_path_up(leaf, stop_ancestor_idx)
                else:
                    path = self.select_path_up(leaf)
                self.export_node_set(path, file_path)
                seen.update(path)
            return seen

        # 3) If file_path is provided and device is cpu/cuda → IMPORT mode
        if (not special_export) and file_path is not None:
            # device is a real torch.device here
            imported_map = self.import_node_set(file_path, attach_parent_idx=stop_ancestor_idx)
            return imported_map

        # 4) Otherwise: real CPU↔CUDA migration
        for leaf in leaf_indices:
            node: BeamTreeNode = self.tree.nodes[leaf] # type: ignore
            current_dev = node.token_tensor.device  # e.g. torch.device("cuda") or torch.device("cpu")

            # If it's already on the exact same device, skip
            if current_dev == target_dev:
                continue

            # Determine upgrade vs downgrade
            is_upgrade = (current_dev.type == 'cpu' and target_dev.type == 'cuda')
            is_downgrade = (current_dev.type == 'cuda' and target_dev.type == 'cpu')

            if is_upgrade:
                # Move all ancestors up to root (or stop_ancestor_idx if forced)
                if stop_ancestor_idx is not None:
                    if not force:
                        raise RuntimeError(
                            "Upgrade: stop_ancestor_idx without force is not allowed."
                        )
                    path = self.select_path_up(leaf, stop_ancestor_idx)
                else:
                    path = self.select_path_up(leaf)

            elif is_downgrade:
                # Move **only** this leaf to CPU (unless force+stop matches)
                if stop_ancestor_idx is not None:
                    if not force or stop_ancestor_idx != leaf:
                        raise RuntimeError("Downgrade: cannot downgrade ancestors.")
                path = [leaf]
            else:
                # E.g. if current_dev is some other device (rare) or mismatched type
                continue

            # 5) Finally, move each node in 'path' onto target_dev
            for idx in path:
                n: BeamTreeNode = self.tree.nodes[idx] # type: ignore
                # skip if it's already on the target device (just to be safe)
                if n.token_tensor.device == target_dev:
                    continue
                n.token_tensor = self.tensor_ops.to_device(n.token_tensor, target_dev)
                n.score_tensor = self.tensor_ops.to_device(n.score_tensor, target_dev)
                seen.add(idx)

        return seen


    # --------- GRAPH SURGERY ---------
    def export_node_set(self, node_indices: List[int], file_path: str):
        """Export selected node indices (and all needed fields for resurrection) to a JSON file."""
        export_data = []
        for idx in node_indices:
            node: BeamTreeNode = self.tree.nodes[idx] # type: ignore
            export_data.append({ # type: ignore
                "node_idx": idx,
                "token": node.token_tensor.item(),
                "score": node.score_tensor.item(),
                "parent": node.parent_node_idx,
                "children": list(node.children_node_indices),
                "depth": node.depth,
            })
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def import_node_set(self, file_path: str, attach_parent_idx: Optional[int] = None) -> Dict[int, int]:
        """
        Import a node set from file. If attach_parent_idx is specified, attach imported root(s) under that node.
        Returns a map from old node idx to new node idx.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        idx_map: Dict[int, int] = {}
        # Create all nodes first (to have their indices for wiring children)
        for node_entry in data:
            parent = node_entry["parent"]
            node = BeamTreeNode(
                token=node_entry["token"],
                score=node_entry["score"],
                parent_node_idx=None,  # We'll wire this after all nodes exist
                depth=node_entry["depth"],
                device=self.tree.device,
                tensor_ops=self.tree.tensor_ops,
            )
            self.tree.nodes.append(node) # type: ignore
            new_idx = len(self.tree.nodes) - 1
            idx_map[node_entry["node_idx"]] = new_idx
        # Now set parents and children
        for node_entry in data:
            new_idx = idx_map[node_entry["node_idx"]]
            node: BeamTreeNode = self.tree.nodes[new_idx] # type: ignore
            if node_entry["parent"] is not None:
                node.parent_node_idx = idx_map.get(node_entry["parent"])
            if attach_parent_idx is not None and node_entry["parent"] is None:
                node.parent_node_idx = attach_parent_idx
            node.children_node_indices = [idx_map[c] for c in node_entry["children"]] # type: ignore
        return idx_map

    def split_subtree(self, root_idx: int) -> List[int]:
        """Detach a subtree (returns all node indices and unlinks from parent)."""
        nodes = self.select_subtree(root_idx)
        parent_idx: Optional[int] = self.tree.nodes[root_idx].parent_node_idx # type: ignore
        if parent_idx is not None:
            # Remove from parent's children
            self.tree.nodes[parent_idx].children_node_indices.remove(root_idx) # type: ignore
            self.tree.nodes[root_idx].parent_node_idx = None # type: ignore
        return nodes

    # --------- SINGLE ACCESS FUNCTION ---------
    def operate(self,
                select: Union[Callable[[], List[int]], List[int], Set[int]],
                move: Optional[str] = None,
                export: Optional[str] = None,
                import_: Optional[str] = None,
                attach_parent_idx: Optional[int] = None,
                device: Optional[str] = None, # Redundant with move
                file_path: Optional[str] = None # Redundant with export/import_
                ) -> Optional[Union[List[int], Set[int], Dict[int, int]]]:
        """
        Master access function.
        - select: function or lambda returning node indices (or just pass list/set).
        - move: device string (if moving to device).
        - export: file path (if exporting to file).
        - import_: file path (if importing from file).
        - attach_parent_idx: index to attach imported nodes under.
        - device: device string, can be used if `move` is not.
        - file_path: file path, can be used if `export` or `import_` are not.
        """
        node_indices: Union[List[int], Set[int]] = select() if callable(select) else select
        result: Optional[Union[List[int], Set[int], Dict[int, int]]] = None
        if move:
            self.move_nodes_to_device(node_indices, move)
            result = node_indices
        if export:
            self.export_node_set(node_indices, export)
            result = node_indices
        if import_:
            result = self.import_node_set(import_, attach_parent_idx=attach_parent_idx)
        return result
