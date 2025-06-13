# Standard library imports
import collections
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from ..util.lazy_loader import lazy_install
# --- END HEADER ---

if TYPE_CHECKING:
    from .beam_search import BeamSearch
    from .compressed_beam_tree import CompressedBeamTree
    from torch_geometric.data import Data as PyGData
    from sentence_transformers import SentenceTransformer

class BeamTreeVisualizer:
    def visualize_subtree(
        self,
        pyg_data: 'PyGData',
        tokenizer: 'PreTrainedTokenizer',
        beam_search_obj: Optional['BeamSearch'],
        root_pyg_node_id: int,
        title: str = "Beam Subtree",
    ) -> None:
        """Visualizes a subtree of PyGData starting from root_pyg_node_id using NetworkX and Matplotlib."""
        plt = lazy_install('matplotlib.pyplot', 'matplotlib')
        nx = lazy_install('networkx')
        if not pyg_data or pyg_data.num_nodes == 0:
            print("No PyG data to visualize.")
            return

        if not (0 <= root_pyg_node_id < pyg_data.num_nodes):
            print(f"Invalid root_pyg_node_id: {root_pyg_node_id}. Must be between 0 and {pyg_data.num_nodes - 1}.")
            return

        # 1. Subtree Node Identification using BFS
        subtree_pyg_node_ids = set()
        queue = collections.deque([root_pyg_node_id])
        
        # Build an adjacency list for efficient traversal
        adj = [[] for _ in range(pyg_data.num_nodes)]
        for i in range(pyg_data.edge_index.shape[1]):
            u, v = pyg_data.edge_index[0, i].item(), pyg_data.edge_index[1, i].item()
            adj[u].append(v)

        while queue:
            current_node = queue.popleft()
            if current_node not in subtree_pyg_node_ids:
                subtree_pyg_node_ids.add(current_node)
                for neighbor in adj[current_node]:
                    if neighbor not in subtree_pyg_node_ids: # Check to avoid re-adding
                        queue.append(neighbor)
        
        if not subtree_pyg_node_ids:
            print(f"Subtree from root {root_pyg_node_id} is empty (or root itself is invalid).")
            return

        # Determine min/max depth for color normalization within the subtree
        min_depth_subtree = float('inf')
        max_depth_subtree = float('-inf')
        if subtree_pyg_node_ids:
            for pyg_id_for_depth in subtree_pyg_node_ids:
                depth_val = int(pyg_data.x[pyg_id_for_depth, 2].item())
                min_depth_subtree = min(min_depth_subtree, depth_val)
                max_depth_subtree = max(max_depth_subtree, depth_val)
        depth_range_subtree = max_depth_subtree - min_depth_subtree
        if depth_range_subtree == 0: depth_range_subtree = 1 # Avoid division by zero


        # 2. NetworkX Graph Construction for the Subtree
        G = nx.DiGraph()
        for pyg_id in subtree_pyg_node_ids:
            token_id = int(pyg_data.x[pyg_id, 0].item())
            score = float(pyg_data.x[pyg_id, 1].item())
            depth = int(pyg_data.x[pyg_id, 2].item())
            is_leaf = pyg_data.pyg_node_is_leaf[pyg_id].item()
            beam_idx_val = pyg_data.pyg_node_to_beam_idx[pyg_id].item()
            try:
                token_str = tokenizer.decode([token_id])
            except: token_str = f"ID:{token_id}"
            
            label = f"{token_str}\nd:{depth},s:{score:.1f}"
            
            # Depth-based color
            normalized_depth = (depth - min_depth_subtree) / depth_range_subtree
            color_rgba = plt.cm.viridis(normalized_depth) # Get RGBA from colormap
            node_color = color_rgba[:3] # Use RGB for NetworkX


            if is_leaf:
                label += f"\nL,B:{beam_idx_val}"
                if beam_search_obj and hasattr(beam_search_obj, 'active_leaf_indices') and beam_idx_val in beam_search_obj.active_leaf_indices:
                    node_color = (0.6, 1.0, 0.6) # Light green for active leaves, overrides depth color
                elif beam_search_obj and hasattr(beam_search_obj, 'dead_end_indices') and beam_idx_val in beam_search_obj.dead_end_indices:
                    node_color = (1.0, 0.6, 0.6) # Light red for dead-end leaves, overrides depth color
            G.add_node(pyg_id, label=label, color=node_color)


        # 3. Edge Addition for the Subtree
        for i in range(pyg_data.edge_index.shape[1]):
            u, v = pyg_data.edge_index[0, i].item(), pyg_data.edge_index[1, i].item()
            if u in subtree_pyg_node_ids and v in subtree_pyg_node_ids:
                G.add_edge(u, v)

        if G.number_of_nodes() > 0:
            plt.figure(figsize=(18, 12))
            try:
                pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42) # Example: spring_layout
            except Exception as e:
                print(f"Warning: spring_layout failed ({e}), falling back to random_layout.")
                pos = nx.random_layout(G, seed=42)
            node_labels = nx.get_node_attributes(G, 'label')
            node_colors = [G.nodes[n].get('color', 'skyblue') for n in G.nodes()]
            nx.draw(G, pos, with_labels=False, node_size=1500, node_color=node_colors, edge_color='gray', alpha=0.8, arrows=True, font_size=8)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)
            plt.title(title)
            plt.axis('off')
            plt.show()
        else:
            print("NetworkX graph for subtree is empty, skipping visualization.")

    def _get_token_path_for_node(self, tree: 'CompressedBeamTree', node_idx: int) -> List[int]:
        tokens_list: List[int] = []
        current_node_idx: Optional[int] = node_idx
        while current_node_idx is not None:
            node = tree.nodes[current_node_idx]
            tokens_list.append(node.token_tensor.item())
            current_node_idx = node.parent_node_idx
        tokens_list.reverse()
        return tokens_list

    def visualize_sentence_embeddings(
        self,
        tree: 'CompressedBeamTree',
        sentence_model: 'SentenceTransformer',
        tokenizer: 'PreTrainedTokenizer',
        title: str = "Beam Tree Node Sentence Embeddings (PCA)",
    ) -> None:
        plt = lazy_install('matplotlib.pyplot', 'matplotlib')
        PCA = lazy_install('sklearn.decomposition', 'scikit-learn').PCA
        if not tree.nodes:
            print("Tree is empty. Nothing to visualize with PCA plot.")
            return

        path_strings = []
        node_indices_for_plot = [] 
        depths_for_plot = [] # Store depths for coloring

        print(f"Preparing {len(tree.nodes)} node paths for sentence embedding...")
        for i, node_obj in enumerate(tree.nodes):
            token_ids = self._get_token_path_for_node(tree, i)
            path_str = tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else ""
            path_strings.append(path_str)
            node_indices_for_plot.append(i)
            depths_for_plot.append(node_obj.depth)
        
        if not path_strings:
            print("No valid path strings generated. Cannot visualize.")
            return

        print(f"Embedding {len(path_strings)} path strings using SentenceTransformer...")
        embeddings = sentence_model.encode(path_strings, convert_to_tensor=True, show_progress_bar=True)
        embeddings_np = embeddings.cpu().numpy() # Ensure embeddings are on CPU for PCA

        if embeddings_np.shape[0] < 3:
            print(f"Need at least 3 data points for 3D PCA, got {embeddings_np.shape[0]}. Skipping PCA visualization.")
            return
        
        print("Performing PCA...")
        n_components = min(3, embeddings_np.shape[0], embeddings_np.shape[1])
        if n_components < 3:
             print(f"PCA n_components reduced to {n_components} due to data shape. Plotting in {n_components}D.")
        
        pca = PCA(n_components=n_components)
        pca_transformed_embeddings = pca.fit_transform(embeddings_np)

        print("Plotting PCA results...")
        fig = plt.figure(figsize=(15, 12))

        # Calculate colors based on depth
        node_colors_for_plot = None
        if depths_for_plot:
            min_d = min(depths_for_plot)
            max_d = max(depths_for_plot)
            depth_range = max_d - min_d if max_d > min_d else 1.0
            normalized_depths = [(d - min_d) / depth_range for d in depths_for_plot]
            node_colors_for_plot = [plt.cm.viridis(nd) for nd in normalized_depths]
        
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_transformed_embeddings[:, 0], pca_transformed_embeddings[:, 1], pca_transformed_embeddings[:, 2], c=node_colors_for_plot, alpha=0.6, s=10)
            ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2"); ax.set_zlabel("PCA Component 3")
            original_node_idx_to_pca_idx = {orig_idx: pca_idx for pca_idx, orig_idx in enumerate(node_indices_for_plot)}
            for current_original_idx, pca_idx_current in original_node_idx_to_pca_idx.items():
                parent_original_idx = tree.nodes[current_original_idx].parent_node_idx
                if parent_original_idx is not None and parent_original_idx in original_node_idx_to_pca_idx:
                    pca_idx_parent = original_node_idx_to_pca_idx[parent_original_idx]
                    x_coords = [pca_transformed_embeddings[pca_idx_parent, 0], pca_transformed_embeddings[pca_idx_current, 0]]
                    y_coords = [pca_transformed_embeddings[pca_idx_parent, 1], pca_transformed_embeddings[pca_idx_current, 1]]
                    z_coords = [pca_transformed_embeddings[pca_idx_parent, 2], pca_transformed_embeddings[pca_idx_current, 2]]
                    ax.plot(x_coords, y_coords, z_coords, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
        elif n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(pca_transformed_embeddings[:, 0], pca_transformed_embeddings[:, 1], c=node_colors_for_plot, alpha=0.6, s=10)
            ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
            original_node_idx_to_pca_idx = {orig_idx: pca_idx for pca_idx, orig_idx in enumerate(node_indices_for_plot)}
            for current_original_idx, pca_idx_current in original_node_idx_to_pca_idx.items():
                parent_original_idx = tree.nodes[current_original_idx].parent_node_idx
                if parent_original_idx is not None and parent_original_idx in original_node_idx_to_pca_idx:
                    pca_idx_parent = original_node_idx_to_pca_idx[parent_original_idx]
                    x_coords = [pca_transformed_embeddings[pca_idx_parent, 0], pca_transformed_embeddings[pca_idx_current, 0]]
                    y_coords = [pca_transformed_embeddings[pca_idx_parent, 1], pca_transformed_embeddings[pca_idx_current, 1]]
                    ax.plot(x_coords, y_coords, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
        plt.title(title); plt.tight_layout(); plt.show()
