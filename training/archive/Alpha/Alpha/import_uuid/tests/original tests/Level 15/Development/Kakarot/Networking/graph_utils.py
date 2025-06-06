# graph_utils.py

import torch
import logging
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

# ========================================
# Functions for Graph Data Creation
# ========================================

def create_graph_data(simulators, configs):
    num_nodes = len(simulators)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                              dtype=torch.long).t()
    num_edges = edge_index.size(1)

    # Initialize edge attributes by combining configurations of connected nodes
    bounds_list = []
    force_limits_list = []
    exchange_rates_list = []
    debt_threshold_list = []
    exchange_types_list = []

    for edge in edge_index.t():
        src, dst = edge[0].item(), edge[1].item()
        src_config, dst_config = configs[src], configs[dst]

        # Combine node configurations for the edge
        combined_bounds = torch.min(src_config.bounds, dst_config.bounds)
        combined_force_limits = torch.min(src_config.force_limits, dst_config.force_limits)
        combined_exchange_rates = (src_config.exchange_rates + dst_config.exchange_rates) / 2
        combined_debt_threshold = max(src_config.debt_threshold.item(), dst_config.debt_threshold.item())
        # Determine exchange type per edge
        bounds_list.append(combined_bounds)
        force_limits_list.append(combined_force_limits)
        exchange_rates_list.append(combined_exchange_rates)
        debt_threshold_list.append(combined_debt_threshold)

        src_type = 1.3 if src_config.exchange_type == "float" else 1.0
        dst_type = 1.3 if dst_config.exchange_type == "float" else 1.0
        combined_type = src_type + dst_type  # Sum the encodings
        exchange_types_list.append(combined_type)

    # Convert to tensor
    exchange_types_tensor = torch.tensor(exchange_types_list, dtype=torch.float)

    # Add to edge_attr
    edge_attr = {
        "bounds": torch.stack(bounds_list),
        "force_limits": torch.stack(force_limits_list),
        "exchange_rates": torch.stack(exchange_rates_list),
        "debt_threshold": torch.tensor(debt_threshold_list),
        "current_debt": torch.zeros(num_edges),
        "energy_history": [],
        "exchange_types": exchange_types_tensor,  # Now a tensor
    }

    # Initialize node features
    x = torch.stack([sim.state_vector for sim in simulators])

    # Compute connectivity
    connectivity = num_edges

    logging.info(f"Graph created with {num_nodes} nodes and {num_edges} edges.")
    logging.info(f"Initial edge attributes: {edge_attr}")
    logging.info(f"Initial node states: {x}")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), connectivity

def prune_graph(graph_data, simulators, positions):
    """
    Prunes disconnected edges and removes nodes with no connected edges.

    Args:
        graph_data (torch_geometric.data.Data): The graph data object.
        simulators (list): List of RandomWalkSimulator instances.
        positions (dict): Dictionary mapping node indices to (x, y) positions.

    Returns:
        bool: True if pruning was performed, False otherwise.
    """
    # Step 1: Prune edges with all bounds set to zero
    bounds = graph_data.edge_attr["bounds"]
    edges_to_prune = torch.all(bounds == 0, dim=1)
    mask = ~edges_to_prune  # Keep edges not fully pruned

    if edges_to_prune.sum().item() > 0:
        logging.info(f"Pruning {edges_to_prune.sum().item()} disconnected edges.")
        # Apply mask directly to edge_index and edge attributes
        graph_data.edge_index = graph_data.edge_index[:, mask]
        for key in graph_data.edge_attr:
            if isinstance(graph_data.edge_attr[key], torch.Tensor):
                graph_data.edge_attr[key] = graph_data.edge_attr[key][mask]
    else:
        logging.info("No edges to prune based on bounds.")

    # Step 2: Remove isolated nodes
    new_edge_index, node_mask = remove_isolated_nodes(graph_data.edge_index, num_nodes=graph_data.num_nodes)[:2]

    nodes_removed = graph_data.num_nodes - node_mask.sum().item() if node_mask is not None else 0
    if nodes_removed > 0:
        logging.info(f"Removing {nodes_removed} isolated nodes.")
        graph_data.edge_index = new_edge_index
        graph_data.x = graph_data.x[node_mask]
        simulators[:] = [sim for sim, keep in zip(simulators, node_mask.tolist()) if keep]
        positions = {new_idx: pos for new_idx, (old_idx, pos) in enumerate(positions.items()) if node_mask[old_idx]}
    else:
        logging.info("No isolated nodes to remove.")

    # Update num_nodes
    graph_data.num_nodes = graph_data.x.size(0)

    # Step 3: Check if the graph is empty
    if graph_data.num_nodes == 0 or graph_data.edge_index.size(1) == 0:
        logging.info("No nodes or edges remain in the graph. Terminating simulation.")
        return False, graph_data, simulators, positions

    return True, graph_data, simulators, positions