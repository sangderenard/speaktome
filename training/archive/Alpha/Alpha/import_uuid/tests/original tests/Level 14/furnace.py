import torch
from torch_geometric.nn import MessagePassing
import logging
import uuid
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import random
import os
import pygame
import sys
from sklearn.decomposition import PCA  # For determining leading dimensions
import matplotlib.colors as mcolors
from torch_geometric.utils import remove_isolated_nodes
import matplotlib.pyplot as plt

# ===========================
# 1. Logging Configuration
# ===========================

# Create a 'logs' directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logger to write to a file instead of the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/simulation.log"),
    ],
)

# ===========================
# 2. Pygame Visualization Setup
# ===========================

# Initialize Pygame
pygame.init()

# Define window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_TITLE = "Graph Network Visualization"

# Create the Pygame window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption(WINDOW_TITLE)

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
EDGE_COLOR = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)

# Define node properties
NODE_RADIUS = 10

# Define font for text
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 14)

# Clock to control the frame rate
clock = pygame.time.Clock()
FPS = 3  # Frames per second

# Define node box dimensions
NODE_BOX_WIDTH = 100
NODE_BOX_HEIGHT = 80

# ===========================
# 3. Helper Functions for Visualization
# ===========================

def draw_graph(screen, positions, node_states, error_magnitudes, simulators, include_bounds=True, cmap='jet'):
    """
    Draws a grid of node status summaries on the Pygame screen.

    Args:
        screen (pygame.Surface): The Pygame screen surface.
        positions (dict): Dictionary mapping node indices to (x, y) positions.
        node_states (torch.Tensor): Tensor containing node state vectors.
        error_magnitudes (torch.Tensor): Tensor containing error magnitudes for color mapping.
        simulators (list): List of RandomWalkSimulator instances.
        include_bounds (bool): Whether to include bound axes in error magnitude calculation.
        cmap (str): Matplotlib colormap name for node coloring.
    """
    screen.fill(WHITE)  # Clear screen with white background

    # Normalize error magnitudes for color mapping
    norm = mcolors.Normalize(vmin=torch.min(error_magnitudes).item(),
                             vmax=torch.max(error_magnitudes).item())
    cmap = plt.get_cmap(cmap)

    for node_idx, pos in positions.items():
        # Determine node color based on error magnitude
        error = error_magnitudes[node_idx].item()
        color = cmap(norm(error))  # Returns RGBA
        color = tuple(int(255 * c) for c in color[:3])  # Convert to RGB

        # Draw rectangle (box) for the node
        rect = pygame.Rect(pos[0], pos[1], NODE_BOX_WIDTH, NODE_BOX_HEIGHT)
        pygame.draw.rect(screen, color, rect)

        # Get number of bound and unbound axes
        sim = simulators[node_idx]
        bounds = sim.config.bounds
        num_bound_axes = int(torch.sum(bounds).item())
        num_unbound_axes = bounds.numel() - num_bound_axes

        # Render node ID and bound/unbound axes
        text_id = FONT.render(f"Node {node_idx}", True, TEXT_COLOR)
        text_bounds = FONT.render(f"Bound: {num_bound_axes}", True, TEXT_COLOR)
        text_unbound = FONT.render(f"Unbound: {num_unbound_axes}", True, TEXT_COLOR)

        # Position text inside the box
        text_x = pos[0] + 5  # Small padding
        text_y = pos[1] + 5

        screen.blit(text_id, (text_x, text_y))
        screen.blit(text_bounds, (text_x, text_y + 20))
        screen.blit(text_unbound, (text_x, text_y + 40))

    pygame.display.flip()  # Update the full display Surface to the screen

def generate_grid_positions(num_nodes, box_width, box_height, margin=50):
    """
    Generates grid positions for nodes.

    Args:
        num_nodes (int): Number of nodes.
        box_width (int): Width of each box.
        box_height (int): Height of each box.
        margin (int): Margin from the window edges.

    Returns:
        dict: Mapping from node index to (x, y) positions.
    """
    grid_cols = int(np.ceil(np.sqrt(num_nodes)))
    grid_rows = int(np.ceil(num_nodes / grid_cols))

    positions = {}
    node_idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if node_idx >= num_nodes:
                break
            x = margin + col * (box_width + margin)
            y = margin + row * (box_height + margin)
            positions[node_idx] = [x, y]
            node_idx += 1
    return positions

def compute_error_magnitudes(node_states, simulators, include_bounds=True):
    """
    Computes error magnitudes for all nodes, optionally including bound axes.

    Args:
        node_states (torch.Tensor): Tensor containing node state vectors.
        simulators (list): List of RandomWalkSimulator instances.
        include_bounds (bool): Whether to include bound axes in error magnitude calculation.

    Returns:
        torch.Tensor: Tensor containing error magnitudes.
    """
    error_magnitudes = []
    for node_idx, state in enumerate(node_states):
        sim = simulators[node_idx]
        bounds = sim.config.bounds
        if include_bounds:
            # Include all axes
            error = torch.sum(torch.abs(state))
        else:
            # Exclude bound axes
            unbound_axes = (bounds == 0)
            error = torch.sum(torch.abs(state * unbound_axes))
        error_magnitudes.append(error.item())
    return torch.tensor(error_magnitudes)

# Constants
DT_INITIAL = .0001
NUM_NODES = 17
STATE_VECTOR_DIM = 7
STATE_DIM_FEATURES = 2
RANDOM_WALK_STEP = 1.1
NUM_ITERATIONS = 5000
ENERGY_BUDGET = 10000000.1

# Dual-mode configuration
EXCHANGE_MODE = "integer"  # Options: "integer" or "float"
TENSOR_TYPE = torch.float  # if EXCHANGE_MODE == "float" else torch.long

def conditional_round(tensor):
    """
    Rounds the tensor if EXCHANGE_MODE is 'integer', otherwise returns the tensor unchanged.
    
    Args:
    - tensor (torch.Tensor): Input tensor to conditionally round.
    
    Returns:
    - torch.Tensor: Rounded tensor if in integer mode, unchanged otherwise.
    """
    if EXCHANGE_MODE == "integer":
        return torch.round(tensor)
    return tensor

class NodeConfiguration:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold):
        """
        Configuration for a node's simulation capacity.
        
        Args:
        - bounds (torch.Tensor): Axes the node is bound by (binary mask).
        - force_limits (torch.Tensor): Maximum allowable force for each axis.
        - exchange_rates (torch.Tensor): Energy cost per unit correction for each axis.
        - debt_threshold (float): Maximum allowable energy debt.
        """
        self.bounds = bounds.to(TENSOR_TYPE)
        self.force_limits = force_limits.to(TENSOR_TYPE)
        self.exchange_rates = exchange_rates.to(TENSOR_TYPE)
        self.debt_threshold = torch.tensor(debt_threshold, dtype=TENSOR_TYPE)

    def __repr__(self):
        return (f"NodeConfiguration(bounds={self.bounds}, force_limits={self.force_limits}, "
                f"exchange_rates={self.exchange_rates}, debt_threshold={self.debt_threshold})")

class StateNegotiationContract:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold):
        self.bounds = bounds.to(TENSOR_TYPE)
        self.force_limits = force_limits.to(TENSOR_TYPE)
        self.exchange_rates = exchange_rates.to(TENSOR_TYPE)
        self.debt_threshold = torch.tensor(debt_threshold, dtype=TENSOR_TYPE)
        self.current_debt = torch.zeros_like(self.debt_threshold)
        self.energy_history = []
        self.error_history = []

    def apply_correction(self, correction, error):
        scaled_correction = torch.min(torch.abs(correction), self.force_limits) * torch.sign(correction)
        bounded_correction = scaled_correction * self.bounds
        bounded_correction = conditional_round(bounded_correction)  # Conditional rounding
        energy_cost = torch.sum(torch.abs(bounded_correction) * self.exchange_rates)

        self.error_history.append(error.clone().detach())

        if self.current_debt + energy_cost > self.debt_threshold:
            logging.info(
                f"Correction rejected. Correction: {correction}, "
                f"Energy cost: {energy_cost}, Current debt: {self.current_debt}, "
                f"Debt threshold: {self.debt_threshold}"
            )
            return torch.zeros_like(correction)
        else:
            self.current_debt += energy_cost
            self.energy_history.append(self.current_debt.clone().detach())
            logging.info(
                f"Correction accepted. Correction: {correction}, "
                f"Bounded correction: {bounded_correction}, "
                f"Energy cost: {energy_cost}, Current debt: {self.current_debt}"
            )
            return bounded_correction

    def regenerate_energy(self, radiation, dt):
        regen_amount = torch.tensor(radiation * dt, dtype=TENSOR_TYPE)
        old_debt = self.current_debt.clone()
        self.current_debt = torch.max(torch.tensor(0, dtype=TENSOR_TYPE), self.current_debt - regen_amount)
        logging.info(
            f"Energy regenerated. Regen amount: {regen_amount}, "
            f"Old debt: {old_debt}, New debt: {self.current_debt}"
        )

class ContractRenegotiator:
    def __init__(self, renegotiation_threshold=0.8):
        self.renegotiation_threshold = renegotiation_threshold

    def suggest_renegotiation(self, edge_attr):
        current_debt = edge_attr["current_debt"]
        debt_threshold = edge_attr["debt_threshold"]
        bounds = edge_attr["bounds"]

        renegotiation_mask = current_debt / debt_threshold > self.renegotiation_threshold
        logging.info(f"Renegotiation mask calculated: {renegotiation_mask}")

        renegotiated_bounds = bounds.clone()
        for i in range(renegotiated_bounds.size(1)):
            if renegotiation_mask.any():
                renegotiated_bounds[renegotiation_mask, i] = 0
                logging.info(f"Released bound on axis {i} for edges: {torch.where(renegotiation_mask)[0]}")

        return renegotiated_bounds

class DebtArbitor:
    def __init__(self):
        self.correction_logs = []
        self.fault_history = []
    def arbitrate(self, error, bounds, force_limits, exchange_rates, current_debt, debt_threshold):
        correction_suggestion = error * bounds
        correction_acceptance = torch.min(torch.abs(correction_suggestion), force_limits) * torch.sign(correction_suggestion)
        correction_acceptance = conditional_round(correction_acceptance)  # Conditional rounding
        energy_cost = torch.sum(torch.abs(correction_acceptance) * exchange_rates, dim=-1)

        debt_exceeded = current_debt + energy_cost > debt_threshold
        correction_acceptance[debt_exceeded] = 0
        updated_debt = current_debt.clone()
        updated_debt[~debt_exceeded] += energy_cost[~debt_exceeded]

        self.correction_logs.append({
            "error": error.clone(),
            "suggestion": correction_suggestion.clone(),
            "acceptance": correction_acceptance.clone(),
            "energy_cost": energy_cost.clone(),
            "debt_exceeded": debt_exceeded.clone(),
            "updated_debt": updated_debt.clone(),
        })

        logging.info(f"Debt arbitration performed. Correction acceptance: {correction_acceptance}, Updated debt: {updated_debt}")

        return correction_acceptance, updated_debt

class RandomWalkSimulator:
    def __init__(self, state_vector_dim, config):
        self.id = uuid.uuid4()
        self.state_vector = torch.randn(state_vector_dim, dtype=torch.float)  # Keep float for compatibility
        self.velocity = torch.zeros(state_vector_dim, dtype=torch.float)
        self.acceleration = torch.zeros(state_vector_dim, dtype=torch.float)
        self.config = config
        logging.info(f"Simulator {self.id} initialized with state vector {self.state_vector} and config {self.config}")

    def random_walk(self, dt):
        old_state_vector = self.state_vector.clone()
        self.acceleration = torch.randn_like(self.state_vector) * dt
        self.velocity += self.acceleration
        self.state_change = self.velocity * dt
        self.state_vector += self.state_change
        self.state_vector = conditional_round(self.state_vector)  # Apply conditional rounding here
        logging.info(
            f"Simulator {self.id} performed random walk. "
            f"Old state vector: {old_state_vector}, New state vector: {self.state_vector}, "
            f"Velocity: {self.velocity}, Acceleration: {self.acceleration}"
        )
        return torch.norm(self.state_change)

    def apply_second_derivative(self, correction):
        old_acceleration = self.acceleration.clone()
        self.acceleration += correction
        self.acceleration = conditional_round(self.acceleration)  # Apply conditional rounding here
        logging.info(
            f"Simulator {self.id} applied second derivative correction. "
            f"Correction: {correction}, Old acceleration: {old_acceleration}, "
            f"New acceleration: {self.acceleration}"
        )

class PillbugNetwork(MessagePassing):
    def __init__(self, num_nodes, num_features, num_subfeatures, temperature=0, gestalt_matrix_function=None):
        super().__init__(aggr="mean")  # Aggregation for message passing
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.temperature = temperature
        self.features = []

        # Initialize features
        if num_subfeatures > 0:
            # Use nested PillbugNetwork for each feature
            self.features = [PillbugNetwork(num_features, num_subfeatures, 0) for _ in range(num_nodes)]
        else:
            # Use RandomWalkSimulator for each node
            self.features = [
                RandomWalkSimulator(num_features, NodeConfiguration(
                    bounds=torch.ones(num_features),
                    force_limits=torch.ones(num_features) * 100000.0,
                    exchange_rates=torch.ones(num_features) * 1.0,
                    debt_threshold=100.0
                ))
                for _ in range(num_nodes)
            ]

        # Internal states for the network
        self.state_matrix = torch.zeros((num_nodes, num_features), dtype=torch.float)
        self.velocity_matrix = torch.zeros((num_nodes, num_features), dtype=torch.float)
        self.acceleration_matrix = torch.zeros((num_nodes, num_features), dtype=torch.float)
        self.gestalt_matrix = torch.zeros((num_features,), dtype=torch.float)
        self.gestalt_matrix_function = gestalt_matrix_function or self.default_gestalt_matrix_function
        self.renegotiator = ContractRenegotiator()
        self.arbitor = DebtArbitor()
        self.previous_corrections = torch.zeros((num_nodes, num_features), dtype=torch.float)
        self.id = uuid.uuid4()

    def random_walk(self, dt):
        for i, feature in enumerate(self.features):
            if isinstance(feature, PillbugNetwork):
                state_change = feature.random_walk(dt)  # Pass dt to nested networks
            elif isinstance(feature, RandomWalkSimulator):
                state_change = feature.random_walk(dt)  # Pass dt and capture state change
                self.state_matrix[i] = feature.state_vector
        logging.info(f"Pillbug Network {self.id} random walk applied. Updated state matrix:\n{self.state_matrix}")
        return  torch.norm(state_change)


    def apply_corrections(self, corrections):
        """
        Applies corrections to the acceleration matrix or delegates to subfeatures.
        Args:
            corrections (torch.Tensor): Matrix of corrections to be applied.
        """
        for i, feature in enumerate(self.features):
            if isinstance(feature, PillbugNetwork):
                # Apply corrections recursively in nested networks
                feature.apply_corrections(corrections[i])
            elif isinstance(feature, RandomWalkSimulator):
                # Apply corrections directly to the simulator
                feature.apply_second_derivative(corrections[i])

        logging.info(f"Corrections applied to Pillbug Network {self.id}.")

    def update_gestalt_matrix(self):
        """
        Updates the gestalt matrix using the specified gestalt matrix function.
        """
        self.gestalt_matrix = self.gestalt_matrix_function(self.state_matrix, self.velocity_matrix, self.acceleration_matrix)
        logging.info(f"Gestalt matrix updated for Pillbug Network {self.id}:\n{self.gestalt_matrix}")

    @staticmethod
    def default_gestalt_matrix_function(state_matrix, velocity_matrix, acceleration_matrix):
        """
        Default gestalt matrix function: Computes the sum of all flux (second derivatives).
        """
        return torch.sum(acceleration_matrix, dim=0)  # Sum of all flux across features

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Message-passing logic for error calculation and correction negotiation.
        """
        error = x_j - x_i
        logging.info(f"Message passing: Error calculated: {error}")

        renegotiated_bounds = self.renegotiator.suggest_renegotiation(edge_attr)
        edge_attr["bounds"] = renegotiated_bounds
        logging.info(f"Bounds after renegotiation: {renegotiated_bounds}")

        correction_acceptance, updated_debt = self.arbitor.arbitrate(
            error,
            renegotiated_bounds,
            edge_attr["force_limits"],
            edge_attr["exchange_rates"],
            edge_attr["current_debt"],
            edge_attr["debt_threshold"],
        )
        edge_attr["current_debt"] = updated_debt

        logging.info(
            f"Correction acceptance: {correction_acceptance}, Updated debt: {updated_debt}"
        )

        return correction_acceptance

    def energy_function(self, error, correction):
        """
        Computes energy as a function of error and correction.
        """
        return error / correction

    def radiate(self):
        """
        Reduces temperature based on a radiation coefficient and updates the radiation pool.
        """
        prev_temp = self.temperature
        self.temperature *= self.radiation_coefficient
        self.radiation_pool += (self.temperature - prev_temp)
        logging.info(f"Pillbug Network {self.id} radiated energy. Temperature: {self.temperature}, Radiation Pool: {self.radiation_pool}")

    def accept_energy(self, error, correction, dt):
        energy = self.energy_function(error, correction)
        self.temperature += energy
        self.radiate()
        self.random_walk(dt)  # Pass dt to random_walk

    def provide_energy(self, error, corrections):
        for feature, correction in zip(self.features, corrections):
            feature.accept_energy(error, correction)

    def process_iteration(self, edge_index, edge_attr, error, corrections, dt):
        self.provide_energy(error, self.previous_corrections, dt)  # Pass dt
        self.previous_corrections = self.forward(self.state_matrix, edge_index, edge_attr)
        self.apply_corrections(corrections)
        self.update_gestalt_matrix()

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

    for edge in edge_index.t():
        src, dst = edge[0].item(), edge[1].item()
        src_config, dst_config = configs[src], configs[dst]

        # Combine node configurations for the edge
        combined_bounds = torch.min(src_config.bounds, dst_config.bounds)
        combined_force_limits = torch.min(src_config.force_limits, dst_config.force_limits)
        combined_exchange_rates = (src_config.exchange_rates + dst_config.exchange_rates) / 2
        combined_debt_threshold = max(src_config.debt_threshold.item(), dst_config.debt_threshold.item())

        bounds_list.append(combined_bounds)
        force_limits_list.append(combined_force_limits)
        exchange_rates_list.append(combined_exchange_rates)
        debt_threshold_list.append(combined_debt_threshold)

    # Convert lists to tensors
    edge_attr = {
        "bounds": torch.stack(bounds_list),
        "force_limits": torch.stack(force_limits_list),
        "exchange_rates": torch.stack(exchange_rates_list),
        "debt_threshold": torch.tensor(debt_threshold_list, dtype=TENSOR_TYPE),
        "current_debt": torch.zeros(num_edges, dtype=TENSOR_TYPE),
        "energy_history": [],
    }

    # Initialize node features
    x = torch.stack([sim.state_vector for sim in simulators])

    logging.info(f"Graph created with {num_nodes} nodes and {num_edges} edges.")
    logging.info(f"Initial edge attributes: {edge_attr}")
    logging.info(f"Initial node states: {x}")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
    
    if edges_to_prune.sum().item() > 0:
        logging.info(f"Pruning {edges_to_prune.sum().item()} disconnected edges.")
        mask = ~edges_to_prune
        graph_data.edge_index = graph_data.edge_index[:, mask]
        for key in graph_data.edge_attr:
            if isinstance(graph_data.edge_attr[key], torch.Tensor):
                graph_data.edge_attr[key] = graph_data.edge_attr[key][mask]
            else:
                pass
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

if __name__ == "__main__":
    # Function to generate diverse configurations
    def generate_random_config(state_vector_dim, max_force=100000, max_debt=1000):
        bounds = torch.tensor([random.choice([0, 1]) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        force_limits = torch.tensor([random.uniform(max_force * 0.5, max_force) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        exchange_rates = torch.tensor([random.uniform(0.8, 1.2) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        debt_threshold = random.uniform(100, max_debt)
        return NodeConfiguration(bounds, force_limits, exchange_rates, debt_threshold)

    # Create diverse configurations
    configs = [generate_random_config(STATE_VECTOR_DIM) for _ in range(NUM_NODES)]

    # Initialize simulators with the provided configurations
    simulators = [RandomWalkSimulator(STATE_VECTOR_DIM, config) for config in configs]

    # Create the graph data
    graph_data = create_graph_data(simulators, configs)

    # Initialize the graph network
    graph_network = PillbugNetwork(NUM_NODES, STATE_VECTOR_DIM, STATE_DIM_FEATURES)

    # Generate grid positions for visualization
    positions = generate_grid_positions(NUM_NODES, NODE_BOX_WIDTH, NODE_BOX_HEIGHT)

    # Simulation main loop
    data_summary = []
    running = True  # Flag to control the main loop
    dt = DT_INITIAL  # Initialize dt

    # Store previous state change for dt scaling
    previous_state_change = 1.0  # Initialize with a non-zero value

    for iteration in range(NUM_ITERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Pygame window closed by user.")
                running = False
                break

        if not running:
            break

        logging.info(f"Iteration {iteration + 1} starting.")

        # Scale dt based on inverse of previous state change
        epsilon = 1e-5  # To prevent division by zero
        dt = 1.0 / (previous_state_change + epsilon)

        logging.info(f"Scaled dt for iteration {iteration + 1}: {dt}")

        # Perform random walk and update graph data
        total_state_change = graph_network.random_walk(dt)  # Pass dt
        graph_data.x = torch.stack([sim.state_vector for sim in simulators])

        # Update dt based on total state change
        previous_state_change = total_state_change

        # Apply corrections using the graph network
        corrections = graph_network(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        for i, sim in enumerate(simulators):
            sim.apply_second_derivative(corrections[i])

        # Regenerate energy for all edges
        for edge_id, (debt, bounds) in enumerate(zip(graph_data.edge_attr["current_debt"], graph_data.edge_attr["bounds"])):
            if torch.all(bounds == 0):
                logging.info(f"Edge {edge_id} completely severed. Skipping.")
                continue
            data_summary.append({
                "iteration": iteration + 1,
                "edge_id": edge_id,
                "current_debt": debt.item(),
                "bounds": bounds.tolist(),
            })

        logging.info(f"Iteration {iteration + 1} completed.")

        # Prune the graph
        continue_simulation, graph_data, simulators, positions = prune_graph(graph_data, simulators, positions)
        if not continue_simulation:
            logging.info("No nodes or edges remain. Exiting simulation loop.")
            break

        # Compute error magnitudes for color mapping
        error_magnitudes = compute_error_magnitudes(graph_data.x, simulators, include_bounds=True)

        # Update visualization
        draw_graph(screen, positions, graph_data.x, error_magnitudes, simulators, include_bounds=True)

        # Control the frame rate
        clock.tick(FPS)

    # Convert results to a DataFrame for visualization
    df = pd.DataFrame(data_summary)
    logging.info("Data Summary:\n" + str(df.head()))

    # Plot results using Matplotlib
    plt.figure(figsize=(12, 6))
    for edge_id in df["edge_id"].unique():
        edge_data = df[df["edge_id"] == edge_id]
        plt.plot(edge_data["iteration"], edge_data["current_debt"], label=f"Edge {edge_id}")
    plt.title("Energy Debt Over Time by Edge")
    plt.xlabel("Iteration")
    plt.ylabel("Current Debt")
    plt.legend()
    plt.grid()
    plt.show()

    # Clean up Pygame
    pygame.quit()
    sys.exit()
