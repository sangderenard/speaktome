import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import uuid

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Constants
NUM_NODES = 17
STATE_VECTOR_DIM = 7
RANDOM_WALK_STEP = 1.1
NUM_ITERATIONS = 500
ENERGY_BUDGET = 10000000.1
REGEN_RATE = 1.1
DT = 1.1

# Dual-mode configuration
EXCHANGE_MODE = "integer"  # Options: "integer" or "float"
TENSOR_TYPE = torch.float# if EXCHANGE_MODE == "float" else torch.long
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

    def regenerate_energy(self, dt):
        regen_amount = torch.tensor(REGEN_RATE * dt, dtype=TENSOR_TYPE)
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

    def random_walk(self):
        old_state_vector = self.state_vector.clone()
        self.acceleration = torch.randn_like(self.state_vector) * RANDOM_WALK_STEP
        self.velocity += self.acceleration
        self.state_vector += self.velocity
        self.state_vector = conditional_round(self.state_vector)  # Apply conditional rounding here
        logging.info(
            f"Simulator {self.id} performed random walk. "
            f"Old state vector: {old_state_vector}, New state vector: {self.state_vector}, "
            f"Velocity: {self.velocity}, Acceleration: {self.acceleration}"
        )

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
    def __init__(self, renegotiator, arbitor):
        super().__init__(aggr="mean")
        self.renegotiator = renegotiator
        self.arbitor = arbitor

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
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
        combined_debt_threshold = max(src_config.debt_threshold, dst_config.debt_threshold)

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



if __name__ == "__main__":
    import random

    # Function to generate diverse configurations
    def generate_random_config(state_vector_dim, max_force=10000, max_debt=100):
        bounds = torch.tensor([random.choice([0, 1]) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        force_limits = torch.tensor([random.uniform(1, max_force) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        exchange_rates = torch.tensor([random.uniform(0.1, 2.0) for _ in range(state_vector_dim)], dtype=TENSOR_TYPE)
        debt_threshold = random.uniform(10, max_debt)
        return NodeConfiguration(bounds, force_limits, exchange_rates, debt_threshold)

    # Create diverse configurations
    configs = [generate_random_config(STATE_VECTOR_DIM) for _ in range(NUM_NODES)]

    
    # Initialize simulators with the provided configurations
    simulators = [RandomWalkSimulator(STATE_VECTOR_DIM, config) for config in configs]
    
    # Create the graph data
    graph_data = create_graph_data(simulators, configs)
    
    # Initialize the graph network components
    renegotiator = ContractRenegotiator()
    arbitor = DebtArbitor()
    graph_network = PillbugNetwork(renegotiator, arbitor)

    # Simulation main loop
    data_summary = []
    for iteration in range(NUM_ITERATIONS):
        logging.info(f"Iteration {iteration + 1} starting.")
        
        # Perform random walk and update graph data
        for sim in simulators:
            sim.random_walk()
        graph_data.x = torch.stack([sim.state_vector for sim in simulators])
        
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
                "bounds_x": bounds[0].item(),
                "bounds_y": bounds[1].item(),
                "bounds_z": bounds[2].item(),
            })
        logging.info(f"Iteration {iteration + 1} completed.")
    
    # Convert results to a DataFrame for visualization
    df = pd.DataFrame(data_summary)
    logging.info("Data Summary:\n" + str(df.head()))
    
    # Plot results
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
