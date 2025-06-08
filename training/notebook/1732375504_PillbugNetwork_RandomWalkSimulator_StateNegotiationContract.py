import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_NODES = 5  # Number of nodes (simulators)
STATE_VECTOR_DIM = 3  # Dimensionality of the state vector (e.g., x, y, z)
RANDOM_WALK_STEP = 0.1  # Step size for the random walk
NUM_ITERATIONS = 50  # Number of graph updates
DEBT_THRESHOLD = 5.0  # Maximum allowable energy debt
ENERGY_BUDGET = 100.0  # Total energy budget
REGEN_RATE = 1.0  # Energy regeneration per step
DT = 1.0  # Time step for energy regeneration

# Random Seed for Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# State Negotiation Contract Class
class StateNegotiationContract:
    def __init__(self, energy_budget, regen_rate, debt_threshold, bounds,
                 force_limits, exchange_rates):
        """
        Initialize a state negotiation contract.
        :param energy_budget: Total energy available for adjustments.
        :param regen_rate: Energy regeneration per step.
        :param debt_threshold: Maximum energy debt before rejection.
        :param bounds: Tensor defining bound features (1 for bound, 0 for free).
        :param force_limits: Tensor defining force limits per feature.
        :param exchange_rates: Tensor expressing energy cost per unit correction per feature.
        """
        self.energy_budget = energy_budget
        self.regen_rate = regen_rate
        self.debt_threshold = debt_threshold
        self.bounds = bounds  # Mask for bound features
        self.force_limits = force_limits
        self.exchange_rates = exchange_rates
        self.current_debt = 0  # Track accumulated debt
        self.energy_history = []  # For plotting

    def apply_correction(self, correction):
        """
        Apply a correction to the state and track energy usage.
        :param correction: Correction vector (second derivative).
        :return: Accepted correction vector.
        """
        # Scale correction to not exceed force limits
        scaled_correction = torch.min(torch.abs(correction), self.force_limits) * torch.sign(correction)
        # Apply bounds mask
        bounded_correction = scaled_correction * self.bounds
        # Compute energy cost
        energy_cost = torch.sum(torch.abs(bounded_correction) * self.exchange_rates)
        # Check energy debt
        if self.current_debt + energy_cost > self.debt_threshold:
            # Reject correction
            bounded_correction = torch.zeros_like(correction)
            energy_cost = 0.0
        else:
            # Accept correction and update debt
            self.current_debt += energy_cost

        self.energy_history.append(self.current_debt)
        return bounded_correction

    def regenerate_energy(self, dt):
        """
        Regenerate energy over time.
        :param dt: Time step.
        """
        self.current_debt = max(0, self.current_debt - self.regen_rate * dt)

# Random Walk Simulator Class
class RandomWalkSimulator:
    def __init__(self, state_vector_dim, initial_state=None):
        """
        Initialize a simulator with a random or provided state vector.
        :param state_vector_dim: Dimensionality of the state vector.
        :param initial_state: Optional initial state vector.
        """
        if initial_state is not None:
            self.state_vector = initial_state.clone()
        else:
            self.state_vector = torch.randn(state_vector_dim)
        self.velocity = torch.zeros(state_vector_dim)
        self.acceleration = torch.zeros(state_vector_dim)
        self.state_history = []  # For plotting

    def random_walk(self):
        """
        Apply a random walk to the state vector.
        """
        self.acceleration = torch.randn_like(self.state_vector) * RANDOM_WALK_STEP
        self.velocity += self.acceleration
        self.state_vector += self.velocity

    def apply_second_derivative(self, correction):
        """
        Apply a second derivative correction to the state.
        :param correction: Correction vector (second derivative).
        """
        self.acceleration += correction

    def record_state(self):
        """
        Record the current state for analysis.
        """
        self.state_history.append(self.state_vector.clone())

# Pillbug Graph Network Class
class PillbugNetwork(MessagePassing):
    def __init__(self):
        super().__init__(aggr="mean")  # Aggregate messages using mean

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the graph network.
        :param x: Node features (state vectors).
        :param edge_index: Graph connectivity (COO format).
        :param edge_attr: Edge features (negotiation contracts and parameters).
        :return: Corrections for each node.
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages between nodes.
        :param x_i: Features of the receiving node.
        :param x_j: Features of the sending node.
        :param edge_attr: Edge features (contracts and parameters).
        """
        # Compute error (difference in bound features)
        error = x_j - x_i
        error = error * edge_attr["bounds"]  # Apply bounds mask

        # Correction suggestion (proportional to error)
        correction_suggestion = error  # For simplicity, could include gains

        # Apply force limits and energy constraints
        correction_acceptance = torch.min(torch.abs(correction_suggestion), edge_attr["force_limits"]) \
                                * torch.sign(correction_suggestion)

        # Compute energy cost
        energy_cost = torch.sum(torch.abs(correction_acceptance) * edge_attr["exchange_rates"], dim=1)

        # Reject corrections exceeding debt threshold
        debt_exceeded = energy_cost + edge_attr["current_debt"] > edge_attr["debt_threshold"]
        correction_acceptance[debt_exceeded] = 0.0

        # Update current debt
        edge_attr["current_debt"][~debt_exceeded] += energy_cost[~debt_exceeded]

        # Log energy history
        edge_attr["energy_history"].append(edge_attr["current_debt"].clone())

        return correction_acceptance

    def update(self, aggr_out, x):
        """
        Update node accelerations based on aggregated corrections.
        :param aggr_out: Aggregated corrections.
        :param x: Original node features.
        :return: Corrections to apply.
        """
        return aggr_out  # Return corrections to be applied as second derivatives

# Setup the Graph Data
def create_graph_data(simulators, contract):
    """
    Create graph data for the simulators and their connections.
    :param simulators: List of RandomWalkSimulator instances.
    :param contract: StateNegotiationContract instance.
    :return: PyTorch Geometric Data object.
    """
    num_nodes = len(simulators)
    # Fully connected graph (excluding self-connections)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                              dtype=torch.long).t()

    # Edge attributes
    num_edges = edge_index.size(1)
    edge_attr = {
        "bounds": contract.bounds.repeat(num_edges, 1),
        "force_limits": contract.force_limits.repeat(num_edges, 1),
        "exchange_rates": contract.exchange_rates.repeat(num_edges, 1),
        "debt_threshold": torch.tensor([contract.debt_threshold] * num_edges),
        "current_debt": torch.tensor([contract.current_debt] * num_edges),
        "energy_history": []
    }

    x = torch.stack([sim.state_vector for sim in simulators])  # Node features
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Instantiate Simulators and Graph Network
# Initial state for forking
initial_state = torch.randn(STATE_VECTOR_DIM)

# Create simulators with forking
simulators = [RandomWalkSimulator(STATE_VECTOR_DIM, initial_state=initial_state) for _ in range(NUM_NODES)]

# Define Negotiation Parameters
bounds = torch.tensor([1, 1, 0], dtype=torch.float32)  # Bind x and y, free z
force_limits = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)  # Force limits per feature
exchange_rates = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # Energy cost per unit correction

# Create contract
contract = StateNegotiationContract(
    energy_budget=ENERGY_BUDGET,
    regen_rate=REGEN_RATE,
    debt_threshold=DEBT_THRESHOLD,
    bounds=bounds,
    force_limits=force_limits,
    exchange_rates=exchange_rates
)

# Create graph data
graph_data = create_graph_data(simulators, contract)

# Initialize graph network
graph_network = PillbugNetwork()

# Statistics for plotting
energy_histories = []
debt_histories = []
state_histories = [[] for _ in range(NUM_NODES)]

# Main Loop: Random Walk with Negotiation
for iteration in range(NUM_ITERATIONS):
    # Step random walks and record states
    for sim in simulators:
        sim.random_walk()
        sim.record_state()

    # Update graph node features with simulator state vectors
    graph_data.x = torch.stack([sim.state_vector for sim in simulators])

    # Regenerate energy
    contract.regenerate_energy(DT)

    # Update edge attributes with current debt
    num_edges = graph_data.edge_index.size(1)
    graph_data.edge_attr["current_debt"] = torch.tensor([contract.current_debt] * num_edges)
    graph_data.edge_attr["energy_history"].append(contract.current_debt)

    # Forward pass through the graph network to get corrections
    corrections = graph_network(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

    # Apply corrections to simulators
    for i, sim in enumerate(simulators):
        sim.apply_second_derivative(corrections[i])
        sim.record_state()

    # Collect energy and debt histories
    energy_histories.append(contract.current_debt)
    debt_histories.append(contract.current_debt)

# Plotting
iterations = np.arange(NUM_ITERATIONS)
plt.figure(figsize=(12, 8))

# Plot energy debt over iterations
plt.subplot(2, 1, 1)
plt.plot(iterations, energy_histories, label='Energy Debt')
plt.xlabel('Iteration')
plt.ylabel('Energy Debt')
plt.title('Energy Debt Over Time')
plt.legend()

# Plot state vectors over iterations for each simulator
for idx, sim in enumerate(simulators):
    states = torch.stack(sim.state_history).numpy()
    plt.subplot(2, NUM_NODES, NUM_NODES + idx + 1)
    plt.plot(iterations, states[:, 0], label='x')
    plt.plot(iterations, states[:, 1], label='y')
    plt.plot(iterations, states[:, 2], label='z')
    plt.xlabel('Iteration')
    plt.ylabel('State Value')
    plt.title(f'Simulator {idx + 1}')
    plt.legend()

plt.tight_layout()
plt.show()
