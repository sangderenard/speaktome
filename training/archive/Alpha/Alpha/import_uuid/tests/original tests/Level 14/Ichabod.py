# Ichabod.py
import pymunk
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import uuid
import random
import os
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DT_INITIAL = 0.0001
STATE_VECTOR_DIM = 7

def conditional_round(tensor, exchange_type):
    """
    Rounds the tensor if exchange_type is 'integer', otherwise returns the tensor unchanged.

    Args:
    - tensor (torch.Tensor): Input tensor to conditionally round.
    - exchange_type (str): 'integer' or 'float'.

    Returns:
    - torch.Tensor: Rounded tensor if in integer mode, unchanged otherwise.
    """
    if exchange_type == "integer":
        return torch.round(tensor)
    return tensor

# ========================================
# Modified RandomWalkSimulator Class
# ========================================

class RandomWalkSimulator:
    def __init__(self, state_vector_dim, config):
        # Existing initialization (from Phoenix.py)
        self.id = uuid.uuid4()
        self.temperature = random.uniform(50.0, 100.0)  # Initial temperature
        self.emissivity = random.uniform(0.1, 1.0)      # Emissivity coefficient
        self.absorbed_radiation = 0.0                   # Radiation absorbed this iteration
        self.state_vector = torch.randn(state_vector_dim)
        self.velocity = torch.zeros(state_vector_dim)
        self.acceleration = torch.zeros(state_vector_dim)
        self.config = config
        logging.info(f"Simulator {self.id} initialized with state vector {self.state_vector} and config {self.config}")

        # Additive changes: Embed ChipmunkSlice and NNWrapper
        self.chipmunk_slice = ChipmunkSlice()
        self.nn_wrapper = ChipmunkSliceNNWrapper()

    # Existing random walk methods are not included as per instruction

    # New methods for ChipmunkSlice and NNWrapper interaction

    def generate_chipmunk_projection(self, dt):
        """
        Generates a future projection using the ChipmunkSlice.

        Args:
            dt (float): Time step for the simulation.

        Returns:
            dict: Projected state from ChipmunkSlice.
        """
        # Prepare state_dict for ChipmunkSlice
        state_dict = {
            'position': self.state_vector[:2].tolist(),
            'velocity': self.velocity[:2].tolist(),
            'mass': self.config.mass if hasattr(self.config, 'mass') else 1.0,
            # Include other necessary parameters as needed
        }
        # Generate projection
        projection = self.chipmunk_slice.simulate(state_dict, dt)
        return projection

    def set_chipmunk_dt(self, dt):
        """
        Sets the dt (time step) of the ChipmunkSlice.

        Args:
            dt (float): Time step value.
        """
        self.chipmunk_slice.dt = dt

# ========================================
# ChipmunkSlice Class (from Icarus.py)
# ========================================

class ChipmunkSlice:
    def __init__(self):
        # Initialize as per Icarus.py
        self.space = pymunk.Space()
        self.space.gravity = (0, -981)  # Gravity in negative Y direction

        # Dynamic body
        self.body = pymunk.Body(mass=1, moment=10)
        self.body.position = (0, 10)  # Starting position
        self.shape = pymunk.Circle(self.body, radius=1)
        self.space.add(self.body, self.shape)

        # Walls (static bodies)
        self.walls = []
        self.create_walls()

        # Simulation control
        self.dt = 0.001  # Default time step duration

    def create_walls(self):
        static_body = self.space.static_body
        walls = [
            pymunk.Segment(static_body, (-10, 0), (-10, 20), 1),  # Left wall
            pymunk.Segment(static_body, (10, 0), (10, 20), 1),    # Right wall
            pymunk.Segment(static_body, (-10, 0), (10, 0), 1),    # Ground
            pymunk.Segment(static_body, (-10, 20), (10, 20), 1),  # Ceiling
        ]
        for wall in walls:
            wall.friction = 0.9
            wall.elasticity = 0.9
            self.space.add(wall)
            self.walls.append(wall)

    def simulate(self, state_dict, dt):
        """
        Simulate the ChipmunkSlice for a given state and time step.

        Args:
            state_dict (dict): Dictionary containing state variables.
            dt (float): Time step.

        Returns:
            dict: Projected state after dt time.
        """
        # Apply initial state
        self.body.position = state_dict['position']
        self.body.velocity = state_dict['velocity']
        self.body.mass = state_dict['mass']

        # Simulate for dt
        self.space.step(dt)

        # Get projected state
        projected_state = {
            'position': [self.body.position.x, self.body.position.y],
            'velocity': [self.body.velocity.x, self.body.velocity.y],
        }
        return projected_state

# ========================================
# NNWrapper Class
# ========================================

class ChipmunkSliceNNWrapper(nn.Module):
    def __init__(self):
        super(ChipmunkSliceNNWrapper, self).__init__()
        # Neural network architecture
        input_size = 7  # As per Icarus.py
        output_size = 4
        self.model = SimpleNN(input_size, output_size)
        # Load pre-trained model if available
        model_path = 'model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logging.info(f"Model loaded from {model_path}")
        else:
            logging.info("No existing model found for NNWrapper. Starting fresh.")

    def forward(self, x):
        return self.model(x)

# SimpleNN class from Icarus.py

class SimpleNN(nn.Module):
    def __init__(self, input_size=7, output_size=4):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# ========================================
# TrainingCoordinator Class
# ========================================

class TrainingCoordinator:
    def __init__(self, simulators):
        self.simulators = simulators
        self.state_logs = {sim.id: [] for sim in simulators}
        self.projection_logs = {sim.id: [] for sim in simulators}
        self.losses = []
        # Optimizer and loss function for each NNWrapper
        self.optimizers = {sim.id: optim.Adam(sim.nn_wrapper.parameters()) for sim in simulators}
        self.criterion = nn.MSELoss()

    def collect_states(self):
        """
        Collects the current state of all simulators.
        """
        for sim in self.simulators:
            state = sim.state_vector.clone()
            self.state_logs[sim.id].append(state)

    def generate_projections(self, dt):
        """
        Generates projections for all simulators using ChipmunkSlice.

        Args:
            dt (float): Time step for the projection.
        """
        for sim in self.simulators:
            projection = sim.generate_chipmunk_projection(dt)
            self.projection_logs[sim.id].append(projection)

    def compute_losses(self):
        """
        Computes the loss between the projections and actual states for each simulator.
        """
        for sim in self.simulators:
            # Get the latest actual state and projection
            actual_state = self.state_logs[sim.id][-1]
            projected_state = self.projection_logs[sim.id][-1]
            # Prepare input for NNWrapper
            input_state = actual_state
            # Forward pass through NNWrapper
            nn_output = sim.nn_wrapper(input_state)
            # Convert projected_state to tensor
            projected_tensor = torch.tensor(projected_state['position'] + projected_state['velocity'])
            # Compute loss between NN output and projected state
            loss = self.criterion(nn_output, projected_tensor)
            self.losses.append(loss.item())
            # Backpropagate and update NNWrapper
            optimizer = self.optimizers[sim.id]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def adjust_chipmunk_dt(self, dt):
        """
        Adjusts the dt of the ChipmunkSlice for all simulators.

        Args:
            dt (float): New time step value.
        """
        for sim in self.simulators:
            sim.set_chipmunk_dt(dt)

# ========================================
# HeadTrainer Class
# ========================================

class HeadTrainer(MessagePassing):
    def __init__(self, training_coordinators):
        super().__init__(aggr="mean")
        self.training_coordinators = training_coordinators
        self.edge_index = self.create_edges()
        self.edge_attr = self.create_edge_attributes()

    def create_edges(self):
        """
        Creates edges between the HeadTrainer and each TrainingCoordinator.

        Returns:
            torch.Tensor: Edge indices.
        """
        num_coordinators = len(self.training_coordinators)
        edge_indices = []
        # Edge from HeadTrainer (node 0) to each TrainingCoordinator (nodes 1..N)
        for i in range(1, num_coordinators + 1):
            edge_indices.append([0, i])  # Edge from HeadTrainer to Coordinator
            edge_indices.append([i, 0])  # Edge from Coordinator to HeadTrainer
        return torch.tensor(edge_indices, dtype=torch.long).t()

    def create_edge_attributes(self):
        """
        Creates edge attributes for the edges.

        Returns:
            dict: Edge attributes.
        """
        num_edges = self.edge_index.size(1)
        edge_attr = {
            'edge_type': torch.zeros(num_edges)  # Placeholder attribute
        }
        return edge_attr

    def forward(self, x):
        """
        Forward pass of the HeadTrainer.

        Args:
            x (torch.Tensor): Node features.

        Returns:
            torch.Tensor: Updated node features.
        """
        return self.propagate(self.edge_index, x=x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Message passing function.

        Args:
            x_i (torch.Tensor): Features of source node.
            x_j (torch.Tensor): Features of target node.
            edge_attr (dict): Edge attributes.

        Returns:
            torch.Tensor: Message to pass.
        """
        # For simplicity, pass x_j as the message
        return x_j

    def update(self, aggr_out):
        """
        Update function.

        Args:
            aggr_out (torch.Tensor): Aggregated messages.

        Returns:
            torch.Tensor: Updated node features.
        """
        # Return the aggregated messages as updated features
        return aggr_out

# ========================================
# NodeConfiguration Class (from Phoenix.py)
# ========================================

class NodeConfiguration:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold, exchange_type="float"):
        """
        Configuration for a node's simulation capacity.

        Args:
        - bounds (torch.Tensor): Axes the node is bound by (binary mask).
        - force_limits (torch.Tensor): Maximum allowable force for each axis.
        - exchange_rates (torch.Tensor): Energy cost per unit correction for each axis.
        - debt_threshold (float): Maximum allowable energy debt.
        - exchange_type (str): 'integer' or 'float' specifying the exchange type.
        """
        self.bounds = bounds
        self.force_limits = force_limits
        self.exchange_rates = exchange_rates
        self.debt_threshold = torch.tensor(debt_threshold)
        self.exchange_type = exchange_type

    def __repr__(self):
        return (f"NodeConfiguration(bounds={self.bounds}, force_limits={self.force_limits}, "
                f"exchange_rates={self.exchange_rates}, debt_threshold={self.debt_threshold}, "
                f"exchange_type={self.exchange_type})")

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    # Initialize configurations for simulators
    def generate_random_config(state_vector_dim):
        bounds = torch.ones(state_vector_dim)
        force_limits = torch.ones(state_vector_dim) * 100000.0
        exchange_rates = torch.ones(state_vector_dim) * 1.0
        debt_threshold = 100.0
        exchange_type = "float"
        # Add mass attribute
        mass = random.uniform(1.0, 10.0)
        config = NodeConfiguration(bounds, force_limits, exchange_rates, debt_threshold, exchange_type)
        config.mass = mass
        return config

    # Number of simulators
    NUM_SIMULATORS = 5  # Example number

    # Create simulators with configurations
    configs = [generate_random_config(STATE_VECTOR_DIM) for _ in range(NUM_SIMULATORS)]
    simulators = [RandomWalkSimulator(STATE_VECTOR_DIM, config) for config in configs]

    # Initialize TrainingCoordinator with simulators
    coordinator = TrainingCoordinator(simulators)

    # Adjust dt of ChipmunkSlice via TrainingCoordinator
    coordinator.adjust_chipmunk_dt(DT_INITIAL)

    # Collect states and generate projections
    coordinator.collect_states()
    coordinator.generate_projections(DT_INITIAL)

    # Compute losses and train NNWrappers
    coordinator.compute_losses()

    # Initialize HeadTrainer with the coordinator
    head_trainer = HeadTrainer([coordinator])

    # Prepare node features (dummy features for example)
    x = torch.randn(len(head_trainer.training_coordinators) + 1, STATE_VECTOR_DIM)  # +1 for HeadTrainer node

    # Perform a forward pass
    updated_x = head_trainer(x)

    # The rest of the execution logic would follow as per simulation requirements
