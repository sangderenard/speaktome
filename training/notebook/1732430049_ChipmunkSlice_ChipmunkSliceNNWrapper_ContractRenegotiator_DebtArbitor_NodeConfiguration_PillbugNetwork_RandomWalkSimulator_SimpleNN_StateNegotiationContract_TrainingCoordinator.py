# Kakarot.py

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
import pymunk
import threading
import time
import queue
import torch.nn as nn
import torch.optim as optim

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

def draw_graph(screen, positions, node_states, error_magnitudes_with_bounds, error_magnitudes_without_bounds, simulators, include_bounds=True, cmap='jet', network=None):
    """
    Draws a grid of node status summaries on the Pygame screen.

    Args:
        screen (pygame.Surface): The Pygame screen surface.
        positions (dict): Dictionary mapping node indices to (x, y) positions.
        node_states (torch.Tensor): Tensor containing node state vectors.
        error_magnitudes_with_bounds (torch.Tensor): Error magnitudes including bound axes.
        error_magnitudes_without_bounds (torch.Tensor): Error magnitudes excluding bound axes.
        simulators (list): List of RandomWalkSimulator instances.
        include_bounds (bool): Whether to include bound axes in error magnitude calculation.
        cmap (str): Matplotlib colormap name for node coloring.
        network (PillbugNetwork): The top-level network instance.
    """
    # Compute background color based on total system temperature, total radiation, and connectivity
    if network:
        # Normalize values to [0, 1]
        temp_norm = min(network.total_temperature() / network.initial_total_temperature, 1.0)
        rad_norm = min(network.total_radiation() / network.initial_total_radiation, 1.0) if network.initial_total_radiation > 0 else 0
        conn_norm = min(network.connectivity / network.max_connectivity, 1.0)

        # Convert to RGB scale [0, 255]
        background_color = (
            int(temp_norm * 255),
            int(rad_norm * 255),
            int(conn_norm * 255)
        )
    else:
        background_color = WHITE

    screen.fill(background_color)  # Fill screen with computed background color

    # Normalize error magnitudes for color mapping
    norm_with_bounds = mcolors.Normalize(vmin=error_magnitudes_with_bounds.min().item(),
                                         vmax=error_magnitudes_with_bounds.max().item())
    norm_without_bounds = mcolors.Normalize(vmin=error_magnitudes_without_bounds.min().item(),
                                            vmax=error_magnitudes_without_bounds.max().item())
    cmap = plt.get_cmap(cmap)

    for node_idx, pos in positions.items():
        # Determine colors based on error magnitudes
        error_with_bounds = error_magnitudes_with_bounds[node_idx].item()
        error_without_bounds = error_magnitudes_without_bounds[node_idx].item()

        color_with_bounds = cmap(norm_with_bounds(error_with_bounds))  # Returns RGBA
        color_without_bounds = cmap(norm_without_bounds(error_without_bounds))

        color_with_bounds = tuple(int(255 * c) for c in color_with_bounds[:3])  # Convert to RGB
        color_without_bounds = tuple(int(255 * c) for c in color_without_bounds[:3])

        # Draw split rectangle (box) for the node
        rect_left = pygame.Rect(pos[0], pos[1], NODE_BOX_WIDTH // 2, NODE_BOX_HEIGHT)
        rect_right = pygame.Rect(pos[0] + NODE_BOX_WIDTH // 2, pos[1], NODE_BOX_WIDTH // 2, NODE_BOX_HEIGHT)

        pygame.draw.rect(screen, color_with_bounds, rect_left)
        pygame.draw.rect(screen, color_without_bounds, rect_right)

        # Draw borders
        pygame.draw.rect(screen, BLACK, rect_left, 1)
        pygame.draw.rect(screen, BLACK, rect_right, 1)

        # Get number of bound and unbound axes
        sim = simulators[node_idx]
        bounds = sim.config.bounds
        num_bound_axes = int(torch.sum(bounds).item())
        num_unbound_axes = bounds.numel() - num_bound_axes

        # Render node ID, temperature, and bound/unbound axes
        text_id = FONT.render(f"Node {node_idx}", True, TEXT_COLOR)
        text_temp = FONT.render(f"Temp: {sim.temperature:.2f}", True, TEXT_COLOR)
        text_bounds = FONT.render(f"Bound: {num_bound_axes}", True, TEXT_COLOR)
        text_unbound = FONT.render(f"Unbound: {num_unbound_axes}", True, TEXT_COLOR)

        # Position text inside the box
        text_x = pos[0] + 5  # Small padding
        text_y = pos[1] + 5

        screen.blit(text_id, (text_x, text_y))
        screen.blit(text_temp, (text_x, text_y + 15))
        screen.blit(text_bounds, (text_x, text_y + 30))
        screen.blit(text_unbound, (text_x, text_y + 45))

        # Draw tiny circles representing acceleration of each feature
        acc = sim.acceleration
        num_features = acc.numel()
        circle_radius = 5
        circle_spacing = (NODE_BOX_WIDTH - 10) // num_features  # Spacing between circles

        for i in range(num_features):
            acc_value = acc[i].item()
            # Map acceleration value to color intensity
            acc_norm = min(abs(acc_value) / 10.0, 1.0)
            acc_color = (int(255 * acc_norm), 0, int(255 * (1 - acc_norm)))  # Red to blue gradient
            circle_x = pos[0] + 5 + i * circle_spacing
            circle_y = pos[1] + NODE_BOX_HEIGHT - 10
            pygame.draw.circle(screen, acc_color, (circle_x, circle_y), circle_radius)

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
DT_INITIAL = 0.0001
NUM_NODES = 17
STATE_VECTOR_DIM = 7
STATE_DIM_FEATURES = 2
RANDOM_WALK_STEP = 1.1
NUM_ITERATIONS = 5000
ENERGY_BUDGET = 10000000.1

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

class StateNegotiationContract:
    def __init__(self, bounds, force_limits, exchange_rates, debt_threshold, exchange_type="float"):
        self.bounds = bounds
        self.force_limits = force_limits
        self.exchange_rates = exchange_rates
        self.debt_threshold = torch.tensor(debt_threshold)
        self.current_debt = torch.zeros_like(self.debt_threshold)
        self.energy_history = []
        self.error_history = []
        self.exchange_type = exchange_type

    def apply_correction(self, correction, error):
        scaled_correction = torch.min(torch.abs(correction), self.force_limits) * torch.sign(correction)
        bounded_correction = scaled_correction * self.bounds
        bounded_correction = conditional_round(bounded_correction, self.exchange_type)
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

    def regenerate_energy(self, radiation_absorbed, dt):
        regen_amount = torch.tensor(radiation_absorbed * dt)
        old_debt = self.current_debt.clone()
        self.current_debt = torch.max(torch.tensor(0.0), self.current_debt - regen_amount)
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

def interpret_exchange_types(exchange_types_tensor):
    exchange_type_strings = []
    # Map tensor values to human-readable strings
    for value in exchange_types_tensor:
        if value != 2.6:  # Integer + Integer
            exchange_type_strings.append("integer")
        else:
            exchange_type_strings.append("float")

    return exchange_type_strings

class DebtArbitor:
    def __init__(self):
        self.correction_logs = []
        self.fault_history = []

    def arbitrate(self, error, bounds, force_limits, exchange_rates, current_debt, debt_threshold, exchange_types):
        exchange_types = interpret_exchange_types(exchange_types)
        correction_suggestion = error * bounds
        scaled_correction = torch.min(torch.abs(correction_suggestion), force_limits) * torch.sign(correction_suggestion)

        # Determine exchange types per edge
        final_corrections = []
        updated_debts = []
        for i in range(scaled_correction.size(0)):
            exchange_type = "float" if exchange_types[i][0] == exchange_types[i][1] == "float" else "integer"
            correction_acceptance = conditional_round(scaled_correction[i], exchange_type)
            energy_cost = torch.sum(torch.abs(correction_acceptance) * exchange_rates[i])

            if current_debt[i] + energy_cost > debt_threshold[i]:
                correction_acceptance = torch.zeros_like(correction_acceptance)
            else:
                current_debt[i] += energy_cost

            final_corrections.append(correction_acceptance)
            updated_debts.append(current_debt[i])

        final_corrections = torch.stack(final_corrections)
        updated_debt = torch.tensor(updated_debts)

        self.correction_logs.append({
            "error": error.clone(),
            "suggestion": correction_suggestion.clone(),
            "acceptance": final_corrections.clone(),
            "energy_cost": energy_cost.clone(),
            "updated_debt": updated_debt.clone(),
        })

        logging.info(f"Debt arbitration performed. Correction acceptance: {final_corrections}, Updated debt: {updated_debt}")

        return final_corrections, updated_debt

# ========================================
# Modified RandomWalkSimulator Class
# ========================================

class RandomWalkSimulator:
    def __init__(self, state_vector_dim, config):
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
        # Initialize the mapping of features to axes
        self.mapped_axes = random.sample(range(state_vector_dim), 2)  # Pick any two features

    def random_walk(self, dt):
        old_state_vector = self.state_vector.clone()
        self.acceleration = torch.randn_like(self.state_vector) * dt
        self.velocity += self.acceleration
        self.state_change = self.velocity * dt
        self.state_vector += self.state_change
        self.state_vector = conditional_round(self.state_vector, self.config.exchange_type)
        logging.info(
            f"Simulator {self.id} performed random walk. "
            f"Old state vector: {old_state_vector}, New state vector: {self.state_vector}, "
            f"Velocity: {self.velocity}, Acceleration: {self.acceleration}"
        )
        return torch.norm(self.state_change)

    def apply_second_derivative(self, correction):
        old_acceleration = self.acceleration.clone()
        self.acceleration += correction
        self.acceleration = conditional_round(self.acceleration, self.config.exchange_type)
        logging.info(
            f"Simulator {self.id} applied second derivative correction. "
            f"Correction: {correction}, Old acceleration: {old_acceleration}, "
            f"New acceleration: {self.acceleration}"
        )

    def emit_radiation(self, dt):
        # Calculate emitted radiation based on temperature and emissivity
        emitted_radiation = self.emissivity * self.temperature * dt
        self.temperature -= emitted_radiation
        logging.info(f"Node {self.id} emitted radiation: {emitted_radiation}, New temperature: {self.temperature}")
        return emitted_radiation

    def absorb_radiation(self, radiation):
        # Absorb radiation and increase temperature
        self.temperature += radiation
        self.absorbed_radiation += radiation
        logging.info(f"Node {self.id} absorbed radiation: {radiation}, New temperature: {self.temperature}")

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
        # Map selected features to axes
        state_dict = {
            'position': self.state_vector[self.mapped_axes].tolist(),
            'velocity': self.velocity[self.mapped_axes].tolist(),
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
# ChipmunkSlice Class using CricketSimulation
# ========================================

class ChipmunkSlice:
    def __init__(self):
        # Use the environment settings from cricket.py, with gravity (0,0)
        self.width = 800
        self.height = 800
        self.gravity = 0  # Gravity magnitude set to 0
        self.color = (0, 0, 1, 0.5)  # Blue with alpha
        self.pymunk_width = 40
        self.pymunk_height = 40

        # Define Pymunk simulation bounds
        self.pymunk_bounds = (
            (-self.pymunk_width / 2 + self.width / 2, self.pymunk_width / 2 + self.width / 2),
            (-self.pymunk_height / 2 + self.height / 2, self.pymunk_height / 2 + self.height / 2),
        )

        # Initialize Pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity

        # Add walls and dynamic body
        self.create_walls()
        self.body = self.create_dynamic_body()

    def create_walls(self):
        """
        Create rigid, smooth walls around the simulation.
        """
        thickness = 10
        (left, right), (bottom, top) = self.pymunk_bounds
        walls = [
            pymunk.Segment(self.space.static_body, (left, bottom), (right, bottom), thickness),
            pymunk.Segment(self.space.static_body, (right, bottom), (right, top), thickness),
            pymunk.Segment(self.space.static_body, (right, top), (left, top), thickness),
            pymunk.Segment(self.space.static_body, (left, top), (left, bottom), thickness),
        ]
        for wall in walls:
            wall.elasticity = 1.0  # Perfectly smooth and rigid
            wall.friction = 0.0
        self.space.add(*walls)

    def create_dynamic_body(self):
        """
        Create a single dynamic body in the simulation.
        :return: Pymunk body object
        """
        mass = 1
        radius = 0.1
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)

        # Center the dynamic body in Pymunk space
        body.position = (
            (self.pymunk_bounds[0][0] + self.pymunk_bounds[0][1]) / 2,
            (self.pymunk_bounds[1][0] + self.pymunk_bounds[1][1]) / 2,
        )
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 1.0
        shape.friction = 0.0
        self.space.add(body, shape)
        return body

    def simulate(self, state_dict, dt):
        """
        Simulate the ChipmunkSlice for a given state and time step.

        Args:
            state_dict (dict): Dictionary containing state variables.
            dt (float): Time step.

        Returns:
            dict: Projected state after dt time.
        """
        # Map the provided position and velocity to the Pymunk body
        pos_x, pos_y = state_dict['position']
        vel_x, vel_y = state_dict['velocity']

        # Convert to Pymunk coordinates
        body_position = pymunk.Vec2d(pos_x, pos_y)
        body_velocity = pymunk.Vec2d(vel_x, vel_y)

        self.body.position = body_position
        self.body.velocity = body_velocity
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
            mapped_axes = sim.mapped_axes
            # Prepare input for NNWrapper
            input_state = actual_state
            # Forward pass through NNWrapper
            nn_output = sim.nn_wrapper(input_state)
            # Convert projected_state to tensor
            projected_state = self.projection_logs[sim.id][-1]
            projected_tensor = torch.tensor(projected_state['position'] + projected_state['velocity'])
            # Compute loss between NN output and projected state
            loss = self.criterion(nn_output, projected_tensor)
            print(loss.item())
            self.losses.append(loss.item())
            # Backpropagate and update NNWrapper
            optimizer = self.optimizers[sim.id]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def adjust_chipmunk_dt(self, dt):
        """
        Adjusts the dt of the ChipmunkSlice for all simulators.
        """
        for sim in self.simulators:
            sim.set_chipmunk_dt(dt)

# ========================================
# PillbugNetwork Class (from Phoenix.py)
# ========================================

class PillbugNetwork(MessagePassing):
    def __init__(self, num_nodes, num_features, num_subfeatures, temperature=0, radiation_coefficient=0.1, gestalt_matrix_function=None):
        super().__init__(aggr="mean")  # Aggregation for message passing
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.temperature = random.uniform(100.0, 200.0)  # Initial temperature
        self.emissivity = random.uniform(0.1, 1.0)       # Emissivity coefficient
        self.absorbed_radiation = 0.0                    # Radiation absorbed this iteration
        self.radiation_coefficient = radiation_coefficient
        self.radiation = 0.0
        self.connectivity = 0
        self.max_connectivity = num_nodes * (num_nodes - 1)
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
                    debt_threshold=100.0,
                    exchange_type="float"
                ))
                for _ in range(num_nodes)
            ]

        # Internal states for the network
        self.state_matrix = torch.zeros((num_nodes, num_features))
        self.velocity_matrix = torch.zeros((num_nodes, num_features))
        self.acceleration_matrix = torch.zeros((num_nodes, num_features))
        self.gestalt_matrix = torch.zeros((num_features,))
        self.gestalt_matrix_function = gestalt_matrix_function or self.default_gestalt_matrix_function
        self.renegotiator = ContractRenegotiator()
        self.arbitor = DebtArbitor()
        self.previous_corrections = torch.zeros((num_nodes, num_features))
        self.id = uuid.uuid4()

        # For energy conservation tracking
        self.initial_total_temperature = self.temperature + sum(f.temperature for f in self.features)
        self.initial_total_radiation = 0.0

    def random_walk(self, dt):
        total_state_change = 0.0
        for i, feature in enumerate(self.features):
            if isinstance(feature, PillbugNetwork):
                state_change = feature.random_walk(dt)
            elif isinstance(feature, RandomWalkSimulator):
                state_change = feature.random_walk(dt)
                self.state_matrix[i] = feature.state_vector
                self.velocity_matrix[i] = feature.velocity
                self.acceleration_matrix[i] = feature.acceleration
            total_state_change += state_change
        logging.info(f"Pillbug Network {self.id} random walk applied. Updated state matrix:\n{self.state_matrix}")
        return total_state_change

    def apply_corrections(self, corrections):
        """
        Applies corrections to the acceleration matrix or delegates to subfeatures.
        Args:
            corrections (torch.Tensor): Matrix of corrections to be applied.
        """
        for i, feature in enumerate(self.features):
            if isinstance(feature, PillbugNetwork):
                feature.apply_corrections(corrections[i])
            elif isinstance(feature, RandomWalkSimulator):
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

        # Collect exchange types from both nodes
        exchange_types = edge_attr["exchange_types"]

        correction_acceptance, updated_debt = self.arbitor.arbitrate(
            error,
            renegotiated_bounds,
            edge_attr["force_limits"],
            edge_attr["exchange_rates"],
            edge_attr["current_debt"],
            edge_attr["debt_threshold"],
            exchange_types
        )
        edge_attr["current_debt"] = updated_debt

        logging.info(
            f"Correction acceptance: {correction_acceptance}, Updated debt: {updated_debt}"
        )

        return correction_acceptance

    def emit_radiation(self, dt):
        # Calculate emitted radiation based on temperature and emissivity
        emitted_radiation = self.emissivity * self.temperature * dt
        self.temperature -= emitted_radiation
        self.radiation += emitted_radiation
        logging.info(f"Network {self.id} emitted radiation: {emitted_radiation}, New temperature: {self.temperature}")
        return emitted_radiation

    def absorb_radiation(self, radiation):
        # Absorb radiation and increase temperature
        self.temperature += radiation
        self.absorbed_radiation += radiation
        logging.info(f"Network {self.id} absorbed radiation: {radiation}, New temperature: {self.temperature}")

    def exchange_radiation(self, dt):
        # Nodes exchange radiation based on inverse distance
        total_emissions = []
        for node in self.features:
            emitted = node.emit_radiation(dt)
            total_emissions.append(emitted)

        # Compute distances between nodes (using positions)
        num_nodes = len(self.features)
        positions = np.array([np.array(pos) for pos in generate_grid_positions(num_nodes, NODE_BOX_WIDTH, NODE_BOX_HEIGHT).values()])

        # Calculate absorption
        for i, node_i in enumerate(self.features):
            absorbed = 0.0
            for j, node_j in enumerate(self.features):
                if i != j:
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance > 0:
                        absorption_fraction = 1.0 / distance
                        absorbed_radiation = total_emissions[j] * absorption_fraction
                        node_i.absorb_radiation(absorbed_radiation)
                        absorbed += absorbed_radiation
            # Subtract absorbed radiation from radiation pool
            self.radiation -= absorbed
            logging.info(f"Node {node_i.id} total absorbed radiation: {absorbed}")

    def process_iteration(self, edge_index, edge_attr, error, corrections, dt):
        self.exchange_radiation(dt)
        self.previous_corrections = self.forward(self.state_matrix, edge_index, edge_attr)
        self.apply_corrections(corrections)
        self.update_gestalt_matrix()

    def total_temperature(self):
        return self.temperature + sum(node.temperature for node in self.features)

    def total_radiation(self):
        return self.radiation + sum(node.radiation for node in self.features if hasattr(node, 'radiation'))

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

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    # Function to generate diverse configurations
    def generate_random_config(state_vector_dim, max_force=100000, max_debt=1000):
        bounds = torch.tensor([random.choice([0, 1]) for _ in range(state_vector_dim)])
        force_limits = torch.tensor([random.uniform(max_force * 0.5, max_force) for _ in range(state_vector_dim)])
        exchange_rates = torch.tensor([random.uniform(0.8, 1.2) for _ in range(state_vector_dim)])
        debt_threshold = random.uniform(100, max_debt)
        exchange_type = random.choice(["integer", "float"])  # Randomly assign exchange type
        config = NodeConfiguration(bounds, force_limits, exchange_rates, debt_threshold, exchange_type)
        config.mass = random.uniform(1.0, 10.0)  # Add mass attribute
        return config

    # Create diverse configurations
    configs = [generate_random_config(STATE_VECTOR_DIM) for _ in range(NUM_NODES)]

    # Initialize simulators with the provided configurations
    simulators = [RandomWalkSimulator(STATE_VECTOR_DIM, config) for config in configs]

    # Initialize TrainingCoordinator with simulators
    coordinator = TrainingCoordinator(simulators)

    # Adjust dt of ChipmunkSlice via TrainingCoordinator
    coordinator.adjust_chipmunk_dt(DT_INITIAL)

    # Create the graph data and compute connectivity
    graph_data, connectivity = create_graph_data(simulators, configs)

    # Initialize the graph network
    graph_network = PillbugNetwork(NUM_NODES, STATE_VECTOR_DIM, STATE_DIM_FEATURES)
    graph_network.connectivity = connectivity  # Set initial connectivity

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

        # Update connectivity
        graph_network.connectivity = graph_data.edge_index.size(1)

        # Compute error magnitudes for color mapping
        error_magnitudes_with_bounds = compute_error_magnitudes(graph_data.x, simulators, include_bounds=True)
        error_magnitudes_without_bounds = compute_error_magnitudes(graph_data.x, simulators, include_bounds=False)

        # Update visualization
        draw_graph(screen, positions, graph_data.x, error_magnitudes_with_bounds, error_magnitudes_without_bounds, simulators, include_bounds=True, network=graph_network)

        # Control the frame rate
        clock.tick(FPS)

        # TrainingCoordinator collects states and generates projections
        coordinator.collect_states()
        coordinator.generate_projections(dt)
        # Compute losses and train NNWrappers
        coordinator.compute_losses()

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
