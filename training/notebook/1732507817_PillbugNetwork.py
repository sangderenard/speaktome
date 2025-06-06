# pillbug_network.py
import numpy as np
import torch
import random
import logging
import uuid
from torch_geometric.nn import MessagePassing
from .contract_renegotiator import ContractRenegotiator
from .debt_arbitor import DebtArbitor
from .helper_functions import conditional_round
from .random_walk_simulator import RandomWalkSimulator
from .node_configuration import NodeConfiguration
from .head_trainer import HeadTrainer

# ========================================
# PillbugNetwork Class
# ========================================

class PillbugNetwork(MessagePassing):
    def __init__(self, num_nodes, num_features, num_subfeatures, temperature=0, edge_index = None, edge_attr = None, radiation_coefficient=0.1, gestalt_matrix_function=None, head_trainer=None):
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


        self.edge_index = edge_index if edge_index is not None else self.initialize_edge_index()

        self.edge_attr = edge_attr if edge_attr is not None else self.initialize_edge_attr()

        logging.info(f"Pillbug Network initialized with {num_nodes} nodes and {self.edge_index.size(1)} edges.")
        logging.debug(f"Initial edge_index:\n{self.edge_index}")
        logging.debug(f"Initial edge_attr:\n{self.edge_attr}")
        
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

        logging.info(f"Pillbug Network ID: {self.id}")

        if head_trainer is None:
                    self.head_trainer = HeadTrainer(self)

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
                ), self.head_trainer)
                for _ in range(num_nodes)
            ]

        # For energy conservation tracking
        self.initial_total_temperature = self.temperature + sum(f.temperature for f in self.features)
        self.initial_total_radiation = 0.0
        logging.info(f"Initial total temperature: {self.initial_total_temperature}")
        logging.info(f"Initial total radiation: {self.initial_total_radiation}")

    def initialize_edge_index(self):
        """
        Initialize a fully connected graph.
        """
        edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).T
        logging.debug(f"Initialized edge_index:\n{edge_index}")
        return edge_index

    def compute_loss(self, corrections, edge_attr):
        """
        Compute loss based on corrections and edge attributes.
        """
        # Example: Loss is proportional to the magnitude of corrections
        correction_loss = torch.sum(torch.norm(corrections, dim=1))

        # Example: Temperature deviation loss
        temperature_loss = torch.var(self.state_matrix[:, 0])  # Assuming temperature is the first feature

        # Example: Radiation conservation loss
        radiation_loss = np.abs(self.total_radiation() - self.initial_total_radiation)

        # Combine losses with weights
        return correction_loss + 0.1 * temperature_loss + 0.01 * radiation_loss

    
    def initialize_edge_attr(self):
                
        edge_attr = {
            "bounds": torch.ones(self.edge_index.size(1)),
            "force_limits": torch.ones(self.edge_index.size(1)),
            "exchange_rates": torch.ones(self.edge_index.size(1)),
            "current_debt": torch.zeros(self.edge_index.size(1)),
            "debt_threshold": torch.ones(self.edge_index.size(1)) * 100.0,
            "exchange_types": ["float"] * self.edge_index.size(1),
        }
        logging.debug(f"Initialized edge_attr:\n{edge_attr}")
        return edge_attr

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
        logging.info(f"Random walk applied. Total state change: {total_state_change:.4f}.")
        logging.debug(f"Updated state matrix:\n{self.state_matrix}")
        logging.debug(f"Updated velocity matrix:\n{self.velocity_matrix}")
        logging.debug(f"Updated acceleration matrix:\n{self.acceleration_matrix}")
        
 
        return total_state_change

    def apply_corrections(self, corrections):
        """
        Applies corrections to the acceleration matrix or delegates to subfeatures.
        Args:
            corrections (torch.Tensor): Matrix of corrections to be applied.
        """
        logging.info(f"\n--- Corrections Applied to Pillbug Network {self.id} ---")
        logging.info(f"{'Node':<10}{'Correction':<30}{'Magnitude':<15}{'Direction':<30}")
        logging.info("-" * 85)

        for i, feature in enumerate(self.features):
            correction = corrections[i]
            if isinstance(feature, PillbugNetwork):
                feature.apply_corrections(correction)
                # Assuming sub-network's state can be represented similarly
                state = feature.state_matrix[i] if feature.state_matrix.size(0) > i else torch.zeros_like(correction)
            elif isinstance(feature, RandomWalkSimulator):
                feature.apply_second_derivative(correction)
                state = self.state_matrix[i]
            else:
                state = torch.zeros_like(correction)
            if correction.dim() == 0:  # Scalar correction
                direction = [0.0] if correction == 0 else [1.0]
                magnitude = correction
            else:  # Vector correction
                magnitude = torch.norm(correction).item()
                direction = (correction / magnitude).numpy() if magnitude > 0 else [0.0 for _ in range(len(correction))]


            # Convert tensors to lists for better readability
            correction_list = correction.tolist()
            state_list = state.tolist()

            # Log the correction details
            logging.info(f"{i:<10}{str(correction_list):<30}{magnitude:<15.4f}{str(direction):<30}")

        logging.info("-" * 85)
        logging.info(f"--- End of Corrections for Pillbug Network {self.id} ---\n")


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
        """
        Nodes exchange radiation based on the distance between their state vectors in feature space.
        Args:
            dt (float): Time step for radiation calculation.
        """
        logging.info(f"Exchange radiation initiated for Pillbug Network {self.id}.")
        
        total_emissions = []
        for node in self.features:
            emitted = node.emit_radiation(dt)
            total_emissions.append(emitted)
        
        # Compute pairwise distances in feature space
        state_vectors = self.state_matrix  # Each row is a node's feature vector
        num_nodes = state_vectors.size(0)
        distances = torch.cdist(state_vectors, state_vectors, p=2)  # Euclidean distance
        
        # Handle self-loops (distance to itself should not matter)
        distances.fill_diagonal_(float('inf'))
        
        for i, node_i in enumerate(self.features):
            absorbed = 0.0
            for j, node_j in enumerate(self.features):
                if i != j:
                    distance = distances[i, j].item()
                    if distance > 0:  # Avoid division by zero
                        absorption_fraction = 1.0 / distance
                        absorbed_radiation = total_emissions[j] * absorption_fraction
                        node_i.absorb_radiation(absorbed_radiation)
                        absorbed += absorbed_radiation
            
            # Subtract absorbed radiation from the total radiation pool
            self.radiation -= absorbed
            logging.info(f"Node {node_i.id} absorbed radiation: {absorbed:.4f}. Total pool radiation: {self.radiation:.4f}")

    def backward_pass(self, loss):
        # Compute gradients
        loss.backward()
        # Update parameters using optimizer
        self.optimizer.step()
        self.optimizer.zero_grad()

    def process_iteration(self, dt, edge_index=None, edge_attr=None, error=0, corrections=None):
        self.exchange_radiation(dt)
        self.random_walk(dt)
        if edge_index is None:
            edge_index = self.edge_index
        if edge_attr is None:
            edge_attr = self.edge_attr
        outputs = self.forward(self.state_matrix, edge_index, edge_attr)
        if corrections is not None:
            self.previous_corrections = corrections
        else:
            self.previous_corrections = outputs
        self.apply_corrections(self.previous_corrections)
        self.update_gestalt_matrix()

        # Compute loss based on outputs and a target (if available)
        loss = self.compute_loss(outputs, edge_attr)
        self.backward_pass(loss)


    def total_temperature(self):
        return self.temperature + sum(node.temperature for node in self.features)

    def total_radiation(self):
        return self.radiation + sum(node.radiation for node in self.features if hasattr(node, 'radiation'))
