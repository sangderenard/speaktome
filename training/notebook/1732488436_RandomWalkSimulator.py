# random_walk_simulator.py

import torch
import random
import logging
import uuid
from .training_coordinator import TrainingCoordinator
from .helper_functions import conditional_round

# ========================================
# RandomWalkSimulator Class (modified to include TrainingCoordinator)
# ========================================

class RandomWalkSimulator:
    def __init__(self, state_vector_dim, config, head_trainer):
        self.head_trainer = head_trainer
        self.id = uuid.uuid4()
        self.temperature = random.uniform(50.0, 100.0)  # Initial temperature
        self.emissivity = random.uniform(0.1, 1.0)      # Emissivity coefficient
        self.absorbed_radiation = 0.0                   # Radiation absorbed this iteration
        self.state_vector = torch.randn(state_vector_dim)
        self.velocity = torch.zeros(state_vector_dim)
        self.acceleration = torch.zeros(state_vector_dim)
        self.config = config
        logging.info(f"Simulator {self.id} initialized with state vector {self.state_vector} and config {self.config}")

        # Each RandomWalkSimulator has its own TrainingCoordinator
        self.training_coordinator = TrainingCoordinator(self, self.head_trainer)

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

        # TrainingCoordinator performs its cycle action
        self.training_coordinator.perform_cycle_action(dt)

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
