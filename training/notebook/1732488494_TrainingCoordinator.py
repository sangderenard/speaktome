# training_coordinator.py

import torch
import random
import logging
import torch.nn as nn
import torch.optim as optim
from .nn_wrapper import NNWrapper
from Primitives.chipmunk_slice import ChipmunkSlice

EFFICIENCY_THRESHOLD = 0.1  # For loss comparison

# ========================================
# TrainingCoordinator Class (per RandomWalkSimulator)
# ========================================

class TrainingCoordinator:
    def __init__(self, simulator, head_trainer):
        self.head_trainer = head_trainer
        self.simulator = simulator
        self.state_logs = []
        self.projection_logs = []
        self.losses = []
        self.energy_budget = 0.0
        # Initialize NNWrapper (generic)
        self.nn_wrapper = NNWrapper()
        self.optimizer = optim.Adam(self.nn_wrapper.parameters())
        self.criterion = nn.MSELoss()
        # Pick any two features to map to axes in a ChipmunkSlice
        self.state_vector_dim = self.simulator.state_vector.size(0)
        self.mapped_axes = random.sample(range(self.state_vector_dim), min(2, self.state_vector_dim))
        # Initialize ChipmunkSlice using cricket.py's environment settings with gravity (0,0)
        self.chipmunk_slice = ChipmunkSlice()
        # Cycle counter
        self.cycle_state = 1
        # Efficiency threshold for loss
        self.efficiency_threshold = EFFICIENCY_THRESHOLD

    def perform_cycle_action(self, dt):
        if self.cycle_state == 1:
            # Slightly increase local temperature
            self.simulator.temperature += 0.1
            # Produce a ChipmunkSlice with initial conditions identical to the state of random two features
            state_dict = {
                'position': self.simulator.state_vector[self.mapped_axes].tolist(),
                'velocity': self.simulator.velocity[self.mapped_axes].tolist(),
                'mass': self.simulator.config.mass if hasattr(self.simulator.config, 'mass') else 1.0,
            }
            projection = self.chipmunk_slice.simulate(state_dict, dt)
            self.projection_logs.append(projection)
            logging.info(f"Cycle 1: Produced ChipmunkSlice projection.")

        elif self.cycle_state == 2:
            # Slightly increase local temperature
            self.simulator.temperature += 0.1
            # Supply the NN wrapper with the initial state parameters and obtain a prediction
            input_state = torch.cat([
                self.simulator.state_vector[self.mapped_axes],
                self.simulator.velocity[self.mapped_axes]
            ])

            self.nn_prediction = self.nn_wrapper(input_state)
            logging.info(f"Cycle 2: Obtained NN prediction.")

        elif self.cycle_state == 3:
            # Compare the real output states and evaluate loss
            projected_state = torch.tensor(self.projection_logs[-1]['position'] + self.projection_logs[-1]['velocity'])
            # Compute distance between real output and NN prediction
            distance_real = torch.norm(projected_state)
            distance_pred = torch.norm(self.nn_prediction)
            # Compute loss as factor of magnitude and direction similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(projected_state, self.nn_prediction, dim=0)
            loss = (distance_real - distance_pred).abs() * (1 - cosine_similarity)
            self.losses.append(loss.item())
            logging.info(f"Cycle 3: Computed loss: {loss.item()}")

            # Adjust temperature based on loss and accuracy
            if loss.item() < self.efficiency_threshold:
                self.simulator.temperature -= 0.1  # Reduce temperature slightly
            else:
                self.simulator.temperature += 0.1  # Increase temperature slightly

            # Accuracy significantly lowers local temperature
            accuracy = torch.norm(projected_state - self.nn_prediction)
            if accuracy < self.efficiency_threshold:
                self.simulator.temperature -= 0.5  # Significantly reduce temperature

            # Energy budget proportional to the distance of the error
            error_distance = torch.norm(projected_state - self.nn_prediction).item()
            self.energy_budget += error_distance

            # Option to cool any system or bind unbound axes
            # For simplicity, we can assume energy is used to reduce temperature further
            self.simulator.temperature -= self.energy_budget * 0.01  # Use some energy to cool down
            self.energy_budget = 0.0  # Reset energy budget after use

        elif self.cycle_state == 4:
            # Slight temperature increase
            self.simulator.temperature += 0.1
            # Generate raw training material
            if len(self.state_logs) > 1:
                # Compare system state logs to simulated evolutions
                input_state = torch.cat([
                    self.state_logs[-2][self.mapped_axes],
                    self.simulator.velocity[self.mapped_axes]
                ])
                target_state = torch.cat([
                    self.state_logs[-1][self.mapped_axes],
                    self.simulator.velocity[self.mapped_axes]
                ])
                self.training_data = {'input': input_state, 'target': target_state}
                logging.info("Cycle 4: Generated raw training material.")

        elif self.cycle_state == 5:
            # Slight temperature increase
            self.simulator.temperature += 0.1
            # Train the NN module for an epoch on historical data
            if hasattr(self, 'training_data'):
                self.optimizer.zero_grad()
                output = self.nn_wrapper(self.training_data['input'])
                loss = self.criterion(output, self.training_data['target'])
                loss.backward()
                self.optimizer.step()
                logging.info(f"Cycle 5: Trained NN module. Loss: {loss.item()}")

        elif self.cycle_state == 6:
            # Submit weights, gradients, and loss logs to the HeadTrainer
            weights = self.nn_wrapper.state_dict()
            gradients = {name: param.grad for name, param in self.nn_wrapper.named_parameters() if param.grad is not None}
            self.head_trainer.receive_update(weights, gradients, self.losses)
            # HeadTrainer reduces the lead PillbugNetwork temperature
            self.head_trainer.reduce_temperature()
            logging.info("Cycle 6: Submitted updates to HeadTrainer.")

        # Update cycle state
        self.cycle_state = (self.cycle_state % 6) + 1

        # Collect state logs for future cycles
        self.state_logs.append(self.simulator.state_vector.clone())
