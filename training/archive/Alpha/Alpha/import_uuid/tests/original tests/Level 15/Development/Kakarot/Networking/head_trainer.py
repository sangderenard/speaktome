# head_trainer.py

import torch
import logging
from torch_geometric.nn import MessagePassing

class HeadTrainer(MessagePassing):
    def __init__(self, pillbug_network):
        super().__init__(aggr="mean")
        self.pillbug_network = pillbug_network
        # Create a unique edge to the lead PillbugNetwork
        self.edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Edge from HeadTrainer to PillbugNetwork
        self.edge_attr = {"edge_type": torch.tensor([1])}  # Placeholder edge attribute
        self.x = torch.zeros(2, pillbug_network.state_matrix.size(1))  # Node features
        self.updates_received = []

    def forward(self):
        return self.propagate(self.edge_index, x=self.x, edge_attr=self.edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def receive_update(self, weights, gradients, losses):
        self.updates_received.append({'weights': weights, 'gradients': gradients, 'losses': losses})
        logging.info("HeadTrainer received updates from a TrainingCoordinator.")

    def reduce_temperature(self):
        temperature_reduction = len(self.updates_received) * 0.1  # Example logic
        self.pillbug_network.temperature -= temperature_reduction
        logging.info(f"HeadTrainer reduced PillbugNetwork temperature by {temperature_reduction}.")
