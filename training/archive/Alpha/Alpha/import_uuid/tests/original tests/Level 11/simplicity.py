import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

class Node:
    def __init__(self, node_type="default", features=None):
        self.id = self._generate_id()
        self.node_type = node_type
        self.features = features if features is not None else torch.empty(0)
        self.edges = []
        self.task_completed = False
        self.loss_threshold = 0.05

        # Boredom and complacency
        self.boredom = 0
        self.complacency = 0
        self.boredom_threshold = 500
        self.complacency_threshold = 3000

        # Task-specific networks
        self.generator_model = self._create_generator()
        self.discriminator_model = self._create_discriminator()

        # Task-specific optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator_model.parameters(), lr=0.01)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator_model.parameters(), lr=0.001)

        # Loss functions
        self.generator_loss_fn = nn.MSELoss()
        self.discriminator_loss_fn = nn.BCELoss()

    def _generate_id(self):
        return uuid.uuid4().hex

    def _create_generator(self):
        """Generator-specific network architecture with upsampling."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64 * 64),  # Initial dense layer
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),  # Reshape to feature map for Conv2D

            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # Feature extraction
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Upsample back to 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Upsample to 32x32
            nn.Sigmoid()
        )


    def _create_discriminator(self):
        """Discriminator-specific network architecture."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Sigmoid()
        )
    def add_edge(self, edge):
            self.edges.append(edge)
    def switch_role(self):
        """Switch between generator and discriminator roles."""
        print(f"[{self.id}] Switching role from {self.node_type}")
        if self.node_type == "generator":
            self.node_type = "discriminator"
        elif self.node_type == "discriminator":
            self.node_type = "generator"

    def update_boredom_and_complacency(self, improved):
        """Update boredom and complacency metrics based on task success."""
        if improved:
            self.boredom = 0
            self.complacency += 1.01
        else:
            self.boredom += 1
            self.complacency = max(0, self.complacency - 1)

        if self.boredom >= self.boredom_threshold and self.complacency <= self.complacency_threshold:
            self.switch_role()
            self.boredom *= 0.99
            self.complacency = 0

    def process_task(self, input_tensor, reference_tensor=None):
        """Perform task using the active network based on current role."""
        if self.node_type == "generator":
            # Generator forward pass
            output = self.generator_model(input_tensor)
            if reference_tensor is not None:
                loss = self.generator_loss_fn(output, reference_tensor)
                #self.generator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                #self.generator_optimizer.step()
                return output, loss.item()
            return output, None

        elif self.node_type == "discriminator":
            # Discriminator forward pass
            output = self.discriminator_model(input_tensor)
            if reference_tensor is not None:
                loss = self.discriminator_loss_fn(output, reference_tensor)
                self.discriminator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                #self.discriminator_optimizer.step()
                return output, loss.item()
            return output, None

    def __repr__(self):
        return f"Node(id={self.id}, type={self.node_type}, num_edges={len(self.edges)})"

class Edge:
    def __init__(self, source, target):
        self.id = uuid.uuid4().hex
        self.source = source
        self.target = target
        self.cache = None

    def forward_pass(self, data):
        """Forward pass data along the edge."""
        self.cache = data
        self.target.receive_message(data)

    def __repr__(self):
        return f"Edge(id={self.id}, source={self.source.id}, target={self.target.id})"

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_type, features=None):
        """Add a node to the graph."""
        node = Node(node_type=node_type, features=features)
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target):
        """Add an edge to the graph."""
        edge = Edge(source, target)
        self.edges[edge.id] = edge
        source.add_edge(edge)
        return edge

    def execute(self, input_tensor, reference_tensor):
        """Execute the graph by processing tasks."""
        generator_node = next((n for n in self.nodes.values() if n.node_type == "generator"), None)
        discriminator_node = next((n for n in self.nodes.values() if n.node_type == "discriminator"), None)

        if generator_node is None or discriminator_node is None:
            raise ValueError("Graph must have both generator and discriminator nodes.")

        # Process generator and discriminator tasks
        generated_output, gen_loss = generator_node.process_task(input_tensor, reference_tensor)
        noise = torch.randn_like(generated_output) * 0.01
        noisy_output = generated_output + noise
        real_labels = torch.ones_like(torch.tensor([[1.0]]))
        disc_output, disc_loss = discriminator_node.process_task(noisy_output, real_labels)

        # Update boredom and complacency metrics
        generator_node.update_boredom_and_complacency(gen_loss < generator_node.loss_threshold)
        discriminator_node.update_boredom_and_complacency(disc_loss < discriminator_node.loss_threshold)

        print(f"[GRAPH] Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
        return gen_loss, disc_loss

def main(num_rounds=2500):
    graph = Graph()
    generator_node = graph.add_node(node_type="generator")
    discriminator_node = graph.add_node(node_type="discriminator")
    graph.add_edge(generator_node, discriminator_node)

    input_tensor = torch.rand((1, 3, 32, 32))
    reference_tensor = torch.rand((1, 3, 32, 32))

    for round_num in range(num_rounds):
        gen_loss, disc_loss = graph.execute(input_tensor, reference_tensor)
        if (round_num + 1) % 100 == 0:
            print(f"Completed {round_num + 1}/{num_rounds} rounds")

if __name__ == "__main__":
    main()
