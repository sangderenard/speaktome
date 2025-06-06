import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import logging

torch.autograd.set_detect_anomaly(True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, node_type="default", features=None):
        self.id = self._generate_id()
        self.node_type = node_type
        self.features = features if features is not None else torch.empty(0)
        self.edges = []
        self.message_queue = []
        self.task_completed = False
        self.loss_threshold = 0.05

        # Boredom and complacency
        self.boredom = 0
        self.complacency = 0
        self.boredom_threshold = 500
        self.complacency_threshold = 3000

        # Node-specific models
        if self.node_type == "generator":
            self.model = self._create_generator()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
        elif self.node_type == "discriminator":
            self.model = self._create_discriminator()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = nn.BCELoss()

    def _generate_id(self):
        return uuid.uuid4().hex

    def _create_generator(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _create_discriminator(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def add_edge(self, edge):
        self.edges.append(edge)

    def receive_message(self, data):
        self.message_queue.append(data)

    def log_state(self, stage, tensor=None, loss=None):
        message = f"[{self.node_type.upper()}] Stage: {stage}"
        if tensor is not None:
            message += f", Tensor stats: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
        if loss is not None:
            message += f", Loss: {loss:.4f}"
        logger.info(message)

    def update_boredom_and_complacency(self, improved):
        if improved:
            self.boredom = 0
            self.complacency += 1
        else:
            self.boredom += 1
            self.complacency = max(0, self.complacency - 1)

        if self.boredom >= self.boredom_threshold and self.complacency <= self.complacency_threshold:
            self.switch_role()
            self.boredom *= 0.99
            self.complacency = 0

    def switch_role(self):
        logger.info(f"[{self.id}] Switching role from {self.node_type}")
        if self.node_type == "generator":
            self.node_type = "discriminator"
            self.model = self._create_discriminator()
        elif self.node_type == "discriminator":
            self.node_type = "generator"
            self.model = self._create_generator()

    def process_task(self, input_tensor, reference_tensor=None):
        if self.node_type == "generator":
            output = self.model(input_tensor)
            self.log_state("Generator Output", output)
            if reference_tensor is not None:
                loss = self.loss_fn(output, reference_tensor)
                loss.backward(retain_graph=True)
                self.task_completed = loss.item() < self.loss_threshold
                self.log_state("Generator Loss Computation", output, loss)
                return output, loss.item()
            return output, None

        elif self.node_type == "discriminator":
            output = self.model(input_tensor)
            self.log_state("Discriminator Output", output)
            if reference_tensor is not None:
                loss = self.loss_fn(output, reference_tensor)
                loss.backward(retain_graph=True)
                self.task_completed = loss.item() < self.loss_threshold
                self.log_state("Discriminator Loss Computation", output, loss)
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
        logger.info(f"[EDGE] Forwarding data from {self.source.id} to {self.target.id}")
        self.cache = data
        self.target.receive_message(data)

    def backward_pass(self):
        if self.cache is not None:
            logger.info(f"[EDGE] Backward passing data from {self.target.id} to {self.source.id}")
            self.source.receive_message(self.cache)
            self.cache = None

    def __repr__(self):
        return f"Edge(id={self.id}, source={self.source.id}, target={self.target.id})"

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_type, features=None):
        node = Node(node_type=node_type, features=features)
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target):
        edge = Edge(source, target)
        self.edges[edge.id] = edge
        source.add_edge(edge)
        return edge

    def execute(self, mode, num_outputs, input_tensor, reference_tensor):
        logger.info("[GRAPH] Starting execution cycle")
        generator_node = next((n for n in self.nodes.values() if n.node_type == "generator"), None)
        discriminator_node = next((n for n in self.nodes.values() if n.node_type == "discriminator"), None)

        if generator_node is None or discriminator_node is None:
            raise ValueError("Graph must have both generator and discriminator nodes.")

        generated_output, gen_loss = generator_node.process_task(input_tensor, reference_tensor)
        noise = torch.randn_like(generated_output) * 0.1
        noisy_output = generated_output + noise
        real_labels = torch.ones_like(noisy_output)
        disc_output, disc_loss = discriminator_node.process_task(noisy_output, real_labels)

        generator_node.update_boredom_and_complacency(gen_loss < generator_node.loss_threshold)
        discriminator_node.update_boredom_and_complacency(disc_loss < discriminator_node.loss_threshold)

        logger.info(f"[GRAPH] Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
        return gen_loss, disc_loss

    def visualize_results(self, input_tensor, generated_output):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Generated Output")
        plt.imshow(generated_output.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.axis("off")

        plt.show()

def main(num_rounds=2500):
    graph = Graph()
    generator_node = graph.add_node(node_type="generator")
    discriminator_node = graph.add_node(node_type="discriminator")
    graph.add_edge(generator_node, discriminator_node)

    input_tensor = torch.rand((1, 3, 32, 32))
    reference_tensor = torch.rand((1, 3, 32, 32))

    for round_num in range(num_rounds):
        gen_loss, disc_loss = graph.execute(
            mode="feature",
            num_outputs=len(input_tensor),
            input_tensor=input_tensor,
            reference_tensor=reference_tensor
        )
        if (round_num + 1) % 100 == 0:
            logger.info(f"Completed {round_num + 1}/{num_rounds} rounds")

    generated_output, _ = generator_node.process_task(input_tensor)
    graph.visualize_results(reference_tensor, generated_output)

if __name__ == "__main__":
    main()
