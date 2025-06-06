import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import pygame

BUFFER_SIZE = 10  # Define buffer size for references

class Reference:
    def __init__(self, reference_tensor):
        self.id = uuid.uuid4().hex
        self.reference_tensor = reference_tensor
        self.output_buffer = []  # Stores recent node outputs
        self.average_output = None
        self.lock = threading.Lock()  # To handle concurrent access

    def add_output(self, output_tensor):
        with self.lock:
            self.output_buffer.append(output_tensor.detach().cpu())
            if len(self.output_buffer) > BUFFER_SIZE:
                self.output_buffer.pop(0)
            self.calculate_average()

    def calculate_average(self):
        if self.output_buffer:
            stacked = torch.stack(self.output_buffer)
            self.average_output = torch.mean(stacked, dim=0)

class Node:
    def __init__(self, features=None):
        """
        Initializes a node with both generator and discriminator models.
        """
        
        self.id = self._generate_id()
        self.features = features if features is not None else torch.empty(0)
        self.edges = []
        self.message_queue = []
        self.task_completed = False
        self.current_task = None  # Task descriptor
        self.loss_threshold = 1e-15  # Arbitrary low-loss threshold
        self.telomere = 10  # Limiting depth for replication
        self.gradient_history = []

        # Models
        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()

        # Optimizers
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        # Loss functions
        self.gen_loss_fn = nn.MSELoss()
        self.disc_loss_fn = nn.MSELoss()  # Difference texture loss

        self.reference_errors = {}  # Maps reference_id to (error, timestamp)
        self.reference_copies = None
        self.best_losses = []
        self.local_weights = {}
        self.edge_weights = {}

    def _generate_id(self):
        return uuid.uuid4().hex

    def attach_hooks(self):
        """Attach hooks to generator and discriminator for gradient monitoring."""
        for name, param in self.generator.named_parameters():
            if param.requires_grad:
                param.register_hook(self._capture_gradient_hook(name, "generator"))
        for name, param in self.discriminator.named_parameters():
            if param.requires_grad:
                param.register_hook(self._capture_gradient_hook(name, "discriminator"))

    def _capture_gradient_hook(self, name, focus):
        """
        Hook to capture gradient norms and store them in gradient history.
        """
        def hook(grad):
            timestamp = time.time()
            grad_norm = grad.norm().item()
            # Log gradient history as low-bit depth (e.g., 8-bit) with timestamp and focus (generator/discriminator)
            self.gradient_history.append({
                "timestamp": timestamp,
                "focus": focus,
                "name": name,
                "grad_norm": float(np.clip(grad_norm, 0, 255))  # Example low-bit depth clipping
            })
        return hook

    def _create_generator(self):
        """Creates a simple convolutional generator."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _create_discriminator(self):
        """Creates a discriminator with a focus on generating difference textures."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def decrement_telomere(self, high_loss=False):
        """Decrements the telomere counter for the node."""
        if high_loss:
            self.telomere -= 1
        if self.telomere <= 0:
            print(f"Node {self.id} pruned due to telomere exhaustion.")
            return True  # Indicate pruning
        return False

    def process_task(self, input_tensor, reference_tensor, reference_id, focus="generator"):
        if focus == "generator":
            output = self.generator(input_tensor)
            loss = self.gen_loss_fn(output, reference_tensor)
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()
            self.task_completed = loss.item() < self.loss_threshold


            # Track error with weighting
            current_time = time.time()
            weighted_error = current_time * loss.item()
            if reference_id in self.reference_errors:
                existing_error, _ = self.reference_errors[reference_id]
                if weighted_error < existing_error:
                    self.reference_errors[reference_id] = (weighted_error, current_time)
            else:
                self.reference_errors[reference_id] = (weighted_error, current_time)

            return output, loss.item()
        elif focus == "discriminator":
            output = self.discriminator(input_tensor)
            loss = self.disc_loss_fn(output, reference_tensor)
            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            # Track error with weighting
            current_time = time.time()
            weighted_error = current_time * loss.item()
            if reference_id in self.reference_errors:
                existing_error, _ = self.reference_errors[reference_id]
                if weighted_error < existing_error:
                    self.reference_errors[reference_id] = (weighted_error, current_time)
            else:
                self.reference_errors[reference_id] = (weighted_error, current_time)

            return output, loss.item()

    def replicate(self, graph):
        if self.telomere > 5 and graph.food_supply >= graph.food_cost_per_replicate:
            child = Node(features=self.features.clone())
            child.telomere = self.telomere - 1
            graph.food_supply -= graph.food_cost_per_replicate
            print(f"Node {self.id} replicated to {child.id}. Food supply: {graph.food_supply}")
            return child
        return None


    def heal(self):
        if self.task_completed:
            self.telomere = min(self.telomere + 1, 10)
            print(f"Node {self.id} healed to telomere {self.telomere}.")

    def generate_data_id(self):
        """
        Generates a unique ID for the data produced by this node.
        """
        return f"{self.id}_{uuid.uuid4().hex}"

    def get_lowest_error_reference(self):
        if not self.reference_errors:
            return None, None
        reference_id, (error, timestamp) = min(self.reference_errors.items(), key=lambda x: x[1][0])
        return reference_id, error

    def __repr__(self):
        return f"Node(id={self.id}, telomere={self.telomere})"

class Edge:
    def __init__(self, source, target, weight=1.0):
        self.id = uuid.uuid4().hex
        self.source = source
        self.target = target
        self.weight = weight  # Weight for self-interaction or specific connections
        self.cache = []  # Holds intermediate results for generator output

    def forward_pass(self, data, data_id=None):
        """Sends data from source to target, tracking data IDs."""
        if len(self.cache) < 10:  # Cache size limit
            self.cache.append((data, data_id))
        self.target.receive_message(data)

    def backward_pass(self):
        """Feeds back data from cache to the source."""
        if self.cache:
            data, data_id = self.cache.pop(0)
            self.source.receive_message(data, data_id)

    def __repr__(self):
        return f"Edge(id={self.id}, source={self.source.id}, target={self.target.id}, weight={self.weight})"

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.references = {}
        self.reference_to_nodes = {}
        self.node_map_buffer = {}
        self.display_node = None
        self.food_supply = 100  # Initial food supply
        self.food_per_task = 5  # Food gained per completed task
        self.food_cost_per_replicate = 10  # Food cost per replication
        
    def add_node(self, features=None):
        node = Node(features=features)
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target, weight=1.0):
        edge = Edge(source, target, weight=weight)
        self.edges[edge.id] = edge
        source.edges.append(edge)
        return edge

    def add_reference(self, reference_tensor):
        reference = Reference(reference_tensor)
        self.references[reference.id] = reference
        self.reference_to_nodes[reference.id] = set()  # Initialize as empty set
        return reference


    def compile_gradient_history(self):
        """
        Compiles gradient history from all nodes into a centralized log.
        """
        compiled_history = []
        for node_id, node in self.nodes.items():
            compiled_history.extend(node.gradient_history)
        return compiled_history

    def save_gradient_history(self, filename="gradient_history.json"):
        """
        Saves compiled gradient history to a file for diagnostics.
        """
        import json
        compiled_history = self.compile_gradient_history()
        with open(filename, "w") as f:
            json.dump(compiled_history, f)
        print(f"Gradient history saved to {filename}.")

    def execute(self, input_tensor):
        current_nodes = list(self.nodes.values())

        for node in current_nodes:
            for reference_id, reference in self.references.items():
                # Track node holding the reference
                self.reference_to_nodes[reference_id].add(node.id)

                # Process tasks for generator and discriminator
                gen_output, gen_loss = node.process_task(input_tensor, reference.reference_tensor, reference_id, focus="generator")
                disc_output, disc_loss = node.process_task(input_tensor, reference.reference_tensor, reference_id, focus="discriminator")

                # Determine the focus based on higher loss
                if gen_loss > disc_loss:
                    node.process_task(input_tensor, reference.reference_tensor, reference_id, focus="generator")
                else:
                    node.process_task(input_tensor, reference.reference_tensor, reference_id, focus="discriminator")

                # Add generator output to reference buffer
                reference.add_output(gen_output)

                if gen_loss < node.loss_threshold:  # Task completed successfully
                    reference.add_output(gen_output)
                    self.food_supply += self.food_per_task
                    print(f"Task completed! Food supply increased to {self.food_supply}.")

                    # Allow replication if conditions are met
                    child = node.replicate(self)
                    if child:
                        self.add_node(features=child.features)


    def centralize_weight_updates(self):
        """
        Centralizes weight updates using PyTorch Geometric utilities, ignoring pruned nodes.
        """
        edge_index = []
        edge_weights = []

        for edge in self.edges.values():
            if edge.source.id in self.nodes and edge.target.id in self.nodes:
                source_idx = list(self.nodes.keys()).index(edge.source.id)
                target_idx = list(self.nodes.keys()).index(edge.target.id)
                edge_index.append([source_idx, target_idx])
                edge_weights.append(edge.weight)

        if not edge_index:
            print("No valid edges for weight updates.")
            return

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        data = Data(edge_index=edge_index, edge_attr=edge_weights)

        # Centralized weight update logic (example: normalize edge weights)
        normalized_weights = pyg_utils.degree(edge_index[0], len(self.nodes)).pow(-0.5)
        data.edge_attr = data.edge_attr * normalized_weights[edge_index[0]] * normalized_weights[edge_index[1]]

        print("Centralized weight updates completed.")

    def prune(self):
        nodes_to_remove = []

        for node_id, node in self.nodes.items():
            # Check if this node holds unique references
            unique_references = [
                ref_id for ref_id, node_ids in self.reference_to_nodes.items()
                if len(node_ids) <= 2 and node_id in node_ids
            ]
            if unique_references:
                print(f"Node {node_id} is protected due to unique references: {unique_references}")
                continue  # Skip pruning this node

            # Prune based on telomere exhaustion or other criteria
            if node.decrement_telomere(high_loss=True):
                nodes_to_remove.append(node_id)

        # Remove pruned nodes and update reference mappings
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            for ref_id in self.reference_to_nodes:
                self.reference_to_nodes[ref_id].discard(node_id)

        print(f"Pruned {len(nodes_to_remove)} nodes.")


    def replicate_and_grow(self):
        for node in list(self.nodes.values()):
            child = node.replicate(self)
            if child:
                self.add_node(features=child.features)

    def update_reference_timeout(self):
        current_time = time.time()
        for reference_id, reference in self.references.items():
            # Find the latest update time across all nodes for this reference
            latest_update = max([error_info[1] for error_info in self.get_reference_errors(reference_id)], default=0)
            if current_time - latest_update > self.display_node.request_timeout:
                # Finalize the average if timeout exceeded
                reference.calculate_average()

    def get_reference_errors(self, reference_id):
        errors = []
        for node in self.nodes.values():
            if reference_id in node.reference_errors:
                errors.append(node.reference_errors[reference_id])
        return errors

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"

class DisplayNode:
    def __init__(self, graph, window_size=(1280, 720)):
        """Initialize the PyGame window."""
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Node Outputs")
        self.clock = pygame.time.Clock()
        self.running = True
        self.graph = graph
        self.graph.display_node = self  # To allow Graph to access request_timeout
        self.request_timeout = 5  # Seconds

    def render_images(self):
        """
        Render averaged images from references to the PyGame window.
        """
        if not self.running:
            return

        self.screen.fill((0, 0, 0))  # Clear screen

        x, y = 10, 10
        for reference_id, reference in self.graph.references.items():
            if reference.average_output is not None:
                self._blit_image(reference.average_output, (x, y), label=f"Ref {reference_id[:6]}")
                y += 250
                if y > self.screen.get_height() - 250:
                    y = 10
                    x += 250

        pygame.display.flip()  # Update display
        self.clock.tick(3)  # Limit frame rate

    def _blit_image(self, tensor, position, label=None):
        """
        Convert a tensor to a PyGame-compatible surface and blit it to the screen.
        """
        img = tensor.squeeze(0).permute(1, 2, 0).numpy()  # Convert to HWC format
        img = np.clip(img, 0, 1)  # Ensure values are between 0 and 1
        img = (img * 255).astype(np.uint8)  # Scale to 0-255
        img_surface = pygame.surfarray.make_surface(img)

        # Scale image to fit
        img_surface = pygame.transform.scale(img_surface, (240, 240))
        self.screen.blit(img_surface, position)

        # Add label (optional)
        if label:
            font = pygame.font.SysFont("Arial", 18)
            text_surface = font.render(label, True, (255, 255, 255))
            self.screen.blit(text_surface, (position[0], position[1] - 20))

    def handle_events(self):
        """
        Handle PyGame events like quitting.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

def generate_checkerboard(size, tile_size):
    """
    Generates a checkerboard pattern with random colors for each square.
    """
    board = np.zeros((size, size, 3))
    for i in range(0, size, tile_size):
        for j in range(0, size, tile_size):
            color = np.random.rand(3)  # Random color for each square
            for x in range(tile_size):
                for y in range(tile_size):
                    if i + x < size and j + y < size:
                        board[i + x, j + y] = color
    return torch.tensor(board, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)


def visualize_loss(gen_losses, disc_losses):
    """
    Visualizes generator and discriminator loss over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss", alpha=0.7)
    plt.plot(disc_losses, label="Discriminator Loss", alpha=0.7)
    plt.title("Loss Statistics Over Time")
    plt.xlabel("Training Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(num_rounds=1000, num_references=1):
    """
    Runs the graph with PyGame display integration.
    """
    graph = Graph()
    generator_node = graph.add_node()
    discriminator_node = graph.add_node()
    graph.add_edge(generator_node, discriminator_node)

    # Initialize multiple references
    references = []
    for _ in range(num_references):
        reference_tensor = generate_checkerboard(size=32, tile_size=2)
        reference = graph.add_reference(reference_tensor)
        references.append(reference)

    # Initialize the display node
    display_node = DisplayNode(graph, window_size=(1280, 720))

    

    for round_num in range(num_rounds):
        input_tensor = torch.rand((1, 3, 32, 32))
        graph.execute(input_tensor)

        # Render images in PyGame
        display_node.handle_events()
        display_node.render_images()

        if not display_node.running:
            break

        if (round_num + 1) % 100 == 0:
            print(f"Completed {round_num + 1} rounds.")

    pygame.quit()

if __name__ == "__main__":
    main()
