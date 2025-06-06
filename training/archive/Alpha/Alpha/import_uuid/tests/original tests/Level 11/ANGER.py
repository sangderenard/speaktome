import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


class Node:
    def __init__(self, node_type="default", features=None):
        """
        Initializes a node with a specific type (generator, discriminator).

        Args:
            node_type (str): Type of the node ("generator" or "discriminator").
            features (torch.Tensor or dict, optional): Feature tensor or dictionary.
        """
        self.flip = True
        self.id = self._generate_id()
        self.node_type = node_type
        self.features = features if features is not None else torch.empty(0)
        self.edges = []
        self.message_queue = []
        self.task_completed = False
        self.current_task = None  # Task descriptor
        self.loss_threshold = 0.05  # Arbitrary low-loss threshold

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
        """Creates a simple convolutional generator."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _create_discriminator(self):
        """Creates a discriminator with trigonometric signal handling."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def add_edge(self, edge):
        self.edges.append(edge)

    def receive_message(self, data):
        """
        Receives a message from an edge.
        """
        self.message_queue.append(data)

    def process_feedback(self, feedback, mode="feature", num_outputs=8):
        """
        Processes feedback by remapping it to generator inputs using FeatureDepthTranslator.

        Args:
            feedback (torch.Tensor): Feedback to be remapped.
            mode (str): The mapping mode ('feature', 'byte', 'bit').
            num_outputs (int): The number of outputs for remapping.

        Returns:
            torch.Tensor: Remapped feedback as a new feature tensor.
        """
        return feedback  # Discriminator nodes may handle feedback differently


    def process_task(self, input_tensor, reference_tensor=None):
        """
        Existing generator or discriminator processing logic...
        """
        if self.node_type == "generator":
            # Probabilistically broadcast feedback across input features
            input_tensor = self.broadcast_feedback_to_features(input_tensor)

            # Forward pass and loss calculation
            output = self.model(input_tensor)
            if reference_tensor is not None:
                loss = self.loss_fn(output, reference_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Handle success condition
                if loss.item() < self.loss_threshold:
                    self.task_completed = True
                    self.model.apply(self._reset_weights)
                return output, loss.item()
            else:
                return output, None
        elif self.node_type == "discriminator":
            output = self.model(input_tensor)
            if reference_tensor is not None:
                # Assuming reference_tensor contains labels (1 for real, 0 for fake)
                loss = self.loss_fn(output, reference_tensor)
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                return output, loss.item()
            else:
                return output, None

    def broadcast_feedback_to_features(self, input_tensor):
        """
        Probabilistically broadcast feedback across input features.

        Args:
            input_tensor (torch.Tensor): Generator's input tensor.

        Returns:
            torch.Tensor: Modified input tensor with feedback broadcast.
        """
        feedback = torch.rand(1, device=input_tensor.device)  # Example feedback value
        feature_broadcast = feedback.repeat(input_tensor.size(0), 1, 1, 1)
        return input_tensor + feature_broadcast

    def _reset_weights(self, m):
        """Resets weights of the model."""
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def __repr__(self):
        return f"Node(id={self.id}, type={self.node_type}, num_edges={len(self.edges)})"

class Edge:
    def __init__(self, source, target):
        self.id = uuid.uuid4().hex
        self.source = source
        self.target = target
        self.cache = None  # Holds intermediate results for forward/backward pass

    def forward_pass(self, data):
        """Sends data from source to target."""
        self.cache = data
        self.target.receive_message(data)

    def backward_pass(self):
        """Sends cached data back to source."""
        if self.cache is not None:
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
        """
        Executes a single forward and backward pass for all nodes.

        Args:
            mapper (FeatureDepthTranslator): The mapper instance.
            mode (str): Mapping mode ('bit', 'byte', 'feature').
            num_outputs (int): Number of output features (bins) for mapping.
            input_tensor (torch.Tensor): Input tensor for the generator.
            reference_tensor (torch.Tensor): Reference tensor for loss computation.
        """
        # Generator forward and backward pass
        generator_node = next((n for n in self.nodes.values() if n.node_type == "generator"), None)
        discriminator_node = next((n for n in self.nodes.values() if n.node_type == "discriminator"), None)

        if generator_node is None or discriminator_node is None:
            raise ValueError("Graph must have both generator and discriminator nodes.")

        generated_output, gen_loss = generator_node.process_task(input_tensor, reference_tensor)

        # Noise injection at the junction
        injection_scale = 0.1  # You can parameterize this if needed
        noise = torch.randn_like(generated_output) * injection_scale
        noisy_output = generated_output + noise


        # Prepare labels for discriminator (assuming real labels are 1)
        real_labels = torch.ones_like(noisy_output)

        # Discriminator forward and backward pass
        disc_output, disc_loss = discriminator_node.process_task(
            generated_output.clone().detach(), real_labels  # Adjust dimensions if needed
        )

        return gen_loss, disc_loss

def generate_checkerboard(size, tile_size):
    """
    Generates a checkerboard pattern.

    Args:
        size (int): Size of the checkerboard (image will be size x size).
        tile_size (int): Size of each tile.

    Returns:
        torch.Tensor: Checkerboard image tensor (1, 3, size, size).
    """
    board = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            if (i // tile_size + j // tile_size) % 2 == 0:
                board[i, j] = 1  # White tile
    return torch.tensor(board, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)  # Shape (1, 3, size, size)

def visualize_checkerboard(original, generated):
    """
    Displays the original checkerboard and generated image.

    Args:
        original (torch.Tensor): Original checkerboard image (1, 3, H, W).
        generated (torch.Tensor): Generated image (1, 3, H, W).
    """
    original = original.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HxWx3
    generated = generated.squeeze().permute(1, 2, 0).cpu().detach().numpy()  # Convert to HxWx3

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Checkerboard")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Generated Output")
    plt.imshow(generated)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main(num_rounds=2500, injection_scale=0.1):
    """
    Runs the network across all mapping modes with noise injection and visualizes statistics.

    Args:
        num_rounds (int): Number of training rounds.
        injection_scale (float): Scaling factor for noise injection at junctions.
    """
    modes = ["feature"]  # Modes to test
    results = {}

    # Initialize graph
    graph = Graph()

    # Add generator and discriminator nodes
    generator_node = graph.add_node(node_type="generator")
    discriminator_node = graph.add_node(node_type="discriminator")

    # Add an edge between generator and discriminator
    graph.add_edge(generator_node, discriminator_node)

    # Loss tracking
    gen_losses = []
    disc_losses = []

    # Initialize random input tensor and reference tensor

    input_tensor = torch.rand((1, 3, 32, 32))  # Random input image
    reference_tensor = generate_checkerboard(size=32, tile_size=4)
    
    # Training loop
    for round_num in range(num_rounds):
        # Execute one pass through the graph
        gen_loss, disc_loss = graph.execute(
            mode="feature",
            num_outputs=len(input_tensor),  # You can adjust this as needed
            input_tensor=input_tensor,
            reference_tensor=reference_tensor
        )

        # Store losses
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)

        # Optionally, print progress
        if (round_num + 1) % 100 == 0:
            print(f"  Completed {round_num + 1}/{num_rounds} rounds")

    # Visualize gradient statistics
    visualize_results(results, num_rounds)

    # Visualize loss statistics
    visualize_loss(gen_losses, disc_losses)

    # Generate and visualize checkerboard
    checkerboard = generate_checkerboard(size=32, tile_size=4)
    generated_output, _ = generator_node.process_task(checkerboard)
    visualize_checkerboard(checkerboard, generated_output)
    input_tensor = torch.rand((1, 3, 32, 32))
    generated_output, _ = generator_node.process_task(input_tensor)
    visualize_checkerboard(input_tensor, generated_output)

def visualize_loss(gen_losses, disc_losses):
    """
    Visualizes generator and discriminator loss over time.

    Args:
        gen_losses (list): List of generator losses over time.
        disc_losses (list): List of discriminator losses over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label="Generator Loss", alpha=0.7)
    plt.plot(disc_losses, label="Discriminator Loss", alpha=0.7)
    plt.title("Loss Statistics Over Time")
    plt.xlabel("Training Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def visualize_results(results, num_rounds):
    """
    Visualizes the gradient statistics for each mode as heatmaps.

    Args:
        results (dict): Dictionary of gradient statistics for each mode.
        num_rounds (int): Number of training rounds.
    """
    for mode, data in results.items():
        generator_gradients = data["generator_gradients"]
        discriminator_gradients = data["discriminator_gradients"]

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Mode: {mode.capitalize()}")

        # Generator gradients
        plt.subplot(1, 2, 1)
        plt.imshow(generator_gradients, aspect="auto", cmap="viridis", extent=[0, num_rounds, 0, generator_gradients.shape[0]])
        plt.colorbar(label="Gradient Norm")
        plt.title("Generator Gradients")
        plt.xlabel("Training Round")
        plt.ylabel("Layer")

        # Discriminator gradients
        plt.subplot(1, 2, 2)
        plt.imshow(discriminator_gradients, aspect="auto", cmap="viridis", extent=[0, num_rounds, 0, discriminator_gradients.shape[0]])
        plt.colorbar(label="Gradient Norm")
        plt.title("Discriminator Gradients")
        plt.xlabel("Training Round")
        plt.ylabel("Layer")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    main()
