import torch
import random


class FeatureDepthTranslator:
    """
    A class whose sole responsibility is to translate features probabilistically using these functions,
    and then after the feature match is guaranteed, subsequent passes can happen for bytes or bits.
    """

    def __init__(self, bit_pattern_size=8):
        self.bit_pattern_size = bit_pattern_size

    def map_binary_to_int(self, value, num_outputs):
        """
        Maps integers represented by a specified number of bits to a desired range of outputs.
        Ensures uniform probability distribution across the output range.
        """
        max_representable = 2**self.bit_pattern_size
        pigeonhole_size = max_representable / num_outputs
        value = value + 1  # Adjust for 1-based indexing

        # Generate output ranges
        pigeonholes = [
            (i * pigeonhole_size, i * pigeonhole_size + pigeonhole_size)
            for i in range(num_outputs)
        ]

        random_point = value - 0.5 + random.random() - 0.5  # Random point in range

        # Determine which range (pigeonhole) the point falls into
        for idx, (start, end) in enumerate(pigeonholes):
            idx = idx + 1  # Adjust for 1-based indexing
            if start < random_point < end:
                return idx - 1
            elif random_point in (start, end):
                return random.choice(
                    [idx - 1 if idx > 1 else idx, idx + 1 if idx < len(pigeonholes) else idx]
                ) - 1

    def refine_features(self, input_tensor, num_outputs, mode="feature"):
        """
        First pass: Translates features probabilistically to the desired depth.
        """
        batch_size, *spatial_dims = input_tensor.shape
        flat_input = input_tensor.flatten(start_dim=1)

        output_tensor = torch.zeros((batch_size, num_outputs), device=input_tensor.device)

        for batch_idx in range(batch_size):
            input_indices = range(flat_input.size(1))

            if mode == "feature":
                feature_sum = flat_input[batch_idx].sum().item()
                mapped_bins = [
                    self.map_binary_to_int(feature_sum, num_outputs)
                ] * flat_input.size(1)
            else:
                raise ValueError(f"Unsupported mode for initial pass: {mode}")

            for input_idx, output_idx in zip(input_indices, mapped_bins):
                output_tensor[batch_idx, output_idx] += flat_input[batch_idx, input_idx]

        return output_tensor

    def refine_to_bytes_or_bits(self, input_tensor, num_outputs, mode="byte"):
        """
        Second pass: Refines down further to bytes or bits after features are matched.
        """
        batch_size, *spatial_dims = input_tensor.shape
        flat_input = input_tensor.flatten(start_dim=1)

        output_tensor = torch.zeros((batch_size, num_outputs), device=input_tensor.device)

        for batch_idx in range(batch_size):
            input_indices = range(flat_input.size(1))

            if mode == "byte":
                grouped_indices = [
                    input_indices[i:i + 8] for i in range(0, len(input_indices), 8)
                ]
                mapped_bins = []
                for group in grouped_indices:
                    group_sum = sum(flat_input[batch_idx, idx].item() for idx in group)
                    mapped_bins.append(
                        self.map_binary_to_int(group_sum, num_outputs)
                    )
            elif mode == "bit":
                mapped_bins = [
                    self.map_binary_to_int(idx, num_outputs) for idx in input_indices
                ]
            else:
                raise ValueError(f"Unsupported mode for second pass: {mode}")

            for input_idx, output_idx in zip(input_indices, mapped_bins):
                output_tensor[batch_idx, output_idx] += flat_input[batch_idx, input_idx]

        return output_tensor

class Node:
    def __init__(self, node_type="default", features=None):
        """
        Initializes a node with a specific type (generator, discriminator).

        Args:
            node_type (str): Type of the node ("generator" or "discriminator").
            features (torch.Tensor or dict, optional): Feature tensor or dictionary.
        """
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
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def add_edge(self, edge):
        self.edges.append(edge)


        self.feedback_processor = FeedbackProcessor()  # Add feedback processor

    def process_feedback(self, feedback, mode="feature", num_outputs=None):
        """
        Processes feedback from the discriminator and modifies generator input using the probabilistic mapper.

        Args:
            feedback (torch.Tensor): Scalar feedback from the discriminator.
            mode (str): Mapping mode ('bit', 'byte', 'feature').
            num_outputs (int): Number of output features (bins) for mapping.

        Returns:
            torch.Tensor: Updated generator input features.
        """
        if self.node_type == "generator":
            mapper = ProbabilisticMapper()
            self.features = mapper.map_features(feedback, num_outputs=num_outputs, mode=mode)


    def process_task(self, input_tensor, reference_tensor=None):
        """
        Existing generator processing logic...
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
        elif self.node_type == "discriminator":
            output = self.model(input_tensor)
            loss = self.loss_fn(output, reference_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return output, loss.item()
    def broadcast_feedback_to_features(self, input_tensor):
        """
        Probabilistically broadcast feedback across input features.

        Args:
            input_tensor (torch.Tensor): Generator's input tensor.

        Returns:
            torch.Tensor: Modified input tensor with feedback broadcast.
        """
        feedback = torch.rand(1)  # Example feedback value
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

    def execute(self):
        """Runs tasks dynamically."""
        for node in self.nodes.values():
            if node.node_type == "generator" and node.task_completed:
                node.current_task = None  # Request new task
            elif node.node_type == "discriminator":
                pass  # Logic for discriminator feedback
def main(num_rounds=1000, injection_scale=0.1):
    """
    Runs the network across all mapping modes with noise injection and visualizes statistics.

    Args:
        num_rounds (int): Number of training rounds.
        injection_scale (float): Scaling factor for noise injection at junctions.
    """
    modes = ["bit", "byte", "feature"]  # Modes to test
    results = {}

    # Initialize graph
    graph = Graph()

    # Add generator and discriminator nodes
    generator_node = graph.add_node(node_type="generator")
    discriminator_node = graph.add_node(node_type="discriminator")

    # Add an edge between generator and discriminator
    graph.add_edge(generator_node, discriminator_node)

    # Initialize Probabilistic Mapper
    mapper = ProbabilisticMapper()

    # Iterate over each mode
    for mode in modes:
        print(f"Running mode: {mode}")
        generator_gradients = []
        discriminator_gradients = []

        # Initialize random input tensor and reference tensor
        input_tensor = torch.rand((1, 3, 32, 32))  # Random input image
        reference_tensor = torch.rand((1, 3, 32, 32))  # Random target image

        for round_num in range(num_rounds):
            # Generator forward and backward pass
            generated_output, gen_loss = generator_node.process_task(input_tensor, reference_tensor)

            # Noise injection at the junction
            noise = torch.randn_like(generated_output) * injection_scale
            noisy_output = generated_output + noise

            # Map generator output to next features using probabilistic mapper
            mapped_output = mapper.map_features(noisy_output, num_outputs=3, mode=mode)

            # Discriminator forward and backward pass
            disc_output, disc_loss = discriminator_node.process_task(
                mapped_output, torch.ones_like(mapped_output)
            )

            # Track generator gradients
            gen_grad = []
            for param in generator_node.model.parameters():
                if param.grad is not None:
                    gen_grad.append(param.grad.norm().item())
            generator_gradients.append(gen_grad)

            # Track discriminator gradients
            disc_grad = []
            for param in discriminator_node.model.parameters():
                if param.grad is not None:
                    disc_grad.append(param.grad.norm().item())
            discriminator_gradients.append(disc_grad)

        # Store results for the mode
        results[mode] = {
            "generator_gradients": np.array(generator_gradients).T,
            "discriminator_gradients": np.array(discriminator_gradients).T,
        }

    # Visualize results for all modes
    visualize_results(results, num_rounds)


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

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
