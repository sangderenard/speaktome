import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import time
import threading
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
import pygame
import json

BUFFER_SIZE = 10  # Define buffer size for references

# Constants for maximum configurations
MAX_BATCH_SIZE = 64
MAX_ACCUMULATION_STEPS = 32
MAX_GRADIENT_CLIPPING = 5.0

class Reference:
    def __init__(self, reference_tensor, reference_parameters, reference_function):
        self.id = uuid.uuid4().hex
        self.reference_tensor = reference_tensor
        self.reference_function = reference_function
        self.reference_parameters = reference_parameters
        self.output_buffer = []  # Stores recent node outputs
        self.average_output = None
        self.lock = threading.Lock()  # To handle concurrent access

    def produce_tracked_reference_formation(self, input_tensor):
        return self.reference_function(**self.reference_parameters, input_tensor=input_tensor)

    def compute_contributor_distribution(self):
        """
        Calculate the contribution of each node to the averaged output as percentages.
        """
        if not self.output_buffer:
            return {}
        contributions = {}
        for tensor in self.output_buffer:
            node_id = tensor.get("node_id", "unknown")
            norm = tensor["output"].norm().item()
            contributions[node_id] = contributions.get(node_id, 0) + norm
        total_norm = sum(contributions.values())
        return {k: v / total_norm * 100 for k, v in contributions.items()}

    def add_output(self, output_tensor):
        """
        Add an output to the buffer and update the running average.
        :param output_tensor: A dictionary containing "output" and optionally "loss".
        """
        with self.lock:
            # Add output tensor to the buffer
            self.output_buffer.append(output_tensor)

            # Remove excess items if buffer exceeds size
            if len(self.output_buffer) > BUFFER_SIZE:
                self.output_buffer.pop(0)

            # Update running averages
            self.calculate_average()

    def calculate_average(self):
        """
        Calculates the average of the tensors in the buffer while retaining metadata in the buffer.
        """
        if self.output_buffer:
            # Extract the 'output' tensors from the dictionaries in the buffer
            tensors = [item["output"] for item in self.output_buffer if "output" in item]
            stacked = torch.stack(tensors)  # Perform tensor stacking
            self.average_output = torch.mean(stacked, dim=0)



class DelegatorNode:
    def __init__(self, graph, gan_nodes, override_config=None):
        """
        Delegator Node to assign tasks to GAN nodes.
        :param graph: The graph structure managing all nodes.
        :param gan_nodes: List of GAN nodes to manage.
        :param override_config: Optional configuration overrides for GAN nodes.
        """
        self.id = uuid.uuid4().hex
        self.graph = graph
        self.gan_nodes = gan_nodes
        self.override_config = override_config  # Optional configuration overrides for GAN nodes
        self.task_weights = {node.id: 1.0 for node in gan_nodes}  # Equal weights initially
        self.lock = threading.Lock()

        # Learned weights for normalization
        self.weight_batch = {node.id: 1.0 for node in gan_nodes}
        self.weight_accum = {node.id: 1.0 for node in gan_nodes}
        self.weight_clip = {node.id: 1.0 for node in gan_nodes}

        # Cache to store hashed references and their statistics
        self.reference_cache = {}

    def assign_tasks(self, input_tensor):
        """
        Assign tasks to GAN nodes based on task weights and learned parameters.
        :param input_tensor: The input tensor for task processing.
        """
        with self.lock:
            # Normalize task weights
            total_weight = sum(self.task_weights.values())
            normalized_weights = {k: v / total_weight for k, v in self.task_weights.items()}

            outputs = []
            for node in self.gan_nodes:
                if not node.brain.targets:  # Skip nodes without targets
                    print(f"Node {node.id} has no targets. Skipping.")
                    continue

                # Apply override configurations if provided
                if self.override_config:
                    node.config.update(self.override_config)

                weight = normalized_weights[node.id]
                task_input = input_tensor * weight  # Scale input by weight

                # Determine batch size, accumulation steps, and clipping norm based on learned weights
                batch_size = int(min(MAX_BATCH_SIZE, self.weight_batch[node.id] * MAX_BATCH_SIZE))
                accumulation_steps = int(min(MAX_ACCUMULATION_STEPS, self.weight_accum[node.id] * MAX_ACCUMULATION_STEPS))
                clipping_norm = float(min(MAX_GRADIENT_CLIPPING, self.weight_clip[node.id] * MAX_GRADIENT_CLIPPING))

                # Ensure integer conventions
                batch_size = max(1, batch_size)
                accumulation_steps = max(1, accumulation_steps)
                clipping_norm = max(0.1, clipping_norm)

                # Assign target loss reduction
                target_loss_reduction = node.brain.target_loss_reduction

                # Use the first available target for task processing
                target_id = list(node.brain.targets.keys())[0]

                # Process generator and discriminator with assigned configurations
                output_g, loss_g = node.process_task(
                    task_input, target_id, focus="generator",
                    batch_size=batch_size, accumulation_steps=accumulation_steps,
                    clipping_norm=clipping_norm, target_loss_reduction=target_loss_reduction["generator"]
                )
                output_d, loss_d = node.process_task(
                    task_input, target_id, focus="discriminator",
                    batch_size=batch_size, accumulation_steps=accumulation_steps,
                    clipping_norm=clipping_norm, target_loss_reduction=target_loss_reduction["discriminator"]
                )

                # Collect outputs and losses
                outputs.append((node.id, {'generator': output_g, 'discriminator': output_d}, {'generator': loss_g, 'discriminator': loss_d}))

                # Check if tasks are completed and handle caching
                if node.task_completed:
                    # Hash the reference tensor
                    reference = node.brain.targets[target_id]
                    reference_hash = hashlib.sha256(reference.average_output.numpy().tobytes()).hexdigest()
                    # Log statistics with the hash
                    self.reference_cache[reference_hash] = {
                        'statistics': node.brain.generate_report(include_gradients=False),
                        'timestamp': time.time()
                    }
                    print(f"Reference hashed and logged: {reference_hash}")

            return outputs

    def update_weights(self, feedback):
        """
        Update learned weights based on feedback from GAN nodes to achieve target loss reductions.
        :param feedback: Dictionary of feedback from nodes (e.g., loss reductions).
        """
        with self.lock:
            for node_id, losses in feedback.items():
                node = self.graph.nodes.get(node_id)
                if not node:
                    continue

                # Update weights based on how well targets are achieved
                # For generator
                loss_reduction_g = losses.get('generator', 0)
                target_g = node.brain.target_loss_reduction.get('generator', 0.01)
                if loss_reduction_g <= target_g:
                    self.weight_batch[node_id] = min(self.weight_batch[node_id] * 1.05, 2.0)  # Increase weight
                    self.weight_accum[node_id] = min(self.weight_accum[node_id] * 1.05, 2.0)
                    self.weight_clip[node_id] = min(self.weight_clip[node_id] * 1.05, 2.0)
                else:
                    self.weight_batch[node_id] = max(self.weight_batch[node_id] * 0.95, 0.5)  # Decrease weight
                    self.weight_accum[node_id] = max(self.weight_accum[node_id] * 0.95, 0.5)
                    self.weight_clip[node_id] = max(self.weight_clip[node_id] * 0.95, 0.5)

                # For discriminator
                loss_reduction_d = losses.get('discriminator', 0)
                target_d = node.brain.target_loss_reduction.get('discriminator', 0.01)
                if loss_reduction_d <= target_d:
                    self.weight_batch[node_id] = min(self.weight_batch[node_id] * 1.05, 2.0)  # Increase weight
                    self.weight_accum[node_id] = min(self.weight_accum[node_id] * 1.05, 2.0)
                    self.weight_clip[node_id] = min(self.weight_clip[node_id] * 1.05, 2.0)
                else:
                    self.weight_batch[node_id] = max(self.weight_batch[node_id] * 0.95, 0.5)  # Decrease weight
                    self.weight_accum[node_id] = max(self.weight_accum[node_id] * 0.95, 0.5)
                    self.weight_clip[node_id] = max(self.weight_clip[node_id] * 0.95, 0.5)


class NodeBrain:
    def __init__(self, node_id, node):
        self.node_id = node_id
        self.node = node  # Reference to the parent Node
        self.targets = {}  # Key: target_id, Value: Reference object
        self.best_losses = []  # Stores best loss metadata
        self.local_weights = {}  # Stores local NN weights
        self.edge_weights = {}  # Personal edge weights for the GNN
        self.active_tasks = {}  # Current tasks for targets
        self.history = []  # Log of completed tasks
        self.loss_history = []
        self.target_loss_reduction = {'generator': 0.0005, 'discriminator': 0.0005}  # Dynamic targets

    def get_moving_average_gradient_norm(self):
        if not self.node.gradient_history:
            return 0
        grad_norms = [entry["grad_norm"] for entry in self.node.gradient_history]
        return sum(grad_norms) / len(grad_norms)

    def get_loss_stats(self):
        if not self.loss_history:
            return 0, 0
        loss_mean = np.mean(self.loss_history)
        loss_std = np.std(self.loss_history)
        return loss_mean, loss_std

    def track_loss(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > BUFFER_SIZE:
            self.loss_history.pop(0)

    def add_target(self, reference):
        """Add a new target for the node to manage."""
        self.targets[reference.id] = reference

    def remove_target(self, reference_id):
        """Remove a target from the node's management."""
        if reference_id in self.targets:
            del self.targets[reference_id]

    def update_best_losses(self, loss, output, target_id):
        """Update best loss history for a specific target."""
        self.best_losses.append({
            "loss": loss,
            "output": output.detach().cpu(),
            "timestamp": time.time(),
            "target_id": target_id
        })
        self.best_losses = sorted(self.best_losses, key=lambda x: x["loss"])[:BUFFER_SIZE]  # Keep top results

    def snapshot_weights(self, generator, discriminator, edges):
        """Capture local and edge weights."""
        self.local_weights = {
            "generator": {name: param.clone() for name, param in generator.named_parameters()},
            "discriminator": {name: param.clone() for name, param in discriminator.named_parameters()}
        }
        self.edge_weights = {edge.id: edge.weight for edge in edges}

    def log_task(self, task_id, result):
        """Log the completion of a task."""
        self.history.append({"task_id": task_id, "result": result, "timestamp": time.time()})

    def generate_report(self, include_gradients=True):
        """Generate a self-report of the node's current state, including gradient statistics."""
        report = {
            "node_id": self.node_id,
            "targets": list(self.targets.keys()),
            "best_losses": self.best_losses,
            "local_weights": list(self.local_weights.keys()),
            "edge_weights": self.edge_weights,
            "active_tasks": self.active_tasks
        }

        if include_gradients:
            report["gradient_statistics"] = self.node.get_gradient_statistics()

        return report


class Node:
    def __init__(self, features=None, config=None):
        """
        Initializes a node with both generator and discriminator models.
        :param features: Optional features for the node.
        :param config: Configuration dictionary for gradient accumulation and clipping.
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
        self.generator_grad_history = {}  # Layer-wise gradient history for generator
        self.discriminator_grad_history = {}  # Layer-wise gradient history for discriminator
        self.gradient_buffer_size = BUFFER_SIZE  # Sliding window size

        self.lock = threading.Lock()

        # Configuration for gradient accumulation and clipping
        self.config = config or {
            "gradient_accumulation_steps_gen": 10,
            "gradient_clipping_gen": 100.0,
            "gradient_accumulation_steps_disc": 10,
            "gradient_clipping_dis": 100.0,
        }

        # Delegate brain for organizational logic
        self.brain = NodeBrain(node_id=self.id, node=self)

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
        self.local_weights = {}
        self.edge_weights = {}
        self.gradient_clipping_max_norm = 1.0  # Example parameter
        self.gradient_clipping_function = None  # Optional: Custom clipping function
        self.attach_hooks()

    def _generate_id(self):
        return uuid.uuid4().hex

    def get_gradient_statistics(self):
        stats = {"generator": {}, "discriminator": {}}
        for layer_name, history in self.generator_grad_history.items():
            stats["generator"][layer_name] = {
                "mean": np.mean(history) if history else -1,
                "std": np.std(history) if history else -1,
                "latest": history[-1] if history else -1,
            }
        for layer_name, history in self.discriminator_grad_history.items():
            stats["discriminator"][layer_name] = {
                "mean": np.mean(history) if history else -1,
                "std": np.std(history) if history else -1,
                "latest": history[-1] if history else -1,
            }
        return stats

    def attach_hooks(self):
        """Attach hooks to generator and discriminator layers for tracking and clipping gradients."""
        for name, param in self.generator.named_parameters():
            if param.requires_grad:
                param.register_hook(self._gradient_clipping_hook("generator"))
        for name, param in self.discriminator.named_parameters():
            if param.requires_grad:
                param.register_hook(self._gradient_clipping_hook("discriminator"))

    def _gradient_clipping_hook(self, model_type):
        def hook(grad):
            max_norm = self.config[f"gradient_clipping_{model_type[:3]}"]
            grad_norm = grad.norm().item()
            # Update the correct history
            if model_type == "generator":
                self.generator_grad_history.setdefault(model_type, []).append(grad_norm)
            elif model_type == "discriminator":
                self.discriminator_grad_history.setdefault(model_type, []).append(grad_norm)

            # Perform gradient clipping
            if self.gradient_clipping_function:
                grad = self.gradient_clipping_function(grad)
            elif grad_norm > max_norm:
                grad = grad * (max_norm / grad_norm)

            return grad

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
    def process_task(self, input_tensor, target_id, focus="generator", batch_size=1, accumulation_steps=1, clipping_norm=1.0, target_loss_reduction=0.01):
        """
        Process a task with specified configurations.
        
        This method executes a task for either the generator or discriminator model, performing
        forward passes, loss calculations, and backpropagation over a specified number of 
        accumulation steps. Outputs and losses are averaged to generate a compressed evaluation
        for a hyperspecific task.

        :param input_tensor: The input tensor for task processing.
        :param target_id: The ID of the target reference.
        :param focus: Whether to focus on 'generator' or 'discriminator'.
        :param batch_size: Number of samples in a batch.
        :param accumulation_steps: Number of gradient accumulation steps.
        :param clipping_norm: Gradient clipping norm.
        :param target_loss_reduction: Target loss reduction for this task.
        :return: Tuple of (output_dict, average_loss_reduction)
        """
        # Fetch the reference associated with the task
        reference = self.brain.targets.get(target_id)
        if not reference:
            raise ValueError(f"Target {target_id} not found for Node {self.id}")

        total_loss_reduction = 0.0
        combined_output = None  # Accumulated outputs for averaging

        # Select model, optimizer, and loss function based on the task focus
        optimizer = self.gen_optimizer if focus == "generator" else self.disc_optimizer
        loss_fn = self.gen_loss_fn if focus == "generator" else self.disc_loss_fn
        model = self.generator if focus == "generator" else self.discriminator

        # Zero gradients before beginning the accumulation process
        optimizer.zero_grad()

        for step in range(accumulation_steps):
            # Generate input batch by repeating the input tensor
            batch_input = input_tensor.repeat(batch_size, 1, 1, 1)
            
            # Generate the corresponding reference batch
            batch_reference = reference.produce_tracked_reference_formation(batch_input)
            if isinstance(batch_reference, tuple):  # Extract tensor if batch_reference is a tuple
                batch_reference = batch_reference[0]

            # Forward pass through the model
            output = model(batch_input)
            
            # Compute loss using the specified loss function
            output_mean = output.mean(dim=0, keepdim=True)
            batch_reference_mean = batch_reference.mean(dim=0, keepdim=True)
            loss = loss_fn(output_mean, batch_reference_mean)

            
            # Accumulate the total loss for averaging later
            total_loss_reduction += loss.item()

            # Backpropagation for this step
            loss.backward()

            # Combine outputs for later evaluation
            if combined_output is None:
                combined_output = output.detach()
            else:
                combined_output = torch.cat((combined_output, output.detach()), dim=0)

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_norm)
        optimizer.step()

        # Calculate the average loss over the accumulation steps and batch size
        average_loss_reduction = total_loss_reduction / (accumulation_steps * batch_size)

        # Update the node's brain with the task results
        self.brain.update_best_losses(average_loss_reduction, combined_output, target_id)
        self.brain.track_loss(average_loss_reduction)

        # Prepare the output dictionary containing the averaged outputs
        output_dict = {"node_id": self.id, "output": combined_output.clone(), "loss": average_loss_reduction}

        # Add the task results to the reference for aggregation
        reference.add_output(output_dict)

        # Check if the task meets the target loss reduction threshold
        if average_loss_reduction <= target_loss_reduction:
            self.task_completed = True
            self.brain.history.append({
                "task": focus,
                "loss_reduction": average_loss_reduction,
                "timestamp": time.time(),
                "category": self.categorize_task(average_loss_reduction, focus)
            })

        return output_dict, average_loss_reduction


    def categorize_task(self, loss_reduction, focus):
        """
        Categorize the task based on loss reduction rate.
        :param loss_reduction: The rate of loss reduction achieved.
        :param focus: 'generator' or 'discriminator'.
        :return: Category string.
        """
        if loss_reduction <= 0.02:
            return "High Efficiency"
        elif loss_reduction <= 0.05:
            return "Moderate Efficiency"
        else:
            return "Low Efficiency"

    def replicate(self, graph):
        if self.telomere > 5 and graph.food_supply >= graph.food_cost_per_replicate:
            child = Node(features=self.features.clone(), config=self.config.copy())
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

    def snapshot_state(self):
        """Snapshot the current state into the NodeBrain."""
        self.brain.snapshot_weights(self.generator, self.discriminator, self.edges)

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

    def add_node(self, features=None, config=None):
        node = Node(features=features, config=config)
        self.nodes[node.id] = node
        return node

    def add_edge(self, source, target, weight=1.0):
        edge = Edge(source, target, weight=weight)
        self.edges[edge.id] = edge
        source.edges.append(edge)
        return edge

    def add_reference(self, reference_tensor, reference_parameters, reference_function):
        reference = Reference(reference_tensor, reference_parameters, reference_function)
        self.references[reference.id] = reference
        self.reference_to_nodes[reference.id] = set()  # Initialize as empty set
        return reference

    def collect_reports(self):
        """Collect reports from all nodes, including gradient statistics."""
        return [node.brain.generate_report(include_gradients=True) for node in self.nodes.values()]

    def display_node_health(self):
        """Aggregate and display node-level health metrics for debugging."""
        reports = self.collect_reports()
        for report in reports:
            print(f"Node {report['node_id']}:")
            grad_stats = report.get("gradient_statistics", {})
            for model_type, layers in grad_stats.items():
                print(f"  {model_type} Layer Gradients:")
                for layer, stats in layers.items():
                    print(f"    Layer {layer}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Latest={stats['latest']:.4f}")

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
        compiled_history = self.compile_gradient_history()
        with open(filename, "w") as f:
            json.dump(compiled_history, f)
        print(f"Gradient history saved to {filename}.")

    def execute(self, input_tensor_shape):
        current_nodes = list(self.nodes.values())

        for node in current_nodes:
            for reference_id, reference in self.references.items():
                # Track node holding the reference
                self.reference_to_nodes[reference_id].add(node.id)
                node.brain.add_target(reference)

                # Process tasks for generator and discriminator
                input_tensor = torch.rand(input_tensor_shape)
                gen_output, gen_loss = node.process_task(input_tensor, reference_id, focus="generator")
                input_tensor = torch.rand(input_tensor_shape)
                disc_output, disc_loss = node.process_task(input_tensor, reference_id, focus="discriminator")

                # Determine the focus based on higher loss
                if gen_loss > disc_loss:
                    node.process_task(input_tensor, reference_id, focus="generator")
                else:
                    node.process_task(input_tensor, reference_id, focus="discriminator")

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

    def execute_with_delegator(self, input_tensor_shape):
        delegator = DelegatorNode(self, list(self.nodes.values()))
        current_nodes = list(self.nodes.values())

        for round_num in range(100):  # Example number of rounds
            input_tensor = torch.rand(input_tensor_shape)

            # Delegator assigns tasks
            task_outputs = delegator.assign_tasks(input_tensor)

            # Collect feedback (e.g., losses) for weight updates
            feedback = {}
            for node_id, outputs, losses in task_outputs:
                feedback[node_id] = losses
            delegator.update_weights(feedback)

            # Log the cached references
            if delegator.reference_cache:
                print(f"Delegator Reference Cache: {delegator.reference_cache}")

            # Placeholder for spectral distribution logic
            # Implement spectral analysis here to adjust task assignments if needed

    def centralize_weight_updates(self):
        """
        Centralizes weight updates using PyTorch Geometric utilities, ignoring pruned nodes.
        """
        edge_index = []
        edge_weights = []

        node_keys = list(self.nodes.keys())
        for edge in self.edges.values():
            if edge.source.id in self.nodes and edge.target.id in self.nodes:
                source_idx = node_keys.index(edge.source.id)
                target_idx = node_keys.index(edge.target.id)
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
        """Prune nodes based on gradient and loss metrics."""
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            report = node.brain.generate_report()
            grad_stats = report.get("gradient_statistics", {})
            loss_mean, loss_std = (
                np.mean([entry["loss"] for entry in node.brain.best_losses]) if node.brain.best_losses else 0,
                np.std([entry["loss"] for entry in node.brain.best_losses]) if node.brain.best_losses else 0,
            )

            # Example pruning condition: High gradients and poor performance
            if any(layer_stats["mean"] > 1.0 for layer_stats in grad_stats.get("generator", {}).values()) and loss_mean > 0.1:
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            print(f"Node {node_id} pruned based on gradient and loss metrics.")

    def replicate_and_grow(self):
        for node in list(self.nodes.values()):
            child = node.replicate(self)
            if child:
                self.add_node(features=child.features, config=child.config.copy())

    def update_reference_timeout(self):
        current_time = time.time()
        for reference_id, reference in self.references.items():
            # Find the latest update time across all nodes for this reference
            latest_update = max([error_info["timestamp"] for error_info in self.get_reference_errors(reference_id)], default=0)
            if current_time - latest_update > self.display_node.request_timeout:
                # Finalize the average if timeout exceeded
                reference.calculate_average()

    def get_reference_errors(self, reference_id):
        errors = []
        for node in self.nodes.values():
            if reference_id in node.reference_errors:
                errors.append(node.reference_errors[reference_id])
        return errors

    def collect_reports(self):
        return [node.brain.generate_report() for node in self.nodes.values()]

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

    def render_gradient_statistics(self):
        """Render detailed gradient statistics for each node."""
        for node_id, node in self.graph.nodes.items():
            grad_stats = node.get_gradient_statistics()
            print(f"Node {node_id}: Gradient Statistics")
            for model_type, layers in grad_stats.items():
                print(f"  {model_type}:")
                for layer, stats in layers.items():
                    print(f"    Layer {layer}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Latest={stats['latest']:.4f}")

    def render_images(self):
        self.screen.fill((0, 0, 0))  # Clear screen
        x, y = 50, 50
        for node_id, node in self.graph.nodes.items():
            # Get the reference outputs to display
            reference_outputs = [
                reference.average_output for reference in self.graph.references.values() if reference.average_output is not None
            ]

            if reference_outputs:
                for output in reference_outputs:
                    metadata = f"Node ID: {node_id[:6]}\n"
                    grad_stats = node.get_gradient_statistics()
                    for model, layers in grad_stats.items():
                        metadata += f"{model.capitalize()}:\n"
                        for layer, stats in layers.items():
                            metadata += (
                                f"  {layer}: Mean={stats['mean']:.4f}, "
                                f"Std={stats['std']:.4f}, "
                                f"Latest={stats['latest']:.4f}\n"
                            )
                    self._blit_image(output, (x, y), label=f"Node {node_id[:6]}", metadata=metadata)
                    x += 400
                    if x > self.screen.get_width() - 250:
                        x = 50
                        y += 250
        pygame.display.flip()

    def _render_node_stats(self, x, y, label, metadata):
        font = pygame.font.SysFont("Arial", 18)
        text_surface = font.render(label, True, (255, 255, 255))
        self.screen.blit(text_surface, (x, y - 20))

        font = pygame.font.SysFont("Arial", 14)
        y_offset = y
        for line in metadata.split("\n"):
            text_surface = font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (x, y_offset))
            y_offset += 20

    def _blit_image(self, tensor, position, label=None, metadata=None):
        img = tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img_surface = pygame.surfarray.make_surface(img)
        img_surface = pygame.transform.scale(img_surface, (240, 240))
        self.screen.blit(img_surface, position)

        # Render label
        if label:
            font = pygame.font.SysFont("Arial", 18)
            text_surface = font.render(label, True, (255, 255, 255))
            self.screen.blit(text_surface, (position[0], position[1] - 20))

        # Render metadata
        if metadata:
            font = pygame.font.SysFont("Arial", 14)
            y_offset = position[1] + 250
            for line in metadata.split("\n"):
                text_surface = font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (position[0], y_offset))
                y_offset += 20

    def handle_events(self):
        """
        Handle PyGame events like quitting.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()


def generate_checkerboard(size, tile_size, color=None, input_tensor=torch.zeros(1)):
    """
    Generates a checkerboard pattern with random colors for each square while respecting gradients.
    """
    # Create the board using an outer tensor operation
    board = torch.zeros((size, size, 3), requires_grad=True)
    mask = torch.zeros((size, size), requires_grad=False)

    # Iterate to set the tile colors
    for i in range(0, size, tile_size):
        for j in range(0, size, tile_size):
            if color is None:
                tile_color = torch.rand(3, requires_grad=True)  # Random color for each square
            else:
                tile_color = color
            # Use broadcasting to apply the color to the tile without in-place operations
            x_end = min(size, i + tile_size)
            y_end = min(size, j + tile_size)
            current_tile_mask = torch.zeros_like(mask)
            current_tile_mask[i:x_end, j:y_end] = 1
            mask = mask + current_tile_mask  # Aggregating tile masks
            board = board + current_tile_mask.unsqueeze(-1) * tile_color.unsqueeze(0).unsqueeze(0)

    # Combine input_tensor scaling without mutating
    scaled_board = board * torch.mean(input_tensor)

    # Reshape and return outputs
    return scaled_board.permute(2, 0, 1).unsqueeze(0), tile_size, color


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
        reference_tensor, reference_size, reference_color = generate_checkerboard(size=32, tile_size=2)
        reference_parameters = {"size": 32, "tile_size": reference_size, "color": reference_color}
        reference = graph.add_reference(reference_tensor, reference_parameters, generate_checkerboard)
        references.append(reference)

    # Assign references to GAN nodes and set dynamic target loss reductions
    for node in graph.nodes.values():
        for reference in references:
            node.brain.add_target(reference)
            # Set dynamic target loss reductions
            node.brain.target_loss_reduction = {
                'generator': 0.0005,       # Example target loss reduction for generator
                'discriminator': 0.0005    # Example target loss reduction for discriminator
            }

    # Initialize the display node
    display_node = DisplayNode(graph, window_size=(1280, 720))
    graph.display_node = display_node  # Link graph to display node

    while display_node.running:
        input_tensor_shape = (1, 3, 32, 32)
        graph.execute_with_delegator(input_tensor_shape)

        # Render images in PyGame
        display_node.handle_events()
        display_node.render_images()

        # Optional: Visualize loss statistics
        # visualize_loss(gen_losses, disc_losses)

        if not display_node.running:
            break

        if (graph.food_supply % 100 == 0):
            print(f"Current Food Supply: {graph.food_supply}")

    pygame.quit()


if __name__ == "__main__":
    main()
