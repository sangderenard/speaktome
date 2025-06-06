import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np

import hashlib
import time
class DiagnosticWrapper:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - {self.name} - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)  # Adjust granularity here
        return logger

    def __call__(self, *args, **kwargs):
        # Log input
        self.logger.debug(f"Input Args: {[(arg.shape if isinstance(arg, torch.Tensor) else type(arg)) for arg in args]}")
        try:
            # Execute wrapped model
            output = self.model(*args, **kwargs)
            # Log output
            self.logger.debug(f"Output: {output}")
            return output
        except Exception as e:
            self.logger.error(f"Exception in {self.name}: {e}")
            raise e

    def __getattr__(self, attr):
        return getattr(self.model, attr)

class RollingCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = []
        self.metadata_cache = []
        self.hash_history = []  # Stores (hash, timestamp) tuples

    def _generate_hash(self, tensor):
        """
        Generate a hash for the tensor.
        """
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()

    def _add_hash_event(self, hash_value):
        """
        Add a hash event with a timestamp to the history.
        """
        timestamp = time.time()
        self.hash_history.append((hash_value, timestamp))
        if len(self.hash_history) > self.max_size:
            self.hash_history = self.hash_history[-self.max_size:]  # Keep history within max size
    def clone(self):
        """
        Create a new RollingCache instance with all tensors cloned.
        """
        cloned_cache = RollingCache(self.max_size)
        for item, metadata in zip(self.cache, self.metadata_cache):
        #    print(f"type of item being cloned: {type(item)}")
        #    print(f"contents of item being cloned: {item}")
            
            cloned_cache.cache.append(item.clone())
            cloned_cache.metadata_cache.append(metadata)  # Metadata is not deep-copied
        cloned_cache.hash_history = self.hash_history.copy()  # Copy hash history
        return cloned_cache

    def detach(self):
        """
        Create a new RollingCache instance with all tensors detached.
        """
        detached_cache = RollingCache(self.max_size)
        for item, metadata in zip(self.cache, self.metadata_cache):
            detached_cache.cache.append(item.detach())
            detached_cache.metadata_cache.append(metadata)  # Metadata is not deep-copied
        detached_cache.hash_history = self.hash_history.copy()  # Copy hash history
        return detached_cache
    def add(self, item, metadata, isbatch=True):
        #if not isinstance(metadata, list):
        #    print(type(metadata))
        #    exit()
        #if item.shape[0] > 10:
        #    exit()
        """
        Add tensors and metadata to the cache.
        """
        #if item.shape[0] != len(metadata):
        #    print("Metadata corruption")
        #for metadat in metadata:
        #    print(f"Metadata ID: {id(metadat)}")
        if isbatch:
            for i in range(item.shape[0]):
                self.cache.append(item[i])

                self.metadata_cache.append(metadata[i])
                hash_value = self._generate_hash(item[i])
                self._add_hash_event(hash_value)
        else:
            self.cache.append(item)
            self.metadata_cache.append(metadata[0])
            hash_value = self._generate_hash(item)
            self._add_hash_event(hash_value)

        # Prune to max size
        if len(self.cache) > self.max_size:
            self.cache = self.cache[-self.max_size:]
            self.metadata_cache = self.metadata_cache[-self.max_size:]
            
            



    def get_all(self):
        """
        Retrieve all tensors and their metadata.
        """
        return list(self.cache), list(self.metadata_cache)

    def clear(self):
        """
        Clear the cache and metadata.
        """
        self.cache.clear()
        self.metadata_cache.clear()
        self.hash_history.clear()

    def get_hash_history(self):
        """
        Retrieve the hash history.
        """
        return self.hash_history

import sys
def handle_quit_events():
    """
    Handles pygame quit events to ensure graceful exit.
    Call this function inside your game loop.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Window close button
            print("Exiting game (window close).")
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:  # Key press
            if event.key == pygame.K_ESCAPE:  # Escape key to quit
                print("Exiting game (ESC key pressed).")
                pygame.quit()
                sys.exit()

# Predictor Network: 2D Conv to 2 1D values output
class PredictorNetwork(nn.Module):
    def __init__(self, input_channels, input_height, input_width, conv_height, conv_width, stride=1, dilation=1):
        super(PredictorNetwork, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=1,  # Single feature map for detecting density changes
            kernel_size=(conv_height, conv_width),
            stride=stride,
            dilation=dilation,
            padding=0  # No padding
        )

        # Dynamically calculate the flattened size after convolution
        self.flatten_size = self._calculate_flatten_size(
            input_height, input_width, conv_height, conv_width, stride, dilation
        )

        # Fully connected layer to reduce to two float parameters
        self.fc = nn.Linear(self.flatten_size, 2)

    def forward(self, x, metadata):
        # Apply convolution
        #if x.shape[0] > 10:
        # #   print("processing too many inputs in predictor")
        #    exit()
        x = F.relu(self.conv(x))

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)

        # Pass through the fully connected layer
        x = self.fc(x)
        return (x, metadata), (None,None), (None,None)

    @staticmethod
    def _calculate_flatten_size(input_height, input_width, kernel_height, kernel_width, stride, dilation):
        """
        Calculate the size of the flattened tensor after applying a convolution.
        """
        H_out = ((input_height - dilation * (kernel_height - 1) - 1) // stride) + 1
        W_out = ((input_width - dilation * (kernel_width - 1) - 1) // stride) + 1
        return H_out * W_out * 1  # Multiply by the number of output channels (always 1 here)
# Reproducer Network: 2 1D input to custom 2D output
class ReproducerNetwork(nn.Module):
    def __init__(self, output_channels, height, width):
        super(ReproducerNetwork, self).__init__()
        self.fc = nn.Linear(2, height * width * output_channels)  # Flattened to 2D output
        self.reshape_size = (output_channels, height, width)

    def forward(self, x, metadata):
        #if x.shape[0] > 10:
        #    print("processing too many inputs in reproducer")
        #    exit()
        r_x = F.relu(self.fc(x))
        return_x = r_x.reshape(-1, *self.reshape_size)
        return (return_x, list(metadata)), (None, None), (None, None)

class ClassifierNetwork(nn.Module):
    def __init__(self, input_channels, input_height, input_width, conv_height, conv_width, stride=1, dilation=1):
        super(ClassifierNetwork, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=1,  # Single feature map
            kernel_size=(conv_height, conv_width),
            stride=stride,
            dilation=dilation,
            padding=0  # No padding
        )
        
        # Dynamically calculate the flattened size
        self.flatten_size = self._calculate_flatten_size(
            input_height, input_width, conv_height, conv_width, stride, dilation
        )

        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, 2)

    def forward(self, x, metadata):
        #if x.shape[0] > 10:
        #    print("processing too many inputs in classifier")
        #    exit()
        # Infer batch size and reshape input if needed
        batch_size = x.size(0)
        if x.dim() == 2:  # Input is flat, reshape it
            x_reshape = x.reshape(batch_size, 3, 128, 128)  # Assumes input size is known; adapt as needed
        else:
            x_reshape = x
        x_copy = x_reshape.clone()  # Retain a copy of the input for later use
        rel_x = F.relu(self.conv(x_reshape))
        flat_x = torch.flatten(rel_x, start_dim=1)
        linear_x = self.fc(flat_x)
        print(f"classifier output: linear_x")
        return (linear_x, metadata), (x_copy, metadata), (None, None)

    @staticmethod
    def _calculate_flatten_size(input_height, input_width, kernel_height, kernel_width, stride, dilation):
        """
        Dynamically calculate the flattened size after convolution.
        """
        H_out = ((input_height - dilation * (kernel_height - 1) - 1) // stride) + 1
        W_out = ((input_width - dilation * (kernel_width - 1) - 1) // stride) + 1
        return H_out * W_out * 1  # Multiply by the number of output channels (always 1 here)


class DiscriminatorNetwork(nn.Module):
    def __init__(self, upscale_height, upscale_width, feedback_whitelist):
        super(DiscriminatorNetwork, self).__init__()
        self.upscale_height = upscale_height
        self.upscale_width = upscale_width
        self.fc = nn.Linear(upscale_height * upscale_width * 3, 1)  # Binary classification
        self.feedback_whitelist = feedback_whitelist

    def forward(self, x, metadata):
        # Initial batch size check
        initial_batch_size = x.shape[0]
        #print(f"[Discriminator] Initial input batch size: {initial_batch_size}")
        
        #if initial_batch_size > 10:
        #    print("[Discriminator] Processing too many inputs in discriminator")
        #    exit()
        
        # Unpack tensor dimensions
        batch_size, channels, height, width = x.size()
        #print(f"[Discriminator] Tensor shape: Batch size={batch_size}, Channels={channels}, Height={height}, Width={width}")
        x_residual = x.clone()
        # Upscale to the specified resolution
        x = F.interpolate(x, size=(self.upscale_height, self.upscale_width), mode='bilinear', align_corners=True)
        upscaled_batch_size = x.shape[0]
        #print(f"[Discriminator] After upscaling: {x.shape} (Batch size={upscaled_batch_size})")
        
        # Flatten and apply the classifier
        x = torch.flatten(x, start_dim=1)
        #print(f"[Discriminator] After flattening: {x.shape}")
        
        decision = torch.sigmoid(self.fc(x))  # Shape: [batch_size, 1]
        #rint(f"[Discriminator] Decision tensor shape: {decision.shape}")
        
        # Determine affirm and deny based on decision
        affirm = decision.squeeze(1) >= 0.5
        deny = decision.squeeze(1) < 0.5
        #print(f"[Discriminator] Affirm count: {affirm.sum().item()}, Deny count: {deny.sum().item()}")
        
        # Split the inputs into accepted and rejected based on the decision
        accepted = x_residual[affirm]
        rejected = x_residual[deny]
        #print(f"[Discriminator] Accepted tensor shape: {accepted.shape}")
        #print(f"[Discriminator] Rejected tensor shape: {rejected.shape}")
        
        # Reshape accepted and rejected to original input dimensions
        try:
            accepted = accepted.reshape(-1, channels, height, width)
            #print(f"[Discriminator] Reshaped accepted tensor shape: {accepted.shape}")
        except RuntimeError as e:
            #print(f"[Discriminator] Error reshaping accepted tensor: {e}")
            exit()
        
        try:
            rejected = rejected.reshape(-1, channels, height, width)
            #print(f"[Discriminator] Reshaped rejected tensor shape: {rejected.shape}")
        except RuntimeError as e:
            #print(f"[Discriminator] Error reshaping rejected tensor: {e}")
            exit()
        
        # Filter metadata manually
        #print(type(metadata))
        #print("METADATA TYPE WAS")
        accepted_metadata = [metadata[i] for i in range(len(metadata)) if affirm[i].item()]
        rejected_metadata = [metadata[i] for i in range(len(metadata)) if deny[i].item()]
        #if isinstance(accepted_metadata, dict):
        #    exit()

        #print(f"[Discriminator] Accepted metadata count: {len(accepted_metadata)}")
        #print(f"[Discriminator] Rejected metadata count: {len(rejected_metadata)}")
        
        # Final batch size checks before returning
        total_processed = len(accepted_metadata) + len(rejected_metadata)
        #rint(f"[Discriminator] Total processed metadata count: {total_processed} (Original batch size={batch_size})")
        
        #if total_processed != batch_size:
            #print(f"[Discriminator] Mismatch in metadata count: Processed={total_processed}, Original={batch_size}")
            #exit()
        
        # Returning the outputs
        #print(f"[Discriminator] Forward pass completed successfully with decision: {decision}.\n")
        return (decision, metadata), (accepted, accepted_metadata), (rejected, rejected_metadata)


class LossObject:
    def __init__(self, value, source, metadata, blend_mode='sum', category='general'):
        """
        Parameters:
        - value (torch.Tensor): The loss value.
        - source (str): Identifier for the source node or module.
        - blend_mode (str): 'sum', 'mean', 'max', etc., to combine this loss.
        - category (str): Loss category (e.g., classification, reconstruction).
        """
        self.value = value
        self.source = source
        self.blend_mode = blend_mode
        self.category = category
        self.metadata = metadata
        #if isinstance(value, list):
            #print("Unexpected list loss")
            #e#xit()
import hashlib

def generate_tensor_hash(tensor):
    """
    Generate a unique hash for a tensor based on its contents and shape.
    """
    tensor_bytes = tensor.cpu().numpy().tobytes()
    tensor_shape = str(tensor.shape).encode('utf-8')
    hash_value = hashlib.md5(tensor_bytes + tensor_shape).hexdigest()  # MD5 is fast; replace with SHA256 if needed
    return hash_value
class LossManager:
    def __init__(self, optimizer):
        self.loss_queue = []
        self.optimizer = optimizer  # Global optimizer

    def add_loss(self, loss_obj):
        """
        Add a loss object to the queue.
        """
        print(f"[LossManager] Adding loss from {loss_obj.source} with blend_mode={loss_obj.blend_mode}")
        
        self.loss_queue.append(loss_obj)
    def prune_losses(self):
        pruned_losses = []
        for loss_obj in self.loss_queue:
            if "gaussian_score" in loss_obj.metadata and loss_obj.metadata["gaussian_score"] < 0.5:
                print(f"Pruned invalid loss: {loss_obj.source}")
                continue
            pruned_losses.append(loss_obj)
        self.loss_queue = pruned_losses

    def compute_total_loss(self):
        """
        Aggregate all losses in the queue.
        """
        if not self.loss_queue:
            return torch.tensor(0.0, requires_grad=True)  # Default to zero if no losses

        total_loss = torch.tensor(0.0, requires_grad=True).to(self.loss_queue[0].value.device)
        for loss_obj in self.loss_queue:
            if loss_obj.blend_mode == 'sum':
                total_loss = total_loss + loss_obj.value
            elif loss_obj.blend_mode == 'mean':
                total_loss = total_loss + loss_obj.value.mean()
            elif loss_obj.blend_mode == 'max':
                total_loss = total_loss + loss_obj.value.max()
            else:
                raise ValueError(f"Unknown blend mode: {loss_obj.blend_mode}")
        return total_loss

    def optimize(self):
        #self.prune_losses()
        total_loss = self.compute_total_loss()
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

torch.autograd.set_detect_anomaly(True)
class IsolationMap:
    def __init__(self, max_cache_size=10):
        self.graph = nx.DiGraph()
        self.models = {}
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.channels = {}  # Mapping of channel names to lists of edges
        self.channel_sequence = []  # Sequence of channel activations
        self.current_channel_index = 0  # Index for active channel in sequence

    def add_model(self, name, model):
        # Wrap the model with diagnostics
        wrapped_model = DiagnosticWrapper(name, model)
        self.models[name] = wrapped_model
        self.cache[name] = RollingCache(self.max_cache_size)
        self.graph.add_node(name)

    def connect(self, source, target, modes, channels=None, out_index=None):
        """
        Connect nodes with directional data flow and isolation modes.

        Parameters:
        - source (str): Source node.
        - target (str): Target node.
        - modes (list[int]): Isolation modes.
        - channels (list[str]): Names of channels this edge belongs to.
        """
        self.graph.add_edge(source, target, modes=modes, out_index=out_index)
        if channels:
            for channel in channels:
                if channel not in self.channels:
                    self.channels[channel] = []
                self.channels[channel].append((source, target, out_index))

    def set_channel_sequence(self, sequence):
        """
        Define the sequence of channel activations.

        Parameters:
        - sequence (list[str]): List of channel names in activation order.
        """
        self.channel_sequence = sequence

    def activate_next_channel(self):
        """
        Activate the next channel in the sequence.
        """
        if not self.channel_sequence:
            raise ValueError("Channel sequence is not defined.")
        self.current_channel_index = (self.current_channel_index + 1) % len(self.channel_sequence)
        active_channel = self.channel_sequence[self.current_channel_index]
        
        # Deactivate all edges
        nx.set_edge_attributes(self.graph, False, "active")

        # Activate edges in the current channel
        if active_channel in self.channels:
            for edge_l, edge_r, _ in self.channels[active_channel]:
                if self.graph.has_edge(edge_l, edge_r):
                    self.graph.edges[(edge_l, edge_r)]["active"] = True
                    #print("activated")
                    

    def isolate_tensor(self, tensor, modes):
        if isinstance(tensor, tuple):
            return tuple(self.isolate_tensor(t, modes) for t in tensor)  # Remove dim=1
        result = tensor
        for mode in modes:
            if mode == 0:
                result = result.detach().clone().requires_grad_()
            elif mode == 1:
                result = result.clone()
            elif mode == 2:
                result = result.detach()
            elif mode == 3:
                result = torch.tensor(result.cpu().numpy(), device=tensor.device)
            else:
                raise ValueError(f"Unknown isolation mode: {mode}")
        return result

    def forward(self, input_tensor, start_node):
        """
        Forward pass through the IsolationMap with metadata propagation.
        """
        #print(f"[IsolationMap] Starting forward pass from node: {start_node}")
        #print(f"[IsolationMap] Input tensor shape: {input_tensor.shape}")
        # Prepopulate return objects with empty lists
        node_loss_outputs = {node: [] for node in self.graph.nodes}
        garbage_outputs = {node: [] for node in self.graph.nodes}
        outputs = {node: [] for node in self.graph.nodes}
        active_edges = [(u, v) for u, v, attrs in self.graph.edges(data=True) if attrs.get("active", False)]
        print(active_edges)
        


        if start_node not in self.cache or not self.cache[start_node].cache:
            #print(f"[IsolationMap] Adding initial input to cache for node {start_node}")
            initial_metadata = {"source": "input", "generation_parameters": {}, "evaluation_scores": {}, "tag": "unknown"}
            self.cache[start_node].add(input_tensor, [initial_metadata] * input_tensor.size(0))

        for source, target in active_edges:
            if source not in self.cache or not self.cache[source].cache:
                print(f"[IsolationMap] No cached inputs for {source}. Skipping.")
                continue
            #exit()
            cached_inputs, metadata_inputs = self.cache[source].get_all()
            print(f"[IsolationMap] Cached inputs for {source}: {[t.shape for t in cached_inputs]}")
            #print(f"[IsolationMap] Cached metadata inputs for {source}: {[t for t in metadata_inputs]}")

            model = self.models[source]
            batched_inputs = torch.stack(cached_inputs, dim=0)
            #print(f"[IsolationMap] Batched inputs for {source}: {batched_inputs.shape}")

            edge_attrs = self.graph.edges[source, target]
            out_index = edge_attrs.get("out_index", (0,0))

            # Node output
            #print(f"batched inputs shape: {batched_inputs.shape}")
            gross_output = model(batched_inputs, metadata_inputs)
            #print(gross_output)
            #print(out_index[0])
            
            node_output, output_metadata = gross_output[out_index[0]]
            
            node_loss_outputs[source].append(gross_output[out_index[1]])
            if len(out_index) > 2:
                garbage_outputs[source].append(gross_output[out_index[2]])
            #print(f"[IsolationMap] Node output from {source} -> {target}: {node_output.shape if isinstance(node_output, torch.Tensor) else type(node_output)}")

            # Isolate and cache
            modes = self.graph.edges[source, target]["modes"]
            isolated_output = self.isolate_tensor(node_output, modes)
            print(f"[IsolationMap] Isolated output for {target}: {isolated_output.shape if isinstance(isolated_output, torch.Tensor) else type(isolated_output)}")

            if target not in self.cache:
                self.cache[target] = RollingCache(self.max_cache_size)
            self.cache[target].add(isolated_output, output_metadata)
            
            outputs[source].append((isolated_output, output_metadata))
            if source == "Descriminator":
                print(f"discriminator at end of isolation map forward: {outputs['Descriminator']}")



        return outputs, node_loss_outputs, garbage_outputs


import pygame
def visualize_loss_map(loss_manager, screen_width=800, screen_height=600):
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Loss Visualization")
    screen.fill((0, 0, 0))

    font = pygame.font.Font(None, 24)
    y_offset = 10

    for loss_obj in loss_manager.loss_queue:
        loss_text = f"{loss_obj.source} [{loss_obj.category}]: {loss_obj.value.item():.4f}"
        text_surface = font.render(loss_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 30

    pygame.display.flip()
def visualize_combined(cache, loss_manager, screen_width=800, screen_height=600, tile_size=50):
    """
    Visualize the rolling cache and the loss map side by side.

    Parameters:
    - cache: RollingCache object containing tensors.
    - loss_manager: LossManager object containing loss data.
    - screen_width: Width of the pygame window.
    - screen_height: Height of the pygame window.
    - tile_size: Size of each tile to display tensor data.
    """
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Cache and Loss Visualization")
    screen.fill((0, 0, 0))

    # Display rolling cache on the left
    cached_items, metadata = cache.get_all()
    rows = screen_height // tile_size
    cols = (screen_width // 2) // tile_size  # Left half of the screen
    tile_count = 0


    for i, tensor in enumerate(cached_items):
        if len(tensor.size()) == 2:
            tensor_np = tensor.clone().detach().unsqueeze(0).cpu().numpy()
        elif len(tensor.size()) == 3 and tensor.size(0) in (1, 3):
            tensor_np = tensor.clone().detach().cpu().numpy()
        else:
            continue

        tensor_np = tensor_np.transpose(1, 2, 0)

        if tensor_np.shape[2] == 1:
            tensor_np = np.repeat(tensor_np, 3, axis=2)

        tensor_min = tensor_np.min()
        tensor_max = tensor_np.max()
        if tensor_max > tensor_min:
            tensor_np = ((tensor_np - tensor_min) / (tensor_max - tensor_min) * 255).astype(np.uint8)
        else:
            tensor_np = np.zeros_like(tensor_np, dtype=np.uint8)

        if tile_count >= rows * cols:
            break

        tensor_resized = pygame.surfarray.make_surface(tensor_np)
        row, col = divmod(tile_count, cols)
        x, y = col * tile_size, row * tile_size
        tensor_resized = pygame.transform.scale(tensor_resized, (tile_size, tile_size))
        screen.blit(tensor_resized, (x, y))
        tile_count += 1

    # Display loss map on the right
    font = pygame.font.Font(None, 24)
    y_offset = 10
    x_offset = screen_width // 2 + 10  # Right half of the screen

    for loss_obj in loss_manager.loss_queue:
        loss_text = f"{loss_obj.source} [{loss_obj.category}]: {loss_obj.value.item():.4f}"
        text_surface = font.render(loss_text, True, (255, 255, 255))
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += 30

    pygame.display.flip()

def dump_cache_to_pygame(cache, screen_width=800, screen_height=600, tile_size=50):
    """
    Dumps the contents of the cache onto a pygame screen, tiling the screen.
    Only displays tensors with valid texture data (2D or 3D with channels).
    
    Parameters:
    - cache: RollingCache object containing tensors.
    - screen_width: Width of the pygame window.
    - screen_height: Height of the pygame window.
    - tile_size: Size of each tile to display tensor data.
    """
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Cache Visualization")
    screen.fill((0, 0, 0))

    # Get all cached tensors
    cached_items = cache.get_all()
    if not cached_items:
        print("Cache is empty!")
        return

    # Normalize and convert tensors for display
    rows = screen_height // tile_size
    cols = screen_width // tile_size
    tile_count = 0

    for i, tensor in enumerate(cached_items):
        #print(f"Processing tensor with shape: {tensor.shape}")
        # Check if the tensor is 2D or 3D with a valid channel dimension
        if len(tensor.size()) == 2:
            # For 2D tensors, add a singleton dimension for channel
            tensor_np = tensor.clone().detach().unsqueeze(0).cpu().numpy()  # Shape: (1, H, W)
        elif len(tensor.size()) == 3 and tensor.size(0) in (1, 3):  # Valid channel dim: 1 (grayscale) or 3 (RGB)
            tensor_np = tensor.clone().detach().cpu().numpy()  # Shape: (C, H, W)
        else:
            #print(f"Skipping tensor at index {i} with shape {tensor.size()} - Invalid texture data.")
            continue

        # Permute to (H, W, C) for pygame
        tensor_np = tensor_np.transpose(1, 2, 0)

        # Handle grayscale (1-channel) by converting to RGB-like
        if tensor_np.shape[2] == 1:  # If single channel
            tensor_np = np.repeat(tensor_np, 3, axis=2)  # Convert to (H, W, 3)

        # Normalize tensor data to [0, 255]
        tensor_min = tensor_np.min()
        tensor_max = tensor_np.max()
        if tensor_max > tensor_min:  # Avoid division by zero
            tensor_np = ((tensor_np - tensor_min) / (tensor_max - tensor_min) * 255).astype(np.uint8)
        else:
            tensor_np = np.zeros_like(tensor_np, dtype=np.uint8)

        if tile_count >= rows * cols:
            break  # Prevent overflow if screen space is exhausted

        # Resize tensor to fit tile
        tensor_resized = pygame.surfarray.make_surface(tensor_np)

        # Determine tile position
        row, col = divmod(tile_count, cols)
        x, y = col * tile_size, row * tile_size

        # Blit to screen
        tensor_resized = pygame.transform.scale(tensor_resized, (tile_size, tile_size))
        screen.blit(tensor_resized, (x, y))

        tile_count += 1

    # Update display
    pygame.display.flip()

import functools

class DiagnosticWrapper:
    def __init__(self, name, wrapped_object):
        self.name = name
        self.wrapped_object = wrapped_object

    def __call__(self, *args, **kwargs):
        #print(f"[DiagnosticWrapper] Calling {self.name} with:")
        #print(f"  args: {[arg.shape if isinstance(arg, torch.Tensor) else type(arg) for arg in args]}")
        #print(f"  kwargs: {kwargs}")

        try:
            result = self.wrapped_object(*args, **kwargs)
            #if isinstance(result, tuple):
            #    print(f"[DiagnosticWrapper] {self.name} returned a tuple with shapes: {[r.shape if isinstance(r, torch.Tensor) else type(r) for r in result]}")
            #elif isinstance(result, torch.Tensor):
            #    print(f"[DiagnosticWrapper] {self.name} returned a tensor with shape: {result.shape}")
            #else:
                #print(f"[DiagnosticWrapper] {self.name} returned type: {type(result)}")
            return result
        except Exception as e:
            #print(f"[DiagnosticWrapper] ERROR in {self.name}: {e}")
            raise

    def __getattr__(self, attr):
        return getattr(self.wrapped_object, attr)
def compute_mean_error(predicted, target):
    """
    Compute the mean error between the predicted and target centers.
    """
    pred_center = torch.mean(predicted, dim=(1, 2))
    target_center = torch.mean(target, dim=(1, 2))
    return torch.norm(pred_center - target_center, dim=1).mean()

def compute_std_dev_error(predicted, target):
    """
    Compute the standard deviation error between the predicted and target distributions.
    """
    pred_std = torch.std(predicted, dim=(1, 2))
    target_std = torch.std(target, dim=(1, 2))
    return torch.norm(pred_std - target_std, dim=1).mean()

def compute_shape_agreement(predicted, target):
    """
    Measure how well the predicted texture matches the target Gaussian using MSE.
    """
    return F.mse_loss(predicted, target)

def compute_confidence_score(predicted, gaussian_target, noise_target):
    """
    Compute a confidence score for whether the predicted distribution matches the Gaussian target.
    """
    gaussian_similarity = -F.mse_loss(predicted, gaussian_target)
    noise_similarity = -F.mse_loss(predicted, noise_target)
    return torch.sigmoid(gaussian_similarity - noise_similarity)

def compute_reproducibility(predicted, original_input):
    """
    Evaluate how well the network reproduces the original input's characteristics.
    """
    return F.mse_loss(predicted, original_input)
def visualize_combined_with_metrics(cache, loss_manager, metrics, screen_width=800, screen_height=600, tile_size=50):
    """
    Visualize rolling cache, loss map, and computed metrics.
    """
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Cache, Loss, and Metrics Visualization")
    screen.fill((0, 0, 0))

    # Display cache on the left
    #visualize_cache_part(cache, screen, tile_size)

    # Display losses and metrics on the right
    font = pygame.font.Font(None, 24)
    y_offset = 10
    x_offset = screen_width // 2 + 10  # Right half of the screen

    # Display losses
    for loss_obj in loss_manager.loss_queue:
        loss_text = f"{loss_obj.source} [{loss_obj.category}]: {loss_obj.value.item():.4f}"
        text_surface = font.render(loss_text, True, (255, 255, 255))
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += 30

    # Display metrics
    for metric_name, metric_value in metrics.items():
        metric_text = f"{metric_name}: {metric_value:.4f}"
        text_surface = font.render(metric_text, True, (200, 200, 255))
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += 30

    pygame.display.flip()
def generate_center_weighted_gaussian(resolution=128.0, mu=None, sigma=16.0, channels=3):
    """
    Generate a high-accuracy center-weighted Gaussian distribution texture.
    
    Parameters:
    - resolution (int): The resolution of the square texture (width and height).
    - mu (tuple): Mean (center) of the Gaussian. Defaults to the center of the texture.
    - sigma (float): Standard deviation of the Gaussian.
    - channels (int): Number of color channels for the texture.

    Returns:
    - torch.Tensor: A tensor of shape (channels, resolution, resolution) with Gaussian values.
    """
    if mu is None:
        mu = (resolution / 2.0, resolution / 2.0)
    
    x = torch.arange(0, resolution).float()
    y = torch.arange(0, resolution).float()
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    
    # Gaussian formula
    gaussian = torch.exp(-((x_grid - mu[0])**2 + (y_grid - mu[1])**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.max()  # Normalize to range [0, 1]
    
    # Expand to multiple channels
    gaussian_texture = gaussian.unsqueeze(0).repeat(channels, 1, 1)
    
    return gaussian_texture
def analyze_spectral_and_radial_symmetry(texture):
    """
    Analyze the spectral and radial symmetry of a texture.
    
    Parameters:
    - texture (torch.Tensor): A tensor of shape (channels, height, width) representing the texture.

    Returns:
    - dict: Analysis results containing:
        - "radial_profile": Radial intensity profile.
        - "spectral_power": Power spectral density.
    """
    # Convert to grayscale for analysis (average across channels)
    if texture.dim() == 3:
        texture_grayscale = texture.mean(dim=0)
    elif texture.dim() == 2:
        texture_grayscale = texture
    else:
        print(texture.shape)
        raise ValueError("Invalid texture shape. Expected 2D or 3D tensor.")

    # Compute Fourier Transform
    fft_result = torch.fft.fftshift(torch.fft.fft2(texture_grayscale))
    power_spectral_density = torch.abs(fft_result)**2

    # Compute radial profile
    resolution = texture_grayscale.shape[0]
    center = resolution / 2.0
    y, x = torch.meshgrid(torch.arange(0, resolution), torch.arange(0, resolution), indexing="ij")
    radius = torch.sqrt((x - center)**2 + (y - center)**2)

    # Radial profile
    radial_profile = torch.zeros(int(center + 1))
    counts = torch.zeros(int(center + 1))
    for r in range(int(center + 1)):
        mask = (radius == r)
        radial_profile[r] += power_spectral_density[mask].sum()
        counts[r] += mask.sum()
    radial_profile /= counts.clamp(min=1)

    return {
        "radial_profile": radial_profile,
        "spectral_power": power_spectral_density,
    }
def generate_metadata_for_generator(output_tensor, generation_params=None):
    """
    Generate metadata for a generator network's output.

    Parameters:
    - output_tensor (torch.Tensor): The generated output from the network (shape: [channels, height, width]).
    - generation_params (dict): Parameters used to guide the generator (e.g., sigma, mu). Default is None.

    Returns:
    - dict: Metadata for the generated output.
    """
    # Default generation parameters if not provided
    if generation_params is None:
        generation_params = {
            "sigma": torch.randint(10, 20, (1,)).item(),
            "mu": (output_tensor.shape[1] // 2, output_tensor.shape[2] // 2),
            "channels": output_tensor.shape[0]
        }

    # Analyze the output tensor
    analysis = analyze_spectral_and_radial_symmetry(output_tensor.mean(dim=0))
    radial_profile = analysis["radial_profile"]
    spectral_power = analysis["spectral_power"]

    # Evaluation scores
    gaussian_score = torch.exp(-F.mse_loss(radial_profile, generate_center_weighted_gaussian(
        resolution=output_tensor.shape[1],
        sigma=generation_params["sigma"],
        mu=generation_params["mu"],
        channels=1
    ).mean(dim=0))).item()

    symmetry_score = radial_profile.std().item()  # Lower std deviation suggests higher symmetry
    spectral_score = spectral_power.mean().item() / spectral_power.max().item()  # Normalized spectral density

    # Quality tag
    quality_tag = "high_quality" if gaussian_score > 0.8 and symmetry_score > 0.8 else "low_quality"

    # Metadata dictionary
    metadata = {
        "source": "generator",
        "generation_parameters": generation_params,
        "evaluation_scores": {
            "gaussian_score": gaussian_score,
            "symmetry_score": symmetry_score,
            "spectral_score": spectral_score
        },
        "tag": quality_tag
    }

    return metadata

def main():
    # Create networks
    predictor = PredictorNetwork(input_channels=3, input_height=128, input_width=128, conv_height=12, conv_width=12)
    reproducer = ReproducerNetwork(output_channels=3, height=128, width=128)
    discriminator = DiscriminatorNetwork(upscale_height=256, upscale_width=256, feedback_whitelist=["Whitelist Item 1", "Whitelist Item 2"])
    classifier = ClassifierNetwork(input_channels=3, input_height=128, input_width=128, conv_height=12, conv_width=12)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(predictor.parameters()) + list(reproducer.parameters()) + list(discriminator.parameters()) + list(classifier.parameters()),
        lr=0.001
    )

    # Loss Manager
    loss_manager = LossManager(optimizer)

    # Create IsolationMap
    iso_map = IsolationMap(max_cache_size=10)
    iso_map.add_model("Predictor", predictor)
    iso_map.add_model("Classifier", classifier)
    iso_map.add_model("Reproducer", reproducer)
    iso_map.add_model("Discriminator", discriminator)

    # Connect nodes
    iso_map.connect("Discriminator", "Predictor", [0], channels=["channel_1"], out_index=(1,0,2))
    iso_map.connect("Predictor", "Reproducer", [0], channels=["channel_2"], out_index=(0,0))
    iso_map.connect("Reproducer", "Classifier", [0], channels=["channel_3"], out_index=(0,0))
    iso_map.connect("Classifier", "Discriminator", [0], channels=["channel_4"], out_index=(1,0))
    #iso_map.connect("Discriminator", "Classifier", [2], channels=["channel_1"], out_index=(2,0,1))

    # Channel sequence
    iso_map.set_channel_sequence(["channel_1", "channel_2", "channel_3", "channel_4"])

    # Input tensors and metadata generation
    batch_size = 10
    resolution = 128


    # Training loop
    for step in range(10000):
        input_tensor = torch.randn(batch_size, 3, resolution, resolution)

        metadata_list = []
        for i in range(batch_size):
            if i % 2 == 0:  # Alternate between Gaussian and noise generation
                source = "gaussian"
                sigma = torch.randint(10, 20, (1,)).item()
                mu = (resolution / 2.0, resolution / 2.0)
                generated_tensor = generate_center_weighted_gaussian(resolution=resolution, mu=mu, sigma=sigma)
                evaluation = analyze_spectral_and_radial_symmetry(generated_tensor.mean(dim=0))
                gaussian_score = evaluation["radial_profile"].mean().item()
                #print(f"gaussian score: {gaussian_score}")
                metadata = {
                    "source": source,
                    "generation_parameters": {"sigma": sigma, "mu": mu, "channels": 3},
                    "evaluation_scores": {"gaussian_score": gaussian_score, "symmetry_score": 0.9, "spectral_score": 0.88},
                    "tag": "high_quality" if gaussian_score > 0.8 else "low_quality"
                }
            else:
                source = "noise"
                generated_tensor = torch.rand(3, resolution, resolution)  # Random noise
                metadata = {
                    "source": source,
                    "generation_parameters": {"sigma": -0.0, "mu": (0.0,0.0), "channels": 3},
                    "evaluation_scores": {"gaussian_score": 0.1, "symmetry_score": 0.1, "spectral_score": 0.1},
                    "tag": "low_quality"
                }

            input_tensor[i] = generated_tensor
            metadata_list.append(metadata)
        # Add initial data to IsolationMap cache
        iso_map.cache["Classifier"].add(input_tensor, metadata_list, isbatch=True)
        print(iso_map.cache["Classifier"].get_all())
        iso_map.activate_next_channel()
        propagated_outputs, evaluated_outputs, rejected_outputs = iso_map.forward(input_tensor, start_node="Classifier")
        

        # Generate Gaussian for loss comparison
        gaussian_target = generate_center_weighted_gaussian(resolution=128.0, sigma=16.0, channels=3)
        reproduction_loss = torch.tensor([0.0])
        classification_loss = torch.tensor([0.0])
        
        for node, all_outputs in evaluated_outputs.items():
            print(f"pre pre output: {node}, {all_outputs}")
            print(propagated_outputs)
            print(rejected_outputs)
            for these_outputs, this_metadata in all_outputs:
                print(f"pre-output")
                for output, metadata in zip(these_outputs, this_metadata):
                    #if isinstance(output, list):
                        #print("unexpected list output in main")
                        #exit()
                    #print(f"output: {output}")
                    #print(f"metadata: {metadata}")
                    if node == "Classifier":
                        #analysis = analyze_spectral_and_radial_symmetry(output)
                        #radial_loss = F.mse_loss(analysis["radial_profile"], gaussian_target.mean(dim=0))
                        #spectral_loss = F.mse_loss(analysis["spectral_power"], gaussian_target)

                        #loss_manager.add_loss(LossObject(radial_loss, source=node, blend_mode="sum", category="symmetry_analysis"))
                        #loss_manager.add_loss(LossObject(spectral_loss, source=node, blend_mode="sum", category="spectral_analysis"))

                        # Classification Loss
                        if metadata["source"] == "gaussian":
                            classification_loss = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))
                        elif metadata["source"] == "noise":
                            classification_loss = F.binary_cross_entropy_with_logits(output, torch.zeros_like(output))
                        
                        loss_manager.add_loss(LossObject(classification_loss, source=node, metadata=metadata, blend_mode="sum", category="classification"))

                    #if node == "Predictor":
                        #predicted_mu, predicted_sigma = output[..., :2], output[..., 2:]
                        #true_mu, true_sigma = metadata["generation_parameters"]["mu"], metadata["generation_parameters"]["sigma"]
                        #print("true_sigma dtype:", type(true_sigma))
                        #print("true_mu dtype:", type(true_mu))
                        
                        #prediction_loss = F.mse_loss(predicted_mu, torch.tensor(true_mu).to(output.device)) + \
                                        #F.mse_loss(predicted_sigma, torch.tensor(true_sigma).to(output.device))
                        
                        #loss_manager.add_loss(LossObject(prediction_loss, source=node, metadata=metadata, blend_mode="sum", category="prediction"))

                    #if node == "Reproducer":
                        #reproduction_loss = F.mse_loss(output, gaussian_target)
                        #gaussian_radial_profile = analyze_spectral_and_radial_symmetry(gaussian_target)["radial_profile"]

                        #radial_loss = F.mse_loss(analyze_spectral_and_radial_symmetry(output)["radial_profile"], gaussian_radial_profile)
                        
                        #loss_manager.add_loss(LossObject(reproduction_loss, source=node, metadata=metadata, blend_mode="sum", category="reproduction"))
                        #loss_manager.add_loss(LossObject(radial_loss, source=node, metadata=metadata, blend_mode="sum", category="symmetry_analysis"))
                        
                    #elif node == "Discriminator":
                        # Assuming metadata is a list of dictionaries
                        
                        #gaussian_score = torch.tensor([metadata["evaluation_scores"]["gaussian_score"]])
                        #discriminator_loss = F.binary_cross_entropy_with_logits(output, gaussian_score)
                        #print(discriminator_loss.shape)
                        #loss_manager.add_loss(LossObject(discriminator_loss, source=node, metadata=metadata, blend_mode="sum", category="classification"))

        # Optimize and visualize
        loss_manager.optimize()
        (cache.clear() for cache in iso_map.cache.items())
        visualize_combined(iso_map.cache["Classifier"], loss_manager)
        #(entry.clone().detach() for entry in iso_map.cache.values())
        handle_quit_events()

        # Logging
        print(f"Step {step}: Optimization completed. loss: {classification_loss}")

main()