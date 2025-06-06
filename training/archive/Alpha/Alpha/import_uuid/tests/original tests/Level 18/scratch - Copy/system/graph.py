import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph

class MetaNetwork:
    def __init__(self, vertices=None, edges=None, fields=None, edge_types=None, device="cuda"):
        self.device = torch.device(device)
        self.vertices = vertices.to(self.device) if vertices is not None else None
        self.edges = edges.to(self.device) if edges is not None else None
        self.edge_types = edge_types.to(self.device) if edge_types is not None else None
        self.edge_types = {}
        # Initialize edge layers
        self.edge_type_index = 0
        self.edge_type_map = torch.tensor([], dtype=torch.float32, device=device)
        self.edge_layers = {}
        self.add_layer("default")  # Default layer for edges
        self.add_layer("fully_connected")  # Fully connected edge layer
        
        # Initialize fully connected edges if vertices are present
        if self.vertices is not None:
            self._initialize_fully_connected_layer()

        # Save ordered field names
        self.field_names = (
            SPACE["raw_dimensionality"]["scalar_fields"] +
            SPACE["raw_dimensionality"]["vector_fields"]
        )
        self.total_fields = len(SPACE["raw_dimensionality"]["scalar_fields"]) + 3 * len(SPACE["raw_dimensionality"]["vector_fields"])
        
        self.fields = self.initialize_fields(fields)
        
        # Initialize subnetwork cache
        self.subnetwork_cache = []
        if self.vertices is not None:
            self._initialize_subnetwork_cache()

                # Create PyG Data object with edge types
        self.data = Data(
            pos=self.vertices, 
            edge_index=self.edges, 
            edge_type=self.edge_type_map,  # New edge type attribute
            x=self.fields
        )

    def _initialize_subnetwork_cache(self):
        """Initialize the subnetwork cache with each vertex as an isolated subnetwork."""
        num_vertices = self.vertices.shape[0]
        self.subnetwork_cache = [(torch.tensor([i], device=self.device), "default") for i in range(num_vertices)]
        print(f"Initialized subnetwork cache with {num_vertices} isolated subnetworks.")


    def _initialize_fully_connected_layer(self):
        """
        Initialize the 'fully_connected' layer with edges connecting all vertex pairs.
        """
        num_vertices = self.vertices.shape[0]
        if num_vertices < 2:
            print("Fully connected layer initialization skipped: Not enough vertices.")
            return

        # Generate all possible vertex pairs (excluding self-loops)
        row, col = torch.meshgrid(torch.arange(num_vertices, device=self.device),
                                torch.arange(num_vertices, device=self.device), indexing='ij')
        mask = row != col  # Exclude self-loops
        fully_connected_edges = torch.stack([row[mask], col[mask]], dim=0)

        # Add edges to the 'fully_connected' layer
        self.add_layer_edge("fully_connected", fully_connected_edges)
        print(f"Fully connected layer initialized with {fully_connected_edges.shape[1]} edges.")
    def _incrementally_update_fully_connected_layer(self, new_vertex_count):
        """
        Incrementally add fully connected edges for new vertices to the 'fully_connected' layer.

        Args:
            new_vertex_count (int): Number of vertices newly added to the network.
        """
        if new_vertex_count == 0:
            return  # No new vertices, nothing to update.

        total_vertex_count = self.vertices.shape[0]
        old_vertex_count = total_vertex_count - new_vertex_count

        # Generate edges between old vertices and new vertices
        old_vertices = torch.arange(0, old_vertex_count, device=self.device)
        new_vertices = torch.arange(old_vertex_count, total_vertex_count, device=self.device)

        row, col = torch.meshgrid(old_vertices, new_vertices, indexing='ij')
        new_edges = torch.cat([row.flatten().unsqueeze(0), col.flatten().unsqueeze(0)], dim=0)

        # Add edges for new vertex pairs within themselves (excluding self-loops)
        intra_new_edges = torch.combinations(new_vertices, r=2).t()

        if intra_new_edges.numel() > 0:  # Only append if there's more than one new vertex
            new_edges = torch.cat([new_edges, intra_new_edges], dim=1)

        # Add the new edges incrementally to the 'fully_connected' layer
        self.add_layer_edge("fully_connected", new_edges)

    def initialize_vertex_states(self, state_dict, num_vertices):
        """
        Initialize vertex states with all fields aligned to the default structure.
        """
        # Start with zero tensors for all fields
        field_tensor = torch.zeros((num_vertices, self.total_fields), device=self.device)
        
        # Fill in known fields based on state_dict
        scalar_fields = SPACE["raw_dimensionality"]["scalar_fields"]
        vector_fields = SPACE["raw_dimensionality"]["vector_fields"]
        
        for field_name in state_dict["fields"]:
            if field_name in scalar_fields:
                idx = scalar_fields.index(field_name)
                field_tensor[:, idx] = state_dict["fields"][field_name]
            elif field_name in vector_fields:
                idx = len(scalar_fields) + vector_fields.index(field_name) * 3
                field_tensor[:, idx:idx + 3] = torch.tensor([state_dict["fields"][field_name]], device=self.device)
        # Add new vertices to subnetwork cache as isolated nodes with default layer
        old_vertex_count = self.vertices.shape[0] if self.vertices is not None else 0
        for i in range(old_vertex_count, old_vertex_count + num_vertices):
            self.subnetwork_cache.append((torch.tensor([i], device=self.device), "default"))
        # Incrementally update the fully connected layer
        self._incrementally_update_fully_connected_layer(num_vertices)

        return field_tensor
    def initialize_fields(self, fields=None):
        """
        Initialize vertex state tensor based on SPACE config.
        Fields are organized as scalar and vector components.
        If no vertices are present, initialize an empty state tensor.
        """
        scalar_fields = len(SPACE["raw_dimensionality"]["scalar_fields"])
        vector_fields = len(SPACE["raw_dimensionality"]["vector_fields"])
        total_fields = scalar_fields + 3 * vector_fields  # Scalars + 3D vectors

        if self.vertices is None:  # No vertices yet
            print("Warning: No vertices provided. Initializing empty state tensor.")
            return torch.empty((0, total_fields), device=self.device)  # Empty tensor

        num_vertices = self.vertices.shape[0]

        # Allocate default zero fields: Scalars (num_vertices, S), Vectors (num_vertices, 3 * V)
        state = torch.zeros((num_vertices, total_fields), device=self.device)

        # If custom fields are provided, insert into state tensor
        if fields:
            for i, field in enumerate(fields):
                state[:, i] = field

        return state


    def field_index(self, field_name):
        """
        Retrieve the index slice for a specific field in the state tensor.
        """
        scalar_fields = SPACE["raw_dimensionality"]["scalar_fields"]
        vector_fields = SPACE["raw_dimensionality"]["vector_fields"]
        total_scalars = len(scalar_fields)

        if field_name in scalar_fields:
            idx = scalar_fields.index(field_name)
            return slice(idx, idx + 1)
        elif field_name in vector_fields:
            idx = vector_fields.index(field_name)
            return slice(total_scalars + idx * 3, total_scalars + (idx + 1) * 3)
        else:
            raise ValueError(f"Field {field_name} not recognized.")

    def update_scalar_field(self, field_name, update_function):
        """
        Apply an update to a scalar field based on an external function.
        Args:
            field_name: The name of the scalar field to update.
            update_function: Function that computes the update.
        """
        field_idx = self.field_index(field_name)
        self.data.x[:, field_idx] = update_function(self.data.x[:, field_idx])

    def update_vector_field(self, field_name, update_function):
        """
        Apply an update to a vector field based on an external function.
        Args:
            field_name: The name of the vector field to update.
            update_function: Function that computes the update.
        """
        field_idx = self.field_index(field_name)
        self.data.x[:, field_idx] = update_function(self.data.x[:, field_idx])

    def apply_charge_dynamics(self):
        """
        Update charge distribution across vertices and edges.
        """
        charge_idx = self.field_index("charge")
        charge = self.data.x[:, charge_idx]

        # Coulomb repulsion forces
        forces = torch.zeros_like(self.vertices)
        for i in range(self.vertices.size(0)):
            r_ij = self.vertices - self.vertices[i]
            distances = torch.norm(r_ij, dim=1).clamp(min=1e-10)  # Prevent division by zero
            repulsion = SPACE["constants"]["coulomb_constant"] * charge[i] * charge[:, 0] / distances**2
            forces += (r_ij / distances.unsqueeze(1)) * repulsion.unsqueeze(1)

        # Apply forces to vertex positions
        self.data.pos += forces * SPACE["gamification_modifiers"]["charge_attraction_modifier"]

    def diffuse_field(self, field_name, diffusion_coefficient=0.1):
        """
        Perform scalar field diffusion across the edges.
        Args:
            field_name: Name of the scalar field to diffuse.
            diffusion_coefficient: Degree of field sharing through connections.
        """
        field_idx = self.field_index(field_name)
        field_values = self.data.x[:, field_idx].squeeze()  # Extract the scalar field

        # Prepare tensor for updated field values
        new_field = field_values.clone()

        # Iterate over edges to compute diffusion updates
        for edge in self.data.edge_index.t():
            i, j = edge
            delta_ij = diffusion_coefficient * (field_values[j] - field_values[i])
            new_field[i] += delta_ij
            new_field[j] -= delta_ij  # Symmetrical sharing

        # Update the state tensor
        self.data.x[:, field_idx] = new_field.unsqueeze(1)


    def extract_connected_components(edge_index, num_nodes, device="cuda"):
        """
        Extract connected components in a graph using Depth-First Search (DFS).
        Args:
            edge_index: Tensor of shape (2, E) representing edges.
            num_nodes: Total number of nodes in the graph.
            device: Device for computations.
        Returns:
            List of tensors, each containing indices of nodes in a connected component.
        """
        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        components = []

        def dfs(node, current_component):
            """Recursive DFS to find connected nodes."""
            visited[node] = True
            current_component.append(node)
            neighbors = edge_index[1][edge_index[0] == node]  # Outgoing edges
            for neighbor in neighbors:
                if not visited[neighbor]:
                    dfs(neighbor, current_component)

        for node in range(num_nodes):
            if not visited[node]:  # Unvisited nodes are new components
                current_component = []
                dfs(node, current_component)
                components.append(torch.tensor(current_component, device=device))

        return components

    def instantiate_shape(self, geometry_type="tetrahedron", state_dict=None, position_offset=None, layer_name="default"):
        """
        Instantiate a predefined shape into the MetaNetwork.
        Args:
            geometry_type: Name of the shape (e.g., "tetrahedron", "cube").
            state_dict: Dictionary defining the vertex state (default: HYDROGEN_STATE).
            position_offset: Offset position for shape placement.
        """
        # Defaults
        state_dict = state_dict or HYDROGEN_STATE
        position_offset = position_offset.to(self.device) if position_offset is not None else torch.zeros(3, device=self.device)

        # Load geometry
        geometry = CompositeGeometry(geometry=geometry_type, device=self.device)
        offsets, _, edge_pairs, _, _ = geometry.configure_geometry(density=1)

        # Shift positions by offset
        shifted_offsets = offsets + position_offset

        # Append vertices
        if self.vertices is not None:
            old_vertex_count = self.vertices.shape[0]
            self.vertices = torch.cat([self.vertices, shifted_offsets], dim=0)
        else:
            old_vertex_count = 0
            self.vertices = shifted_offsets

        # Update vertex states
        num_new_vertices = shifted_offsets.shape[0]
        new_fields = self.initialize_vertex_states(state_dict, num_new_vertices)
        
        self.fields = torch.cat([self.fields, new_fields], dim=0) if self.fields is not None else new_fields

        # Add edges via add_edge
        edge_pairs = edge_pairs + old_vertex_count  # Offset edges for new vertices
        self.add_edge(edge_pairs)

        # Update Data object
        self.data = Data(pos=self.vertices, edge_index=self.data.edge_index, edge_types=self.edge_type_map, x=self.fields)
    def add_layer(self, layer_name):
        """
        Add a new edge layer. Errors if the layer already exists.
        Args:
            layer_name (str): Name of the new edge layer.
        """
        if layer_name in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' already exists. Duplicate layers are not allowed.")
        self.edge_layers[layer_name] = torch.empty((2, 0), dtype=torch.long, device=self.device)
        self.edge_types[layer_name] = (self.edge_type_index, uuid.uuid4())
        self.edge_type_index += 1
        print(f"Layer '{layer_name}' added successfully.")

    def remove_layer(self, layer_name):
        """
        Remove an existing edge layer. Errors if the layer does not exist.
        Args:
            layer_name (str): Name of the layer to remove.
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist. Cannot remove.")
        del self.edge_layers[layer_name]
        print(f"Layer '{layer_name}' removed successfully.")

    def add_layer_edge(self, layer_name, edges):
        """
        Add edges to a specific layer.
        Args:
            layer_name (str): The name of the edge layer.
            edges (Tensor or list): Edge pairs to add. Shape: (2, N) or (N, 2).
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist. Add the layer first.")

        edges = torch.as_tensor(edges, dtype=torch.long, device=self.device)
        if edges.dim() == 1 and edges.shape[0] == 2:
            edges = edges.unsqueeze(-1)
        elif edges.dim() == 2 and edges.shape[1] == 2:
            edges = edges.t()
        elif edges.dim() != 2 or edges.shape[0] != 2:
            raise ValueError("Edges must have shape (2, N) or (N, 2).")

        #print(f"\n--- Debug: Adding Edges to Layer '{layer_name}' ---")

        # Input edges before any operation
        #print("Edges Being Added (Input):", edges)

        # Existing edges in the layer before addition
        existing_edges = self.edge_layers[layer_name]
        #print("Existing Edges Before Addition:", existing_edges)

        # Combine edges
        combined_edges = torch.cat([existing_edges, edges], dim=1)
        #print("Combined Edges After Addition:", combined_edges)

        # Verify shapes and data
        #print("Edge Tensor Shape:", combined_edges.shape)
        #print("Edge Tensor Max Index:", combined_edges.max(), "Vertex Count:", self.vertices.shape[0])


        edge_type_index = torch.full((edges.shape[1],), self.edge_types[layer_name][0], dtype=torch.long, device=self.device)
        self.edge_type_map = torch.cat([self.edge_type_map, edge_type_index])


        # Update the layer
        self.edge_layers[layer_name] = combined_edges
        #print(f"Updated '{layer_name}' Edges:", self.edge_layers[layer_name])


        # Verify edge layers are non-empty
        if self.data.edge_index is not None and self.data.edge_index.shape[0] > 0:
            
            self.data.edge_index = torch.cat([self.data.edge_index, edges], dim=1)
        else:
            self.data.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device)
        
        #print(f"Edges added to layer '{layer_name}'. Total edges: {combined_edges.shape[1]}")


    def remove_layer_edge(self, layer_name, edges):
        """
        Remove edges from a specific layer.
        Args:
            layer_name (str): The name of the edge layer.
            edges (Tensor or list): Edge pairs to remove. Shape: (2, N) or (N, 2).
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist.")

        edges = torch.as_tensor(edges, dtype=torch.long, device=self.device)
        if edges.dim() == 1 and edges.shape[0] == 2:
            edges = edges.unsqueeze(-1)
        elif edges.dim() == 2 and edges.shape[1] == 2:
            edges = edges.t()
        elif edges.dim() != 2 or edges.shape[0] != 2:
            raise ValueError("Edges must have shape (2, N) or (N, 2).")

        # Filter out edges to remove
        layer_edges = self.edge_layers[layer_name]
        mask = torch.ones(layer_edges.shape[1], dtype=torch.bool, device=self.device)
        
        for edge in edges.t():
            mask &= ~((layer_edges[0] == edge[0]) & (layer_edges[1] == edge[1]) |
                    (layer_edges[0] == edge[1]) & (layer_edges[1] == edge[0]))
        self.edge_layers[layer_name] = layer_edges[:, mask]
        
        # Build the keeping mask for edge_index that matches BOTH the edge vertices AND the edge type
        edge_type_index = self.edge_types[layer_name][0]  # Retrieve the type index for this layer

        # Condition to keep edges: NOT matching provided edges AND matching the correct type
        keeping_mask = (
            (self.data.edge_index[0].unsqueeze(1) != edges[0].unsqueeze(0)) | 
            (self.data.edge_index[1].unsqueeze(1) != edges[1].unsqueeze(0))
        ).all(dim=1) & (self.edge_type_map == edge_type_index)

        # Apply the keeping mask to edge_index and edge_type_map
        self.data.edge_index = self.data.edge_index[:, keeping_mask]
        self.edge_type_map = self.edge_type_map[keeping_mask]

        print(f"Edges removed from layer '{layer_name}'. Remaining edges: {self.edge_layers[layer_name].shape[1]}")

    def add_edge(self, edges, layer_name="default"):
        """
        Add edges to a specified edge layer.
        Args:
            edges (Tensor or list): Edge pairs to add.
            layer_name (str): Name of the edge layer.
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist. Add the layer first.")
        self.add_layer_edge(layer_name, edges)
        
        self.incremental_subnetwork_update(edges, "add", layer_name)


    def remove_edge(self, edges, layer_name="default"):
        """
        Remove edges from a specified edge layer.
        Args:
            edges (Tensor or list): Edge pairs to remove.
            layer_name (str): Name of the edge layer.
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist.")
        self.remove_layer_edge(layer_name, edges)
        self.incremental_subnetwork_update(edges, "remove", layer_name)


    def edge_exists(self, vertex_pair, layer_name="default"):
        """
        Check if an edge exists between two vertices in the default or a specified layer.
        Args:
            vertex_pair (tuple): Tuple of two vertex indices.
            layer_name (str): Optional name of the edge layer to check. If None, checks all layers.

        Returns:
            bool: True if the edge exists, otherwise False.
        """
        
        edges = self.edge_layers.get(layer_name, torch.empty((2, 0), device=self.device))
        
        exists = ((edges[0] == vertex_pair[0]) & (edges[1] == vertex_pair[1])) | \
                    ((edges[0] == vertex_pair[1]) & (edges[1] == vertex_pair[0]))
        return exists.any()

    def display_summary(self):
        """Print a summary of the current MetaNetwork."""
        print("MetaNetwork Summary:")
        print(f" - Total Vertices: {self.vertices.shape[0]}")
        print(f" - Total Edges: {self.data.edge_index.shape[1]}")
        print(f" - State Tensor Shape: {self.data.x.shape}")
    def incremental_subnetwork_update(self, edges, operation, layer_name="default"):
        """
        Update the subnetwork cache incrementally based on added or removed edges.
        Tracks edge layers to manage subnetwork-layer relationships.

        Args:
            edges (Tensor): Tensor of edges or vertex pairs.
            operation (str): "add" or "remove" to specify the operation.
            layer_name (str): Name of the edge layer involved in this update.
        """
        if layer_name not in self.edge_layers:
            raise ValueError(f"Layer '{layer_name}' does not exist. Cannot update subnetworks.")

        if operation == "add":
            self._handle_edge_addition(edges, layer_name)
        elif operation == "remove":
            self._handle_edge_removal(edges, layer_name)
        else:
            raise ValueError("Operation must be 'add' or 'remove'.")


    def _handle_edge_addition(self, new_edges, layer_name):
        """
        Merge subnetworks if new edges connect them.
        Associates subnetworks with the specified edge layer.

        Args:
            new_edges (Tensor): Tensor of edges (2, num_edges).
            layer_name (str): Name of the edge layer.
        """
        vertex_to_subnet = {}
        for idx, (subnetwork, layer) in enumerate(self.subnetwork_cache):
            for v in subnetwork:
                vertex_to_subnet[v.item()] = (idx, layer)

        updated_subnets = []
        merged_indices = set()
        
        for edge in new_edges:
            v1, v2 = edge.tolist()
            s1, l1 = vertex_to_subnet.get(v1, (None, None))
            s2, l2 = vertex_to_subnet.get(v2, (None, None))

            if s1 is not None and s2 is not None and s1 != s2:
                # Merge two subnetworks and update to the current layer
                updated_subnets.append((torch.cat([self.subnetwork_cache[s1][0], 
                                                self.subnetwork_cache[s2][0]]).unique(), 
                                        layer_name))
                merged_indices.update([s1, s2])
            elif s1 is not None:
                updated_subnets.append((torch.cat([self.subnetwork_cache[s1][0], 
                                                torch.tensor([v2], device=self.device)]), 
                                        layer_name))
                merged_indices.add(s1)
            elif s2 is not None:
                updated_subnets.append((torch.cat([self.subnetwork_cache[s2][0], 
                                                torch.tensor([v1], device=self.device)]), 
                                        layer_name))
                merged_indices.add(s2)
            else:
                # Both vertices are new
                updated_subnets.append((torch.tensor([v1, v2], device=self.device), layer_name))

        # Keep unmerged subnetworks and update cache
        self.subnetwork_cache = [
            (sub, layer) for idx, (sub, layer) in enumerate(self.subnetwork_cache)
            if idx not in merged_indices
        ] + updated_subnets

        #print(f"Subnetworks updated (add). Total subnetworks: {len(self.subnetwork_cache)}")

    def _handle_edge_removal(self, edges, layer_name):
        """
        Check if edge removal splits a subnetwork and update the cache.
        Keeps track of the layer associated with the split subnetworks.

        Args:
            edges (Tensor): Tensor of edges (2, num_edges).
            layer_name (str): Name of the edge layer.
        """
        for edge in edges.t():
            v1, v2 = edge.tolist()

            # Identify the subnetwork containing the edge
            target_idx = None
            for idx, (subnetwork, layer) in enumerate(self.subnetwork_cache):
                if v1 in subnetwork and v2 in subnetwork:
                    target_idx = idx
                    break

            if target_idx is None:
                print(f"Edge {v1}-{v2} not found in any subnetwork for layer '{layer_name}'.")
                continue

            subnetwork, current_layer = self.subnetwork_cache.pop(target_idx)
            visited = torch.zeros_like(subnetwork, dtype=torch.bool, device=self.device)

            def dfs(node_idx):
                visited[node_idx] = True
                node = subnetwork[node_idx]
                neighbors = self.edge_layers[layer_name][1][self.edge_layers[layer_name][0] == node]
                for neighbor in neighbors:
                    if neighbor in subnetwork:
                        neighbor_idx = (subnetwork == neighbor).nonzero().item()
                        if not visited[neighbor_idx]:
                            dfs(neighbor_idx)

            # Perform DFS starting from v1
            v1_idx = (subnetwork == v1).nonzero().item()
            dfs(v1_idx)

            # Split the subnetwork if disconnected
            if not visited[(subnetwork == v2).nonzero().item()]:
                connected_part = subnetwork[visited]
                disconnected_part = subnetwork[~visited]
                self.subnetwork_cache.append((connected_part, current_layer))
                self.subnetwork_cache.append((disconnected_part, current_layer))
                print(f"Edge {v1}-{v2} removed, splitting a subnetwork.")
            else:
                self.subnetwork_cache.append((subnetwork, current_layer))
                print(f"Edge {v1}-{v2} removed without splitting.")

    def forward(self, field_config = {"charge":{"type":"inverse_square", "magnitude":-1, "falloff_exponent":2, "layer_name":"fully_connected"},
                                      "mass":{"type": "inverse_square", "magnitude": 1, "falloff_exponent":2, "layer_name":"fully_connected"},
                                      "bond_spring":{"type": "spring_force", "magnitude": 1, "layer_name": "default" }},
                time_step=1e-10):
        """
        Perform a forward step to evaluate scalar field contributions and update states.

        Args:
            field_config (dict): Dictionary of scalar field definitions and parameters.

        Returns:
            Updated vertex states and edge data.
        """

        pos = self.data.pos  # Vertex positions
        edges = self.data.edge_index  # Edge data
        fields = self.data.x  # Vertex states
        
        # Initialize force accumulation tensor
        forces = torch.zeros_like(pos)


        mass_index = -1
        for field_name, params in field_config.items():
            state_idx = self.field_index(field_name)
            vertex_state = self.data.x[:, state_idx]
            if vertex_state.shape[0] == 0:
                continue

            layer_name = params["layer_name"]
            edge_type_index = self.edge_types[layer_name][0]

            # Filter edges for the current layer
            mask = self.edge_type_map == edge_type_index  # Identify relevant edges
            filtered_edges = self.data.edge_index[:, mask]

            if params["type"] == "inverse_square":
                forces += self._compute_inverse_square(pos, filtered_edges, params, vertex_state)
            elif params["type"] == "spring_force":
                forces += self._compute_spring_force(pos, filtered_edges, params, vertex_state)
        # Convert forces to acceleration
        mass_index = self.field_index("mass")
        velocity_index = self.field_index("velocity")
        acceleration_index = self.field_index("acceleration")
        force_index = self.field_index("force")
        vertex_mass = self.data.x[:,mass_index]
        self.data.x[:,force_index] = forces
        
        self.data.x[:,acceleration_index] = self.data.x[:,force_index] / vertex_mass
        
        
        # Integrate acceleration to update velocity and position
        self.data.x[:,velocity_index] = self.data.x[:,velocity_index] + self.data.x[:,acceleration_index] * time_step
        self.data.pos = self.data.pos + self.data.x[:,velocity_index] * time_step
        #print(f"force: {torch.max(self.data.x[:,force_index])} acceleration: {torch.max(self.data.x[:,acceleration_index])} velocity: {torch.max(self.data.x[:,velocity_index])} position: {torch.max(self.data.pos)}")        
        return self.data
