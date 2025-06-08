import uuid
import torch
SPACE = {
    "spatial_dimensionality": 3,
    "raw_dimensionality": {
        "position": "3D",  # Cartesian space
        "distance_scale": 1,#1e-10,
        "quaternion_fields": ["orientation"],
        "scalar_fields": ["charge", "temperature", "density", "potential_energy", "mass", "bond_spring", "barrier"],
        "vector_fields": ["flux", "force", "spin", "position", "velocity", "acceleration", "gravity", "momentum"],
        "latent_fields": ["bond_potential", "statistical_energy"],
        "temporal_fields": []
    },
    "constants": {
        # Fundamental Constants
        "electric_constant": 8.854e-12,   # Vacuum permittivity, ε₀ (F·m⁻¹)
        "coulomb_constant": 8.987e9,      # Coulomb's constant, k_e (N·m²·C⁻²)
        "electron_charge": 1.602e-19,     # Elementary charge, e (C)
        "boltzmann_constant": 1.381e-23,  # Boltzmann constant (J·K⁻¹)
        "spring_constant": 1e2,           # Hookean spring constant for edge relaxation
    },
    "gamification_modifiers": {
        "charge_attraction_modifier": 1e3,
        "repulsion_strength_modifier": 1e-2,
        "bond_strength_modifier": 1e2,
        "stochastic_field_scaling": 1e-1,
        "preferred_length_scale": 1.0,
        "angular_relaxation_force": 1e-1,  # Force to relax edge angles
    },
    "simplicial_complex": {
        "vertices": {
            "active": True,
            "fields": ["scalar_fields", "vector_fields"],
            "dipole_properties": {"charge": True, "magnetism": True},
            "spontaneous_generation": True,
            "configurable_options": {
                "position_lock": False,              # Vertex immobility
                "edge_angle_lock": False,            # Lock edge angles
                "edge_angle_relaxation": True,       # Relaxation of angles under force
                "charge_freeze": False,              # Static charge state
                "temperature_freeze": False,         # Static temperature state
                "velocity_cap": None,                # Optional maximum velocity
                "user_tag": None                     # Custom tags for gameplay
            }
        },
        "edges": {
            "active": True,
            "fields": ["vector_fields"],
            "spontaneous_generation": True,
            "spontaneous_dissolution": True,
            "conductivity_function": "bond_strength * flux_gradient",
            "max_conductivity": 1.0,
            "configurable_options": {
                "preferred_length": 1.5e-10,        # Resting bond length
                "spring_force_override": None,      # Optional spring force constant
                "stress_threshold": 1e3,            # Max stress before dissolution
                "is_rigid": False,                  # Overrides edge deformation
                "conductivity_override": None,      # Edge-specific conductivity
                "user_tag": None                    # Custom tags for gameplay
            }
        },
        "faces": {
            "active": True,
            "fields": ["latent_fields", "statistical_energy"],
            "spontaneous_generation": False
        }
    },
    "geometry_rules": {
        "coordinate_system": "spherical",
        "vertex_relationships": {
            "bond_length_min": 1e-10,
            "bond_length_max": 2e-10,
            "bond_probability_function": "charge_interaction_threshold"
        },
        "edge_dynamics": {
            "edge_weight_function": "scalar_field_gradient + bond_potential",
            "preferred_length": 100.8,#1.5e-10,
            "dissolution_condition": "flux_stress_exceed_threshold"
        },
        "force_scaling": {
            "coulomb_force": "coulomb_constant * charge_product / r^2",
            "repulsive_force": "repulsion_constant / r^2"
        }
    },
    "field_dynamics": {
        "scalar_field_diffusion": {
            "temperature": 0.05,
            "charge": 0.2,
            "density": 0.01
        }
    }
}
HYDROGEN_STATE = {
    "name": "hydrogen",
    "fields": {
        "charge": 1.602e-19,             # Elementary charge (+1e)
        "spin": [0, 0, 0],
        "temperature": 1e-10, #K
        "potential_energy": 0.0,
        "density": 1.67e3,
        "mass": 1.67e-27,
        "bond_spring" :15
    },
    "bond_sites": {
        "count": 4,                     # Tetrahedral max bonds
        "preferred_angle": 109.5,       # Degrees
        "bond_overload_policy": "soft", # Policy: soft allows overloads under stress
    },
    "metadata": {
        "pressure_requirement": None,
        "temperature_requirement": None,
    }
}
HYDROGEN_STATE = {
    "name": "hydrogen",
    "fields": {
        "charge": 1,             # Elementary charge (+1e)
        "spin": [0, 0, 0],
        "temperature": 1e-10, #K
        "potential_energy": 0.0,
        "density": 1.67e3,
        "mass": 1.0,
        "bond_spring" :1
    },
    "bond_sites": {
        "count": 4,                     # Tetrahedral max bonds
        "preferred_angle": 109.5,       # Degrees
        "bond_overload_policy": "soft", # Policy: soft allows overloads under stress
    },
    "metadata": {
        "pressure_requirement": None,
        "temperature_requirement": None,
    }
}

import torch

import torch

class RelativisticInertialOperator:
    """
    Relativistic Inertial Operator that updates data.x in-place.
    Each particle/node has its own mass, velocity, momentum, etc.
    We'll rely on your self.field_index(...) to find the correct slices.
    """

    def __init__(self, params={}):
        """
        Args:
            params (dict): Engine config.
                - speed_of_light (float): speed of light in m/s
                - velocity_threshold (float): fraction of c
                - history_length (int): how many timesteps to store for advanced usage
        """
        self.params = params
        self.speed_of_light     = params.get("speed_of_light", 3.0e8)
        self.velocity_threshold = params.get("velocity_threshold", 0.01)
        self.history_length     = params.get("history_length", 10)

        # Keep some optional state history for advanced usage
        self.state_history = {}
        self.dt_history    = []

    def update_state(self, data_x, field_index_func, edges, dt):
        """
        In-place update of velocity & position from momentum, with per-node mass.
        
        Args:
            data_x           (torch.Tensor): shape [N, total_fields], the global state
            field_index_func (callable): e.g. self.field_index(name) -> slice
            edges            (torch.Tensor): shape [2, E], if adjacency needed
            dt               (float): time step
        """

        # 1) Get slices for each field
        pos_slice       = field_index_func("position")     # e.g. slice(...)
        vel_slice       = field_index_func("velocity")     # e.g. slice(...)
        mom_slice       = field_index_func("momentum")     # slice for momentum
        mass_slice      = field_index_func("mass")         # slice for per-node mass
        acc_slice       = field_index_func("acceleration") # optional
        jerk_slice      = field_index_func("jerk")         # optional

        # 2) Reshape for convenience
        #    Assume 'position', 'velocity', 'momentum' are (N,3), mass is (N,1) or (N)
        pos       = data_x[:, pos_slice].view(-1, 3)
        vel       = data_x[:, vel_slice].view(-1, 3)
        momentum  = data_x[:, mom_slice].view(-1, 3)
        mass      = data_x[:, mass_slice].view(-1, 1)   # shape [N,1]

        # 3) Possibly use edges if you have adjacency-based logic. Otherwise, ignore.
        i, j = edges  # shape [2, E]

        # 4) Compute velocity magnitude & gamma factor
        v_mag = torch.norm(vel, dim=1, keepdim=True)                # shape [N,1]
        c_sq  = self.speed_of_light ** 2
        rel_mask = (v_mag / self.speed_of_light) > self.velocity_threshold
        gamma     = torch.ones_like(v_mag)
        gamma[rel_mask] = 1.0 / torch.sqrt(1.0 - (v_mag[rel_mask] ** 2) / c_sq)

        # 5) Update velocity from momentum:
        #    simplest: vel_new = vel + (momentum / mass)*dt
        dvel = (momentum / mass) * dt

        
        new_vel = dvel#vel + dvel

        # 6) Update position
        new_pos = pos + new_vel * dt

        # 7) Optionally compute new acceleration, jerk, etc.
        if acc_slice is not None:
            old_acc = data_x[:, acc_slice].view(-1, 3)  # previous acceleration
            new_acc = (new_vel - vel) / dt
            data_x[:, acc_slice].view(-1, 3).copy_(new_acc)

        if jerk_slice is not None:
            if acc_slice is not None:
                new_acc = data_x[:, acc_slice].view(-1, 3)
                # If we stored old_acc in self.state_history last step:
                old_acc = self.state_history.get("acceleration", new_acc)
                new_jerk = (new_acc - old_acc) / dt
                data_x[:, jerk_slice].view(-1, 3).copy_(new_jerk)

        # 8) Write back new position & velocity
        data_x[:, pos_slice].view(-1, 3).copy_(new_pos)
        data_x[:, vel_slice].view(-1, 3).copy_(new_vel)

        # 9) Optionally store for history
        self.state_history["position"]     = new_pos.detach().clone()
        self.state_history["velocity"]     = new_vel.detach().clone()
        if acc_slice is not None:
            self.state_history["acceleration"] = data_x[:, acc_slice].view(-1,3).detach().clone()

        self.dt_history.append(dt)
        if len(self.dt_history) > self.history_length:
            self.dt_history.pop(0)

        # done in-place
        return data_x

    def reset_history(self):
        self.state_history.clear()
        self.dt_history.clear()

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph
import queue
import threading

class GeneralNode:
    def __init(self, state_id, layers):
        self.state_id = state_id
        self.layers = layers
class MetaNetwork:
    def __init__(self, vertices=None, edges=None, fields=None, edge_types=None, device="cuda"):
        self.device = torch.device(device)
        self.invariant_mask = None
        self.vertices = vertices.to(self.device) if vertices is not None else None
        self.edges = edges.to(self.device) if edges is not None else None
        self.edge_types = edge_types.to(self.device) if edge_types is not None else None
        self.edge_types = {}
        self.rio = RelativisticInertialOperator()
        # Initialize edge layers
        self.edge_type_index = 0
        self.edge_type_map = torch.tensor([], dtype=torch.float32, device=device)
        self.edge_layers = {}
        self.add_layer("default")  # Default layer for edges
        self.add_layer("fully_connected")  # Fully connected edge layer
        self.time = 0
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
        animation_rate = 1/6.0
        self.runners = []
        self.data_queue = queue.Queue()
        self.data_queue_lock = threading.Lock()
        self.forward_runner = threading.Thread(
            target=self._forward_runner,
            args=(animation_rate,),
            daemon=True
        )

    def _forward_runner(self, animation_rate):
        import time as tm
        time = 0
        intertime = 0
        if len(self.runners) > 0:
            id = self.runners[-1]+1
        else:
            id = 0
        
        self.runners.append(True)

        while self.runners[id] == True:
            #print(f"calculating time: {time}")
            if self.data_queue.qsize() > 100:
                
                continue
            nextTime = time % animation_rate
            if nextTime <= animation_rate / 40.0:
                nextTime += animation_rate - intertime
            data, time_elapsed = self.forward(time_step=nextTime)
            intertime += time_elapsed
            if intertime >= nextTime:
                intertime -= nextTime
                with self.data_queue_lock:
                    self.data_queue.put((data.x[...,self.field_index("position")].cpu()))
            
            time += time_elapsed
            
        
    def initialize_invariant_mask(self, vertices):
        """
        Create a mask indicating which vertices are positionally invariant.
        By default, only the first vertex (index 0) is fixed, and all others are free.
        """
        num_vertices = vertices.shape[0]
        # Default mask: True for fixed points, False for free points
        invariant_mask = torch.zeros(num_vertices, dtype=torch.bool, device=self.device)
        #invariant_mask[0] = True  # Fix the first point (index 0) by default
        
        print(f"Initialized invariant mask with {num_vertices} vertices. Fixed point: 0")
        return invariant_mask

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
        field_tensor = torch.zeros((num_vertices, self.total_fields), dtype=torch.float64, device=self.device)
        
        # Fill in known fields based on state_dict
        quaternion_fields = SPACE["raw_dimensionality"]["quaternion_fields"]
        scalar_fields = SPACE["raw_dimensionality"]["scalar_fields"]
        vector_fields = SPACE["raw_dimensionality"]["vector_fields"]
        
        for field_name in state_dict["fields"]:
            if field_name in quaternion_fields:
                idx = quaternion_fields.index(field_name)
                field_tensor[:, idx] = state_dict["fields"][field_name]
            elif field_name in scalar_fields:
                idx = scalar_fields.index(field_name)
                field_tensor[:, idx] = state_dict["fields"][field_name]# * ((torch.rand_like(field_tensor[:,idx])-.5)*2) if field_name == "charge" else state_dict["fields"][field_name] * (torch.rand_like(field_tensor[:,idx])+1e-20)

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
            return None
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
        
        position_offset = position_offset.to(self.device) if position_offset is not None else torch.zeros(3, dtype=torch.float64, device=self.device)

        # Load geometry
        geometry = CompositeGeometry(geometry=geometry_type, precision=torch.float64, device=self.device)
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
        if self.invariant_mask is None:
            self.invariant_mask = torch.tensor([], dtype=torch.bool, device=self.device)
        self.invariant_mask = torch.cat([self.invariant_mask, self.initialize_invariant_mask(offsets)])
        #print(self.invariant_mask)
        self.fields = torch.cat([self.fields, new_fields], dim=0) if self.fields is not None else new_fields

        # Add edges via add_edge
        edge_pairs = edge_pairs + old_vertex_count  # Offset edges for new vertices
        self.add_edge(edge_pairs)
        self.fields[...,self.field_index("position")] = self.vertices
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
            return
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

    def forward(self, field_config = {"charge":{"type":"inverse_square", "magnitude":-1e2, "falloff_exponent":2, "layer_name":"fully_connected"},
                                      "mass":{"type": "inverse_square", "magnitude": 1e2, "falloff_exponent":2, "layer_name":"fully_connected"},
                                      "bond_spring":{"type": "spring_force", "magnitude": 1, "damping":[0.0,0.0], "layer_name": "default" },
                                      "barrier": {"type": "barrier", "magnitude":1},
                                      "gravity": {"type": "gravity", "magnitude":1*0},
                                      "induction": {"type": "induction", "magnitude":1}},
                time_step=1/6.0):
        """
        Perform a forward step to evaluate scalar field contributions and update states.

        Args:
            field_config (dict): Dictionary of scalar field definitions and parameters.

        Returns:
            Updated vertex states and edge data.
        """
        time_step = time_step / torch.mean(torch.abs(self.data.x[...,self.field_index("velocity")])**(1/6.0)+1)
        #print(time_step)
        #time_step = time_step / (self.data.x.shape[0]  * 100)**.75
        pos = self.data.pos  # Vertex positions
        edges = self.data.edge_index  # Edge data
        fields = self.data.x  # Vertex states
        
        # Initialize force accumulation tensor
        forces = torch.zeros_like(pos)
        force_gradients = torch.zeros_like(pos)

        mass_index = -1
        for field_name, params in field_config.items():
            state_idx = self.field_index(field_name)
            vertex_state = self.data.x[:, state_idx]
            vel = self.data.x[:,self.field_index("velocity")]
            if vertex_state.shape[0] == 0:
                continue

            layer_name = params.get("layer_name", "fully_connected")
            edge_type_index = self.edge_types[layer_name][0]

            # Filter edges for the current layer
            mask = self.edge_type_map == edge_type_index  # Identify relevant edges
            filtered_edges = self.data.edge_index[:, mask]
            
            if params["type"] == "inverse_square":
                force_return, force_gradient_return = self._compute_inverse_square(pos, filtered_edges, params, vertex_state)
                forces += force_return
                force_gradients += force_gradient_return
                
            elif params["type"] == "spring_force":
                force_return, force_gradient_return = self._compute_spring_force(vel, pos, filtered_edges, params, vertex_state)
                forces += force_return
                force_gradients += force_gradient_return

            elif params["type"] == "barrier":
                force_return = self._compute_barrier_rejection_force(vel, pos, params)
                forces += force_return
                #force_gradients += force_gradient_return

            elif params["type"] == "gravity":
                mass_index = self.field_index("mass")
                gravity_vector = torch.tensor([0, -params["magnitude"]*time_step, 0], dtype=pos.dtype, device=pos.device)
                forces += self.data.x[:, mass_index] * gravity_vector
            #elif params["type"] == "induction":
            #    candidate_charge = torch.sin((pos[...,0]**2 + pos[...,1]**2 + pos[...,2]**2)**.5 * self.time)*10
            #    #print(candidate_charge.shape)
            #    self.data.x[..., self.field_index("charge")] = candidate_charge.unsqueeze(1)


        # Convert forces to acceleration
        mass_index = self.field_index("mass")
        velocity_index = self.field_index("velocity")
        acceleration_index = self.field_index("acceleration")
        force_index = self.field_index("force")
        vertex_mass = self.data.x[:,mass_index]
        self.data.x[:,force_index] = forces
        
                # Use proportional interpolation for improved integration
        force_magnitude = torch.norm(forces, dim=1, keepdim=True) + 1e-10
        gradient_magnitude = torch.norm(force_gradients, dim=1, keepdim=True) + 1e-10
        alpha = torch.clamp(gradient_magnitude / (force_magnitude + gradient_magnitude), 0, 1)
        scaled_gradient = (force_gradients / gradient_magnitude) * force_magnitude
        interpolated_force = (1 - alpha) * forces + alpha * scaled_gradient

        # Compute magnitudes
        force_magnitude = torch.norm(forces, dim=1, keepdim=True) + 1e-10
        gradient_magnitude = torch.norm(force_gradients, dim=1, keepdim=True) + 1e-10

        # Compute the projection of the gradient onto the force vector (parallel component)
        gradient_parallel = (torch.sum(force_gradients * forces, dim=1, keepdim=True) / force_magnitude**2) * forces

        # Use the parallel component of the gradient to scale the force
        # If parallel gradient points uphill (positive projection), resist motion
        # If parallel gradient points downhill (negative projection), enhance motion
        scaling_factor = torch.clamp(1.0 - gradient_parallel / force_magnitude, min=0.1, max=1.0)
        scaled_force = forces * scaling_factor

        # Update momentum using the scaled force
        dp = scaled_force * time_step
        self.data.x[:, self.field_index("momentum")] += dp


        #print(forces > interpolated_force)
        #pos = pos.requires_grad_(False)
        # Compute change in momentum and update state
        momentum_index = self.field_index("momentum")
        dp = interpolated_force * time_step
        self.data.x[:, momentum_index] += dp

        # Update positions and velocities using the Relativistic Inertial Operator
        updated_state = self.rio.update_state(
            self.data.x, self.field_index, self.data.edge_index, time_step
        )

        # Update position directly from the updated state
        self.data.pos = updated_state[:, self.field_index("position")]

        # Enforce positional invariance for invariant vertices
        self.data.pos[self.invariant_mask] = self.vertices[self.invariant_mask]
        
        return self.data, time_step
    def _proposed_forward(self, time_step):
        #set sample points to include all vertices
        #set sample points to include all points in the force field stencil, the freedom of movement arcs of the edges
        # as well as extensions and contraction of the edges, all of these considered from each vertex's perspective
        # as the local gradients of freedom taken in isolation of any other movements
        # then overlay all these gradients in a blurred additive process
        #set gradient tracking
        #evaluate forces with dimensional explosion in the near field, a high resolution band of values along the edge as if the vertex displaced the edge to that point
        #extrapolate arcs that smear the high resolution bands into the same expression over the near-vertex region
        #obtain gradients at all points identified in the overlapping smearing
        #exclude all points which violate hameltonian while retaining the information of which is closest
        #use our relativistic integrator and the best possible methods to micro-evolve the path until the forces accumulated are fully and exactly dissipated


        # Build sample points

        sample_points = self.data.pos






        return self.data, time_step
    def _compute_barrier_rejection_force(self, vel, pos, params):
        """
        Compute rejection forces and apply extreme dampening for vertices interacting with a barrier.

        Args:
            vel (torch.Tensor): Vertex velocities (N, 3).
            pos (torch.Tensor): Vertex positions (N, 3).
            params (dict): Parameters defining the barrier.
                - "type": Type of barrier ("plane", "sphere", "box").
                - "magnitude": Rejection strength.
                - "distance_threshold": Threshold for activation.
                - "falloff": "linear", "exponential", or "hard".
                - "barrier_params": Specific parameters for the barrier (e.g., normal, center).

        Returns:
            torch.Tensor: Rejection force contributions for each vertex.
        """
        forces = torch.zeros_like(pos)
        barrier_type = params.get("barrier_type", "box")
        magnitude = params.get("magnitude", 1.0)
        distance_threshold = params.get("distance_threshold", 0.1)
        falloff = params.get("falloff", "linear")
        dampening_factor = params.get("dampening_factor", 1.0)  # Extreme dampening scaling factor
        barrier_params = params.get("barrier_params", {})

        if barrier_type == "plane":
            # Plane defined by normal vector and a point
            normal = torch.tensor(barrier_params.get("normal", [0, 1, 0]), device=pos.device, dtype=pos.dtype)
            point_on_plane = torch.tensor(barrier_params.get("point", [0, -1, 0]), device=pos.device, dtype=pos.dtype)

            # Compute signed distance to plane
            distances = torch.matmul(pos - point_on_plane, normal)
            penetration_mask = distances < distance_threshold

            # Compute rejection forces
            rejection_forces = torch.zeros_like(pos)
            if falloff == "linear":
                rejection_forces[penetration_mask] = -magnitude * (distances[penetration_mask] - distance_threshold).unsqueeze(1) * normal
            elif falloff == "exponential":
                rejection_forces[penetration_mask] = -magnitude * torch.exp(-distances[penetration_mask].unsqueeze(1)) * normal
            elif falloff == "hard":
                rejection_forces[penetration_mask] = -magnitude * normal

            # Apply rejection forces
            forces[penetration_mask] += rejection_forces[penetration_mask]

            # Apply extreme dampening
            velocity_projection = torch.sum(vel[penetration_mask] * normal, dim=1, keepdim=True)
            dampening = -dampening_factor * velocity_projection * normal
            forces[penetration_mask] += dampening

        elif barrier_type == "sphere":
            # Sphere defined by center and radius
            center = torch.tensor(barrier_params.get("center", [0, 0, 0]), device=pos.device, dtype=pos.dtype)
            radius = barrier_params.get("radius", 1.0)

            # Compute distance to sphere surface
            r_ij = pos - center
            distances = torch.norm(r_ij, dim=1) - radius
            penetration_mask = distances < distance_threshold

            # Compute rejection forces
            direction = r_ij / (distances.unsqueeze(1) + 1e-10)  # Avoid division by zero
            rejection_forces = torch.zeros_like(pos)
            if falloff == "linear":
                rejection_forces[penetration_mask] = -magnitude * (distances[penetration_mask] - distance_threshold).unsqueeze(1) * direction[penetration_mask]
            elif falloff == "exponential":
                rejection_forces[penetration_mask] = -magnitude * torch.exp(-distances[penetration_mask].unsqueeze(1)) * direction[penetration_mask]
            elif falloff == "hard":
                rejection_forces[penetration_mask] = -magnitude * direction[penetration_mask]

            # Apply rejection forces
            forces[penetration_mask] += rejection_forces[penetration_mask]

            # Apply extreme dampening
            velocity_projection = torch.sum(vel[penetration_mask] * direction[penetration_mask], dim=1, keepdim=True)
            dampening = -dampening_factor * velocity_projection * direction[penetration_mask]
            forces[penetration_mask] += dampening

        elif barrier_type == "box":
            # Box defined by min and max bounds
            min_bounds = torch.tensor(barrier_params.get("min_bounds", [-2.5, -2.5, -2.5]), device=pos.device, dtype=pos.dtype)
            max_bounds = torch.tensor(barrier_params.get("max_bounds", [2.5, 2.5, 2.5]), device=pos.device, dtype=pos.dtype)

            # Check for penetration in each dimension
            below_min = pos < min_bounds.unsqueeze(0)
            above_max = pos > max_bounds.unsqueeze(0)
            
            # Compute rejection forces for each violated dimension
            for dim in range(pos.shape[1]):
                if falloff == "linear":
                    forces[:, dim] -= magnitude * (pos[:, dim] - min_bounds[dim]) * below_min[:, dim]
                    forces[:, dim] += magnitude * (max_bounds[dim] - pos[:, dim]) * above_max[:, dim]
                elif falloff == "exponential":
                    forces[:, dim] += magnitude * torch.exp(-(pos[:, dim] - min_bounds[dim])) * below_min[:, dim]
                    forces[:, dim] -= magnitude * torch.exp(-(max_bounds[dim] - pos[:, dim])) * above_max[:, dim]
                elif falloff == "hard":
                    forces[:, dim] += magnitude * below_min[:, dim]
                    forces[:, dim] -= magnitude * above_max[:, dim]

                # Apply extreme dampening
                velocity_projection = vel[:, dim] * below_min[:, dim] + vel[:, dim] * above_max[:, dim]
                forces[:, dim] -= dampening_factor * velocity_projection * 1e-18 * 0

        return forces
    def _compute_inverse_square(self, pos, edges, params, vertex_state):
        """
        Compute forces and gradients based on the inverse square law.

        Args:
            pos (torch.Tensor): Vertex positions.
            edges (torch.Tensor): Edge indices.
            params (dict): Field parameters including magnitude and falloff.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Force contributions and force gradients for each vertex.
        """
        i, j = edges  # Source and target vertices
#        print(f"Edges: {edges.shape}, i: {i.shape}, j: {j.shape}")

        r_ij = (pos[j] - pos[i]) * SPACE["raw_dimensionality"]["distance_scale"]  # Displacements
#        print(f"r_ij (displacements): {r_ij.shape}")

        state_ij = params["magnitude"] * vertex_state[j] * vertex_state[i]
#        print(f"state_ij: {state_ij.shape}")

        distances = torch.norm(r_ij, dim=1, keepdim=True) + 1e-10  # Shape: (num_edges, 1)
#        print(f"distances: {distances.shape}")

        # Compute forces
        force_magnitudes = state_ij / distances ** params["falloff_exponent"]
#        print(f"force_magnitudes: {force_magnitudes.shape}")

        forces = (r_ij / distances) * force_magnitudes  # Normalize and scale forces
#        print(f"forces: {forces.shape}")

        # Gradient components
        identity = torch.eye(3, device=pos.device).unsqueeze(0).expand(r_ij.size(0), -1, -1)  # Shape: (num_edges, 3, 3)
#        print(f"identity: {identity.shape}")

        distances_expanded = distances.unsqueeze(-1).expand(-1, 3, 3)  # Shape: (num_edges, 3, 3)
#        print(f"distances_expanded (broadcasted): {distances_expanded.shape}")

        # Correct the numerator to match the denominator's shape
        force_magnitudes_expanded = force_magnitudes.unsqueeze(-1).expand(-1, 3, 3)  # Shape: (num_edges, 3, 3)
#        print(f"force_magnitudes_expanded: {force_magnitudes_expanded.shape}")

        grad_forces = (
            (-2 * params["falloff_exponent"] * force_magnitudes_expanded / distances_expanded)
            * (identity - (r_ij.unsqueeze(2) * r_ij.unsqueeze(1)) / (distances_expanded ** 2))
        )  # Shape: (num_edges, 3, 3)
#        print(f"grad_forces: {grad_forces.shape}")

        # Aggregate forces and gradients at source and target vertices
        vertex_forces = torch.zeros_like(pos)
        vertex_forces.index_add_(0, i, forces)
        vertex_forces.index_add_(0, j, -forces)  # Opposite direction for target
#        print(f"vertex_forces after aggregation: {vertex_forces.shape}")

        vertex_gradients = torch.zeros_like(pos)
        vertex_gradients.index_add_(0, i, grad_forces.sum(dim=1))  # Sum along the matrix axis for each vertex
        vertex_gradients.index_add_(0, j, -grad_forces.sum(dim=1))  # Opposite direction for target
#        print(f"vertex_gradients after aggregation: {vertex_gradients.shape}")

        return vertex_forces, vertex_gradients
    def _compute_spring_force(self, vel, pos, edges, params, vertex_state):
        """
        Compute forces and gradients based on Hooke's Law for spring-like interactions.

        Args:
            pos (torch.Tensor): Vertex positions.
            edges (torch.Tensor): Edge indices.
            params (dict): Field parameters including spring constant and rest length.
            vertex_state (torch.Tensor): State tensor for bond_spring field.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Force contributions and force gradients for each vertex.
        """
        i, j = edges  # Source and target vertices
        rest_length = params.get("rest_length", SPACE["geometry_rules"]["edge_dynamics"]["preferred_length"])
        spring_constant = params["magnitude"] * (vertex_state[i] + vertex_state[j]) / 2  # Hookean spring constant

        # Compute displacements and distances
        r_ij = (pos[j] - pos[i]) * SPACE["raw_dimensionality"]["distance_scale"]  # Displacements
        distances = torch.norm(r_ij, dim=1, keepdim=True) + 1e-10  # Avoid division by zero
        direction = r_ij / distances  # Normalized direction vector

        # Hooke's Law: F = -k * (|distance - rest_length|) * direction
        extension = distances - rest_length  # Difference from rest length
        force_magnitudes = spring_constant * extension
        forces = direction * force_magnitudes  # Apply direction to force magnitudes

        # Compute gradients
        spring_constant_expanded = spring_constant.unsqueeze(-1).unsqueeze(-1)  # Shape: [num_edges, 1, 1]
        distances_expanded = distances.unsqueeze(-1)  # Shape: [num_edges, 1, 1]
        identity = torch.eye(3, device=pos.device).unsqueeze(0).expand(r_ij.size(0), -1, -1)  # Shape: [num_edges, 3, 3]

        grad_forces = (
            (spring_constant_expanded / distances_expanded)
            * (identity - direction.unsqueeze(2) * direction.unsqueeze(1))
        )  # Shape: [num_edges, 3, 3]
        
        # Aggregate forces and gradients at source and target vertices
        vertex_forces = torch.zeros_like(pos)
        vertex_forces.index_add_(0, i, forces)
        vertex_forces.index_add_(0, j, -forces)  # Opposite direction for target

        vertex_gradients = torch.zeros_like(pos)
        vertex_gradients.index_add_(0, i, grad_forces.sum(dim=(1, 2)))  # Sum over all axes except batch
        vertex_gradients.index_add_(0, j, -grad_forces.sum(dim=(1, 2)))  # Opposite direction for target
        
        return vertex_forces, vertex_gradients


class SphereGraph:
    def __init__(self, num_points=10, device="cuda"):
        self.num_points = num_points
        self.device = torch.device(device)
        self.precision = torch.float64
        self.offsets = None
        self.edge_pairs = None
    def fibonacci_sphere(self, num_points, device='cpu', dtype=torch.float64):
        import math
        """Generate approximately evenly distributed points on the sphere using Fibonacci sampling."""
        indices = torch.arange(0, num_points, dtype=dtype, device=device) + 0.5
        phi = (2 * math.pi) * indices / num_points  # Azimuthal angle
        theta = torch.acos(1 - 2 * indices / num_points)  # Polar angle

        # Convert to Cartesian coordinates
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack((x, y, z), dim=-1)
    def generate_graph(self):
        # Central point at the origin
        center = torch.tensor([[0.0, 0.0, 0.0]], dtype=self.precision, device=self.device)

        # Generate random points on the unit sphere
        #phi = torch.rand(self.num_points, dtype=self.precision, device=self.device) * 2 * torch.pi  # Random azimuth angle
        #theta = torch.acos(2 * torch.rand(self.num_points, device=self.device) - 1)  # Random polar angle
        # Generate an even mesh of points on the unit sphere
        phi = torch.linspace(0, 2 * torch.pi, int(self.num_points ** 0.5), dtype=self.precision, device=self.device)
        theta = torch.linspace(0, torch.pi, int(self.num_points ** 0.5), dtype=self.precision, device=self.device)

        # Create a meshgrid for phi and theta
        phi, theta = torch.meshgrid(phi[0:-1], theta[1:-1], indexing='ij')

        # Compute Cartesian coordinates
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        # Flatten the coordinates if a single list of points is needed
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()


        #sphere_points = torch.cat([torch.stack([x, y, z], dim=1), self.fibonacci_sphere(self.num_points, self.device, self.precision)])
        #sphere_points = self.fibonacci_sphere(self.num_points, self.device, self.precision)
        sphere_points = torch.stack([x, y, z], dim=1)
        # Combine the central point with the sphere points
        self.offsets = torch.cat([center, sphere_points], dim=0)

        # Generate edges: connect each sphere point to the center (index 0)
        self.edge_pairs = torch.tensor(
            [(0, i) for i in range(1, sphere_points.shape[0] + 1)], dtype=torch.int64, device=self.device
        )

    def get_graph(self):
        return {
            "offsets": self.offsets,
            "edge_pairs": self.edge_pairs
        }


class CompositeGeometry:
    def __init__(self, geometry="tetrahedron", precision=torch.float64, device="cuda"):
        self.geometry = geometry
        self.precision = precision
        self.device = torch.device(device)
        self.network_geometry = None
        if self.geometry == "sphere":
            self.sphere_count = 640
            self.sphere = SphereGraph(self.sphere_count, device)
            self.sphere.generate_graph()
            self.sphere_properties = self.sphere.get_graph()
            self.sphere_offsets = self.sphere_properties["offsets"]
            self.sphere_edges = self.sphere_properties["edge_pairs"]
    def load_network_override(self, new_edges, new_offsets):
        self.network_geometry = {"offsets": new_offsets, "edge_pairs":new_edges}
    def define_offsets(self, density, micro_jitter=False):
        if self.geometry == "network_configured" and self.network_geometry is not None:
            offsets = self.network_geometry["offsets"].clone()
        elif self.geometry == "ray":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0]
            ], dtype = self.precision, device = self.device)
        elif self.geometry == "triangle":
            sqrt3 = torch.sqrt(torch.tensor(3.0, dtype=self.precision, device=self.device))
            offsets = torch.tensor([
                [0, 0, 0],            # Vertex A
                [1, 0, 0],            # Vertex B
                [0.5, sqrt3 / 2, 0]   # Vertex C (equilateral height)
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "sphere":
            offsets = self.sphere_offsets
        elif self.geometry == "cube":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "tetrahedron":
            offsets = torch.tensor([
                [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "square":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "octahedron":
            offsets = torch.tensor([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "icosahedron":
            phi = (1 + 5 ** 0.5) / 2  # Golden ratio
            offsets = torch.tensor([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ], dtype=self.precision, device=self.device) / phi
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        
        offsets /= density

        # Optionally apply micro jitter
        if micro_jitter:
            jitter_strength = 1e-8
            jitter = torch.randn_like(offsets) * jitter_strength
            offsets += jitter

        # Calculate tile size and centered offsets
        tile_size = offsets.max(dim=0).values - offsets.min(dim=0).values
        centroid = offsets.mean(dim=0)
        centered_offsets = offsets - centroid

        return centered_offsets, tile_size

    def configure_geometry(self, geometry=None, density=1, micro_jitter=False):
        if geometry is not None:
            self.geometry = geometry
        """Configure vertex and edge definitions for the geometry."""
        if self.geometry == "network_configured":
            vertex_count = self.network_geometry["offsets"].shape[0]
            edge_pairs = self.network_geometry["edge_pairs"].clone()
        elif self.geometry == "ray":
            vertex_count = 2
            edge_pairs = torch.tensor([(0,1)], dtype=torch.int64, device=self.device)
        elif self.geometry == "square":
            vertex_count = 4
            edge_pairs = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=torch.int64, device=self.device)
        elif self.geometry == "triangle":
            vertex_count = 3
            edge_pairs = torch.tensor([(0,1), (1,2), (2, 0)], dtype=torch.int64, device=self.device)
        elif self.geometry == "cube":
            vertex_count = 8
            edge_pairs = torch.tensor([
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ], dtype=torch.int64, device=self.device)
        elif self.geometry == "tetrahedron":
            vertex_count = 4
            edge_pairs = torch.tensor([
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ], dtype=torch.int64, device=self.device)
        elif self.geometry == "octahedron":
            vertex_count = 6
            edge_pairs = torch.tensor([
                (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 4), (2, 5), (3, 4), (3, 5)
            ], dtype=torch.int64, device=self.device)
        elif self.geometry == "icosahedron":
            vertex_count = 12
            edge_pairs = torch.tensor([
                (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
                (1, 5), (1, 6), (1, 8), (1, 9),
                (2, 3), (2, 4), (2, 6), (2, 9), (2, 11),
                (3, 4), (3, 7), (3, 8), (3, 10),
                (4, 5), (4, 7), (4, 9),
                (5, 10), (6, 8), (6, 11),
                (7, 8), (7, 9), (8, 11),
                (9, 10), (9, 11), (10, 11)
                
            ], dtype=torch.int64, device=self.device)
        elif self.geometry == "sphere":
            vertex_count = self.sphere_count
            edge_pairs = self.sphere_edges#torch.tensor([], dtype=torch.int64, device=self.device)#self.sphere_edges
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        offsets, tile_size = self.define_offsets(density, micro_jitter)
        edge_lengths = torch.norm(
            offsets[edge_pairs[:, 0]] - offsets[edge_pairs[:, 1]],
            dim=1
        ) if edge_pairs.shape[0] > 0 else None
        return offsets.requires_grad_(False), torch.tensor(vertex_count, device=self.device, dtype=self.precision), edge_pairs, edge_lengths, tile_size


import torch

class FaceMapGenerator:
    def __init__(self, vertices, edges, device="cpu"):
        """
        Initialize the FaceMapGenerator.
        Args:
            vertices: Tensor of vertex positions (N, 3).
            edges: Tensor of edge pairs (E, 2).
            device: 'cuda' or 'cpu'.
        """
        self.vertices = vertices.to(device)
        self.edges = edges.to(device)
        self.device = device
        self.graph = self.build_edge_graph()

    def build_edge_graph(self):
        """Build a graph (adjacency list) from edges."""
        from collections import defaultdict
        graph = defaultdict(list)
        for edge in self.edges:
            u, v = edge.tolist()
            graph[u].append(v)
            graph[v].append(u)  # Undirected graph
        return graph

    def find_edge_loops(self, max_face_size=6):
        """
        Find unique edge circuits (candidate faces) using DFS.
        Args:
            max_face_size: Max allowable vertices in a face.
        Returns:
            List of valid edge loops.
        """
        visited_edges = set()
        faces = []

        def dfs(current, start, path):
            """Recursive depth-first search to find loops."""
            if len(path) > max_face_size:
                return
            for neighbor in self.graph[current]:
                edge = tuple(sorted((current, neighbor)))
                if neighbor == start and len(path) > 2:
                    faces.append(path + [start])  # Found a loop
                    return
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    dfs(neighbor, start, path + [neighbor])
                    visited_edges.remove(edge)

        for start in range(len(self.vertices)):
            dfs(start, start, [start])

        # Eliminate duplicates (loops in reverse order)
        unique_faces = set(tuple(sorted(face)) for face in faces)
        return [list(face) for face in unique_faces]

    def is_planar(self, face):
        """
        Check if a face is planar using cross-products.
        Args:
            face: List of vertex indices.
        Returns:
            True if planar, else False.
        """
        v0 = self.vertices[face[0]]
        normal = None
        for i in range(1, len(face) - 1):
            edge1 = self.vertices[face[i]] - v0
            edge2 = self.vertices[face[i + 1]] - v0
            cross_product = torch.cross(edge1, edge2)
            if normal is None:
                normal = cross_product
            else:
                if not torch.allclose(normal, cross_product, atol=1e-6):
                    return False
        return True

    def generate_face_map(self, check_planarity=True):
        """
        Generate the face map for the geometry.
        Args:
            check_planarity: Whether to enforce planarity.
        Returns:
            Dictionary mapping face indices to vertex indices.
        """
        face_map = {}
        candidate_faces = self.find_edge_loops()
        face_index = 0

        for face in candidate_faces:
            if check_planarity and not self.is_planar(face):
                continue
            face_map[face_index] = face
            face_index += 1

        return face_map

import torch
import hashlib

class HodgeStarBuilder:
    """
    Constructs Hodge star operators for a given simplicial complex and caches them
    based on the hashed topology for reuse.
    """
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.hodge_cache = {}  # Cache for Hodge star matrices keyed by topology hash

    def _hash_topology(self, vertices, edges, faces):
        """
        Hash the graph topology using vertices, edges, and faces for caching.
        """
        topology_data = (vertices.shape, edges.cpu().numpy().tobytes(), faces.cpu().numpy().tobytes())
        return hashlib.sha256(b''.join([str(t).encode() for t in topology_data])).hexdigest()

    def compute_vertex_volumes(self, vertices, faces):
        """Compute vertex volumes for 0-forms."""
        volumes = torch.zeros(vertices.shape[0], device=self.device)
        for face in faces:
            v0, v1, v2 = vertices[face]
            area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
            for vertex in face:
                volumes[vertex] += area / 3.0
        return volumes

    def compute_edge_dual_areas(self, vertices, edges, faces):
        """Compute dual areas for edges (1-forms)."""
        edge_dual_areas = torch.zeros(edges.shape[0], device=self.device)
        for i, edge in enumerate(edges):
            shared_faces = faces[(faces == edge[0]).any(dim=1) & (faces == edge[1]).any(dim=1)]
            dual_area = 0.0
            for face in shared_faces:
                v0, v1, v2 = vertices[face]
                area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
                dual_area += area / 3.0
            edge_dual_areas[i] = dual_area
        return edge_dual_areas

    def compute_face_areas(self, vertices, faces):
        """Compute areas for 2-forms."""
        areas = torch.zeros(faces.shape[0], device=self.device)
        for i, face in enumerate(faces):
            v0, v1, v2 = vertices[face]
            areas[i] = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
        return areas

    def build_hodge_star(self, vertices, edges, faces):
        """
        Build Hodge star operators for 0-forms, 1-forms, and 2-forms, with caching.
        """
        hash_key = self._hash_topology(vertices, edges, faces)
        if hash_key in self.hodge_cache:
            print("Reusing cached Hodge star operators.")
            return self.hodge_cache[hash_key]

        print("Computing new Hodge star operators.")
        hodge_0 = self.compute_vertex_volumes(vertices, faces)
        hodge_1 = self.compute_edge_dual_areas(vertices, edges, faces)
        hodge_2 = self.compute_face_areas(vertices, faces)

        hodge_stars = (torch.diag(hodge_0), torch.diag(hodge_1), torch.diag(hodge_2))
        self.hodge_cache[hash_key] = hodge_stars
        return hodge_stars


class CompositeGeometryDEC(CompositeGeometry):
    def __init__(self, base_geometry, precision=torch.float64, device="cpu"):
        self.base_geometry = base_geometry
        super().__init__(base_geometry, precision, device)
        self.hodge_star_builder = HodgeStarBuilder(device)
        self.hodge_stars = None
        self.grad_matrix = None
        self.curl_matrix = None

    def initialize_DEC(self, faces):
        """
        Initialize DEC operators: Hodge stars and incidence matrices.
        Args:
            faces (Tensor): Face definitions for the simplicial complex.
        """
        self.faces = faces.to(self.device)
        self.hodge_stars = self.hodge_star_builder.build_hodge_star(
            self.vertices, self.edges, self.faces
        )
        self.grad_matrix, self.curl_matrix = self.compute_exterior_derivative()
        print("DEC operators initialized.")

    def compute_exterior_derivative(self):
        """Compute incidence matrices for the exterior derivative."""
        num_vertices = self.vertices.shape[0]
        num_edges = self.edges.shape[0]
        num_faces = self.faces.shape[0]

        # Build gradient matrix
        grad_matrix = torch.zeros((num_edges, num_vertices), device=self.device)
        for i, (v0, v1) in enumerate(self.edges):
            grad_matrix[i, v0] = -1
            grad_matrix[i, v1] = 1

        # Build curl matrix
        curl_matrix = torch.zeros((num_faces, num_edges), device=self.device)
        for i, face in enumerate(self.faces):
            for j in range(3):  # Loop over edges of the face
                edge = tuple(sorted((face[j], face[(j + 1) % 3])))
                edge_idx = torch.where((self.edges == edge).all(dim=1))[0]
                if edge_idx.numel() > 0:
                    curl_matrix[i, edge_idx] = 1
        return grad_matrix, curl_matrix

    def laplace_beltrami_operator(self):
        """
        Compute the Laplace-Beltrami operator Δ = d* d for scalar fields.
        """
        hodge_0, hodge_1, _ = self.hodge_stars
        grad, _ = self.compute_exterior_derivative()

        # Laplace-Beltrami operator: Δ = *d*d
        laplace = grad.T @ hodge_1 @ grad @ torch.inverse(hodge_0)
        return laplace





import torch

# -------------------------------
# Base Shape Constant Definitions
# -------------------------------

SHAPE_DEFINITIONS = {
    "cube": {
        "vertices": [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ],
        "edges": [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
    },
    "tetrahedron": {
        "vertices": [
            [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
        ],
        "edges": [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        ]
    },
    "octahedron": {
        "vertices": [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ],
        "edges": [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (2, 5), (3, 4), (3, 5)
        ]
    }
}

# ----------------
# Shape Class
# ----------------

class Shape:
    def __init__(self, name, device="cpu"):
        """
        Represents a geometric structure with vertices and edges.

        Args:
            name (str): Name of the shape ('cube', 'tetrahedron', etc.).
            device (str): Target device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        if name not in SHAPE_DEFINITIONS:
            raise ValueError(f"Shape '{name}' is not defined.")
        shape_data = SHAPE_DEFINITIONS[name]
        self.vertices = torch.tensor(shape_data["vertices"], dtype=torch.float64, device=self.device)
        self.edges = torch.tensor(shape_data["edges"], dtype=torch.int64, device=self.device)
        self.bitmask = None  # For activation mapping
        self.interpolated_edges = None  # For interpolated values

    def calculate_bitmask(self, activation_values, threshold):
        """
        Calculate the activation bitmask for vertices.

        Args:
            activation_values (torch.Tensor): Scalar field values for each vertex.
            threshold (float): Threshold value for activation.

        Returns:
            torch.Tensor: Bitmask of activated vertices.
        """
        self.bitmask = (activation_values > threshold).int()
        return self.bitmask

    def interpolate_edges(self, activation_values, threshold):
        """
        Calculate interpolated edge crossings for a target value.

        Args:
            activation_values (torch.Tensor): Scalar field values for each vertex.
            threshold (float): Threshold value for interpolation.

        Returns:
            torch.Tensor: Interpolated edge crossings.
        """
        self.interpolated_edges = []
        for edge in self.edges:
            v0, v1 = edge
            f0, f1 = activation_values[v0], activation_values[v1]
            if (f0 - threshold) * (f1 - threshold) < 0:  # Edge crosses threshold
                t = (threshold - f0) / (f1 - f0)
                self.interpolated_edges.append((1 - t) * self.vertices[v0] + t * self.vertices[v1])
        self.interpolated_edges = torch.stack(self.interpolated_edges) if self.interpolated_edges else None
        return self.interpolated_edges



# ----------------
# Shuttle Class
# ----------------

class Shuttle:
    def __init__(self, weft, warp, scalar_field, weft_resolution="activation", warp_resolution="activation", device="cpu"):
        """
        Combines weft and warp shapes into a shuttle object.

        Args:
            weft (Shape): The weft geometry.
            warp (Shape): The warp geometry.
            scalar_field (torch.Tensor): Scalar field defining affinities.
            weft_resolution (str): Resolution for weft ('activation' or 'interpolation').
            warp_resolution (str): Resolution for warp ('activation' or 'interpolation').
            device (str): Target device ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.weft = weft
        self.warp = warp
        self.scalar_field = scalar_field
        self.weft_resolution = weft_resolution
        self.warp_resolution = warp_resolution
        self.vertices = self.get_combined_vertices()
        self.edges = self.get_combined_edges()
        self.E3 = self.define_E3()

        # Calculate resolution-specific data
        self.process_resolutions()

    def get_combined_vertices(self):
        """Combine weft and warp vertices."""
        return torch.cat((self.weft.vertices, self.warp.vertices), dim=0)

    def get_combined_edges(self):
        """Combine weft and warp edges, adjusting warp indices."""
        warp_edges = self.warp.edges + self.weft.vertices.shape[0]
        return torch.cat((self.weft.edges, warp_edges), dim=0)

    def define_E3(self):
        """
        Define the asymmetrical edge E3 probabilistically based on scalar field.

        Returns:
            tuple: Indices of vertices forming E3.
        """
        centroid = self.vertices.mean(dim=0)
        probabilities = torch.softmax(self.scalar_field, dim=0)
        edge_idx = torch.multinomial(probabilities, num_samples=1).item()
        candidate_edge = self.edges[edge_idx]
        start, end = candidate_edge
        direction = self.vertices[end] - self.vertices[start]
        chirality = torch.dot(direction, torch.tensor([0, 0, 1], dtype=torch.float64, device=self.device))
        if chirality < 0:
            return (end, start)
        return (start, end)

    def process_resolutions(self):
        """
        Process weft and warp resolutions (activation or interpolation).
        """
        if self.weft_resolution == "activation":
            self.weft.calculate_bitmask(self.scalar_field[:self.weft.vertices.shape[0]], threshold=0.5)
        elif self.weft_resolution == "interpolation":
            self.weft.interpolate_edges(self.scalar_field[:self.weft.vertices.shape[0]], threshold=0.5)

        if self.warp_resolution == "activation":
            self.warp.calculate_bitmask(self.scalar_field[self.weft.vertices.shape[0]:], threshold=0.5)
        elif self.warp_resolution == "interpolation":
            self.warp.interpolate_edges(self.scalar_field[self.weft.vertices.shape[0]:], threshold=0.5)

    def visualize(self):
        """Visualize the shuttle's graph network."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        vertices = self.vertices.cpu().numpy()
        edges = self.edges.cpu().numpy()
        E3 = self.E3

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot edges
        for edge in edges:
            start, end = edge
            x = [vertices[start, 0], vertices[end, 0]]
            y = [vertices[start, 1], vertices[end, 1]]
            z = [vertices[start, 2], vertices[end, 2]]
            ax.plot(x, y, z, color="black", linewidth=0.5)

        # Highlight E3
        x = [vertices[E3[0], 0], vertices[E3[1], 0]]
        y = [vertices[E3[0], 1], vertices[E3[1], 1]]
        z = [vertices[E3[0], 2], vertices[E3[1], 2]]
        ax.plot(x, y, z, color="red", linewidth=2, label="E3 (Chiral Edge)")

        plt.legend()
        plt.show()
