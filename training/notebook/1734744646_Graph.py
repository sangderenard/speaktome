class Graph:
    def __init__(self, nodes=None, edges=None, fields=None, edge_types=None, device="cuda"):
        self.device = torch.device(device)
        self.nodes = nodes.to(self.device) if nodes is not None else None
        self.edges = edges.to(self.device) if edges is not None else None
        self.edge_types = edge_types.to(self.device) if edge_types is not None else None
        self.edge_types = {}
        
        # Initialize edge layers
        self.edge_type_index = 0
        self.edge_type_map = torch.tensor([], dtype=torch.float32, device=self.device)
        self.edge_layers = {}
        self.add_layer("default")  # Default layer for edges
        self.add_layer("fully_connected")  # Fully connected edge layer

        # Initialize fully connected edges if vertices are present
        if self.nodes is not None:
            self._initialize_fully_connected_layer()

        # Build tensor map from fields configuration
        self.tensor_map = self._assemble_tensor_map(fields)
        self.total_fields = self.tensor_map["total_length"]

        # Initialize fields in the state tensor
        self.fields = self.initialize_fields(fields)

        # Initialize subnetwork cache
        self.subnetwork_cache = []
        if self.vertices is not None:
            self._initialize_subnetwork_cache()

        # Create PyTorch Geometric Data object
        self.data = Data(
            pos=self.vertices,
            edge_index=self.edges,
            edge_type=self.edge_type_map,  # Edge type attribute
            x=self.fields
        )

    def _assemble_tensor_map(self, fields):
        """
        Build a mapping of fields to their corresponding state tensor slices.

        Args:
            fields (list): List of field configurations from YAML.
        
        Returns:
            dict: Tensor map containing offsets, lengths, and metadata for each field.
        """
        tensor_map = {
            "field_offsets": {},  # Maps field names to their start indices in the tensor
            "field_lengths": {},  # Maps field names to their lengths in the tensor
            "total_length": 0,    # Total length of the state tensor
        }

        DIMENSIONALITY = 3
        sizes = {
            "vector": DIMENSIONALITY,
            "scalar": 1,
            "tensor": -1,         # Placeholder for dynamically defined sizes
            "parametric": -1,     # Placeholder for runtime-defined output dimensions
            "latent": 1,
            "causal": 8           # Actor/observer ID pair in causal fields
        }

        current_offset = 0

        for field in fields:
            field_name = field["name"]
            mode = field["engine"]["mode"]
            size = sizes.get(mode)

            # Handle dynamically sized fields
            if size == -1:
                if mode == "tensor":
                    size = field.get("tensor_size", 1)  # Default to 1 if unspecified
                elif mode == "parametric":
                    size = len(field["engine"].get("outputs", []))  # Number of outputs defines parametric size
            
            # Validate size
            if size <= 0:
                raise ValueError(f"Invalid size {size} for field '{field_name}' in mode '{mode}'.")

            # Map field to tensor
            tensor_map["field_offsets"][field_name] = current_offset
            tensor_map["field_lengths"][field_name] = size
            current_offset += size

        tensor_map["total_length"] = current_offset
        return tensor_map

    def initialize_fields(self, fields):
        """
        Initialize the state tensor with zeros based on the tensor map.

        Args:
            fields (list): List of field configurations from YAML.
        
        Returns:
            torch.Tensor: Initialized state tensor.
        """
        num_nodes = self.nodes.shape[0] if self.nodes is not None else 0
        tensor_length = self.tensor_map["total_length"]
        return torch.zeros((num_nodes, tensor_length), device=self.device)

    def get_field_slice(self, field_name):
        """
        Retrieve the slice of the state tensor corresponding to a field.

        Args:
            field_name (str): Name of the field.

        Returns:
            slice: Slice object for accessing the field in the tensor.
        """
        offset = self.tensor_map["field_offsets"][field_name]
        length = self.tensor_map["field_lengths"][field_name]
        return slice(offset, offset + length)
