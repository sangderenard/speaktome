import torch
class BitmaskInterpreter:
    def __init__(self, vertex_count, edge_pairs, vertex_offsets, device='cuda', precision=torch.float64):
        """
        Initialize the BitmaskInterpreter with necessary geometric information.

        Args:
            vertex_count (int): Number of vertices in the geometry.
            edge_pairs (torch.Tensor): Tensor of shape (num_edges, 2) defining all valid edges.
            vertex_offsets (torch.Tensor): Tensor of shape (num_vertices, 3) defining the offsets of vertices.
            device (str or torch.device): The device to use for computations.
            precision (torch.dtype): The data type for computations.
        """
        self.vertex_count = vertex_count
        self.edge_pairs = edge_pairs.to(device=device)
        self.vertex_offsets = vertex_offsets.to(device=device)
        self.device = device
        self.precision = precision

        # Check for integer ceiling issues
        max_bitmask = 2 ** vertex_count - 1
        if max_bitmask > torch.iinfo(torch.int64).max:
            raise ValueError("Integer ceiling exceeded for bitmask representation with int64.")
        elif vertex_count > 64:
            raise ValueError("Vertex count exceeds 64 bits, standard integer types may not suffice.")

    def bitmask_to_active_vertices(self, bitmask):
        """
        Convert a bitmask to a list of active vertex indices.

        Args:
            bitmask (int or torch.Tensor): Bitmask representing active vertices.

        Returns:
            torch.Tensor: Tensor of active vertex indices.
        """
        if isinstance(bitmask, int):
            bitmask = torch.tensor([bitmask], device=self.device, dtype=torch.int64)
        else:
            bitmask = bitmask.to(device=self.device, dtype=torch.int64)

        # Generate masks for each vertex position
        power_values = 2 ** torch.arange(self.vertex_count, device=self.device, dtype=torch.int64)
        active_mask = (bitmask.unsqueeze(-1) & power_values) > 0  # Shape: (num_bitmasks, vertex_count)

        # Get active vertex indices
        active_vertices = [torch.nonzero(mask).flatten() for mask in active_mask]

        return active_vertices

    def active_vertices_to_bitmask(self, active_vertices):
        """
        Convert a list of active vertex indices to a bitmask.

        Args:
            active_vertices (list or torch.Tensor): List or tensor of active vertex indices.

        Returns:
            int: Bitmask representing the active vertices.
        """
        if isinstance(active_vertices, list):
            active_vertices = torch.tensor(active_vertices, device=self.device, dtype=torch.int64)
        else:
            active_vertices = active_vertices.to(device=self.device, dtype=torch.int64)

        bitmask = (2 ** active_vertices).sum().item()
        return bitmask

    def bitmask_to_active_edges(self, bitmask):
        """
        Determine which edges are active (cross the isosurface) based on the bitmask.

        Args:
            bitmask (int or torch.Tensor): Bitmask representing active vertices.

        Returns:
            torch.Tensor: Tensor of active edge indices.
        """
        active_vertices = self.bitmask_to_active_vertices(bitmask)
        active_edges_list = []
        for idx, active_vertex_indices in enumerate(active_vertices):
            # Create a mask for active vertices
            vertex_active_mask = torch.zeros(self.vertex_count, device=self.device, dtype=torch.bool)
            vertex_active_mask[active_vertex_indices] = True

            # Determine if each edge crosses the isosurface
            v1_active = vertex_active_mask[self.edge_pairs[:, 0]]
            v2_active = vertex_active_mask[self.edge_pairs[:, 1]]
            edge_crosses = v1_active != v2_active  # Edges where one vertex is active and the other is inactive
            active_edge_indices = torch.nonzero(edge_crosses).flatten()
            active_edges_list.append(active_edge_indices)

        return active_edges_list

    def reconstruct_geometry(self, bitmask):
        """
        Reconstruct the geometry (active vertices and edges) based on the bitmask.

        Args:
            bitmask (int or torch.Tensor): Bitmask representing active vertices.

        Returns:
            dict: Dictionary containing active vertex positions and active edges.
        """
        active_vertices = self.bitmask_to_active_vertices(bitmask)
        active_edges = self.bitmask_to_active_edges(bitmask)

        reconstructed_list = []
        for idx in range(len(active_vertices)):
            vertices = self.vertex_offsets[active_vertices[idx]]
            edges = self.edge_pairs[active_edges[idx]]
            reconstructed_list.append({
                'active_vertex_positions': vertices,
                'active_edges': edges
            })

        return reconstructed_list

    def interpret_bitmask(self, bitmask):
        """
        Provide a comprehensive interpretation of the bitmask, including active vertices,
        active edges, and potential issues.

        Args:
            bitmask (int or torch.Tensor): Bitmask representing active vertices.

        Returns:
            dict: Dictionary containing interpretation details.
        """
        active_vertices = self.bitmask_to_active_vertices(bitmask)
        active_edges = self.bitmask_to_active_edges(bitmask)

        interpretation_list = []
        for idx in range(len(active_vertices)):
            num_active_vertices = len(active_vertices[idx])
            num_active_edges = len(active_edges[idx])

            interpretation = {
                'bitmask': bitmask[idx].item() if isinstance(bitmask, torch.Tensor) else bitmask,
                'active_vertex_indices': active_vertices[idx].tolist(),
                'active_edge_indices': active_edges[idx].tolist(),
                'num_active_vertices': num_active_vertices,
                'num_active_edges': num_active_edges
            }
            interpretation_list.append(interpretation)

        return interpretation_list
