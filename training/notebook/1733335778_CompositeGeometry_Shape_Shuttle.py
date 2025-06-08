import torch
class CompositeGeometry:
    def __init__(self, geometry="tetrahedron", precision=torch.float64, device="cuda"):
        self.geometry = geometry
        self.precision = precision
        self.device = torch.device(device)
        self.network_geometry = None
    def load_network_override(self, new_edges, new_offsets):
        self.network_geometry = {"offsets": new_offsets, "edge_pairs":new_edges}
    def define_offsets(self, density, micro_jitter=False):
        if self.geometry == "network_configured" and self.network_geometry is not None:
            offsets = self.network_geometry["offsets"].clone()
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
        elif self.geometry == "square":
            vertex_count = 4
            edge_pairs = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=torch.int64, device=self.device)
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
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")
        offsets, tile_size = self.define_offsets(density, micro_jitter)
        edge_lengths = torch.norm(
            offsets[edge_pairs[:, 0]] - offsets[edge_pairs[:, 1]],
            dim=1
        )
        return offsets.requires_grad_(True), torch.tensor(vertex_count, device=self.device, dtype=self.precision), edge_pairs, edge_lengths, tile_size
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
    def __init__(self, name, device="cuda"):
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
    def __init__(self, weft, warp, scalar_field, weft_resolution="activation", warp_resolution="activation", device="cuda"):
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
