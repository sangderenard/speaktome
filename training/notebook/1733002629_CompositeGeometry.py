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
