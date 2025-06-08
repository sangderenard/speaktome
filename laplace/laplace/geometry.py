"""Discrete Exterior Calculus utilities."""
# --- END HEADER ---

import hashlib
import torch


class HodgeStarBuilder:
    """Constructs Hodge star operators with simple caching."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.hodge_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _hash_topology(self, vertices: torch.Tensor, edges: torch.Tensor, faces: torch.Tensor) -> str:
        topology_data = (vertices.shape, edges.cpu().numpy().tobytes(), faces.cpu().numpy().tobytes())
        return hashlib.sha256(b"".join([str(t).encode() for t in topology_data])).hexdigest()

    def compute_vertex_volumes(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        volumes = torch.zeros(vertices.shape[0], device=self.device)
        for face in faces:
            v0, v1, v2 = vertices[face]
            area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
            for vertex in face:
                volumes[vertex] += area / 3.0
        return volumes

    def compute_edge_dual_areas(self, vertices: torch.Tensor, edges: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
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

    def compute_face_areas(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        areas = torch.zeros(faces.shape[0], device=self.device)
        for i, face in enumerate(faces):
            v0, v1, v2 = vertices[face]
            areas[i] = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0))
        return areas

    def build_hodge_star(self, vertices: torch.Tensor, edges: torch.Tensor, faces: torch.Tensor):
        hash_key = self._hash_topology(vertices, edges, faces)
        if hash_key in self.hodge_cache:
            return self.hodge_cache[hash_key]

        hodge_0 = self.compute_vertex_volumes(vertices, faces)
        hodge_1 = self.compute_edge_dual_areas(vertices, edges, faces)
        hodge_2 = self.compute_face_areas(vertices, faces)
        stars = (torch.diag(hodge_0), torch.diag(hodge_1), torch.diag(hodge_2))
        self.hodge_cache[hash_key] = stars
        return stars


class CompositeGeometryDEC:
    """Geometry wrapper that exposes Laplace-Beltrami via DEC."""

    def __init__(self, vertices: torch.Tensor, edges: torch.Tensor, faces: torch.Tensor, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.vertices = vertices.to(self.device)
        self.edges = edges.to(self.device)
        self.faces = faces.to(self.device)
        self.hodge_builder = HodgeStarBuilder(device)
        self.hodge_stars = self.hodge_builder.build_hodge_star(self.vertices, self.edges, self.faces)
        self.grad_matrix, self.curl_matrix = self._exterior_derivative()

    def _exterior_derivative(self):
        num_vertices = self.vertices.shape[0]
        num_edges = self.edges.shape[0]
        num_faces = self.faces.shape[0]
        grad = torch.zeros((num_edges, num_vertices), device=self.device)
        for i, (v0, v1) in enumerate(self.edges):
            grad[i, v0] = -1
            grad[i, v1] = 1
        curl = torch.zeros((num_faces, num_edges), device=self.device)
        for i, face in enumerate(self.faces):
            for j in range(3):
                edge = tuple(sorted((face[j], face[(j + 1) % 3])))
                edge_idx = torch.where((self.edges == edge).all(dim=1))[0]
                if edge_idx.numel() > 0:
                    curl[i, edge_idx] = 1
        return grad, curl

    def laplace_beltrami(self) -> torch.Tensor:
        h0, h1, _ = self.hodge_stars
        laplace = self.grad_matrix.T @ h1 @ self.grad_matrix @ torch.inverse(h0)
        return laplace
