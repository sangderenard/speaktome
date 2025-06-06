import pygame
import torch
import random
import math

class CompositeGeometry:
    def __init__(self, geometry="tetrahedron", precision=torch.float64, device="cuda"):
        self.geometry = geometry
        self.precision = precision
        self.device = torch.device(device)
        self.network_geometry = None

    def load_network_override(self, new_edges, new_offsets):
        self.network_geometry = {"offsets": new_offsets, "edge_pairs": new_edges}

    def define_offsets(self, density, micro_jitter=False):
        """Define the vertex positions for the chosen geometry."""
        if self.geometry == "network_configured" and self.network_geometry is not None:
            offsets = self.network_geometry["offsets"].clone()
        elif self.geometry == "cube":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ], dtype=self.precision, device=self.device)
        elif self.geometry == "tetrahedron":
            offsets = torch.tensor([
                [1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1]
            ], dtype=self.precision, device=self.device)
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        offsets /= density
        if micro_jitter:
            jitter_strength = 1e-8
            jitter = torch.randn_like(offsets) * jitter_strength
            offsets += jitter

        centroid = offsets.mean(dim=0)
        centered_offsets = offsets - centroid
        return centered_offsets

    def configure_geometry(self, density=1, micro_jitter=False):
        """Configure the geometry and return vertices and edges."""
        offsets = self.define_offsets(density, micro_jitter)
        if self.geometry == "cube":
            edge_pairs = torch.tensor([
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ], dtype=torch.int64, device=self.device)
        elif self.geometry == "tetrahedron":
            edge_pairs = torch.tensor([
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ], dtype=torch.int64, device=self.device)
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        return offsets.requires_grad_(False), edge_pairs

def interpolate_edges(vertices, edges, steps=5):
    """Interpolate along edges to create additional points."""
    points = []
    for edge in edges:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        for t in torch.linspace(0, 1, steps):
            interpolated = (1 - t) * v0 + t * v1
            points.append(interpolated)
    return torch.stack(points)

def voxelize(points, voxel_size=0.1):
    """Voxelize points to grid coordinates."""
    coords = torch.floor(points / voxel_size).long()
    return coords

def compute_voxel_overlap(coords1, coords2):
    """Compute overlap between voxelized point clouds."""
    vox1 = torch.unique(coords1, dim=0)
    vox2 = torch.unique(coords2, dim=0)
    combined = torch.cat((vox1, vox2), dim=0)
    _, counts = torch.unique(combined, dim=0, return_counts=True)
    return (counts > 1).sum().item()

def triple_bit_code(a_vertex, b_vertex):
    """Generate a 3-bit mask comparing vertex dimensions."""
    mask = 0
    for d in range(3):
        bit = 1 if a_vertex[d] > b_vertex[d] else 0
        mask |= (bit << d)
    return mask

def run_demo():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define geometries
    density = 1
    steps = 5
    tetra = CompositeGeometry("tetrahedron", device=device)
    cube = CompositeGeometry("cube", device=device)
    tetra_vertices, tetra_edges = tetra.configure_geometry(density=density)
    cube_vertices, cube_edges = cube.configure_geometry(density=density)

    running = True
    iteration = 0
    voxel_size = 0.2
    overlap_history = []

    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random cube transformation
        alpha = random.uniform(0, 2 * math.pi)
        beta = random.uniform(0, 2 * math.pi)
        rotation = torch.tensor([[math.cos(alpha), -math.sin(alpha), 0],
                                 [math.sin(alpha), math.cos(alpha), 0],
                                 [0, 0, 1]], device=device, dtype=torch.float64)
        translation = torch.tensor([random.uniform(-0.5, 0.5) for _ in range(3)], device=device)
        cube_transformed = cube_vertices @ rotation.T + translation

        # Voxelize volumes
        tetra_voxels = voxelize(interpolate_edges(tetra_vertices, tetra_edges, steps), voxel_size)
        cube_voxels = voxelize(interpolate_edges(cube_transformed, cube_edges, steps), voxel_size)

        # Compute voxel overlap
        overlap = compute_voxel_overlap(tetra_voxels, cube_voxels)
        overlap_history.append(overlap)
        avg_overlap = sum(overlap_history) / len(overlap_history)

        # Bitmask generation
        final_mask = 0
        for v_tet in tetra_vertices:
            for v_cube in cube_transformed:
                code = triple_bit_code(v_tet, v_cube)
                final_mask = (final_mask << 3) | code

        # Display metrics
        screen.blit(font.render(f"Iteration: {iteration}", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Overlap: {overlap}", True, (255, 255, 255)), (10, 40))
        screen.blit(font.render(f"Avg Overlap: {avg_overlap:.2f}", True, (255, 255, 255)), (10, 70))
        screen.blit(font.render(f"Bitmask: {bin(final_mask)}", True, (200, 200, 200)), (10, 100))

        pygame.display.flip()
        clock.tick(2)
        iteration += 1

    pygame.quit()

if __name__ == "__main__":
    run_demo()
