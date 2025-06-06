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

        tile_size = offsets.max(dim=0).values - offsets.min(dim=0).values
        centroid = offsets.mean(dim=0)
        centered_offsets = offsets - centroid
        return centered_offsets, tile_size

    def configure_geometry(self, geometry=None, density=1, micro_jitter=False):
        if geometry is not None:
            self.geometry = geometry

        if self.geometry == "cube":
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
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        offsets, tile_size = self.define_offsets(density, micro_jitter)
        edge_lengths = torch.norm(
            offsets[edge_pairs[:, 0]] - offsets[edge_pairs[:, 1]],
            dim=1
        )
        return offsets.requires_grad_(False), vertex_count, edge_pairs, edge_lengths, tile_size

def rotation_matrix_3d(alpha, beta):
    """Rotation around Z then Y (simple)"""
    cz = math.cos(alpha)
    sz = math.sin(alpha)
    cy = math.cos(beta)
    sy = math.sin(beta)
    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [0,   0,  1]], dtype=torch.float64)
    Ry = torch.tensor([[cy, 0, sy],
                       [0,  1, 0],
                       [-sy,0, cy]], dtype=torch.float64)
    return Rz @ Ry

def voxelize(points, voxel_size=0.0005):
    coords = torch.floor(points/voxel_size).long()
    keys = coords[:,0]*73856093 + coords[:,1]*19349663 + coords[:,2]*83492791
    return keys

def compute_overlap(vox1, vox2):
    v1 = vox1.cpu().unique()
    v2 = vox2.cpu().unique()
    return torch.isin(v1, v2).sum().item()

def triple_bit_code(a_vertex, b_vertex):
    # Compare each dimension: if a_vertex[d] > b_vertex[d] => bit=1 else bit=0
    mask = 0
    for d in range(3):
        bit = 1 if a_vertex[d] > b_vertex[d] else 0
        mask |= (bit << d)
    return mask

def run_demo():
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    pygame.display.set_caption("Tetrahedron-Cube Overlap (All Vertex Pairs)")
    clock = pygame.time.Clock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tetra = CompositeGeometry("tetrahedron", device=device)
    cube = CompositeGeometry("cube", device=device)
    tet_positions, tet_count, _, _, _ = tetra.configure_geometry(density=1)
    cube_positions, cube_count, _, _, _ = cube.configure_geometry(density=1)

    # Move cube a bit initially
    cube_positions = cube_positions + torch.tensor([0.0,0.0,0.0], dtype=torch.float64, device=device)

    overlap_history = []
    iteration = 0
    running = True

    while running:
        screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random rotation/translation
        alpha = random.uniform(0, 2*math.pi)
        beta = random.uniform(0, 2*math.pi)
        R = rotation_matrix_3d(alpha, beta).to(device)
        translate = torch.tensor([random.uniform(-0.1,0.1),
                                  random.uniform(-0.1,0.1),
                                  random.uniform(-0.1,0.1)], dtype=torch.float64, device=device)
        cube_transformed = (cube_positions @ R.T) + translate

        # Compute overlap
        vox_tet = voxelize(tet_positions)
        vox_cube = voxelize(cube_transformed)
        overlap = compute_overlap(vox_tet, vox_cube)
        overlap_history.append(overlap)
        avg_overlap = sum(overlap_history)/len(overlap_history)

        # Generate a big mask from every permutation of vertices (tet vs cube)
        # For each tet vertex and each cube vertex:
        #   Compute a 3-bit code and append to mask
        final_mask = 0
        bits_per_pair = 3
        total_pairs = tet_count * cube_count

        for i in range(tet_count):
            for j in range(cube_count):
                code = triple_bit_code(tet_positions[i], cube_transformed[j])
                # Shift the final_mask and add the code
                final_mask = (final_mask << bits_per_pair) | code

        # Display metrics
        font = pygame.font.SysFont(None, 24)
        text1 = font.render(f"Iteration: {iteration}", True, (255,255,255))
        text2 = font.render(f"Current Overlap: {overlap}", True, (255,255,255))
        text3 = font.render(f"Average Overlap: {avg_overlap:.2f}", True, (255,255,255))
        screen.blit(text1, (10,10))
        screen.blit(text2, (10,40))
        screen.blit(text3, (10,70))

        # Show a shortened binary mask (if too long, just show tail end)
        mask_bin = bin(final_mask)
        if len(mask_bin) > 100:
            mask_bin = mask_bin[:50] + " ... " + mask_bin[-50:]
        text4 = font.render(f"Final Mask (bin): {mask_bin}", True, (200,200,200))
        screen.blit(text4, (10, 100))

        # 2D Projection
        scale = 60
        offset_screen = torch.tensor([400,300], dtype=torch.float64, device=device)
        def project_points(pts):
            return (pts[:, :2]*scale + offset_screen).cpu().numpy()

        tet_2d = project_points(tet_positions)
        cube_2d = project_points(cube_transformed)

        for p in tet_2d:
            pygame.draw.circle(screen, (0,255,0), p.astype(int), 4)
        for p in cube_2d:
            pygame.draw.circle(screen, (0,0,255), p.astype(int), 4)

        pygame.display.flip()
        clock.tick(2)
        iteration += 1

    pygame.quit()

if __name__ == "__main__":
    run_demo()
