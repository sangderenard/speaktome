import pygame
import torch
import random
import math
from math import gcd
from functools import reduce
import time

# -------------------------
# CompositeGeometry class (from the user's snippet)
# -------------------------
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
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        offsets, tile_size = self.define_offsets(density, micro_jitter)
        edge_lengths = torch.norm(
            offsets[edge_pairs[:, 0]] - offsets[edge_pairs[:, 1]],
            dim=1
        )
        return offsets.requires_grad_(False), vertex_count, edge_pairs, edge_lengths, tile_size

# ---------------------------------
# Utility functions
# ---------------------------------
def rotation_matrix_3d(alpha, beta, gamma=0.0):
    """Create a rotation matrix from Euler angles (Z-Y-X convention)."""
    # Rotate around Z
    cz = math.cos(alpha)
    sz = math.sin(alpha)

    # Rotate around Y
    cy = math.cos(beta)
    sy = math.sin(beta)

    # Rotate around X
    cx = math.cos(gamma)
    sx = math.sin(gamma)

    # Combined rotation Z-Y-X
    # Rz * Ry * Rx
    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [0,   0,  1]], dtype=torch.float64)
    Ry = torch.tensor([[cy, 0, sy],
                       [0,  1, 0 ],
                       [-sy,0, cy]], dtype=torch.float64)
    Rx = torch.tensor([[1, 0, 0],
                       [0, cx,-sx],
                       [0, sx, cx]], dtype=torch.float64)
    return Rz @ Ry @ Rx

def voxelize(points, voxel_size=0.25):
    """Convert point cloud to a set of voxel keys."""
    coords = torch.floor(points / voxel_size).long()
    # Simple hashing
    keys = coords[:,0]*73856093 + coords[:,1]*19349663 + coords[:,2]*83492791
    return keys

def compute_overlap(voxels1, voxels2):
    """Compute overlap count by checking common elements."""
    # Sort and binary search or use set for simplicity if small:
    # For GPU, we can just intersect via isin if on CPU.
    # Move data to CPU if needed because isin might not be available on GPU for older PyTorch.
    v1 = voxels1.cpu()
    v2 = voxels2.cpu()
    unique1 = torch.unique(v1)
    unique2 = torch.unique(v2)
    overlap = torch.isin(unique1, unique2).sum().item()
    return overlap

def dimension_bitmask_compare(vA, vB):
    """
    Compare coordinates of vA and vB along each dimension:
    If vA.x > vB.x => bit x = 1 else 0
    If vA.y > vB.y => bit y = 1 else 0
    If vA.z > vB.z => bit z = 1 else 0
    Combine into a 3-bit integer: (zbit << 2) | (ybit << 1) | xbit
    """
    mask = 0
    for d in range(3):
        bit = 1 if vA[d] > vB[d] else 0
        mask |= (bit << d)
    return mask

# ---------------------------------
# Main Demonstration
# ---------------------------------
def run_demo():
    # Initialize PyGame window
    pygame.init()
    screen = pygame.display.set_mode((800,600))
    pygame.display.set_caption("Tetrahedron-Cube Overlap Monte Carlo")
    clock = pygame.time.Clock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create shapes
    tetra = CompositeGeometry("tetrahedron", device=device)
    cube = CompositeGeometry("cube", device=device)

    # Configure them
    tet_offsets, tet_count, tet_edges, tet_edge_lens, tet_size = tetra.configure_geometry(density=1)
    cube_offsets, cube_count, cube_edges, cube_edge_lens, cube_size = cube.configure_geometry(density=1)

    # Initial positions
    tet_positions = tet_offsets.clone()
    cube_positions = cube_offsets.clone() + torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)

    # Statistics
    overlap_history = []
    bitmask_counts = torch.zeros(8, dtype=torch.float64)  # 3 bits => up to 8 combinations
    iteration = 0

    running = True

    while running:
        screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Monte Carlo: apply a random small rotation and translation to cube
        alpha = random.uniform(0, 2*math.pi)
        beta = random.uniform(0, 2*math.pi)
        R = rotation_matrix_3d(alpha, beta, 0.0).to(device)

        translate = torch.tensor([random.uniform(-0.1,0.1),
                                  random.uniform(-0.1,0.1),
                                  random.uniform(-0.1,0.1)], dtype=torch.float64, device=device)

        # Apply transform to cube
        cube_pos_transformed = (cube_positions @ R.T) + translate

        # Voxel approximation to measure overlap
        vox1 = voxelize(tet_positions)
        vox2 = voxelize(cube_pos_transformed)
        overlap = compute_overlap(vox1, vox2)

        # Compute bitmask: Compare each vertex of tetra with the corresponding vertex of cube
        # (For simplicity, we assume same vertex count. If not, we can loop min(tet_count, cube_count)).
        # If different counts, we can match them by nearest pairs or by index.
        vc = min(tet_count, cube_count)
        current_bitmask_sum = torch.zeros(8, dtype=torch.float64)
        for i in range(vc):
            bm = dimension_bitmask_compare(tet_positions[i], cube_pos_transformed[i])
            current_bitmask_sum[bm] += 1

        bitmask_counts += current_bitmask_sum

        # Record overlap history
        overlap_history.append(overlap)
        avg_overlap = sum(overlap_history)/len(overlap_history)

        # Simple text info
        font = pygame.font.SysFont(None, 24)
        text1 = font.render(f"Iteration: {iteration}", True, (255,255,255))
        text2 = font.render(f"Current Overlap: {overlap}", True, (255,255,255))
        text3 = font.render(f"Average Overlap: {avg_overlap:.2f}", True, (255,255,255))
        screen.blit(text1, (10,10))
        screen.blit(text2, (10,40))
        screen.blit(text3, (10,70))

        # Bitmask frequencies
        total_counts = bitmask_counts.sum().item()
        if total_counts > 0:
            for idx in range(8):
                freq = bitmask_counts[idx].item()/total_counts
                bm_text = font.render(f"Bitmask {idx:03b}: {freq:.3f}", True, (200,200,200))
                screen.blit(bm_text, (10, 100+20*idx))

        # Project and draw the shapes in 2D for visualization
        # Just a simple orthographic projection + scaling
        scale = 60
        offset_screen = torch.tensor([400,300], dtype=torch.float64)

        def project_points(pts):
            # Project from 3D to 2D by ignoring z and scaling
            return (pts[:, :2].cpu().numpy()*scale + offset_screen.cpu().numpy())

        tet_2d = project_points(tet_positions)
        cube_2d = project_points(cube_pos_transformed)

        # Draw tetra (green)
        for p in tet_2d:
            pygame.draw.circle(screen, (0,255,0), p.astype(int), 4)

        # Draw cube (blue)
        for p in cube_2d:
            pygame.draw.circle(screen, (0,0,255), p.astype(int), 4)

        pygame.display.flip()
        clock.tick(2)  # Slow down for visibility
        iteration += 1

    pygame.quit()


if __name__ == "__main__":
    run_demo()
