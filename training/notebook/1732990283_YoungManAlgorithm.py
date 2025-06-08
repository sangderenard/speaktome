import torch
import logging
import numpy as np
import os
import pickle
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import torch
import numpy as np
import logging
import os
import pickle
import time
from triangulator import Triangulator
from compositegeometry import CompositeGeometry
logging.basicConfig(level=logging.INFO, format="%(message)s")


class YoungManAlgorithm:
    def __init__(self, geometry="cube", density=1.0, save_dir="geometry_maps", device="cuda", precision=torch.float64, jitter_enabled=False, jitter_seed=None, jitter_strength = (torch.pi, torch.pi/2, 0), micro_jitter=False):
        assert geometry in ["square", "cube", "tetrahedron", "octahedron", "icosahedron"], \
            "Unsupported geometry. Choose 'square', 'cube', 'tetrahedron', 'octahedron', or 'icosahedron'."
        self.jitter_enabled = jitter_enabled
        self.jitter_once = False  # Determines if jitter is applied once or refreshed dynamically
        self.jitter_seed = jitter_seed  # Seed for reproducibility
        self.jitter_strength = jitter_strength  # Max ranges for (theta, phi, r)
        self.current_jitter = None  # Cached jitter for "once" mode
        self.micro_jitter = micro_jitter

        self.current_geometry = geometry
        self.density = density
        self.save_dir = save_dir
        self.precision = precision
        self.device = torch.device(device)

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Initialize cache for geometry-specific maps
        self.geometry_maps = {}
        # Load or generate maps for the initial geometry
        self._load_and_set_geometry(geometry, density, micro_jitter)
        
    def enable_jitter(self, theta_range=0.0, phi_range=0.0, r_range=0.0, once=False, seed=None):
        """
        Enable jitter with specified spherical ranges and mode.
        
        Args:
            theta_range (float): Maximum deviation for theta (azimuthal angle) in radians.
            phi_range (float): Maximum deviation for phi (polar angle) in radians.
            r_range (float): Maximum radial deviation.
            once (bool): If True, jitter is applied once and cached. Otherwise, it's refreshed dynamically.
            seed (int): Seed for reproducibility.
        """
        self.jitter_enabled = True
        self.jitter_once = once
        self.jitter_seed = seed
        self.jitter_strength = (theta_range, phi_range, r_range)
        if seed is not None:
            torch.manual_seed(seed)
        if once:
            self.current_jitter = self._generate_jitter()

    def disable_jitter(self):
        """Disable jitter."""
        self.jitter_enabled = False
        self.current_jitter = None


    def generate_jitter(self, offsets, jitter_strength=None, jitter_seed=None):
        """
        Generate spherical jitter for a batch of offsets and apply it as rotation and translation.

        Args:
            offsets (torch.Tensor): Tensor of shape (batch_size, V, 3) representing the original offsets for all units.
            jitter_strength (tuple): Strength of jitter for (theta, phi, r), where:
                                    - theta: Angular deviation around the z-axis.
                                    - phi: Polar deviation from the z-axis.
                                    - r: Translation magnitude.
            jitter_seed (int): Seed for reproducibility.

        Returns:
            torch.Tensor: Jittered offsets of shape (batch_size, V, 3).
        """
        device = self.device
        if jitter_strength is None:
            jitter_strength = self.jitter_strength

        # Set the seed if provided
        if jitter_seed is not None:
            self.jitter_seed = jitter_seed
        if self.jitter_seed is not None:
            torch.manual_seed(self.jitter_seed)

        # Ensure jitter strength has 3 components
        assert len(jitter_strength) == 3, "jitter_strength must be a tuple of (theta, phi, r)."

        batch_size, num_vertices, _ = offsets.size()

        # Generate random spherical coordinates per batch
        theta = (torch.rand(batch_size, device=device, dtype=self.precision) - 0.5) * 2 * jitter_strength[0]
        phi = (torch.rand(batch_size, device=device, dtype=self.precision) - 0.5) * 2 * jitter_strength[1]
        r = (torch.rand(batch_size, device=device, dtype=self.precision) - 0.5) * 2 * jitter_strength[2]

        # Calculate rotation matrices for each unit (one per batch)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        rotation_matrices = torch.stack([
            cos_theta * cos_phi, -sin_theta, cos_theta * sin_phi,
            sin_theta * cos_phi, cos_theta, sin_theta * sin_phi,
            -sin_phi, torch.zeros_like(theta), cos_phi
        ], dim=-1).reshape(batch_size, 3, 3)  # Shape: (batch_size, 3, 3)

        # Apply rotation to all offsets in the batch
        rotated_offsets = torch.bmm(rotation_matrices, offsets.transpose(1, 2)).transpose(1, 2)  # Shape: (batch_size, V, 3)

        # Convert r to Cartesian translation
        x_translation = r * torch.sin(phi) * torch.cos(theta)
        y_translation = r * torch.sin(phi) * torch.sin(theta)
        z_translation = r * torch.cos(phi)
        translation_vector = torch.stack([x_translation, y_translation, z_translation], dim=-1)  # Shape: (batch_size, 3)

        # Apply translation to all offsets
        jittered_offsets = rotated_offsets + translation_vector.unsqueeze(1)  # Shape: (batch_size, V, 3)

        return jittered_offsets


    def _load_and_set_geometry(self, geometry, density, micro_jitter):
        """Load maps for the given geometry and set as current attributes."""
        self.vertex_offsets, self.vertex_count, self.edge_pairs, self.edge_lengths, \
            self.active_vertex_map, self.triangle_map, self.triangle_mask, self.centroids, self.tile_size = \
            self._load_geometry_maps_if_cached(geometry, density, micro_jitter)
    def _map_file_path(self, geometry, density, map_name):
        """Get the full path to a specific map file for a given geometry."""
        return os.path.join(self.save_dir, f"{geometry}_{map_name}_{density}.pkl")

    def _save_map(self, geometry, density, map_data, map_name):
        """Save a map to a file for a specific geometry."""
        file_path = self._map_file_path(geometry, density, map_name)
        with open(file_path, 'wb') as f:
            pickle.dump(map_data, f)
        logging.info(f"Saved {map_name} for {geometry} to {file_path}.")

    def _load_map(self, geometry, density, map_name):
        """Load a map from a file for a specific geometry."""
        file_path = self._map_file_path(geometry, density, map_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                logging.info(f"Loaded {map_name} for {geometry} from {file_path}.")
                return torch.tensor(pickle.load(f)).to(device=self.device, dtype=self.precision)
        return None

    def _load_geometry_maps(self, geometry, density, micro_jitter=False):
        """
        Load all necessary maps for a given geometry.
        Returns:
            A tuple containing all required maps.
        """
        
        maps = {}
        map_names = ["vertex_offsets", "vertex_count", "edge_pairs", "edge_lengths",
                    "active_vertex_map", "triangle_map", "triangle_mask", "centroids", "tile_size"]
        for map_name in map_names:
            loaded_map = self._load_map(geometry, density, map_name)
            if loaded_map is not None:
                maps[map_name] = loaded_map

        # Check if all maps are loaded
        if len(maps) == len(map_names) and micro_jitter is False:
            logging.info(f"All maps loaded from cache for geometry: {geometry}")
            return (maps["vertex_offsets"], maps["vertex_count"], maps["edge_pairs"],
                    maps["edge_lengths"], maps["active_vertex_map"], maps["triangle_map"],
                    maps["triangle_mask"], maps["centroids"], maps["tile_size"])
        else:
            logging.info(f"Generating geometry maps for {geometry}...")
            vertex_offsets, vertex_count, edge_pairs, edge_lengths, tile_size = self._configure_geometry(geometry, micro_jitter)
            active_vertex_map, triangle_mask, triangle_map, centroids = self._build_triangle_map(edge_pairs, vertex_count, vertex_offsets)
            centroids = self._centroids_to_tensor(centroids)

            # Save generated maps
            self._save_map(geometry, density, vertex_offsets, "vertex_offsets")
            self._save_map(geometry, density, vertex_count, "vertex_count")
            self._save_map(geometry, density, edge_pairs, "edge_pairs")
            self._save_map(geometry, density, edge_lengths, "edge_lengths")
            self._save_map(geometry, density, active_vertex_map, "active_vertex_map")
            self._save_map(geometry, density, triangle_map, "triangle_map")
            self._save_map(geometry, density, triangle_mask, "triangle_mask")
            self._save_map(geometry, density, centroids, "centroids")
            self._save_map(geometry, density, tile_size, "tile_size")

            return (vertex_offsets, vertex_count, edge_pairs, edge_lengths,
                    active_vertex_map, triangle_map, triangle_mask, centroids, tile_size)
    def _validate_triangle_map(self, triangle_mask, vertex_count, edge_pairs):
        """
        Validates that the triangle map has a consistent active edge count based on edge pairs.

        Args:
            triangle_mask (torch.Tensor): Mask indicating valid triangles for each bitmask.
            vertex_count (int): Total number of vertices in the geometry.
            edge_pairs (list of tuple): List of edge pairs defining the geometry.

        Raises:
            ValueError: If the active edge count in the triangle map does not match the expected count.
        """
        for bitmask in range(2 ** int(vertex_count)):
            # Calculate active edges from the bitmask
            active_edges_count = sum(
                (bitmask & (1 << v1)) != (bitmask & (1 << v2))
                for v1, v2 in edge_pairs.to(dtype=torch.int64)
            )

            # Calculate active triangles from the triangle mask
            active_triangles_count = triangle_mask[bitmask].sum().item()

            # Validate counts
            if active_triangles_count > active_edges_count:
                raise ValueError(
                    f"Validation failed for bitmask {bitmask}: "
                    f"{active_triangles_count} triangles, {active_edges_count} active edges."
                )


    def _load_geometry_maps_if_cached(self, geometry, density, micro_jitter=False):
        """Check if maps for a geometry and density are cached; if not, load/generate them."""
        if geometry not in self.geometry_maps:
            self.geometry_maps[geometry] = {}

        if density not in self.geometry_maps[geometry]:
            maps = self._load_geometry_maps(geometry, density, micro_jitter)
            self.geometry_maps[geometry][density] = maps

        return self.geometry_maps[geometry][density]

    def switch_geometry(self, new_geometry, density=None):
        """
        Switch the current geometry to a new geometry and optionally update density.
        """
        if density is not None:
            self.density = density

        if new_geometry == self.current_geometry and density == self.density:
            logging.info(f"Already using geometry: {new_geometry} with density: {self.density}")
            return

        assert new_geometry in ["square", "cube", "tetrahedron", "octahedron", "icosahedron"], \
            "Unsupported geometry. Choose 'square', 'cube', 'tetrahedron', 'octahedron', or 'icosahedron'."

        #logging.info(f"Switching geometry from {self.current_geometry} to {new_geometry} with density {self.density}...")
        self.current_geometry = new_geometry
        maps = self._load_geometry_maps_if_cached(new_geometry, self.density)
        self.vertex_offsets, self.vertex_count, self.edge_pairs, self.edge_lengths, \
            self.active_vertex_map, self.triangle_map, self.triangle_mask, self.centroids, self.tile_size = maps

    def _apply_micro_jitter(self, offsets, jitter_strength=1e-8):
        """
        Applies an infinitesimal mandatory nonzero jitter to the given offsets.
        Ensures no coordinate is exactly 0 or 1 (phi is acceptable).

        Args:
            offsets (torch.Tensor): The offsets tensor to be jittered.
            jitter_strength (float): The magnitude of the jitter.

        Returns:
            torch.Tensor: The jittered offsets.
        """
        # Generate random jitter of the same shape as offsets
        jitter = torch.randn_like(offsets) * jitter_strength

        # Apply the jitter to the offsets
        jittered_offsets = offsets + jitter

        # Correct any values that are exactly 0 or 1
        # Use a small adjustment proportional to the jitter_strength
        correction = jitter_strength * 0.1
        jittered_offsets = torch.where(
            jittered_offsets == 0,
            jittered_offsets + correction,
            jittered_offsets
        )
        jittered_offsets = torch.where(
            jittered_offsets == 1,
            jittered_offsets - correction,
            jittered_offsets
        )
        jittered_offsets = torch.where(
            jittered_offsets == -1,
            jittered_offsets - correction,
            jittered_offsets
        )
        return jittered_offsets

    def _define_offsets(self, geometry, micro_jitter=False):
        if geometry == "cube":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=self.precision, device=self.device)
        elif geometry == "tetrahedron":
            offsets = torch.tensor([
                [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
            ], dtype=self.precision, device=self.device)
        elif geometry == "square":
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            ], dtype=self.precision, device=self.device)
        elif geometry == "octahedron":
            offsets = torch.tensor([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ], dtype=self.precision, device=self.device)
        elif geometry == "icosahedron":
            phi = (1 + 5 ** 0.5) / 2  # Golden ratio
            offsets = torch.tensor([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ], dtype=self.precision, device=self.device) / phi
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        # Scale by density and center around 0

        offsets /= self.density
        if micro_jitter:
            offsets = self._apply_micro_jitter(offsets)
        tile_size = offsets.max(dim=0).values - offsets.min(dim=0).values

        centroid = offsets.mean(dim=0)  # Calculate the centroid
        centered_offsets = offsets - centroid  # Subtract centroid from all vertices
        return centered_offsets, tile_size

    def _configure_geometry(self, geometry, micro_jitter):
        """Configure vertex and edge definitions for the chosen geometry."""
        if geometry == "square":
            vertex_count = torch.tensor(4.0, device=self.device, dtype=self.precision)
            edge_pairs = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=self.precision, device=self.device)
        elif geometry == "cube":
            vertex_count = torch.tensor(8.0, device=self.device, dtype=self.precision)
            edge_pairs = torch.tensor([
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ], dtype=self.precision, device=self.device)
        elif geometry == "tetrahedron":
            vertex_count = torch.tensor(4.0, device=self.device, dtype=self.precision)
            edge_pairs = torch.tensor([
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ], dtype=self.precision, device=self.device)
        elif geometry == "octahedron":
            vertex_count = torch.tensor(6.0, device=self.device, dtype=self.precision)
            edge_pairs = torch.tensor([
                (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 4), (2, 5), (3, 4), (3, 5)
            ], dtype=self.precision, device=self.device)
        elif geometry == "icosahedron":
            vertex_count = torch.tensor(12.0, device=self.device, dtype=self.precision)
            edge_pairs = torch.tensor([
                (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
                (1, 5), (1, 6), (1, 8), (1, 9),
                (2, 3), (2, 4), (2, 6), (2, 9), (2, 11),
                (3, 4), (3, 7), (3, 8), (3, 10),
                (4, 5), (4, 7), (4, 9),
                (5, 10), (6, 8), (6, 11),
                (7, 8), (7, 9), (8, 11),
                (9, 10), (9, 11), (10, 11)
            ], dtype=self.precision, device=self.device)
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")

        # Calculate edge lengths using the vertex offsets
        edge_lengths = []
        offsets, tile_size = self._define_offsets(geometry, micro_jitter)

        for edge in edge_pairs.to(dtype=torch.int64):
            vertex1, vertex2 = edge
            offset1 = offsets[vertex1]
            offset2 = offsets[vertex2]
            
            # Calculate the Euclidean distance between the two vertices
            distance = torch.norm(offset2 - offset1)  # Euclidean distance in 3D space
            edge_lengths.append(distance.item())  # Convert tensor to float for easier handling

        return offsets, vertex_count, edge_pairs, edge_lengths, tile_size
    def _edge_lookup(self, edges_to_lookup, edge_pairs):
        """
        Create a signed lookup table for edge pairs based on their order
        relative to the definitional edge_pairs tensor.

        Args:
            edges_to_lookup (list of tuples or tensor): List of edges to lookup as vertex index pairs.
            edge_pairs (torch.Tensor): Tensor of shape (num_edges, 2) defining all valid edges.

        Returns:
            torch.Tensor: Signed mapping tensor of shape (num_edges_to_lookup,), where:
                        - Positive values indicate a match in the original order in edge_pairs.
                        - Negative values indicate a match in the reversed order in edge_pairs.
        """
        num_edges_to_lookup = len(edges_to_lookup)
        signed_lookup = torch.full((num_edges_to_lookup,), -1, dtype=torch.int64, device=self.device)

        for i, edge in enumerate(edges_to_lookup):
            edge_tensor = torch.tensor(edge, dtype=torch.int64, device=self.device)
            reverse_edge_tensor = torch.flip(edge_tensor, dims=[0])  # Reverse the edge

            # Find matches in edge_pairs
            forward_match = torch.all(edge_pairs == edge_tensor, dim=1)
            reverse_match = torch.all(edge_pairs == reverse_edge_tensor, dim=1)

            if forward_match.any():
                signed_lookup[i] = torch.nonzero(forward_match, as_tuple=True)[0].item()  # Positive index
            elif reverse_match.any():
                #signed_lookup[i] = -torch.nonzero(reverse_match, as_tuple=True)[0].item()  # Negative index (negative removed)
                signed_lookup[i] = torch.nonzero(reverse_match, as_tuple=True)[0].item()  # invariant returns
            else:
                raise ValueError(f"Edge {edge} not found in the definitional edge_pairs!")

        return signed_lookup



    def _build_triangle_map(self, edge_pairs, vertex_count, vertex_offsets):
        """
        Build a triangle map for all bitmask configurations based on active-inactive edges.

        Returns:
            active_vertex_map: A tensor containing active vertex indices for all bitmask configurations.
            triangle_mask: A mask tensor indicating valid triangles.
            triangle_map: A correlation map of bitmasks to edge involvement and their triangle fans.
            centroids: A list of centroids for each bitmask (optional for refinement).
        """
        num_edges = len(edge_pairs)  # Total number of edges in the geometry
        max_triangles = num_edges  # Maximum triangles correspond to edges
        triangle_map = torch.full((2 ** vertex_count.to(dtype=torch.int64), num_edges, 2), -1, dtype=self.precision, device=self.device)

        triangle_mask = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), max_triangles), dtype=torch.bool, device=self.device)
        active_vertex_map = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), vertex_count.to(dtype=torch.int64)), dtype=torch.bool, device=self.device)  # Active vertices
        centroids = []

        for bitmask in range(2 ** vertex_count.to(dtype=torch.int64)):
            # Step 1: Identify active vertices for the current bitmask
            active_vertex_mask = (bitmask & (1 << torch.arange(vertex_count.to(dtype=torch.int64), device=self.device))) > 0
            active_vertex_map[bitmask] = active_vertex_mask  # Store active vertex map
            active_vertices = vertex_offsets[active_vertex_mask]
            active_indices = torch.nonzero(active_vertex_mask).flatten().tolist()

            

            # Step 2: Calculate centroid (optional)
            try:
                centroid = active_vertices.mean(axis=0)
                centroids.append((bitmask, centroid))
                active_vertices = torch.cat([active_vertices, centroid.unsqueeze(0)])  # Add centroid for fan reference
            except Exception as e:
                logging.debug(f"Error calculating centroid for bitmask {bitmask}: {e}")
                continue

            # Step 3: Determine active-inactive edges
            active_edges = []
            for edge_idx, (v1, v2) in enumerate(edge_pairs):
                v1 = v1.to(dtype=torch.int64)
                v2 = v2.to(dtype=torch.int64)
                v1_active = active_vertex_mask[v1]
                v2_active = active_vertex_mask[v2]
                if v1_active != v2_active:  # One active, one inactive
                    active_edges.append((v1, v2))
            triangles = []
            if len(active_edges) >= 3:  # At least 3 edges needed to form triangles
                # Step 1: Calculate the mean midpoint (origin of the local plane)
                edge_midpoints = [(vertex_offsets[v1] + vertex_offsets[v2]) / 2 for v1, v2 in active_edges]
                plane_origin = torch.mean(torch.stack(edge_midpoints), dim=0)

                # Step 2: Define a consistent planar basis
                # Use the first two edges to construct the plane axes
                e1_vector = edge_midpoints[1] - plane_origin
                e2_vector = edge_midpoints[0] - plane_origin
                plane_x = e1_vector / e1_vector.norm()  # Normalize to unit vector
                plane_y = torch.cross(plane_x, e2_vector)  # Orthogonal axis
                plane_y = plane_y / torch.norm(plane_y)

                # Step 3: Project edge midpoints onto the plane and calculate angles
                projected_edges = []
                for midpoint, edge in zip(edge_midpoints, active_edges):
                    relative_position = midpoint - plane_origin
                    x_coord = torch.dot(relative_position, plane_x)
                    y_coord = torch.dot(relative_position, plane_y)
                    angle = torch.atan2(y_coord, x_coord)  # Angle in the plane
                    if torch.isnan(angle).any():
                        angle = 0
                    projected_edges.append((angle, edge))

                # Step 4: Sort edges by their angle
                projected_edges.sort(key=lambda x: x[0])
                sorted_edges = [edge for _, edge in projected_edges]

                # Step 5: Form triangles from sorted edges
                for i in range(len(sorted_edges)):
                    e1 = sorted_edges[i]  # Current edge
                    e2 = sorted_edges[(i + 1) % len(sorted_edges)]  # Next edge in circular order
                    triangles.append((e1, e2))  # Store edge pair as triangle

            
            # Populate triangle map and mask
            for i, triangle in enumerate(triangles):
                triangle = self._edge_lookup(triangle, edge_pairs)
                triangle_map[bitmask, i] = torch.tensor(triangle, device=self.device, dtype=self.precision)
                #logging.info(f"Bitmask: {bitmask}, Active Edges: {len(active_edges)}, Triangles: {len(triangles)}")

                triangle_mask[bitmask, i] = True

        self._validate_triangle_map(triangle_mask, vertex_count, edge_pairs)

        return active_vertex_map, triangle_mask, triangle_map, centroids
    def prepare_opengl_data(self, evaluation_result):
        """
        Prepare vertex and normal data interleaved for OpenGL.

        Args:
            evaluation_result (dict): Output of the evaluate method.

        Returns:
            np.ndarray: Interleaved vertex and normal data, flattened for OpenGL.
        """
        # Directly use the interleaved structure from evaluation_result
        interleaved_data = evaluation_result["interleaved"]

        # Flatten the interleaved data for OpenGL compatibility
        return interleaved_data.reshape(-1)

    def evaluate(self, scalar_function, domain_bounds, isovalue=1.0, gradient_normals=False, deduplicate=False, compute_vertex_normals=False, centroid_refinement=False, oversampling_grid=(1,1,1), oversampling_spot=1, jitter_strength=None):
        """
        Evaluate the scalar function over the domain to generate the isosurface.

        Args:
            scalar_function (callable): Function defining the scalar field.
            domain_bounds (list of tuples): Bounds for the domain.
            isovalue (float): Isosurface value.
            gradient_normals (bool): Whether to compute gradient-based normals.
            deduplicate (bool): Whether to deduplicate vertices.

        Returns:
            dict: Contains vertices, triangle indices, and interleaved normals.
        """
        # Step 1: Generate the grid and tile vertices
        grid_points = self._generate_grid(domain_bounds, oversampling=oversampling_grid)
        tiled_vertices = self.tile_grid(grid_points,oversampling_factor=oversampling_spot,jitter_strength=jitter_strength)

        # Step 2: Compute bitmasks and active vertices
        bitmasks, relevant_vertices = self._compute_bitmasks(scalar_function, tiled_vertices, isovalue)

        # Step 3: Compute edge intersections
        intersection_tensor = self._compute_dense_intersections(
            relevant_vertices[:, : self.vertex_count.to(dtype=torch.int64), ...],
            scalar_function,
            isovalue,
        )

        # Step 4: Compute dynamic centroids
        active_triangles_map = self.triangle_map[bitmasks]
        active_triangles_mask = self.triangle_mask[bitmasks]
        grid_centers = relevant_vertices[:, self.vertex_count.to(dtype=torch.int64), ...]
        active_centroids = (grid_centers.unsqueeze(1) + self.centroids[bitmasks]).squeeze(1)

        organized_tensor, _ = self._compute_batch_centroids(
            active_triangles_map,
            active_triangles_mask,
            intersection_tensor,
            scalar_function,
            active_centroids,
            isovalue,
            centroid_refinement
        )

        # Deduplicate vertices and generate indices
        result = self._deduplicate_vertices_and_generate_indices(organized_tensor, scalar_function, compute_vertex_normals, gradient_normals, deduplicate)

        return result

    def _deduplicate_vertices_and_generate_indices(
        self,
        triangles_tensor,
        scalar_function,
        compute_vertex_normals=True,
        gradient_normals=True,
        deduplicate=True
    ):
        """
        Deduplicate vertices and interleave their respective normals for OpenGL.

        Args:
            triangles_tensor (torch.Tensor): Tensor of shape (num_triangles, 3, 3) representing vertex positions.
            scalar_function (callable): Function to compute scalar field values.
            vertex_normals (bool): Whether to compute vertex normals using the scalar field gradient.
            deduplicate (bool): Whether to deduplicate vertices.

        Returns:
            dict: Contains deduplicated vertices, triangle indices, and interleaved data for OpenGL.
        """
        # Flatten the triangle tensor to extract all vertices
        all_vertices = triangles_tensor.view(-1, 3)  # Shape: (num_triangles * 3, 3)

        # Deduplicate vertices if requested
        if deduplicate:
            unique_vertices, inverse_indices = torch.unique(all_vertices, dim=0, return_inverse=True)
            triangle_indices = inverse_indices.view(-1, 3)  # Map back to triangle form
        else:
            unique_vertices = all_vertices
            triangle_indices = torch.arange(all_vertices.size(0), device=all_vertices.device).view(-1, 3)

        # Compute triangle normals (fallback)
        v0, v1, v2 = triangles_tensor[:, 0, :], triangles_tensor[:, 1, :], triangles_tensor[:, 2, :]
        edge1, edge2 = v1 - v0, v2 - v0
        triangle_normals = torch.cross(edge1, edge2, dim=1)
        triangle_normals = triangle_normals / triangle_normals.norm(dim=1, keepdim=True)  # Normalize
        triangle_normals = triangle_normals.repeat_interleave(3, dim=0)  # Match vertex structure

        # Initialize normals with triangle normals
        normals = triangle_normals.clone()

        # Compute vertex normals if requested
        if compute_vertex_normals:
            if gradient_normals:
                previously_required_grad = unique_vertices.requires_grad_().all()
                # Enable gradient computation for unique vertices
                if not previously_required_grad:
                    unique_vertices.requires_grad_(True)

                # Compute scalar field values for unique vertices
                scalar_values = scalar_function(unique_vertices[:, 0], unique_vertices[:, 1], unique_vertices[:, 2])

                # Compute gradients to get normals
                vertex_normals = torch.autograd.grad(
                    outputs=scalar_values,
                    inputs=unique_vertices,
                    grad_outputs=torch.ones_like(scalar_values),
                    create_graph=False,
                    retain_graph=False
                )[0]

                if not previously_required_grad:
                    unique_vertices.requires_grad_(False)

                # Normalize the computed vertex normals
                vertex_normals = vertex_normals / vertex_normals.norm(dim=1, keepdim=True)
            else:
                # Batch computation for averaging triangle normals
                vertex_normals = torch.zeros_like(unique_vertices)
                counts = torch.zeros(unique_vertices.size(0), device=unique_vertices.device, dtype=torch.float32)

                # Expand triangle indices to (num_triangles * 3, 1)
                flat_triangle_indices = triangle_indices.view(-1)


                # Accumulate normals for each unique vertex
                vertex_normals = vertex_normals.index_add_(0, flat_triangle_indices, triangle_normals)

                # Count contributions for normalization
                counts = counts.index_add_(0, flat_triangle_indices, torch.ones_like(flat_triangle_indices, dtype=torch.float32))

                # Normalize accumulated normals
                vertex_normals = vertex_normals / counts.unsqueeze(1)


        else:
            # Create empty normals (all zeros) if not computing vertex normals
            vertex_normals = torch.zeros_like(unique_vertices)
        # Interleave vertices and normals for OpenGL

        interleaved_data = torch.cat((unique_vertices, vertex_normals), dim=1)


        # Return deduplicated data as a dictionary
        return {
            "vertices": unique_vertices,
            "indices": triangle_indices,
            "interleaved": interleaved_data,
        }



    def _compute_dense_intersections(self, tiled_vertices, scalar_function, isovalue):
        """
        Compute a dense tensor of intersection vertices for all edges in all tiles.
        """
        num_tiles, num_vertices, _ = tiled_vertices.shape
        num_edges = len(self.edge_pairs)
        intersection_tensor = torch.full((num_tiles, num_edges, 3), float('nan'), device=tiled_vertices.device, dtype=torch.float64)

        # Extract vertex positions for all edges
        v1_indices, v2_indices = zip(*self.edge_pairs.to(dtype=torch.int64))  # Split edge pairs
        v1_positions = tiled_vertices[:, v1_indices, :]  # Shape: (num_tiles, num_edges, dim)
        v2_positions = tiled_vertices[:, v2_indices, :]  # Shape: (num_tiles, num_edges, dim)

        # Compute scalar values for edge vertices
        v1_values = scalar_function(v1_positions[..., 0], v1_positions[..., 1], v1_positions[..., 2])
        v2_values = scalar_function(v2_positions[..., 0], v2_positions[..., 1], v2_positions[..., 2])

        # Determine active edges (cross the isosurface)
        active_edges_mask = (v1_values > isovalue) != (v2_values > isovalue)  # Shape: (num_tiles, num_edges)

        # Interpolate intersection points along active edges
        t = (isovalue - v1_values) / (v2_values - v1_values)  # Shape: (num_tiles, num_edges)
        intersection_points = v1_positions + t.unsqueeze(-1) * (v2_positions - v1_positions)  # Shape: (num_tiles, num_edges, 3)

        # Populate dense tensor with intersection points
        intersection_tensor[active_edges_mask] = intersection_points[active_edges_mask]
        
        return intersection_tensor

    def _compute_batch_centroids(self, active_triangles_map, active_triangles_mask, intersection_tensor, scalar_function, precomputed_centroids, isovalue, centroid_refinement):
        """
        Compute centroids and generate triangles in batch, embedding positional and meta-information for edge-vertex correlation.

        Args:
            active_triangles_map (torch.Tensor): Map of active triangles for each tile.
            active_triangles_mask (torch.Tensor): Mask of valid triangles.
            intersection_tensor (torch.Tensor): Shape (num_tiles, num_edges, dim).
            scalar_function (function): Scalar function to compute field values.

        Returns:
            torch.Tensor: Expanded tensor with positional and meta-information (num_tiles, num_edges + 1, dim + meta_dims).
            dict: Reference dictionary mapping meta-information.
        """
        num_tiles, num_edges, dim = intersection_tensor.shape

        # Step 1: Define meta-information structure
        meta_ref = {
            "triangle_id": dim,         # Unique Triangle ID
            "edge_id_1": dim + 1,       # First edge of the triangle
            "edge_id_2": dim + 2,       # Second edge of the triangle
            "vertex_id_1": dim + 3,     # First vertex of the triangle
            "vertex_id_2": dim + 4,     # Second vertex of the triangle
            "vertex_id_3": dim + 5,     # Third vertex (centroid)
        }
        meta_dims = len(meta_ref)

        # Step 2: Initialize expanded tensor for positional and meta-information
        # Shape: (num_tiles, num_edges + 1, dim + meta_dims)
        expanded_tensor = torch.zeros(
            (num_tiles, num_edges + 1, dim + meta_dims), device=intersection_tensor.device
        )
        expanded_tensor[:, :num_edges, :dim] = intersection_tensor  # Copy intersection positions into spatial dimensions

        # Step 3: Compute valid intersections and edge metadata
        valid_mask = ~torch.isnan(intersection_tensor[..., 0])  # Valid edges per tile
        filled_tensor = torch.where(valid_mask.unsqueeze(-1), intersection_tensor, torch.zeros_like(intersection_tensor))

        # Assign edge IDs (first and second vertex of the edge)
        edge_ids = torch.arange(num_edges, device=intersection_tensor.device)
        expanded_tensor[:, :num_edges, meta_ref["edge_id_1"]] = edge_ids.unsqueeze(0).expand(num_tiles, -1)

        # Step 4: Compute centroids dynamically
        summed_vertices = filled_tensor.sum(dim=1)  # Sum valid vertices
        valid_counts = valid_mask.sum(dim=1, keepdim=True)  # Count valid vertices per tile
        initial_centroids = torch.where(valid_counts > 0, summed_vertices / valid_counts, torch.zeros_like(summed_vertices))

        if centroid_refinement:        
            # Step 2: If precomputed centroids are provided, refine both
            centroids = self._refine_intersections(
                activation_centroids=precomputed_centroids,
                dynamic_centroids=initial_centroids,
                scalar_function=scalar_function,
                isovalue=isovalue
            )
        else:
            centroids = initial_centroids
            
        valid_mask = (active_triangles_map >= 0).all(dim=-1)  # [num_tiles, max_triangles]

        # Filter only valid triangles
        valid_triangles = active_triangles_map[valid_mask]  # [num_valid_triangles, 2]
        valid_tiles = torch.nonzero(valid_mask)[:, 0]       # [num_valid_triangles]

        # Get corresponding tile indices
        valid_tile_intersection_tensors = intersection_tensor[valid_tiles]  # [num_valid_triangles, num_vertices, 3]
        valid_centroids = centroids[valid_tiles]                            # [num_valid_triangles, 3]

        # Gather vertices for each edge pair
        v1_indices = valid_triangles[:, 0].to(dtype=torch.int64)
        v2_indices = valid_triangles[:, 1].to(dtype=torch.int64)

        v1 = valid_tile_intersection_tensors[torch.arange(valid_triangles.shape[0]), v1_indices]  # [num_valid_triangles, 3]
        v2 = valid_tile_intersection_tensors[torch.arange(valid_triangles.shape[0]), v2_indices]  # [num_valid_triangles, 3]

        # Combine vertices and centroids
        expanded_triangles = torch.stack([v1, v2, valid_centroids], dim=1)  # [num_valid_triangles, 3, 3]


        # Convert the expanded triangles list into a dense tensor
        expanded_triangles_tensor = torch.tensor(expanded_triangles, dtype=self.precision, device=intersection_tensor.device)
        
        # Step 6: Return expanded tensor and reference dictionary
        return expanded_triangles_tensor, meta_ref


    def _refine_intersections(self, activation_centroids, dynamic_centroids, scalar_function, isovalue):
        """
        Refine intersections using rays defined by dynamic centroids -> activation centroids.

        Args:
            activation_centroids (torch.Tensor): Starting points for the rays (num_tiles, dim).
            dynamic_centroids (torch.Tensor): Target points for the rays (closer to ideal isovalue).
            scalar_function (function): Scalar field function used for isosurface evaluation.
            isovalue (float): Target isosurface value for refinement.

        Returns:
            torch.Tensor: Refined intersection points (num_tiles, dim).
        """
        # Compute direction vectors
        directions = activation_centroids - dynamic_centroids  # Shape: (num_tiles, dim)

        # Scalar field values at activation centroids and dynamic centroids
        activation_values = scalar_function(
            activation_centroids[:, 0], activation_centroids[:, 1], activation_centroids[:, 2]
        )
        dynamic_values = scalar_function(
            dynamic_centroids[:, 0], dynamic_centroids[:, 1], dynamic_centroids[:, 2]
        )

        # Compute interpolation factor t for the scalar field
        t = (isovalue - dynamic_values) / (activation_values - dynamic_values + 1e-8)  # Shape: (num_tiles,)
        t = t.unsqueeze(-1)  # Reshape to match the dimensionality of directions

        # Interpolate refined intersection points
        refined_intersections = dynamic_centroids + t * directions  # Shape: (num_tiles, dim)

        return refined_intersections

    def _centroids_to_tensor(self, centroids):
        """
        Convert the list of (bitmask, (x, y, z)) tuples into a tensor indexed by bitmask.
        """
        # Step 1: Find the maximum bitmask to create a tensor of appropriate size
        max_bitmask = max([bitmask for bitmask, _ in centroids])
        tensor_size = max_bitmask + 1  # Include 0 as well if bitmask starts from 0
        
        # Step 2: Initialize a tensor with zeros
        # Assuming the centroid values are 3D (x, y, z), so we initialize a 3-column tensor.
        centroids_tensor = torch.zeros((tensor_size, 3), device=self.device, dtype=self.precision)
        
        # Step 3: Assign the centroid values at the positions specified by the bitmask
        for bitmask, (x, y, z) in centroids:
            centroids_tensor[bitmask] = torch.tensor([x, y, z], dtype=self.precision, device=self.device)
        
        centroids_tensor_two = centroids_tensor.unsqueeze(1)
        
        
        return centroids_tensor_two



    def _generate_grid(self, domain_bounds, oversampling=(1,1,1)):
        """
        Generate grid points based on the domain bounds, tile size, and oversampling rates.

        Args:
            domain_bounds (list of tuples): Bounds for each dimension, e.g., [(x_min, x_max), (y_min, y_max), ...].
            oversampling (tuple): Per-dimension oversampling rates, must have the same length as domain_bounds.
                                Each rate is a multiplier for the number of points in that dimension.

        Returns:
            torch.Tensor: Tensor of shape (num_points, num_dimensions) containing grid points.
        """
        # Ensure oversampling rates match the number of dimensions
        num_dimensions = len(domain_bounds)
        if len(oversampling) != num_dimensions:
            raise ValueError("Oversampling tuple must have the same length as domain_bounds.")

        # Calculate grid points for each dimension
        grid_ranges = []
        for i, (start, end) in enumerate(domain_bounds):
            # Adjust step size based on tile size and oversampling rate
            steps = int((end - start) / self.tile_size[i] * oversampling[i]) + 1
            grid_range = torch.linspace(
                start, end, steps=steps, device=self.device, dtype=self.precision
            )
            grid_ranges.append(grid_range)

        # Create mesh grid using the computed ranges
        mesh = torch.meshgrid(*grid_ranges, indexing="ij")

        # Flatten the grid and stack into a single tensor
        grid_points = torch.stack([m.flatten() for m in mesh], dim=-1)

        return grid_points



    def _compute_bitmasks(self, scalar_function, vector_field, isovalue):
        """
        Compute bitmasks for vertices based on scalar field values.

        Args:
            scalar_function (callable): Function to compute scalar field values from spatial coordinates.
            vector_field (torch.Tensor): Tensor of shape (..., 3) representing spatial coordinates.
            isovalue (float): Threshold value for determining active vertices.

        Returns:
            torch.Tensor: Filtered bitmask values on `self.device`.
            torch.Tensor: Filtered relevant portions of the input `vector_field`.
        """
        # Evaluate the scalar field at the given vector field positions
        scalar_field = scalar_function(vector_field[..., 0], vector_field[..., 1], vector_field[..., 2])
        
        # Compute the number of vertices including the centroid
        effective_vertices = torch.round(self.vertex_count + 1).to(dtype=torch.int64)

        # Create a centroid mask to exclude the centroid bit
        centroid_mask = (2 ** effective_vertices - 1)
        centroid_mask &= ~(1 << (effective_vertices - 1))

        # Determine which vertices are above the isovalue
        above_water = (scalar_field > isovalue).to(dtype=torch.int64)

        # Compute bitmask values for all vertices
        power_values = 2 ** torch.arange(effective_vertices, device=self.device, dtype=torch.int64)
        bitmask = (above_water * power_values).sum(dim=1).to(self.device)

        # Apply the centroid mask to clear the centroid bit
        masked_bitmask = bitmask & torch.full_like(bitmask, centroid_mask)

        # Identify valid bitmask values based on relevance
        vector_field_relevance_mask = (bitmask > 0) & (bitmask < (2 ** self.vertex_count))

        # Return filtered bitmask and corresponding vector field values
        return masked_bitmask[vector_field_relevance_mask], vector_field[vector_field_relevance_mask]

    def tile_grid(self, grid_points, oversampling_factor=1, jitter_strength=None):
        """
        Tile the grid into the chosen geometry, applying jitter to expand offsets 
        into a unique field for each grid point.

        Args:
            grid_points (torch.Tensor): Tensor of shape (N, D), where N is the number of points
                                        and D is the dimensionality.
            oversampling_factor (int): Number of independently jittered shapes to apply per grid point
                                        (only used if jitter is enabled). Default is 1.

        Returns:
            torch.Tensor: Tiled vertices of shape (N * oversampling_factor, V, D), where N is the number of tiles,
                            V is the number of vertices per tile, and D is the dimensionality.
        """
        batch_size, num_dimensions = grid_points.shape
        if jitter_strength is not None:
            self.jitter_strength = jitter_strength
        if self.jitter_enabled:
            # Expand grid points for oversampling
            if oversampling_factor > 1:
                grid_points = grid_points.repeat_interleave(oversampling_factor, dim=0)  # Shape: (N * oversampling_factor, D)

            # Expand offsets to match the new grid points
            expanded_offsets = self.vertex_offsets.unsqueeze(0).expand(grid_points.size(0), -1, -1)  # (N * oversampling_factor, V, D)

            # Apply jitter to the expanded offsets
            jittered_offsets = self.generate_jitter(
                expanded_offsets,  # No need to flatten here
                jitter_strength=self.jitter_strength,
                jitter_seed=self.jitter_seed
            )  # (N * oversampling_factor, V, D)

            # Append the centroid offset
            jittered_offsets = torch.cat([
                jittered_offsets,
                torch.tensor([[0] * num_dimensions], device=self.device, dtype=self.precision).expand(grid_points.size(0), 1, num_dimensions)
            ], dim=1)
        else:
            # No jitter; use original offsets and append centroid offset
            expanded_offsets = self.vertex_offsets.unsqueeze(0).expand(batch_size, -1, -1)
            jittered_offsets = torch.cat([
                expanded_offsets,
                torch.tensor([[0] * num_dimensions], device=self.device, dtype=self.precision).expand(batch_size, 1, num_dimensions)
            ], dim=1)

        # Tile the grid points with jittered offsets
        tiled_vertices = grid_points.unsqueeze(1) + jittered_offsets

        return tiled_vertices

    
def scalar_sphere(x, y, z, r=1.0):
    """Example scalar function for a sphere."""
    return torch.sin(x)**2 + torch.sin(y)**2 + torch.sin(z)**2 - r**2

def initialize_pygame(width=800, height=600):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption('Live Isosurface Visualization')

def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    
    # Set up projection matrix
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (800/600), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)



def dynamic_scalar_sphere(x, y, z, t, r=1.0):
    return x ** 2 + y ** 2 + z ** 2 - r ** 2
    """Dynamic scalar function for a sphere with oscillating radius."""
    return torch.sin(x + t)**2 + torch.sin(y + t)**2 + torch.sin(z + t)**2 - r**2

def setup_lighting():
    """Set up basic lighting in the OpenGL scene."""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)  # Enable Light 0

    # Set light position and properties
    light_position = [5.0, 5.0, 5.0, 1.0]  # x, y, z, w
    light_diffuse = [1.0, 1.0, 1.0, 1.0]  # RGB diffuse
    light_specular = [0.5, 0.5, 0.5, 1.0]  # RGB specular
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

    # Set material properties
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])  # High specular reflection
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)  # Shininess exponent


def handle_events(events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, num_modes, rotating):
    """
    Handle all events, including mode switching, geometry switching, and scalar function switching.

    Args:
        events (list): List of pygame events.
        geometries (list): Available geometries.
        scalar_functions (list): Available scalar functions.
        processor (YoungManAlgorithm): Geometry processor.
        current_geometry (str): Current geometry.
        current_scalar_function (int): Current scalar function index.
        current_mode (int): Current rendering mode.
        num_modes (int): Total number of rendering modes.

    Returns:
        tuple: Updated current_geometry, current_scalar_function, and current_mode.
    """
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                # Switch scalar function
                idx = event.key - pygame.K_1
                if idx < len(scalar_functions):
                    logging.info(f"Switching to Scalar Function {idx + 1}")
                    current_scalar_function = idx
            elif pygame.K_a <= event.key <= pygame.K_z:
                # Switch geometry
                idx = event.key - pygame.K_a
                if idx < len(geometries):
                    geometry = geometries[idx]
                    logging.info(f"Switching to Geometry: {geometry}")
                    processor.switch_geometry(geometry)
                    current_geometry = geometry
                if event.key == pygame.K_r:
                    rotating = not rotating
            elif event.key == pygame.K_UP:
                # Increment rendering mode
                current_mode = (current_mode + 1) % num_modes
                logging.info(f"Switched to rendering mode {current_mode}")
            elif event.key == pygame.K_DOWN:
                # Decrement rendering mode
                current_mode = (current_mode - 1) % num_modes
                logging.info(f"Switched to rendering mode {current_mode}")
            elif event.key == pygame.K_ESCAPE:
                # Exit the program
                pygame.quit()
                quit()

    return current_geometry, current_scalar_function, current_mode, rotating


def render_opengl_data(evaluation_result, rotation_angle, mode, triangulator=None, rotating=False):
    """
    Render isosurface using the specified rendering mode.

    Args:
        evaluation_result (dict): Output of YoungManAlgorithm evaluation.
        rotation_angle (float): Rotation angle for the visualization.
        mode (int): Current rendering mode.
        triangulator (Triangulator): Instance for Delaunay-based rendering.
    """
    vertices = evaluation_result["vertices"].detach().cpu().numpy()
    indices = evaluation_result["indices"].detach().cpu().numpy()
    if not rotating:
        rotation_angle = 0
        
    if mode == 0:  # Point rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(rotation_angle, 1, 1, 0)

        glBegin(GL_POINTS)
        for vertex in vertices:
            glVertex3fv(vertex)
        glEnd()
    elif mode == 1:  # Triangle rendering (raw output)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(rotation_angle, 1, 1, 0)

        glBegin(GL_TRIANGLES)
        for tri in indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 2:  # Delaunay triangulation
        triangulated_data = triangulator.apply(vertices, decimation_factor=1)
        triangulated_indices = triangulated_data["indices"]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(rotation_angle, 1, 1, 0)

        glBegin(GL_TRIANGLES)
        for tri in triangulated_indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 3:  # Decimated Delaunay triangulation
        triangulated_data = triangulator.apply(vertices, decimation_factor=2)
        triangulated_indices = triangulated_data["indices"]

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(rotation_angle, 1, 1, 0)

        glBegin(GL_TRIANGLES)
        for tri in triangulated_indices:
            for vertex_idx in tri:
                vertex = vertices[vertex_idx]
                glVertex3fv(vertex)
        glEnd()
    elif mode == 4:  # High-quality rendering (not yet implemented)
        print("High-quality rendering is a placeholder.")
        # Placeholder: Add high-quality rendering logic here.
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    pygame.display.flip()

if __name__ == "__main__":
    # Initialize Pygame and OpenGL
    initialize_pygame()
    setup_opengl()
    setup_lighting()

    # Available geometries and scalar functions
    geometries = ["tetrahedron", "cube", "icosahedron", "octahedron", "square"]
    scalar_functions = [
        lambda x, y, z, t: x**2 + y**2 + z**2 - (1.0 + 0.5 * torch.sin(t)),  # Breathing sphere
        lambda x, y, z, t: torch.sin(x + y + z + t),                        # Sine wave propagation
        lambda x, y, z, t: torch.cos(x * y * z + t),                        # Oscillating cosine wave
        lambda x, y, z, t: x**2 - y**2 + z**2 - (0.5 + 0.3 * torch.cos(t)), # Ellipsoid oscillation
        lambda x, y, z, t: torch.sqrt(x**2 + y**2 + z**2) - (1.0 + 0.2 * torch.sin(2 * t)),  # Breathing shell
        lambda x, y, z, t: torch.sin(x + t) * torch.cos(y + t) * torch.sin(z + t)  # Ripple sphere
    ]

    # Initialize YoungManAlgorithm and Triangulator
    density = 10
    domain_bounds = [
        (-2.5, 2.5),  # x range
        (-2.5, 2.5),  # y range
        (-2.5, 2.5),  # z range
    ]
    isovalue = 0.0
    processor = YoungManAlgorithm(geometry=geometries[0], density=density, jitter_enabled=True, micro_jitter=False)
    triangulator = Triangulator()
    rotating = False

    # Default geometry, scalar function, and rendering mode
    current_geometry = geometries[0]
    current_scalar_function = 0
    current_mode = 0

    rotation_angle = 0
    clock = pygame.time.Clock()
    current_time = torch.tensor([0.0], device="cuda")

    while True:
        # Handle events for mode switching
        events = pygame.event.get()

        current_geometry, current_scalar_function, current_mode, rotating = handle_events(
            events, geometries, scalar_functions, processor, current_geometry, current_scalar_function, current_mode, 5, rotating
        )

        # Update current time for dynamic scalar fields
        current_time += 0.01  # Time progression

        # Get the active scalar function
        scalar_function = scalar_functions[current_scalar_function]

        # Evaluate the scalar field for the current geometry
        evaluation_result = processor.evaluate(
            lambda x, y, z: scalar_function(x, y, z, t=current_time),
            domain_bounds,
            isovalue=isovalue,
            gradient_normals=False,
            compute_vertex_normals=False,
            centroid_refinement=False,
            deduplicate=True,
            oversampling_grid=(1,1,1),
            oversampling_spot=1,
            jitter_strength=(torch.pi, torch.pi/2, 0)

        )

        # Render the updated data using the current mode
        render_opengl_data(evaluation_result, rotation_angle, current_mode, triangulator, rotating)

        # Update rotation angle
        rotation_angle += 1
        if rotation_angle >= 360:
            rotation_angle -= 360

        # Cap the frame rate
        clock.tick(60)
