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
from adaptivegraphnetwork import AdaptiveGraphNetwork


logging.basicConfig(level=logging.INFO, format="%(message)s")


class YoungManAlgorithm:
    def __init__(self, geometry="cube", density=1.0, save_dir="geometry_maps", device="cuda", precision=torch.float64, jitter_enabled=False, jitter_seed=None, jitter_strength = (0.0, 0.0, 0.0), micro_jitter=False, enforce_req_grad=True):
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
        self.translated_offsets = None
        self.historical_filter = []
        self.accumulated_force = None
        self.accumulated_torque = None

        # Create an instance of CompositeGeometry
        self.geometry_wrapper = CompositeGeometry(geometry, precision, device)

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Initialize cache for geometry-specific maps
        self.geometry_maps = {}
        # Load or generate maps for the initial geometry
        self._load_and_set_geometry(geometry, density, micro_jitter)
        # Redirect methods to CompositeGeometry
    
    def _configure_geometry(self, geometry, density, micro_jitter):
        return self.geometry_wrapper.configure_geometry(geometry, density, micro_jitter)        
    
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

    def compute_buoyancy_forces(vertices, scalar_function, isovalue, center_of_mass):
        """
        Compute buoyancy forces and torques on a geometry.
        
        Args:
            vertices (torch.Tensor): Geometry vertices, shape (N, 3).
            scalar_function (callable): Scalar field function.
            isovalue (float): Target scalar field value for equilibrium.
            center_of_mass (torch.Tensor): Center of mass of the geometry, shape (3,).

        Returns:
            torch.Tensor: Net force vector, shape (3,).
            torch.Tensor: Net torque vector, shape (3,).
        """
        # Compute scalar field values at vertex positions
        scalar_values = scalar_function(vertices[:, 0], vertices[:, 1], vertices[:, 2])

        # Compute buoyant forces proportional to scalar field error
        forces = -1.0 * (scalar_values - isovalue).unsqueeze(1) * (vertices - center_of_mass)

        # Sum forces and compute torques
        net_force = forces.sum(dim=0)
        torques = torch.cross(vertices - center_of_mass, forces)
        net_torque = torques.sum(dim=0)

        return net_force, net_torque

    def generate_jitter(self, offsets, jitter_strength=None, jitter_seed=None, historical_filter=True):
        """
        Generate spherical jitter for a batch of offsets and apply it as rotation and translation.
        Historical jitter is always applied.

        Args:
            offsets (torch.Tensor): Tensor of shape (batch_size, V, 3) representing the original offsets for all units.
            jitter_strength (tuple): Strength of jitter for (theta, phi, r), where:
                                    - theta: Angular deviation around the z-axis.
                                    - phi: Polar deviation from the z-axis.
                                    - r: Translation magnitude.
            jitter_seed (int): Seed for reproducibility.
            historical_filter (bool): If True, includes historical transformations.

        Returns:
            torch.Tensor: Jittered offsets of shape (batch_size, V, 3).
        """
        # Apply historical jitter
        historical_jitter = self.anticipate(self.historical_filter, tile_count=offsets.shape[0])
        historical_translation = historical_jitter["translation_vector"]
        historical_rotation_axis = historical_jitter["rotation_axis"]
        historical_rotation_angle = historical_jitter["rotation_angle"]

        # Apply historical rotation to offsets
        if historical_rotation_axis.any() > 0.0:
            # Compute rotation matrix for historical rotation
            cos_angle = torch.cos(historical_rotation_angle)
            sin_angle = torch.sin(historical_rotation_angle)
            one_minus_cos = 1 - cos_angle

            # Normalize axis for batch processing
            #print(historical_rotation_axis.shape)
            
            axis = historical_rotation_axis / historical_rotation_axis.norm(dim=1, keepdim=True)
            #print(axis.shape)
             
            # Compute outer product for batch
            outer_product = torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))

            # Construct rotation matrices
            rotation_matrix = torch.eye(3, device=self.device, dtype=self.precision).unsqueeze(0).repeat(axis.size(0), 1, 1)
            #print(rotation_matrix.shape)
            
            prematrix = torch.stack([
                torch.zeros_like(axis[:, 0]), -axis[:, 2], axis[:, 1],
                axis[:, 2], torch.zeros_like(axis[:, 0]), -axis[:, 0],
                -axis[:, 1], axis[:, 0], torch.zeros_like(axis[:, 2])  # Fix: Add this element
            ], dim=-1).reshape(-1, 3, 3)

            #print(prematrix.shape)
            matrix = sin_angle.unsqueeze(-1).unsqueeze(-1) * prematrix.reshape(-1, 3, 3)

                        
            # Ensure the rotation matrix is correct
            #print("Prematrix shape:", prematrix.shape)  # Should be (20, 3, 3)
            rotation_matrix += one_minus_cos.unsqueeze(-1).unsqueeze(-1) * outer_product
            rotation_matrix += matrix
            #print("Rotation matrix shape:", rotation_matrix.shape)  # Should be (20, 3, 3)

            # Ensure offsets are transposed correctly for batch matrix multiplication
            #print("Offsets shape before transpose:", offsets.shape)  # Should be (20, V, 3)
            offsets = offsets.transpose(1, 2)  # Shape: (20, 3, V)
            #print("Offsets shape after transpose:", offsets.shape)

            # Apply the rotation
            rotated_offsets = torch.bmm(rotation_matrix, offsets).transpose(1, 2)  # Shape: (20, V, 3)

        else:
            rotated_offsets = offsets.clone()

        # Apply historical translation
        def ensure_correct_shape(tensor, reference_tensor, device, dtype):
            if tensor is None or tensor.shape != reference_tensor.shape:
                return torch.zeros_like(reference_tensor, device=device, dtype=dtype)
            return tensor
        self.translated_offsets = ensure_correct_shape(self.translated_offsets, rotated_offsets, self.device, self.precision)
        #print(rotated_offsets.shape)        
        #print(historical_translation.unsqueeze(0).shape)
        self.translated_offsets = rotated_offsets + historical_translation.unsqueeze(1)

        #print(self.translated_offsets)
        # If jitter is disabled, return only the historical transformation
        #if not self.jitter_enabled:# and historical_rotation_angle != 0:
        #    return self.translated_offsets, (rotation_matrix if historical_rotation_angle > 0 else torch.eye(3), historical_translation, None)
        #elif not self.jitter_enabled:
        #    return offsets, (torch.eye(3), historical_translation, None)
        # Generate new jitter if enabled
        device = self.device
        if jitter_strength is None:
            jitter_strength = self.jitter_strength

        if jitter_strength is None:
            jitter_strength = (0,0,0)

        if jitter_seed is not None:
            torch.manual_seed(jitter_seed)

        # Generate random spherical coordinates per batch
        batch_size, num_vertices, _ = offsets.size()
        theta = (torch.rand(batch_size, device=self.device, dtype=self.precision) - 0.5) * 2 * jitter_strength[0]
        phi = (torch.rand(batch_size, device=self.device, dtype=self.precision) - 0.5) * 2 * jitter_strength[1]
        r = (torch.rand(batch_size, device=self.device, dtype=self.precision) - 0.5) * 2 * jitter_strength[2]
        spherical_jitter = torch.stack([r, theta, phi], dim=-1)

        # Calculate rotation matrices for each unit (batch)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        rotation_matrices = torch.stack([
            cos_theta * cos_phi, -sin_theta, cos_theta * sin_phi,
            sin_theta * cos_phi, cos_theta, sin_theta * sin_phi,
            -sin_phi, torch.zeros_like(theta), cos_phi
        ], dim=-1).reshape(batch_size, 3, 3)

        # Apply rotation to offsets
        jittered_offsets = torch.bmm(rotation_matrices, self.translated_offsets.transpose(1, 2)).transpose(1, 2)

        # Convert r to Cartesian translation
        x_translation = r * torch.sin(phi) * torch.cos(theta)
        y_translation = r * torch.sin(phi) * torch.sin(theta)
        z_translation = r * torch.cos(phi)
        translation_vector = torch.stack([x_translation, y_translation, z_translation], dim=-1)
        #print(translation_vector)
        # Apply translation to jittered offsets
        jittered_offsets += translation_vector.unsqueeze(1)

        return jittered_offsets, (rotation_matrices, translation_vector, spherical_jitter)



    def _load_and_set_geometry(self, geometry, density, micro_jitter):
        """Load maps for the given geometry and set as current attributes."""
        self.vertex_offsets, self.vertex_count, self.edge_pairs, self.edge_lengths, \
            self.active_vertex_map, self.triangle_map, self.triangle_mask, self.force_map, self.torque_map, self.centroids, self.tile_size = \
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

    def _load_geometry_maps(self, geometry, density, micro_jitter=False, noload=False):
        """
        Load all necessary maps for a given geometry.
        Returns:
            A tuple containing all required maps.
        """
        
        maps = {}
        map_names = ["vertex_offsets", "vertex_count", "edge_pairs", "edge_lengths", "force_map", "torque_map",
                    "active_vertex_map", "triangle_map", "triangle_mask", "centroids", "tile_size"]
        for map_name in map_names:
            loaded_map = self._load_map(geometry, density, map_name)

            if loaded_map is not None:
                maps[map_name] = loaded_map

        # Check if all maps are loaded
        if not noload and len(maps) == len(map_names) and micro_jitter is False:
            logging.info(f"All maps loaded from cache for geometry: {geometry}")
            return (maps["vertex_offsets"], maps["vertex_count"], maps["edge_pairs"],
                    maps["edge_lengths"], maps["active_vertex_map"], maps["triangle_map"],
                    maps["triangle_mask"], maps["force_map"], maps["torque_map"], maps["centroids"], maps["tile_size"])
        else:
            logging.info(f"Generating geometry maps for {geometry}...")
            vertex_offsets, vertex_count, edge_pairs, edge_lengths, tile_size = self._configure_geometry(geometry, density, micro_jitter)
            active_vertex_map, triangle_mask, triangle_map, force_map, torque_map, centroids = self._build_triangle_map(edge_pairs, vertex_count, vertex_offsets)
            centroids = self._centroids_to_tensor(centroids)

            # Save generated maps
            self._save_map(geometry, density, vertex_offsets, "vertex_offsets")
            self._save_map(geometry, density, vertex_count, "vertex_count")
            self._save_map(geometry, density, edge_pairs, "edge_pairs")
            self._save_map(geometry, density, edge_lengths, "edge_lengths")
            self._save_map(geometry, density, active_vertex_map, "active_vertex_map")
            self._save_map(geometry, density, triangle_map, "triangle_map")
            self._save_map(geometry, density, force_map, "force_map")
            self._save_map(geometry, density, torque_map, "torque_map")
            self._save_map(geometry, density, triangle_mask, "triangle_mask")
            self._save_map(geometry, density, centroids, "centroids")
            self._save_map(geometry, density, tile_size, "tile_size")

            return (vertex_offsets, vertex_count, edge_pairs, edge_lengths,
                    active_vertex_map, triangle_map, triangle_mask, force_map, torque_map, centroids, tile_size)
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
        if geometry == "network_conifgured":
            self._load_network_geometry(self.edge_pairs, self.active_vertex_map, self.geometry_wrapper)
        
        if geometry not in self.geometry_maps:
            self.geometry_maps[geometry] = {}

        if density not in self.geometry_maps[geometry] or geometry == "network_configured":
            maps = self._load_geometry_maps(geometry, density, micro_jitter)
            self.geometry_maps[geometry][density] = maps

        return self.geometry_maps[geometry][density]
    def _load_network_geometry(self, new_edges, new_offsets, geometry_manager):
        geometry_manager.load_network_override(new_edges, new_offsets)
    def switch_geometry(self, new_geometry, density=None):
        """
        Switch the current geometry to a new geometry and optionally update density.
        """
        if density is not None:
            self.density = density

        if new_geometry == self.current_geometry and density == self.density:
            logging.info(f"Already using geometry: {new_geometry} with density: {self.density}")
            return

        assert new_geometry in ["square", "cube", "tetrahedron", "octahedron", "icosahedron", "network_configured"], \
            "Unsupported geometry. Choose 'square', 'cube', 'tetrahedron', 'octahedron', or 'icosahedron'."

        #logging.info(f"Switching geometry from {self.current_geometry} to {new_geometry} with density {self.density}...")
        self.current_geometry = new_geometry
        maps = self._load_geometry_maps_if_cached(new_geometry, self.density)
        self.historical_filter = []
        self.accumulated_force = None
        self.accumulated_torque = None
        self.vertex_offsets, self.vertex_count, self.edge_pairs, self.edge_lengths, \
            self.active_vertex_map, self.triangle_map, self.triangle_mask, self.force_map, self.torque_map, self.centroids, self.tile_size = maps

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
        centroids = []
        force_map = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), 3), dtype=self.precision, device=self.device)
        torque_map = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), 3), dtype=self.precision, device=self.device)

        triangle_mask = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), max_triangles), dtype=torch.bool, device=self.device)
        active_vertex_map = torch.zeros((2 ** vertex_count.to(dtype=torch.int64), vertex_count.to(dtype=torch.int64)), dtype=torch.bool, device=self.device)  # Active vertices
        

        for bitmask in range(2 ** vertex_count.to(dtype=torch.int64)):
            # Step 1: Identify active vertices for the current bitmask
            active_vertex_mask = (bitmask & (1 << torch.arange(vertex_count.to(dtype=torch.int64), device=self.device))) > 0
            active_vertex_map[bitmask] = active_vertex_mask  # Store active vertex map
            active_vertices = vertex_offsets[active_vertex_mask]
            active_indices = torch.nonzero(active_vertex_mask).flatten().tolist()
            if active_vertices.size(0) == 0:  # All vertices inactive (bitmask=0)
                # Apply upward perturbation force to break inactivity
                force_map[bitmask] = torch.tensor([-1.0, -1.0, -1.0], dtype=self.precision, device=self.device)
                torque_map[bitmask] = torch.tensor([0, 0, 0], dtype=self.precision, device=self.device)
                

            elif active_vertices.size(0) == vertex_count:  # All vertices active (full bitmask)
                # Apply downward perturbation force to deactivate some vertices
                force_map[bitmask] = torch.tensor([1.0, 1.0, 1.0], dtype=self.precision, device=self.device)
                torque_map[bitmask] = torch.tensor([0, 0, 0], dtype=self.precision, device=self.device)
                

            # Step 2: Calculate centroid (optional)
            try:
                if active_vertices.size(0) == 0:
                    centroid = torch.tensor([0.0,0.0,0.0], dtype=self.precision, device=self.device)
                else:
                    centroid = active_vertices.mean(axis=0)
        
                    active_vertices = torch.cat([active_vertices, centroid.unsqueeze(0)])  # Add centroid for fan reference
    
                centroids.append((bitmask, centroid))
            except Exception as e:
                print(f"Error calculating centroid for bitmask {bitmask}: {e}")
                exit()
                #continue

            if centroid.all() == 0.0:
                force_direction = torch.rand(3, device=self.device, dtype=self.precision)
            else:
                force_direction = centroid / centroid.norm()  # Normalize direction
            buoyant_force = force_direction * active_vertices.size(0)  # Scale by the number of active vertices
            force_map[bitmask] = buoyant_force

            # Step 4: Compute torque
            if centroid.all() == 0.0:
                torque = torch.rand(3, device=self.device, dtype=self.precision)
            else:
                torque = torch.cross(centroid, buoyant_force)  # Cross product for torque
            torque_map[bitmask] = torque

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

        return active_vertex_map, triangle_mask, triangle_map, force_map, torque_map, centroids
    def reverse_jitter_transformations(self, data, jitter_set, use_spherical=False, spherical_params=None):
        """
        Reverse transformations applied by jitter or apply transformations using spherical parameters.

        Args:
            data (torch.Tensor): Data to transform back or apply transformations to.
            jitter_set (tuple): Rotation matrices and translation vectors for vertices.
            use_spherical (bool): Whether to use spherical parameters instead.
            spherical_params (tuple): Spherical parameters (theta, phi, r) for intersections.

        Returns:
            torch.Tensor: Data with reversed or applied transformations.
        """
        if use_spherical and spherical_params is not None:
            
            theta, phi, r = torch.mean(spherical_params[...,2]), torch.mean(spherical_params[...,1]), torch.mean(spherical_params[...,0])
            

            # Calculate Cartesian translation from spherical parameters
            x_translation = r * torch.sin(phi) * torch.cos(theta)
            y_translation = r * torch.sin(phi) * torch.sin(theta)
            z_translation = r * torch.cos(phi)
            translation_vector = torch.stack([x_translation, y_translation, z_translation], dim=-1)

            # Apply translations
            transformed_data = data + translation_vector.unsqueeze(0)
            return transformed_data
        else:#this isn't working I'm turning it off because why use it
            rotation_matrices, translation_vectors = jitter_set

            # Reverse translation
            data_translated = data - translation_vectors

            # Reverse rotation (apply transpose of rotation matrix)
            reversed_data = torch.bmm(data_translated, rotation_matrices.transpose(1, 2))
            return reversed_data

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
    def prepare_line_diagram(self, edge_archtypes, vertices):
        """
        Prepare a pre-rendered line diagram for archetype edges, instance edges, and intercept points.

        Args:
            edge_archtypes (torch.Tensor): Tensor containing edge indices for archetypal edges ([num_edges, 2]).
            vertices (torch.Tensor): Tensor containing instance data in the shape [B, T, E, 9].

        Returns:
            tuple: (vao, vbo_positions, vbo_colors)
                vao (int): Vertex array object.
                vbo_positions (int): Vertex buffer object for edge positions and intercept points.
                vbo_colors (int): Vertex buffer object for edge and point colors.
        """
        # Step 1: Extract archetype edges
        archetype_starts = self.vertex_offsets[edge_archtypes[:, 0]].detach().cpu().numpy()
        archetype_ends = self.vertex_offsets[edge_archtypes[:, 1]].detach().cpu().numpy()

        archetype_positions = np.empty((len(edge_archtypes) * 2, 3), dtype=np.float32)
        archetype_positions[0::2] = archetype_starts
        archetype_positions[1::2] = archetype_ends

        # Step 2: Extract instance edges from vertices tensor
        instance_starts = vertices[..., :3].reshape(-1, 3).detach().cpu().numpy()
        instance_ends = vertices[..., 3:6].reshape(-1, 3).detach().cpu().numpy()

        instance_positions = np.empty((len(instance_starts) + len(instance_ends), 3), dtype=np.float32)
        instance_positions[0::2] = instance_starts
        instance_positions[1::2] = instance_ends

        # Step 3: Extract intercept points
        intercept_points = vertices[..., 6:].reshape(-1, 3).detach().cpu().numpy()
        #print(intercept_points)
        # Step 4: Calculate mean edge
        mean_start = instance_starts.mean(axis=0)
        mean_end = instance_ends.mean(axis=0)
        mean_positions = np.array([mean_start, mean_end], dtype=np.float32)

        # Step 5: Define colors
        archetype_colors = np.tile([1.0, 1.0, 1.0, 1.0], (len(archetype_positions), 1))  # Yellow
        intercept_colors = np.tile([0.00010, 0.00010, 0.00010, 0.00010], (len(intercept_points), 1))  # Red for intercept points
        instance_colors = np.tile([1.0, 0.0, 0.0, 1.0], (len(instance_positions), 1))  # Pale blue
        mean_colors = np.tile([0.0, 1.0, 0.0, 1.0], (2, 1))  # White for mean edge
        

        # Combine all positions and colors
        all_positions = np.vstack([intercept_points, archetype_positions, instance_positions, mean_positions])
        vbo_lengths = all_positions.shape[0]
        #print(all_positions)
        all_colors = np.vstack([intercept_colors, archetype_colors, instance_colors, mean_colors])
        #all_positions = np.nan_to_num(all_positions, nan=1, posinf=1, neginf=-1)
        
        # Validation: Check for NaNs or Infs in positions and colors
        def validate_data(array, name):
            """Helper function to validate arrays."""
            num_nans = np.isnan(array).sum()
            num_infs = np.isinf(array).sum()
            min_value = np.nanmin(array)
            max_value = np.nanmax(array)
            mean_value = np.nanmean(array)
            std_value = np.nanstd(array)

            #print(f"\nValidation Report for {name}:")
            #print(f"Shape: {array.shape}")
            #print(f"NaNs: {num_nans}, Infs: {num_infs}")
            #print(f"Min: {min_value}, Max: {max_value}")
            #print(f"Mean: {mean_value}, Std Dev: {std_value}")

        # Validate position and color data
        validate_data(all_positions, "All Positions")
        validate_data(all_colors, "All Colors")

        # Step 6: Create OpenGL buffers
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        # Helper function to validate buffer data
        def validate_buffer(buffer_target, original_data, data_type):
            """Validates OpenGL buffer content against original data."""
            buffer_size = original_data.nbytes
            retrieved_data = glGetBufferSubData(buffer_target, 0, buffer_size)
            extracted_data = np.frombuffer(retrieved_data, dtype=data_type)

            # Debugging raw data mismatch
            #print("\nOriginal Data (First 5 Elements):", original_data[:50])
            #print("Retrieved Data (First 5 Elements):", extracted_data[:50])

            if not np.array_equal(original_data, extracted_data):
                # Compare raw byte content
                original_bytes = original_data.tobytes()
                retrieved_bytes = retrieved_data
                #print("Original Bytes (First 20):", original_bytes[:200])
                #print("Retrieved Bytes (First 20):", retrieved_bytes[:200])
                #raise ValueError(f"Validation failed: Buffer content does not match original data.")
            #print("Validation successful: Buffer data matches original data.")

        # Vertex positions VBO
        vbo_positions = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, all_positions.nbytes, all_positions, GL_STATIC_DRAW)

        # Extract and validate the buffer data
        validate_buffer(GL_ARRAY_BUFFER, all_positions, all_positions.dtype)

        # Vertex positions VBO
        vbo_positions = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, all_positions.nbytes, all_positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Colors VBO
        vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, all_colors.nbytes, all_colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)  # Unbind VAO
        return vao, vbo_positions, vbo_colors, vbo_lengths


    def process_evaluation_candidate_dataset(self, evaluation_candidate_dataset, jitter_set, grid_points, tiled_vertices, relevant_indices):
        """
        Process candidate datasets by batch-masking well-activated tiles.

        Args:
            evaluation_candidate_dataset (tuple): Dataset containing candidate tiles and intersections.
            jitter_set (tuple): Rotation matrices, translation vectors, and spherical jitters.
            grid_points (torch.Tensor): The grid points used for evaluation.
            tiled_vertices (torch.Tensor): Tiled vertices of the geometry.

        Returns:
            dict: Processed dataset including normalized geometry, edges, render buffers, and metadata.
        """
        # Extract relevant data from the dataset
        candidate_tiles, candidate_intersections = evaluation_candidate_dataset
        active_tiles_mask = relevant_indices
        # Filter active tiles by mask
        active_grid_points = grid_points[active_tiles_mask]
        
        active_jitter_set = (
            jitter_set[0][active_tiles_mask],
            jitter_set[1][active_tiles_mask],
            jitter_set[2]
        )

        active_candidates = candidate_tiles
        active_intersections = candidate_intersections

        # Reverse transformations for candidates and intersections
        reversed_candidates = self.reverse_jitter_transformations(
            active_candidates, active_jitter_set, use_spherical=True, spherical_params=active_jitter_set[2]
        )

        reversed_intersections = self.reverse_jitter_transformations(
            active_intersections, None, use_spherical=True, spherical_params=active_jitter_set[2]
        )

        # Normalize coordinates to the grid point of the tile
        normalized_candidates = reversed_candidates - active_grid_points.unsqueeze(1)#.expand(-1, reversed_candidates.size(1), -1) #wrong size, active grid points needs to be extended for all the candidate vertices right?
        normalized_intersections = reversed_intersections - active_grid_points.unsqueeze(1)
       
        # Combine vertices for rendering
        blended_vertices = torch.cat([normalized_candidates, normalized_intersections], dim=1)
       
        # Use edge pairs from the current geometry and extract vertices for edges
        edges = self.edge_pairs.to(dtype=torch.int64)  # Access preloaded edge pairs
        if len(normalized_candidates.shape) <= 3:
            normalized_candidates = normalized_candidates.unsqueeze(0)
        if len(normalized_intersections.shape) <= 3:
            normalized_intersections = normalized_intersections.unsqueeze(0)
        # Extract shapes
        B, T, N, D = normalized_candidates.shape  # Batch, Tile, Vertices, Dimensions
        _, _, E, _ = normalized_intersections.shape  # Edges

        # Step 1: Gather edge endpoints using edge pairs
        edge_start = torch.gather(
            normalized_candidates, 2,
            edges[:, 0].view(1, 1, -1, 1).expand(B, T, -1, D)
        )
        edge_end = torch.gather(
            normalized_candidates, 2,
            edges[:, 1].view(1, 1, -1, 1).expand(B, T, -1, D)
        )

        # Step 2: Concatenate start and end points along the feature dimension
        edge_vertices = torch.cat([edge_start, edge_end], dim=-1)  # Shape: (B, T, E, 2 * D)
        
        # Step 3: Combine with normalized_intersections
        combined_edges = torch.cat([edge_vertices, normalized_intersections], dim=-1)  # Shape: (B, T, E + N, 2 * D)

        
        # Metadata for debugging and inspection
        metadata = {
            "original_candidates": active_candidates,
            "original_intersections": active_intersections,
            "jitter_matrices": active_jitter_set,
            "grid_points": active_grid_points,
        }
       
        # Prepare OpenGL buffers for rendering
        vao, vertex_vbo, edge_ebo = self.setup_opengl_buffers(blended_vertices, edges)
        # New: Prepare line diagram (without touching existing render pipeline)
        line_vao, line_vbo_positions, line_vbo_colors, line_vbo_length = self.prepare_line_diagram(
            edge_archtypes=edges,
            vertices=combined_edges
            
        )

        # Return all data, preserving the existing "render" key
        return {
            "normalized_geometry": blended_vertices,
            "edges": edges,
            "render": (vao, vertex_vbo, edge_ebo),  # Existing rendering pipeline untouched
            "line_diagram": (line_vao, line_vbo_positions, line_vbo_colors, line_vbo_length),  # New line diagram
            "metadata": metadata,
        }
    def anticipate(self, history, translation_limits=(0.0, 10.0), rotation_limits=(-torch.pi, torch.pi), tile_count=0):
        """
        Converts historical forces and torques into jitter recommendations for translation and rotation.

        Args:
            history (list): List of tuples containing force and torque tensors.
            translation_limits (tuple): Min and max limits for translation magnitude.
            rotation_limits (tuple): Min and max limits for rotation angle in radians.

        Returns:
            dict: Jitter recommendation containing:
                - "translation_vector": Recommended translation as (x, y, z).
                - "rotation_axis": Recommended rotation axis as a normalized vector (x, y, z).
                - "rotation_angle": Recommended rotation angle in radians.
        """
        if not history:
            # If history is empty, initialize with valid default values
            zero_vector = torch.zeros((tile_count, 3), device=self.device, dtype=self.precision)
            
            # Default rotation axis: Random normalized vector
            rotation_axis = torch.randn((tile_count, 3), device=self.device, dtype=self.precision)
            rotation_axis /= rotation_axis.norm(dim=1, keepdim=True)

            # Default rotation angle: Small random perturbation within limits
            #rotation_angle = torch.rand(tile_count, device=self.device, dtype=self.precision) * (rotation_limits[1] - rotation_limits[0]) + rotation_limits[0]
            rotation_angle = torch.zeros(tile_count, device=self.device, dtype=self.precision)
            #print(f"first pass, tile count:{tile_count} rotation axis shape: {rotation_axis.shape}")
            return {
                "translation_vector": zero_vector,
                "rotation_axis": rotation_axis,
                "rotation_angle": rotation_angle
            }

        # Separate forces and torques from history
        forces, torques = zip(*history)
        if self.accumulated_force is None or self.accumulated_torque is None:
            integrated_force = self.accumulated_force = forces[-1]
            integrated_torque = self.accumulated_torque = torques[-1]
        else:
            integrated_force = self.accumulated_force = self.accumulated_force + forces[-1] * .1
            integrated_torque = self.accumulated_torque = self.accumulated_torque + torques[-1] * .1
        # Print results
        #print(f"Non-zero integrated_force: {non_zero_force}")
        #print(f"Non-zero integrated_torque: {non_zero_torque}")

        # Step 1: Convert force to translation jitter
        translation_magnitude = integrated_force.norm()  # Magnitude of net force
        translation_clamped = torch.clamp(translation_magnitude, translation_limits[0], translation_limits[1])
        translation_vector = integrated_force / translation_magnitude * translation_clamped if translation_magnitude > 0 else torch.zeros((tile_count,3), device=self.device, dtype=self.precision)
        #print(f"translation_magnitude: {translation_magnitude}")
        # Step 2: Convert torque to rotation jitter
        torque_magnitude = integrated_torque.norm()  # Magnitude of net torque
        rotation_angle = torch.clamp(torque_magnitude, rotation_limits[0], rotation_limits[1])
        rotation_axis = integrated_torque / torque_magnitude if torque_magnitude > 0 else torch.zeros((tile_count, 3), device=self.device, dtype=self.precision)
        #print("subsequent passes")
        # Return jitter recommendations
        return {
            "translation_vector": translation_vector,  # Translation offset in Cartesian coordinates
            "rotation_axis": rotation_axis,  # Normalized axis of rotation
            "rotation_angle": rotation_angle  # Angle of rotation in radians
        }


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
        tiled_vertices, jitter_set, oversampled_grid_points = self.tile_grid(grid_points,oversampling_factor=oversampling_spot,jitter_strength=jitter_strength)

        # Step 2: Compute bitmasks and active vertices
        bitmasks, relevant_vertices, relevant_indices, raw_bitmasks = self._compute_bitmasks(scalar_function, tiled_vertices, isovalue)

        # Step 3: Compute edge intersections
        intersection_tensor, evaluation_candidate_dataset = self._compute_dense_intersections(
            relevant_vertices[:, : self.vertex_count.to(dtype=torch.int64), ...],
            scalar_function,
            isovalue,
        )
        evaluation_dataset = self.process_evaluation_candidate_dataset(evaluation_candidate_dataset, jitter_set, oversampled_grid_points, tiled_vertices, relevant_indices)
        passive_data = {
            "evaluation_dataset": evaluation_dataset,
            "bitmasks": bitmasks,  # Active vertices encoded in the bitmask
            "intersection_points": intersection_tensor.clone(),  # Interpolated intersection points
        }

        
        
        # Step 4: Compute dynamic centroids
        active_triangles_map = self.triangle_map[bitmasks]
        active_triangles_mask = self.triangle_mask[bitmasks]
        #print(bitmasks)
        #print(raw_bitmasks)
        
        tile_forces = self.force_map[raw_bitmasks]
        tile_torques = self.torque_map[raw_bitmasks]
        #print(f"tile_forces: {tile_forces}")
        #print(f"tile_torques: {tile_torques}")
        if len(self.historical_filter) > 300:
            # Slice the last 300 elements, then append the new element
            self.historical_filter = self.historical_filter[-300:]
            self.historical_filter.append((tile_forces, tile_torques))
        else:
            self.historical_filter.append((tile_forces, tile_torques))
        grid_centers = relevant_vertices[:, self.vertex_count.to(dtype=torch.int64), ...]
        #print(self.centroids)
        #print(bitmasks.shape)
        #print(bitmasks)
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
        result["passive_data"] = passive_data
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
                counts = torch.zeros(unique_vertices.size(0), device=unique_vertices.device, dtype=self.precision)

                # Expand triangle indices to (num_triangles * 3, 1)
                flat_triangle_indices = triangle_indices.view(-1)


                # Accumulate normals for each unique vertex
                vertex_normals = vertex_normals.index_add_(0, flat_triangle_indices, triangle_normals)

                # Count contributions for normalization
                counts = counts.index_add_(0, flat_triangle_indices, torch.ones_like(flat_triangle_indices, dtype=self.precision))

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
        intersection_tensor = torch.full((num_tiles, num_edges, 3), float('nan'), device=tiled_vertices.device, dtype=self.precision)

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
        intersection_points = v1_positions + t.unsqueeze(-1).to(self.precision) * (v2_positions - v1_positions)  # Shape: (num_tiles, num_edges, 3)

        # Populate dense tensor with intersection points
        intersection_tensor[active_edges_mask] = intersection_points[active_edges_mask]
        
        candidate_tiles = tiled_vertices
        
        candidate_intersections = intersection_tensor
        
        # Return data for further processing
        return intersection_tensor, (candidate_tiles, candidate_intersections)

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
    def preprocess_vertex_data(self, vertex_data, neutral_value=torch.tensor([-1e6, -1e6, -1e6])):
        # Identify and mask NaN values
        mask = ~torch.isnan(vertex_data).any(dim=-1)
        processed_data = vertex_data.clone()
        processed_data[~mask] = neutral_value.to(dtype=vertex_data.dtype, device=vertex_data.device)
        return processed_data, mask
    def setup_opengl_buffers(self, vertices, edges):
        # Ensure vertices tensor is contiguous and on CPU
        if not vertices.is_contiguous():
            vertices = vertices.contiguous()
        if vertices.is_cuda:
            vertices = vertices.cpu()

        vertices = vertices.detach().numpy()
        
        # Ensure edges tensor is contiguous and on CPU
        if not edges.is_contiguous():
            edges = edges.contiguous()
        if edges.is_cuda:
            edges = edges.cpu()

        edges = edges.detach().numpy()

        # Create buffers
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        
        # Vertex buffer
        vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.tobytes(), GL_STATIC_DRAW)

        # Enable vertex attribute pointer
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Edge buffer
        edge_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edge_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edges.nbytes, edges.tobytes(), GL_STATIC_DRAW)

        return vao, vertex_vbo, edge_ebo



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
        grid_points.requires_grad_(True)

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
        raw_bitmasks = masked_bitmask.clone()
        # Identify valid bitmask values based on relevance
        vector_field_relevance_mask = (bitmask > 0) & (bitmask < (2 ** self.vertex_count))

        relevant_indices = torch.nonzero(vector_field_relevance_mask, as_tuple=False).squeeze()

        # Return filtered bitmask, corresponding vector field values, and original indices
        return masked_bitmask[vector_field_relevance_mask], vector_field[vector_field_relevance_mask], relevant_indices, raw_bitmasks

    def tile_grid(self, grid_points, oversampling_factor=1, jitter_strength=None):
        """
        Tile the grid into the chosen geometry, applying jitter to expand offsets 
        into a unique field for each grid point.

        Args:
            grid_points (torch.Tensor): Tensor of shape (N, D), where N is the number of points
                                        and D is the dimensionality.
            oversampling_factor (int): Number of independently jittered shapes to apply per grid point
                                        (only used if jitter is enabled). Default is 1.
            jitter_strength (tuple): Strength of jitter for (theta, phi, r) or None.

        Returns:
            torch.Tensor: Tiled vertices of shape (N * oversampling_factor, V, D), where N is the number of tiles,
                            V is the number of vertices per tile, and D is the dimensionality.
        """
        # Extract grid properties
        batch_size, num_dimensions = grid_points.shape

        # Step 1: Handle oversampling
        if oversampling_factor > 1:
            # Repeat each grid point according to oversampling_factor
            grid_points = grid_points.repeat_interleave(oversampling_factor, dim=0)  # Shape: (N * oversampling_factor, D)

        # Step 2: Generate base offsets (vertex_offsets expanded to match grid points)
        expanded_offsets = self.vertex_offsets.unsqueeze(0).expand(grid_points.size(0), -1, -1)  # Shape: (N * oversampling_factor, V, D)
        #print(f"expanded_offsets: {expanded_offsets.shape}")
        # Step 3: Apply jitter if enabled
        #if self.jitter_enabled:
            # Apply jitter to the offsets
        jittered_offsets, jitter_set = self.generate_jitter(
                expanded_offsets,
                jitter_strength=self.jitter_strength or jitter_strength,
                jitter_seed=self.jitter_seed,
            )  # Shape: (N * oversampling_factor, V, D)
        #else:
        #    # If jitter is not enabled, use the base offsets without modification
        #    jittered_offsets = expanded_offsets
        #    rotation_matrices = torch.eye(3, device=self.device, dtype=self.precision).unsqueeze(0).expand(grid_points.size(0), -1, -1)
        #    translation_vector = torch.zeros(grid_points.size(0), 3, device=self.device, dtype=self.precision)
        #    spherical_jitter = torch.zeros(grid_points.size(0), 3, device=self.device, dtype=self.precision)


        #    jitter_set = (rotation_matrices, translation_vector, spherical_jitter)

        # Step 4: Append the centroid offset to the offsets
        centroid_offset = torch.zeros((1, 1, num_dimensions), device=self.device, dtype=self.precision)
        centroid_offsets = centroid_offset.expand(grid_points.size(0), 1, num_dimensions)  # Shape: (N * oversampling_factor, 1, D)
        final_offsets = torch.cat([jittered_offsets, centroid_offsets], dim=1)  # Shape: (N * oversampling_factor, V+1, D)

        # Step 5: Tile the grid points with the offsets
        tiled_vertices = grid_points.unsqueeze(1) + final_offsets  # Shape: (N * oversampling_factor, V+1, D)

        return tiled_vertices, jitter_set, grid_points


    
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
        lambda x, y, z, t: 0,
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
    processor = YoungManAlgorithm(geometry=geometries[0], density=density, jitter_enabled=False, micro_jitter=False)
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
            jitter_strength=(0, 0, 0)#(torch.pi, torch.pi/2, 0)

        )

        # Render the updated data using the current mode
        render_opengl_data(evaluation_result, rotation_angle, current_mode, triangulator, rotating)

        # Update rotation angle
        rotation_angle += 1
        if rotation_angle >= 360:
            rotation_angle -= 360

        # Cap the frame rate
        clock.tick(60)
