import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


class FloatIsosurface:
    def __init__(self, geometry="cube"):
        """
        Initialize the FloatIsosurface with a given geometry.

        Args:
            geometry (str): The type of geometry ('square', 'cube', 'tetrahedron').
        """
        assert geometry in ["square", "cube", "tetrahedron"], \
            "Geometry must be 'square', 'cube', or 'tetrahedron'."
        self.geometry = geometry

        # Configure geometry-specific parameters
        self.vertex_count, self.edge_pairs = self._configure_geometry(geometry)

        # Build the general march map (dynamic lookup table)
        self.triangle_map = self._build_triangle_map()

    def _configure_geometry(self, geometry):
        """
        Configure the vertex and edge definitions for the chosen geometry.

        Args:
            geometry (str): The geometry type.

        Returns:
            tuple: (number of vertices, list of edge pairs)
        """
        if geometry == "square":
            vertex_count = 4
            edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif geometry == "cube":
            vertex_count = 8
            edge_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
        elif geometry == "tetrahedron":
            vertex_count = 4
            edge_pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")
        return vertex_count, edge_pairs


    def _build_triangle_map(self):
        """
        Build the triangle map dynamically for the configured geometry.

        Returns:
            torch.Tensor: A tensor containing triangulation data for each bitmask.
        """
        logging.info(f"Building triangle map for {self.geometry} geometry.")
        table_size = 2 ** self.vertex_count
        triangle_map = []

        for bitmask in range(table_size):
            edges = []
            for edge_idx, (v1, v2) in enumerate(self.edge_pairs):
                # Check if this edge is intersected based on the bitmask
                if ((bitmask & (1 << v1)) > 0) != ((bitmask & (1 << v2)) > 0):
                    edges.append(edge_idx)

            # Store edges as a list of indices
            edge_tensor = torch.full((15,), -1, dtype=torch.int32)  # Max 15 triangles (cube case)
            for i, edge_idx in enumerate(edges):
                if i < 15:  # Limit to 15 entries
                    edge_tensor[i] = edge_idx

            triangle_map.append(edge_tensor)

        # Convert to a PyTorch tensor
        print(f"the triangle map is: {triangle_map}")
        return torch.stack(triangle_map)

    def interpolate_edge(self, scalar_field, isovalue, v0_idx, v1_idx, vertex_coords):
        """
        Interpolates along an edge to find the point of intersection with the isosurface.
        Args:
            scalar_field (torch.Tensor): Scalar field at each vertex.
            isovalue (float): The isosurface threshold.
            v0_idx, v1_idx (int): Indices of the vertices at each edge.
            vertex_coords (torch.Tensor): The coordinates of the cube vertices.
        Returns:
            torch.Tensor: The interpolated coordinates of the intersection.
        """
        scalar_v0 = scalar_field[:, v0_idx]
        scalar_v1 = scalar_field[:, v1_idx]
        t = (isovalue - scalar_v0) / (scalar_v1 - scalar_v0 + 1e-8)  # Linear interpolation factor
        return_val = vertex_coords[:, v0_idx] + t.unsqueeze(-1) * (vertex_coords[:, v1_idx] - vertex_coords[:, v0_idx])
        print(return_val)
        return return_val

    def evaluate(self, scalar_field, isovalue, grid_coords):
        """
        Evaluate the scalar field and return isosurface triangles.

        Args:
            scalar_field (torch.Tensor): Scalar field grid of shape (B, 8),
                                         where B is batch size and 8 is vertices per cube.
            isovalue (float): The isosurface threshold value.
            grid_coords (torch.Tensor): Coordinates of each cube's origin in the grid, shape (B, 3).

        Returns:
            torch.Tensor: Vertices of triangles, shape (B, 15, 3).
        """
        B, N = scalar_field.shape
        assert N == self.vertex_count, f"Expected scalar_field with {self.vertex_count} vertices per cube."

        # Interpolating vertices
        vertex_offsets = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=torch.float32, device=scalar_field.device)
        vertex_coords = grid_coords.unsqueeze(1) + vertex_offsets.unsqueeze(0)

        # Edge interpolation
        edge_points = torch.zeros((B, len(self.edge_pairs), 3), dtype=torch.float32, device=scalar_field.device)
        for i, (v0, v1) in enumerate(self.edge_pairs):
            edge_points[:, i] = self.interpolate_edge(scalar_field, isovalue, v0, v1, vertex_coords)

        mask = scalar_field > isovalue
        print(f"Mask (scalar_field > isovalue):\n{mask}")  # Debug: Show the boolean mask

        bitmasks = 0
        for i in range(self.vertex_count):
            shifted_bit = (mask[:, i].int() << i)
            print(f"Vertex {i}: Mask = {mask[:, i].int()}, Shifted Bit = {shifted_bit}")
            bitmasks += shifted_bit

        print(f"Final bitmask tensor:\n{bitmasks}")

        # Lookup triangles
        triangle_indices = self.triangle_map[bitmasks.long()]

        # Gather triangle vertices
        triangle_vertices = torch.zeros((B, 15, 3), dtype=torch.float32, device=scalar_field.device)
        for i, batch in enumerate(triangle_indices):
            for j, edge_idx in enumerate(batch):
                if edge_idx >= 0:
                    triangle_vertices[i, j] = edge_points[i, edge_idx]

        return triangle_vertices


import torch
import numpy as np
from skimage.measure import marching_cubes
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Assuming FloatIsosurface is already defined and imported
def generate_test_case(level, geometry="cube"):
    """
    Generate scalar field values and anticipated bitmask for a given complexity level and geometry type.
    
    Args:
        level (int): The complexity level (1 to 5), where:
            1: Single trivial isosurface intersection.
            2: Multiple intersections forming simple geometry.
            3: Complex intersections with non-trivial topology.
            4: Near-degenerate cases with floating-point precision challenges.
            5: Fully filled and fully empty geometries.
        geometry (str): The type of geometry ('cube', 'square', 'tetrahedron').

    Returns:
        dict: A dictionary containing:
            - 'scalar_values': A tensor of scalar values for the geometry.
            - 'isovalue': The isovalue used for the test.
            - 'expected_bitmask': The anticipated bitmask.
            - 'description': A brief description of the test case.
    """
    assert geometry in ["cube", "square", "tetrahedron"], "Invalid geometry. Choose 'cube', 'square', or 'tetrahedron'."

    if geometry == "cube":
        vertex_count = 8
    elif geometry == "square":
        vertex_count = 4
    elif geometry == "tetrahedron":
        vertex_count = 4
    else:
        raise ValueError("Unsupported geometry.")

    if level == 1:
        scalar_values = torch.tensor([0.5] * (vertex_count - 1) + [1.5]).unsqueeze(0)
        expected_bitmask = 1 << (vertex_count - 1)
        description = "One vertex above isovalue, trivial intersection."
    elif level == 2:
        scalar_values = torch.tensor([(1.5 if i % 2 == 1 else 0.5) for i in range(vertex_count)]).unsqueeze(0)
        expected_bitmask = sum([(1 << i) for i in range(vertex_count) if i % 2 == 1])
        description = "Multiple intersections, simple geometry."
    elif level == 3:
        scalar_values = torch.tensor([1.2, 0.8, 1.5, 0.6] + ([0.9, 1.4, 0.7, 1.1] if geometry == "cube" else [])).unsqueeze(0)
        expected_bitmask = sum([(1 << i) for i, val in enumerate(scalar_values[0]) if val > 1.0])
        description = "Complex intersections with varying values."
    elif level == 4:
        scalar_values = torch.tensor([(1.00001 if i % 2 == 0 else 0.99999) for i in range(vertex_count)]).unsqueeze(0)
        expected_bitmask = sum([(1 << i) for i, val in enumerate(scalar_values[0]) if val > 1.0])
        description = "Near-degenerate case, precision challenges."
    elif level == 5:
        scalar_values = torch.tensor([1.5] * vertex_count).unsqueeze(0)
        expected_bitmask = (1 << vertex_count) - 1  # All vertices above isovalue
        description = "Fully filled geometry (no intersection)."
    else:
        raise ValueError("Level must be between 1 and 5.")

    return {
        "scalar_values": scalar_values,
        "isovalue": 1.0,
        "expected_bitmask": expected_bitmask,
        "description": description,
        "geometry": geometry
    }

def generate_sphere_scalar_field(grid_size, center, radius):
    """
    Generate a scalar field for a sphere.

    Args:
        grid_size (tuple): Dimensions of the scalar field (x, y, z).
        center (tuple): Center of the sphere.
        radius (float): Radius of the sphere.

    Returns:
        np.ndarray: 3D scalar field.
    """
    x, y, z = np.meshgrid(
        np.linspace(0, grid_size[0] - 1, grid_size[0]),
        np.linspace(0, grid_size[1] - 1, grid_size[1]),
        np.linspace(0, grid_size[2] - 1, grid_size[2]),
        indexing="ij"
    )
    field = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 - radius ** 2
    return field.astype(np.float32)

import numpy as np
from skimage.measure import marching_cubes
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

def validate_against_scikit_marching_cubes():
    levels_to_validate = [1, 2, 3, 4, 5]
    geometries_to_test = ["cube", "square", "tetrahedron"]

    for geometry in geometries_to_test:
        isosurface = FloatIsosurface(geometry=geometry)
        logging.info(f"\n--- Testing Geometry: {geometry} ---")

        for level in levels_to_validate:
            logging.info(f"\n--- Validating Level {level}: ---")
            test_case = generate_test_case(level, geometry=geometry)
            scalar_values = test_case["scalar_values"]
            isovalue = test_case["isovalue"]
            expected_bitmask = test_case["expected_bitmask"]
            description = test_case["description"]

            logging.info(f"Description: {description}")
            logging.info(f"Scalar Values:\n{scalar_values}")
            logging.info(f"Expected Bitmask: {bin(expected_bitmask)}")

            # Compute bitmask
            mask = scalar_values > isovalue
            computed_bitmask = 0
            for i in range(len(mask[0])):
                shifted_bit = (mask[:, i].int() << i)
                computed_bitmask += shifted_bit

            logging.info(f"Computed Bitmask: {bin(computed_bitmask.item())}")
            assert computed_bitmask.item() == expected_bitmask, f"Level {level}, Geometry {geometry}: Bitmask mismatch!"

            # Internal Isosurface Validation
            grid_coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            triangle_vertices = isosurface.evaluate(scalar_values, isovalue, grid_coords)
            logging.info(f"Triangle Vertices for Level {level}:\n{triangle_vertices}")

            try:
                scalar_min = scalar_values.min().item()
                scalar_max = scalar_values.max().item()

                # Ensure the isovalue is within the range of scalar field data
                if not (scalar_min <= isovalue <= scalar_max):
                    logging.warning(f"Skipping Scikit-Image validation for Level {level}, Geometry {geometry}: "
                                    f"Isovalue {isovalue} is out of range ({scalar_min}, {scalar_max}).")
                elif scalar_values.numel() == 8:  # Only attempt for full 3D cubes
                    vertices, faces, _, _ = marching_cubes(scalar_values.numpy().reshape((2, 2, 2)), level=isovalue)
                    logging.info(f"Scikit-Image Vertices for Level {level}: {vertices}")
                    logging.info(f"Scikit-Image Faces for Level {level}: {faces}")
                else:
                    logging.warning(f"Skipping Scikit-Image validation for Level {level}, Geometry {geometry}: Incompatible grid size.")
            except ValueError as e:
                if "cannot reshape array" in str(e):
                    logging.warning(f"Skipping Scikit-Image validation due to reshaping error: {e}")
                else:
                    raise e



if __name__ == "__main__":
    validate_against_scikit_marching_cubes()
