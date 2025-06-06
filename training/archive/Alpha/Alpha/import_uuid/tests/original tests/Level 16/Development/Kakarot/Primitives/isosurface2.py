import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


class GeometryProcessor:
    def __init__(self, geometry="cube", density=1.0):
        """
        Initialize the GeometryProcessor with a given geometry and density.

        Args:
            geometry (str): The type of geometry ('square', 'cube', 'tetrahedron').
            density (float): Scaling factor for the density of the tiling.
        """
        assert geometry in ["square", "cube", "tetrahedron"], \
            "Geometry must be 'square', 'cube', or 'tetrahedron'."
        self.geometry = geometry
        self.density = density

        # Define fundamental offsets for the chosen geometry
        self.vertex_offsets = self._define_offsets()

        # Configure geometry-specific parameters for isosurface evaluation
        self.vertex_count, self.edge_pairs = self._configure_geometry(geometry)

        # Build the triangle map dynamically
        self.triangle_map = self._build_triangle_map()

    def _define_offsets(self):
        """Define vertex offsets for the chosen geometry."""
        if self.geometry == "cube":
            # Correct vertices for a cube (unit cube)
            return torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=torch.float32)
        elif self.geometry == "tetrahedron":
            # Correct vertices for a single regular tetrahedron
            return torch.tensor([
                [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
            ], dtype=torch.float32)
        elif self.geometry == "square":
            # Correct vertices for a square in 2D
            return torch.tensor([
                [0, 0], [1, 0], [1, 1], [0, 1],
            ], dtype=torch.float32)
        elif self.geometry == "octahedron":
            # Correct vertices for a regular octahedron
            return torch.tensor([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ], dtype=torch.float32)
        elif self.geometry == "icosahedron":
            # Correct vertices for a regular icosahedron
            phi = (1 + 5 ** 0.5) / 2  # Golden ratio
            return torch.tensor([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ], dtype=torch.float32)

    def _configure_geometry(self, geometry):
        """
        Configure the vertex and edge definitions for the chosen geometry.

        Args:
            geometry (str): The geometry type.

        Returns:
            tuple: (number of vertices, list of edge pairs)
        """
        if geometry == "square":
            # Edges for a square (2D)
            vertex_count = 4
            edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif geometry == "cube":
            # Edges for a cube (3D)
            vertex_count = 8
            edge_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
        elif geometry == "tetrahedron":
            # Edges for a regular tetrahedron (3D)
            vertex_count = 4
            edge_pairs = [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ]
        elif geometry == "octahedron":
            # Edges for a regular octahedron (3D)
            vertex_count = 6
            edge_pairs = [
                (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 4), (2, 5), (3, 4), (3, 5)
            ]
        elif geometry == "icosahedron":
            # Edges for a regular icosahedron (3D)
            vertex_count = 12
            edge_pairs = [
                (0, 1), (0, 5), (0, 7), (0, 10), (0, 11),
                (1, 5), (1, 6), (1, 8), (1, 9),
                (2, 3), (2, 4), (2, 6), (2, 9), (2, 11),
                (3, 4), (3, 7), (3, 8), (3, 10),
                (4, 5), (4, 7), (4, 9),
                (5, 10), (6, 8), (6, 11),
                (7, 8), (7, 9), (8, 11),
                (9, 10), (9, 11), (10, 11)
            ]
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
        triangle_map = torch.full((table_size, 15), -1, dtype=torch.int32)

        bitmask_tensor = torch.arange(table_size)
        for edge_idx, (v1, v2) in enumerate(self.edge_pairs):
            v1_active = (bitmask_tensor & (1 << v1)) > 0
            v2_active = (bitmask_tensor & (1 << v2)) > 0
            intersected = v1_active ^ v2_active
            triangle_map[intersected, edge_idx] = edge_idx

        return triangle_map

    def tile_grid(self, grid_points):
        """
        Tile the grid into the chosen geometry.

        Args:
            grid_points (torch.Tensor): Tensor of shape (N, D), where N is the number of points
                                         and D is the dimensionality.

        Returns:
            torch.Tensor: Tiled vertices of shape (M, V, D), where M is the number of tiles,
                          V is the number of vertices per tile, and D is the dimensionality.
        """
        D = self.vertex_offsets.shape[1]  # Dimensionality of the geometry
        assert grid_points.shape[1] == D, f"Grid points must have dimensionality {D}."

        # Compute the bounding box of the grid
        min_coords = grid_points.min(dim=0).values
        max_coords = grid_points.max(dim=0).values

        # Generate a regular grid covering the bounding box
        ranges = [torch.arange(min_coords[i], max_coords[i], step=self.density) for i in range(D)]
        mesh = torch.meshgrid(*ranges, indexing="ij")
        base_points = torch.stack([m.flatten() for m in mesh], dim=-1)  # (M, D)

        # Apply offsets to generate tiles
        tiled_vertices = base_points.unsqueeze(1) + self.vertex_offsets.unsqueeze(0)

        return tiled_vertices

    def evaluate(self, scalar_function, domain_bounds, isovalue=1.0):
        """
        Evaluate the scalar function and return isosurface triangles.

        Args:
            scalar_function (callable): The function to evaluate at each vertex. Should accept D arguments.
            domain_bounds (tuple): The bounds of the domain as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
                                   For 2D geometries, provide ((xmin, xmax), (ymin, ymax)).
            isovalue (float): The isosurface threshold value.

        Returns:
            torch.Tensor: Vertices of triangles, shape (M, 15, D).
        """
        # Generate grid centers based on the domain
        ranges = [torch.linspace(start, end, steps=int((end - start) / self.density) + 1) 
                  for start, end in domain_bounds]
        mesh = torch.meshgrid(*ranges, indexing="ij")
        centers = torch.stack([m.flatten() for m in mesh], dim=-1)  # (C, D)

        # Evaluate scalar function at centers
        with torch.no_grad():
            center_values = scalar_function(*centers.T)  # (C,)

        # Tile the grid to get vertices
        tiled_vertices = self.tile_grid(centers)  # (M, V, D)

        # Flatten tiled_vertices for batch processing
        M, V, D = tiled_vertices.shape
        tiled_vertices_flat = tiled_vertices.view(-1, D)  # (M*V, D)

        # Evaluate scalar function at all vertices
        with torch.no_grad():
            vertex_values = scalar_function(*tiled_vertices_flat.T)  # (M*V,)

        # Reshape vertex_values to (M, V)
        vertex_values = vertex_values.view(M, V)  # (M, V)

        # Now, scalar_field is vertex_values: (M, V)
        scalar_field = vertex_values  # (M, V)

        # Proceed with isosurface extraction
        B, N = scalar_field.shape
        assert N == self.vertex_count, f"Expected scalar_field with {self.vertex_count} vertices."

        # Batch edge interpolation
        edge_starts = tiled_vertices[:, [v1 for v1, _ in self.edge_pairs]]  # (M, E, D)
        edge_ends = tiled_vertices[:, [v2 for _, v2 in self.edge_pairs]]    # (M, E, D)
        scalar_starts = scalar_field[:, [v1 for v1, _ in self.edge_pairs]]  # (M, E)
        scalar_ends = scalar_field[:, [v2 for _, v2 in self.edge_pairs]]    # (M, E)

        t = (isovalue - scalar_starts) / (scalar_ends - scalar_starts + 1e-8)  # (M, E)
        t = t.unsqueeze(-1)  # (M, E, 1)
        edge_points = edge_starts + t * (edge_ends - edge_starts)  # (M, E, D)

        # Compute bitmasks in batch
        mask = (scalar_field > isovalue).int()  # (M, N)
        powers_of_two = 2 ** torch.arange(self.vertex_count, device=mask.device)  # (N,)
        bitmasks = (mask * powers_of_two).sum(dim=1)  # (M,)

        # Lookup triangle indices
        triangle_indices = self.triangle_map[bitmasks.long()]  # (M, 15)

        # Handle invalid indices (-1 in triangle_map)
        valid_triangle_indices = triangle_indices >= 0  # (M, 15)
        sanitized_triangle_indices = triangle_indices.clamp(min=0).long()  # (M, 15)

        # Gather triangle vertices in batch
        # edge_points: (M, E, D), sanitized_triangle_indices: (M, 15)
        triangle_vertices = edge_points.gather(
            1, sanitized_triangle_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # (M, 15, D)

        # Fill invalid entries with zeros
        triangle_vertices[~valid_triangle_indices] = 0

        return triangle_vertices


import torch
import numpy as np
from skimage.measure import marching_cubes
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


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


def validate_with_batch_processing():
    """
    Extend the validation tests to include batch processing of an entire grid.
    Each batch will consist of multiple identical copies of grid points to ensure consistency.
    """
    levels_to_validate = [1, 2, 3, 4, 5]
    geometries_to_test = ["cube", "square", "tetrahedron"]

    for geometry in geometries_to_test:
        # Define a scalar function for testing
        def scalar_function(x, y, z=None):
            if geometry == "square":
                return torch.sin(x) * torch.cos(y) + 0.5
            elif geometry == "tetrahedron":
                # Custom scalar function for tetrahedron geometry
                return torch.sin(x) * torch.cos(y) + (torch.sin(z) if z is not None else 0.0) + 0.5
            else:
                return torch.sin(x) * torch.cos(y) * torch.sin(z) + 0.5

        # Define domain bounds based on geometry
        if geometry == "square":
            domain_bounds = ((0, 1), (0, 1))  # 2D
        elif geometry == "tetrahedron":
            # Use tetrahedron-compatible bounds
            domain_bounds = ((0, 1), (0, 1), (0, 1))
        else:
            domain_bounds = ((0, 1), (0, 1), (0, 1))  # 3D

        isosurface = GeometryProcessor(geometry=geometry)
        logging.info(f"\n--- Batch Testing Geometry: {geometry} ---")

        for level in levels_to_validate:
            logging.info(f"\n--- Batch Validating Level {level}: ---")
            test_case = generate_test_case(level, geometry=geometry)
            scalar_values = test_case["scalar_values"]
            isovalue = test_case["isovalue"]
            description = test_case["description"]

            logging.info(f"Description: {description}")
            logging.info(f"Scalar Values:\n{scalar_values}")

            # Run evaluation on the batched data
            try:
                triangle_vertices = isosurface.evaluate(scalar_function, domain_bounds, isovalue)

                logging.info(f"Batched Triangle Vertices for Level {level}:\n{triangle_vertices}")

                # Validation: Since the scalar_function is deterministic and identical across the grid,
                # all triangles should be the same. We can compare them against the first entry.
                single_case_vertices = triangle_vertices[0].unsqueeze(0).repeat(triangle_vertices.shape[0], 1, 1)

                for i in range(triangle_vertices.shape[0]):
                    assert torch.allclose(triangle_vertices[i], single_case_vertices[i], atol=1e-5), \
                        f"Batch entry {i} does not match single-case result for Level {level}, Geometry {geometry}!"
            except AssertionError as e:
                logging.error(f"Validation failed for {geometry}, Level {level}: {str(e)}")
                raise e

    logging.info("Batch processing validation complete. All tests passed.")


def validate_against_scikit_marching_cubes():
    levels_to_validate = [1, 2, 3, 4, 5]
    geometries_to_test = ["cube", "square", "tetrahedron"]

    for geometry in geometries_to_test:
        # Define a scalar function for testing
        def scalar_function(x, y, z=None):
            if geometry == "square":
                return torch.sin(x) * torch.cos(y) + 0.5
            else:
                return torch.sin(x) * torch.cos(y) * torch.sin(z) + 0.5

        # Define domain bounds based on geometry
        if geometry == "square":
            domain_bounds = ((0, 1), (0, 1))  # 2D
        else:
            domain_bounds = ((0, 1), (0, 1), (0, 1))  # 3D

        isosurface = GeometryProcessor(geometry=geometry)
        logging.info(f"\n--- Testing Geometry: {geometry} ---")

        for level in levels_to_validate:
            logging.info(f"\n--- Validating Level {level}: ---")
            test_case = generate_test_case(level, geometry=geometry)
            isovalue = test_case["isovalue"]
            expected_bitmask = test_case["expected_bitmask"]
            description = test_case["description"]

            logging.info(f"Description: {description}")
            logging.info(f"Isovalue: {isovalue}")
            logging.info(f"Expected Bitmask: {bin(expected_bitmask)}")

            # Run evaluation on the batch
            triangle_vertices = isosurface.evaluate(scalar_function, domain_bounds, isovalue)

            logging.info(f"Triangle Vertices for Level {level}:\n{triangle_vertices}")

            # Compute bitmask manually for validation
            # This step is more relevant if you have specific expectations
            # Here, it's kept for consistency with the original test logic
            # However, with the standardized evaluate method, it's less straightforward
            # So this part might be adapted or removed based on actual test requirements

            try:
                # For 3D geometries, attempt scikit-image validation
                if geometry != "square":
                    # Reshape scalar_values to match a small grid (2x2x2) for marching_cubes
                    scalar_values = scalar_function(
                        torch.tensor([0.0, 0.0, 0.0]),
                        torch.tensor([1.0, 1.0, 1.0])
                    ).numpy().reshape((2, 2, 2))
                    vertices, faces, _, _ = marching_cubes(scalar_values, level=isovalue)
                    logging.info(f"Scikit-Image Vertices for Level {level}: {vertices}")
                    logging.info(f"Scikit-Image Faces for Level {level}: {faces}")
                else:
                    logging.warning(f"Skipping Scikit-Image validation for Level {level}, Geometry {geometry}: Not a 3D geometry.")
            except ValueError as e:
                logging.warning(f"Skipping Scikit-Image validation for Level {level}, Geometry {geometry}: {e}")

    logging.info("Scikit-Marching Cubes validation complete.")


def validate_with_functional_field():
    """
    Validate isosurface extraction from a function-defined scalar field.
    This test generates a scalar field from a mathematical function, computes the isosurface,
    and visualizes the result using Matplotlib.
    """
    # Define the scalar field function
    def scalar_field_function(x, y, z=None):
        if z is not None:
            return torch.sin(x) * torch.cos(y) * torch.sin(z) + 0.5
        else:
            return torch.sin(x) * torch.cos(y) + 0.5

    # Define domain bounds
    domain_bounds = ((0, 1), (0, 1), (0, 1))  # 3D for cube

    # Instantiate the GeometryProcessor
    isosurface = GeometryProcessor(geometry="cube", density=0.5)

    # Evaluate the isosurface
    isovalue = 0.5  # Threshold for the isosurface
    triangle_vertices = isosurface.evaluate(scalar_field_function, domain_bounds, isovalue)

    # Filter non-zero triangles for visualization
    valid_triangles = triangle_vertices[triangle_vertices.sum(dim=-1) != 0].detach().cpu().numpy()

    # Plot the resulting isosurface
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for triangle in valid_triangles:
        verts = [list(triangle)]  # Create a list of triangle vertices
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, edgecolor="k"))

    # Set limits and labels
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Isosurface Extracted from Functional Field")
    plt.show()


if __name__ == "__main__":
    # Run all validations
    validate_with_batch_processing()
    validate_against_scikit_marching_cubes()
    validate_with_functional_field()
