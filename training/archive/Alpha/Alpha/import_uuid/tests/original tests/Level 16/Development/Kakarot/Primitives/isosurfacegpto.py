import torch
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logging.basicConfig(level=logging.INFO, format="%(message)s")


class GeometryProcessor:
    def __init__(self, geometry="cube", density=1.0, device=None):
        """
        Initialize the GeometryProcessor with a given geometry and density.

        Args:
            geometry (str): The type of geometry ('square', 'cube', 'tetrahedron', 'octahedron', 'icosahedron').
            density (float): Scaling factor for the density of the tiling.
            device (str or torch.device, optional): Device to perform computations on.
        """
        assert geometry in ["square", "cube", "tetrahedron", "octahedron", "icosahedron"], \
            "Unsupported geometry. Choose 'square', 'cube', 'tetrahedron', 'octahedron', or 'icosahedron'."
        self.geometry = geometry
        self.density = density
        self.device = device if device else torch.device('cpu')

        # Define fundamental offsets and geometry-specific parameters
        self.vertex_offsets = self._define_offsets().to(self.device)
        self.vertex_count, self.edge_pairs = self._configure_geometry()

        # Precompute the triangle map for isosurface extraction
        self.triangle_map, self.triangle_mask = self._build_triangle_map()

    def _define_offsets(self):
        """Define vertex offsets for the chosen geometry and scale by density."""
        if self.geometry == "cube":
            return torch.tensor([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=torch.float32) * self.density
        elif self.geometry == "tetrahedron":
            return torch.tensor([
                [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
            ], dtype=torch.float32) * self.density
        elif self.geometry == "square":
            return torch.tensor([
                [0, 0], [1, 0], [1, 1], [0, 1],
            ], dtype=torch.float32) * self.density
        elif self.geometry == "octahedron":
            return torch.tensor([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ], dtype=torch.float32) * self.density
        elif self.geometry == "icosahedron":
            phi = (1 + 5 ** 0.5) / 2  # Golden ratio
            return torch.tensor([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ], dtype=torch.float32) * self.density


    def _configure_geometry(self):
        """Configure vertex and edge definitions for the chosen geometry."""
        if self.geometry == "square":
            vertex_count = 4
            edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif self.geometry == "cube":
            vertex_count = 8
            edge_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
        elif self.geometry == "tetrahedron":
            vertex_count = 4
            edge_pairs = [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ]
        elif self.geometry == "octahedron":
            vertex_count = 6
            edge_pairs = [
                (0, 2), (0, 3), (0, 4), (0, 5),
                (1, 2), (1, 3), (1, 4), (1, 5),
                (2, 4), (2, 5), (3, 4), (3, 5)
            ]
        elif self.geometry == "icosahedron":
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
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        return vertex_count, edge_pairs

    def _build_triangle_map(self):
        """
        Precompute the triangle map using a triangular fan method for all geometries.
        Handles degenerate cases, creates a center-mass-based triangulation, and maps
        simplices back to the original vertex indices.
        """
        max_triangles = self.vertex_count  # Maximum triangles for full connectivity
        num_bitmasks = 2 ** self.vertex_count
        # Initialize triangle_map with -1 and triangle_mask with False
        triangle_map = torch.full((num_bitmasks, max_triangles, 3), -1, dtype=torch.int32, device=self.device)
        triangle_mask = torch.zeros((num_bitmasks, max_triangles), dtype=torch.bool, device=self.device)
        vertices = self.vertex_offsets.clone().detach().cpu()  # Ensure vertices are on CPU for processing

        for bitmask in range(num_bitmasks):
            # Extract active vertices based on the bitmask
            active_vertex_mask = ((bitmask >> torch.arange(self.vertex_count, device=self.device)) & 1).bool()
            active_vertices = vertices[active_vertex_mask.cpu()]  # Shape: (num_active, D)

            logging.debug(f"Processing bitmask {bitmask:0{self.vertex_count}b}")
            logging.debug(f"Active vertex mask: {active_vertex_mask.tolist()}")
            logging.debug(f"Active vertices (shape {active_vertices.shape}): {active_vertices}")

            # Skip configurations with fewer than 3 vertices
            if active_vertices.shape[0] < 3:
                logging.debug(f"Skipping bitmask {bitmask} due to insufficient vertices (< 3).")
                continue

            try:
                # Calculate the centroid of the active vertices
                centroid = active_vertices.mean(dim=0)
                logging.debug(f"Centroid of active vertices: {centroid}")

                # Append centroid to active vertices
                original_vertex_count = active_vertices.shape[0]
                active_vertices = torch.cat([active_vertices, centroid.unsqueeze(0)], dim=0)  # Centroid is last index

                # Generate a fan of triangles with the centroid
                for i in range(original_vertex_count):
                    v1_idx = i
                    v2_idx = (i + 1) % original_vertex_count  # Wrap around
                    centroid_idx = original_vertex_count  # Centroid is the last index
                    triangle_map[bitmask, i] = torch.tensor([centroid_idx, v1_idx, v2_idx], dtype=torch.int32)
                    triangle_mask[bitmask, i] = True
                    logging.debug(f"Added triangle ({centroid_idx}, {v1_idx}, {v2_idx}) for bitmask {bitmask}.")

            except Exception as e:
                logging.warning(f"Skipping bitmask {bitmask} due to triangulation error: {e}")
                continue

        return triangle_map, triangle_mask

    def evaluate(self, scalar_function, domain_bounds, isovalue=1.0):
        """
        Evaluate the scalar function and extract isosurface triangles using the precomputed triangle map.
        Returns global vertices and triangles.

        Args:
            scalar_function (callable): Function to evaluate on the grid. Should accept coordinates as inputs.
            domain_bounds (list of tuples): Bounds for each dimension, e.g., [(-1,1), (-1,1), (-1,1)].
            isovalue (float): The isovalue to extract the isosurface.

        Returns:
            global_vertices (torch.Tensor): Tensor of unique vertex coordinates.
            global_triangles (torch.Tensor): Tensor of triangles with indices into global_vertices.
        """
        # Generate the grid of points in the domain
        grid_points = self._generate_grid(domain_bounds)  # Shape: (N, D)
        tiled_vertices = self.tile_grid(grid_points)  # Shape: (M, V, D)

        # Flatten tiled_vertices to shape (M*V, D) for scalar function evaluation
        M, V, D = tiled_vertices.shape
        flattened_vertices = tiled_vertices.view(-1, D)  # Shape: (M*V, D)

        # Separate coordinates
        if self.geometry == "square":
            # For 2D geometries
            x = flattened_vertices[:, 0]
            y = flattened_vertices[:, 1]
            vertex_values = scalar_function(x, y)  # Shape: (M*V,)
        else:
            # For 3D geometries
            x = flattened_vertices[:, 0]
            y = flattened_vertices[:, 1]
            z = flattened_vertices[:, 2]
            vertex_values = scalar_function(x, y, z)  # Shape: (M*V,)

        # Reshape vertex_values to (M, V)
        scalar_field = vertex_values.view(M, V)  # Shape: (M, V)

        # Compute bitmasks based on scalar values
        bitmasks = self._compute_bitmasks(scalar_field, isovalue)  # Shape: (M,)

        # Use the triangle map to extract the triangles for each cell
        triangles = self.triangle_map[bitmasks]  # Shape: (M, T, 3)

        # Filter valid triangles using the mask
        valid_triangles = self.triangle_mask[bitmasks]  # Shape: (M, T)
        triangles = triangles[valid_triangles]  # Shape: (Total_valid_triangles, 3)

        # Convert tiled vertices to global vertices and remap triangle indices
        global_vertices, global_triangles = self.convert_to_global_vertices(tiled_vertices, triangles)

        return global_vertices, global_triangles

    def convert_to_global_vertices(self, tiled_vertices, triangles):
        """
        Convert local tiled vertices and triangles to global unique vertices and updated triangle indices.

        Args:
            tiled_vertices (torch.Tensor): Tiled vertices of shape (M, V, D).
            triangles (torch.Tensor): Triangles with local indices of shape (N, 3).

        Returns:
            global_vertices (torch.Tensor): Tensor of unique vertex coordinates.
            global_triangles (torch.Tensor): Tensor of triangles with indices into global_vertices.
        """
        logging.info("Converting tiled vertices and triangles to global vertices and triangles...")

        # Reshape tiled_vertices to (M*V, D)
        all_vertices = tiled_vertices.view(-1, tiled_vertices.shape[-1])  # Shape: (M*V, D)

        # Round vertices to handle floating point precision issues
        precision = 6  # Number of decimal places to round to
        multiplier = 10 ** precision
        rounded_vertices = torch.round(all_vertices * multiplier) / multiplier  # Shape: (M*V, D)

        # Find unique vertices and get inverse indices
        unique_vertices, inverse_indices = torch.unique(rounded_vertices, dim=0, return_inverse=True)

        logging.info(f"Total vertices before deduplication: {all_vertices.shape[0]}")
        logging.info(f"Unique vertices after deduplication: {unique_vertices.shape[0]}")

        # Map triangles to global indices
        # triangles shape: (N, 3)
        # To map, we first flatten triangles to (N*3,) then use inverse_indices
        flat_triangles = triangles.view(-1)  # Shape: (N*3,)
        global_triangles = inverse_indices[flat_triangles].view(-1, 3)  # Shape: (N, 3)

        logging.info(f"Total triangles before filtering: {triangles.shape[0]}")
        logging.info(f"Total triangles after mapping to global indices: {global_triangles.shape[0]}")

        return unique_vertices, global_triangles

    def _get_intersected_edges(self, bitmask):
        """Identify edges that are intersected based on the bitmask."""
        intersected_edges = []
        for edge_idx, (v1, v2) in enumerate(self.edge_pairs):
            if (bitmask & (1 << v1)) > 0 != (bitmask & (1 << v2)) > 0:
                intersected_edges.append(edge_idx)
        return intersected_edges

    def _generate_grid(self, domain_bounds):
        """Generate grid points based on the domain bounds."""
        # domain_bounds is a list of tuples: [(min1, max1), (min2, max2), ...]
        ranges = [torch.arange(start, end + self.density, step=self.density, device=self.device)
                  for start, end in domain_bounds]
        mesh = torch.meshgrid(*ranges, indexing="ij")  # Returns a list of tensors
        grid = torch.stack([m.flatten() for m in mesh], dim=-1)  # Shape: (N, D)
        return grid

    def _compute_bitmasks(self, scalar_field, isovalue):
        """Compute bitmasks based on scalar field values."""
        # scalar_field: (M, V)
        bitmask = (scalar_field > isovalue).int()  # Shape: (M, V)
        powers_of_two = (1 << torch.arange(self.vertex_count, device=self.device)).int()  # Shape: (V,)
        bitmasks = (bitmask * powers_of_two).sum(dim=1)  # Shape: (M,)
        return bitmasks

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
        ranges = [torch.arange(min_coords[i], max_coords[i] + self.density, step=self.density, device=self.device)
                  for i in range(D)]
        mesh = torch.meshgrid(*ranges, indexing="ij")  # Returns a list of tensors
        base_points = torch.stack([m.flatten() for m in mesh], dim=-1)  # Shape: (M, D)

        # Apply offsets to generate tiles
        # base_points: (M, D)
        # vertex_offsets: (V, D)
        # tiled_vertices: (M, V, D) = base_points.unsqueeze(1) + vertex_offsets.unsqueeze(0)
        tiled_vertices = base_points.unsqueeze(1) + self.vertex_offsets.unsqueeze(0)  # Shape: (M, V, D)

        logging.debug(f"Number of tiles (M): {base_points.shape[0]}")
        logging.debug(f"Tiled vertices shape: {tiled_vertices.shape}")

        return tiled_vertices


# Define scalar function outside the class
def scalar_sphere(x, y, z, r=1.0):
    """Example scalar function for a sphere."""
    return x**2 + y**2 + z**2 - r**2


def visualize_isosurface(vertices, triangles):
    """
    Visualize the isosurface made up of discovered triangles.
    Args:
        vertices (numpy.ndarray): Coordinates of vertices.
        triangles (numpy.ndarray): Extracted triangles (indices into vertices).
    """
    logging.info("Visualizing isosurface from extracted triangles...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract triangle vertices
    triangle_vertices = vertices[triangles]

    # Create a 3D polygon collection
    collection = Poly3DCollection(triangle_vertices, alpha=0.6, edgecolor="k")
    ax.add_collection3d(collection)

    # Auto-scale to fit the geometry
    x, y, z = vertices.T
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    ax.set_title("Extracted Isosurface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


if __name__ == "__main__":
    # Set logging to debug level for detailed output (optional)
    logging.getLogger().setLevel(logging.INFO)

    # Example setup for a cube geometry
    geometry = "cube"  # Example: "cube", "tetrahedron", "octahedron", etc.
    density = 0.5
    processor = GeometryProcessor(geometry=geometry, density=density, device=torch.device('cpu'))

    # Define domain bounds for the grid
    domain_bounds = [
        (-1.5, 1.5),  # x range
        (-1.5, 1.5),  # y range
        (-1.5, 1.5),  # z range
    ]

    # Extract isosurface triangles using the sphere scalar function
    logging.info(f"Evaluating isosurface for geometry: {geometry}...")
    global_vertices, global_triangles = processor.evaluate(
        lambda x, y, z: scalar_sphere(x, y, z, r=1.0),
        domain_bounds,
        isovalue=0.0
    )

    # Output results
    logging.info(f"Extracted {global_triangles.shape[0]} triangles from the isosurface.")
    logging.info(f"Number of unique global vertices: {global_vertices.shape[0]}")
    logging.info(f"Number of triangles: {global_triangles.shape[0]}")

    # Convert to NumPy for visualization
    global_vertices_np = global_vertices.cpu().numpy()
    global_triangles_np = global_triangles.cpu().numpy()

    # Visualize the extracted isosurface
    visualize_isosurface(global_vertices_np, global_triangles_np)
