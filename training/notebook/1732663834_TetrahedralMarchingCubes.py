import torch
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Define edges for tetrahedral decomposition
EDGE_VERTEX_PAIRS = [
    (0, 1), (1, 2), (2, 0),  # Base edges
    (0, 3), (1, 3), (2, 3)   # Edges connecting apex
]

def build_tetrahedron_lookup():
    """
    Build a precise edge and triangle lookup table for tetrahedra.

    Returns:
        dict: Lookup table mapping tetrahedron configurations to intersected edges and triangles.
    """
    logging.info("Building tetrahedron lookup table.")
    tetrahedron_table = {}

    # Iterate through all 2^4 configurations of vertex states
    for tet_index in range(16):
        edges = []
        triangles = []

        # Check which edges are intersected
        for edge_idx, (v1, v2) in enumerate(EDGE_VERTEX_PAIRS):
            # Edge is intersected if vertices have different signs
            if (tet_index & (1 << v1)) != (tet_index & (1 << v2)):
                edges.append(edge_idx)

        # Triangulate intersected edges into valid geometry
        if len(edges) == 3:
            triangles.append(edges)  # Single triangle
        elif len(edges) == 4:
            # Create two triangles from the intersected edges
            triangles.append([edges[0], edges[1], edges[2]])
            triangles.append([edges[0], edges[2], edges[3]])
        elif len(edges) == 5:
            # Divide the 5 edges into three triangles
            triangles.append([edges[0], edges[1], edges[2]])
            triangles.append([edges[0], edges[2], edges[3]])
            triangles.append([edges[0], edges[3], edges[4]])
        elif len(edges) == 6:
            # Divide the 6 edges into four triangles
            triangles.append([edges[0], edges[1], edges[2]])
            triangles.append([edges[2], edges[3], edges[0]])
            triangles.append([edges[3], edges[4], edges[0]])
            triangles.append([edges[4], edges[5], edges[0]])

        # Validate that the configuration is consistent
        if len(edges) > 6:
            raise ValueError(f"Tetrahedron {tet_index} has an invalid edge configuration: {edges}")

        # Store the result in the lookup table
        tetrahedron_table[tet_index] = {"edges": edges, "triangles": triangles}
        logging.debug(f"Tetrahedron {tet_index}: {tetrahedron_table[tet_index]}")

    logging.info("Finished building tetrahedron lookup table.")
    return tetrahedron_table


def interpolate_edge(p1, p2, value1, value2, isovalue):
    """
    Compute the precise intersection point along an edge.

    Args:
        p1, p2 (torch.Tensor): Positions of the edge's endpoints (3,).
        value1, value2 (float): Scalar values at the edge's endpoints.
        isovalue (float): The isosurface value.

    Returns:
        torch.Tensor: Intersection point (3,).
    """
    if value1 == value2:
        raise ValueError("Invalid interpolation: scalar values at both endpoints are identical.")

    t = (isovalue - value1) / (value2 - value1)
    if not (0 <= t <= 1):
        raise ValueError(f"Interpolation factor t={t} is out of bounds.")
    return (1 - t) * p1 + t * p2

class TetrahedralMarchingCubes:
    def __init__(self, isovalue, lookup_table):
        """
        Initialize the Tetrahedral Marching Cubes algorithm.

        Args:
            isovalue (float): The scalar value defining the isosurface.
            lookup_table (dict): Precomputed tetrahedron configurations.
        """
        self.isovalue = isovalue
        self.lookup_table = lookup_table

    def extract_shell(self, scalar_field, vertex_positions, tetrahedra):
        """
        Extract the isosurface as vertices and faces.

        Args:
            scalar_field (torch.Tensor): Scalar values at vertices (N,).
            vertex_positions (torch.Tensor): Positions of vertices in 3D space (N, 3).
            tetrahedra (torch.Tensor): Indices of tetrahedron vertices (M, 4).

        Returns:
            dict: Contains 'vertices' and 'faces' of the extracted isosurface.
        """
        vertices = []
        faces = []

        # Iterate over all tetrahedra
        for tet in tetrahedra:
            # Retrieve scalar values and positions for this tetrahedron
            tet_scalars = scalar_field[tet]
            tet_positions = vertex_positions[tet]

            # Determine tetrahedron configuration
            config = sum((1 << i) for i, value in enumerate(tet_scalars) if value > self.isovalue)
            lookup = self.lookup_table.get(config)
            if not lookup or not lookup["edges"]:
                continue  # Skip if no intersected edges

            # Compute intersection points for edges
            edge_to_vertex = {}
            for edge_idx in lookup["edges"]:
                v1, v2 = EDGE_VERTEX_PAIRS[edge_idx]
                edge_to_vertex[edge_idx] = len(vertices)  # Assign new vertex index
                intersection = interpolate_edge(
                    tet_positions[v1], tet_positions[v2],
                    tet_scalars[v1], tet_scalars[v2],
                    self.isovalue
                )
                vertices.append(intersection)

            # Generate faces using the triangulation
            for triangle in lookup["triangles"]:
                faces.append([edge_to_vertex[edge] for edge in triangle])

        return {
            "vertices": torch.stack(vertices) if vertices else torch.empty((0, 3)),
            "faces": torch.tensor(faces, dtype=torch.int64) if faces else torch.empty((0, 3), dtype=torch.int64)
        }

# Example usage
def generate_test_data():
    """
    Generate scalar field, vertex positions, and tetrahedron definitions for testing.

    Returns:
        tuple: (scalar_field, vertex_positions, tetrahedra)
    """
    scalar_field = torch.tensor([0.0, 0.6, 0.4, 0.8, 0.2, 0.9, 0.3, 0.7])
    vertex_positions = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=torch.float32)
    tetrahedra = torch.tensor([
        [0, 1, 3, 4], [1, 2, 3, 6], [3, 4, 5, 6]
    ])
    return scalar_field, vertex_positions, tetrahedra

def main():
    scalar_field, vertex_positions, tetrahedra = generate_test_data()
    lookup_table = build_tetrahedron_lookup()
    marching_cubes = TetrahedralMarchingCubes(isovalue=0.5, lookup_table=lookup_table)

    result = marching_cubes.extract_shell(scalar_field, vertex_positions, tetrahedra)
    print("Vertices:")
    print(result["vertices"])
    print("Faces:")
    print(result["faces"])

if __name__ == "__main__":
    main()
