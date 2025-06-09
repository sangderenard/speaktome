import torch
import torch
import math
from itertools import product
from typing import List, Tuple, Dict

def build_tetrahedron_lookup():
    """
    Build edge and triangle lookup tables for tetrahedra.

    Returns:
        dict: Lookup table for edge intersections and triangulations.
    """
    EDGE_VERTEX_PAIRS = [
        (0, 1), (1, 2), (2, 0),  # Base triangle edges
        (0, 3), (1, 3), (2, 3)   # Edges connecting the apex
    ]

    tetrahedron_table = {}

    for tet_index in range(16):
        edges = set()
        triangles = []

        for edge_idx, (v1, v2) in enumerate(EDGE_VERTEX_PAIRS):
            if (tet_index & (1 << v1)) != (tet_index & (1 << v2)):
                edges.add(edge_idx)

        edges = sorted(edges)

        if len(edges) > 6:
            raise ValueError(f"Invalid configuration for tetrahedron {tet_index}: too many edges.")

        # Generate triangles based on intersected edges
        if len(edges) == 3:
            triangles.append(edges)
        elif len(edges) == 4:
            triangles.append(edges[:3])
            triangles.append([edges[0], edges[2], edges[3]])
        elif len(edges) == 5:
            triangles.append(edges[:3])
            triangles.append([edges[0], edges[2], edges[3]])
            triangles.append([edges[0], edges[3], edges[4]])
        elif len(edges) == 6:
            triangles.append(edges[:3])
            triangles.append([edges[0], edges[2], edges[3]])
            triangles.append([edges[0], edges[3], edges[4]])
            triangles.append([edges[0], edges[4], edges[5]])

        tetrahedron_table[tet_index] = {"edges": edges, "triangles": triangles}

    return tetrahedron_table

def validate_tetrahedron_lookup(tetrahedron_table):
    """
    Validate the tetrahedron lookup table with random test cases.

    Args:
        tetrahedron_table (dict): The generated lookup table.

    Returns:
        bool: True if the table passes all validations.
    """
    EDGE_VERTEX_PAIRS = [
        (0, 1), (1, 2), (2, 0),
        (0, 3), (1, 3), (2, 3)
    ]

    for _ in range(10):
        scalars = torch.rand(4)
        isovalue = scalars.mean()

        tet_index = sum((1 << i) for i, scalar in enumerate(scalars) if scalar > isovalue)

        entry = tetrahedron_table.get(tet_index)
        if not entry:
            raise ValueError(f"Configuration {tet_index} not found in the table.")

        intersection_points = []
        for edge_idx in entry["edges"]:
            v1, v2 = EDGE_VERTEX_PAIRS[edge_idx]
            t = (isovalue - scalars[v1]) / (scalars[v2] - scalars[v1] + 1e-10)
            intersection = (1 - t) * v1 + t * v2
            intersection_points.append(intersection)

        for triangle in entry["triangles"]:
            if len(triangle) != 3:
                raise ValueError(f"Invalid triangle definition: {triangle}.")
            for edge in triangle:
                if edge not in entry["edges"]:
                    raise ValueError(f"Triangle edge {edge} not in intersected edges.")

    print("All validations passed!")
    return True

# Build and validate the lookup table
tetrahedron_table = build_tetrahedron_lookup()
validate_tetrahedron_lookup(tetrahedron_table)

class TetrahedralMarchingCubes:
    def __init__(self, isovalues: torch.Tensor, lookup_table: Dict[int, Dict], precision=1e-6):
        """
        Initialize the Tetrahedral Marching Cubes utility class.

        Args:
            isovalues (torch.Tensor): Tensor of scalar values at which to extract isosurfaces.
            lookup_table (dict): Precomputed lookup table for tetrahedral configurations.
            precision (float): Precision for vertex deduplication.
        """
        self.isovalues = isovalues  # Shape: (num_isosurfaces,)
        self.lookup_table = lookup_table
        self.precision = precision
        self.vertex_maps = [{} for _ in range(len(isovalues))]  # One map per isovalue

    def extract_shells(self, scalar_field: torch.Tensor, vertex_positions: torch.Tensor, tile_size: int = 8):
        """
        Perform Marching Cubes on a scalar field to extract multiple shell geometries.

        Args:
            scalar_field (torch.Tensor): Scalar field values at the grid vertices (Nx, Ny, Nz).
            vertex_positions (torch.Tensor): 3D positions of grid vertices (Nx, Ny, Nz, 8, 3).
            tile_size (int): Size of each tile to divide the domain.

        Returns:
            List[Dict]: List of dictionaries containing "vertices", "faces", and "normals" for each isovalue.
        """
        Nx, Ny, Nz = scalar_field.shape
        num_isosurfaces = len(self.isovalues)

        # Initialize outputs for each isovalue
        results = [{
            "vertices": [],
            "faces": [],
            "normals": []
        } for _ in range(num_isosurfaces)]

        # Determine the number of tiles in each dimension
        tiles_x = math.ceil(Nx / tile_size)
        tiles_y = math.ceil(Ny / tile_size)
        tiles_z = math.ceil(Nz / tile_size)

        # Iterate through all tiles
        for tile_x in range(tiles_x):
            for tile_y in range(tiles_y):
                for tile_z in range(tiles_z):
                    # Define the bounds of the current tile
                    x_start = tile_x * tile_size
                    y_start = tile_y * tile_size
                    z_start = tile_z * tile_size

                    x_end = min(x_start + tile_size + 1, Nx)
                    y_end = min(y_start + tile_size + 1, Ny)
                    z_end = min(z_start + tile_size + 1, Nz)

                    # Extract the subdomain
                    sub_field = scalar_field[x_start:x_end, y_start:y_end, z_start:z_end]
                    sub_positions = vertex_positions[x_start:x_end, y_start:y_end, z_start:z_end, :, :]

                    # Process the subdomain
                    for isovalue_idx, isovalue in enumerate(self.isovalues):
                        self._extract_subdomain(
                            isovalue,
                            isovalue_idx,
                            sub_field,
                            sub_positions,
                            results[isovalue_idx]
                        )

        # Convert lists to tensors
        for result in results:
            if result["vertices"]:
                result["vertices"] = torch.stack(result["vertices"], dim=0)
                result["faces"] = torch.tensor(result["faces"], dtype=torch.int32)
                result["normals"] = torch.stack(result["normals"], dim=0)
            else:
                result["vertices"] = torch.empty((0, 3))
                result["faces"] = torch.empty((0, 3), dtype=torch.int32)
                result["normals"] = torch.empty((0, 3))

        return results

    def _extract_subdomain(self, isovalue: float, isovalue_idx: int, sub_field: torch.Tensor,
                          sub_positions: torch.Tensor, result: Dict):
        """
        Extract isosurface from a subdomain for a specific isovalue.

        Args:
            isovalue (float): Scalar value at which to extract the isosurface.
            isovalue_idx (int): Index of the isovalue.
            sub_field (torch.Tensor): Scalar field of the subdomain.
            sub_positions (torch.Tensor): Vertex positions of the subdomain.
            result (dict): Dictionary to accumulate the results.
        """
        Nx, Ny, Nz = sub_field.shape

        # Iterate through all cubes in the subdomain
        for x in range(Nx - 1):
            for y in range(Ny - 1):
                for z in range(Nz - 1):
                    # Process each tetrahedron in the current cube
                    for tet_index in range(6):  # 6 tetrahedrons per cube
                        tet_scalars, tet_positions = self._get_tetrahedron(
                            sub_field, sub_positions, x, y, z, tet_index
                        )
                        self._process_tetrahedron(
                            isovalue, isovalue_idx, tet_scalars, tet_positions, result
                        )

    def _get_tetrahedron(self, sub_field: torch.Tensor, sub_positions: torch.Tensor,
                        x: int, y: int, z: int, tet_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve scalar values and positions for a single tetrahedron.

        Args:
            sub_field (torch.Tensor): Scalar field values in the subdomain (Nx, Ny, Nz).
            sub_positions (torch.Tensor): Vertex positions in the subdomain (Nx, Ny, Nz, 8, 3).
            x, y, z (int): Grid indices.
            tet_index (int): Tetrahedron index (0-5 within a cube).

        Returns:
            tuple: (scalar values, vertex positions) for the tetrahedron.
        """
        TETRAHEDRON_DECOMPOSITION = [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 3, 5, 6],
            [0, 3, 4, 7],
            [3, 4, 6, 7],
            [3, 5, 6, 7],
        ]
        tet_vertices = TETRAHEDRON_DECOMPOSITION[tet_index]


        tet_scalars = torch.tensor([sub_field[x + (v >> 0 & 1),
                                              y + (v >> 1 & 1),
                                              z + (v >> 2 & 1)] for v in tet_vertices])

        # Get the positions
        tet_positions = torch.stack([sub_positions[x + (v >> 0 & 1),
                                                   y + (v >> 1 & 1),
                                                   z + (v >> 2 & 1),
                                                   0] for v in tet_vertices])  # Adjusted indexing if necessary

        return tet_scalars, tet_positions

    def _process_tetrahedron(self, isovalue: float, isovalue_idx: int,
                             tet_scalars: torch.Tensor, tet_positions: torch.Tensor,
                             result: Dict):
        """
        Process a single tetrahedron to extract intersecting geometry.

        Args:
            isovalue (float): Scalar value at which to extract the isosurface.
            isovalue_idx (int): Index of the isovalue.
            tet_scalars (torch.Tensor): Scalar values at the tetrahedron vertices (4,).
            tet_positions (torch.Tensor): 3D positions of the tetrahedron vertices (4, 3).
            result (dict): Dictionary to accumulate the results.
        """
        tet_index = sum(
            (1 << i) for i, scalar in enumerate(tet_scalars) if scalar > isovalue
        )
        lookup_entry = self.lookup_table.get(tet_index)
        if not lookup_entry:
            return
        intersected_edges = lookup_entry["edges"]
        triangle_indices = lookup_entry["triangles"]
        if not intersected_edges:
            return

        EDGE_VERTEX_PAIRS = [
            (0, 1), (1, 2), (2, 0),  # Base triangle edges
            (0, 3), (1, 3), (2, 3)   # Edges connecting the apex
        ]

        edge_vertices = {}
        for edge_idx in intersected_edges:
            v1, v2 = EDGE_VERTEX_PAIRS[edge_idx]
            vertex = self._interpolate_edge(
                tet_positions[v1],
                tet_positions[v2],
                tet_scalars[v1],
                tet_scalars[v2],
                isovalue
            )
            vertex_idx = self._deduplicate_vertex(vertex, isovalue_idx, result)
            edge_vertices[edge_idx] = vertex_idx

        for triangle in triangle_indices:
            if len(triangle) != 3:
                continue
            v0, v1, v2 = edge_vertices[triangle[0]], edge_vertices[triangle[1]], edge_vertices[triangle[2]]

            # Calculate normal and enforce consistent orientation
            edge1 = result["vertices"][v1] - result["vertices"][v0]
            edge2 = result["vertices"][v2] - result["vertices"][v0]
            normal = torch.cross(edge1, edge2)
            norm = torch.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = torch.tensor([0.0, 0.0, 0.0])

            # Append the face with the correct winding order
            result["faces"].append([v0, v1, v2])

            # Append the normal
            result["normals"].append(normal)

    def _interpolate_edge(self, p1: torch.Tensor, p2: torch.Tensor, value1: float, value2: float,
                         isovalue: float) -> torch.Tensor:
        """
        Interpolate the position where the isosurface intersects an edge.

        Args:
            p1 (torch.Tensor): Position of the first vertex (3,).
            p2 (torch.Tensor): Position of the second vertex (3,).
            value1 (float): Scalar value at the first vertex.
            value2 (float): Scalar value at the second vertex.
            isovalue (float): The isovalue for the isosurface.

        Returns:
            torch.Tensor: Interpolated vertex position (3,).
        """
        if abs(value1 - value2) > 1e-6:
            t = (isovalue - value1) / (value2 - value1)
        else:
            t = 0.5
        return (1 - t) * p1 + t * p2

    def _deduplicate_vertex(self, vertex: torch.Tensor, isovalue_idx: int, result: Dict) -> int:
        """
        Deduplicate vertices to avoid redundancy.

        Args:
            vertex (torch.Tensor): Vertex position (3,).
            isovalue_idx (int): Index of the isovalue.
            result (dict): Dictionary to accumulate the results.

        Returns:
            int: Index of the deduplicated vertex.
        """
        rounded_vertex = tuple((vertex / self.precision).round().tolist())
        if rounded_vertex in self.vertex_maps[isovalue_idx]:
            return self.vertex_maps[isovalue_idx][rounded_vertex]
        idx = len(result["vertices"])
        result["vertices"].append(vertex)
        self.vertex_maps[isovalue_idx][rounded_vertex] = idx
        return idx

    def validate_meshes(self, results: List[Dict]):
        """
        Validate the extracted meshes for duplicate vertices, disjoint triangles, and incorrect face ordering.

        Args:
            results (List[Dict]): Extracted geometries for each isovalue.

        Returns:
            List[bool]: Validation results for each mesh.
        """
        validation_results = []
        for idx, result in enumerate(results):
            vertices = result["vertices"]
            faces = result["faces"]
            vertex_count = vertices.shape[0]
            face_count = faces.shape[0]
            adjacency = torch.zeros((vertex_count, vertex_count), dtype=torch.bool)
            for face in faces:
                adjacency[face[0], face[1]] = True
                adjacency[face[1], face[2]] = True
                adjacency[face[2], face[0]] = True
            connected_components = torch.any(adjacency, dim=1).sum().item()
            validation = connected_components == vertex_count
            print(f"Mesh {idx}: Vertex Count = {vertex_count}, Face Count = {face_count}, "
                  f"Connected Components = {connected_components}, Valid = {validation}")
            validation_results.append(validation)
        return validation_results

def export_to_obj_with_options(data: Dict, params: Dict, filename: str = "output.obj"):
    """
    Export mesh to an OBJ file with optional features.

    Args:
        data (dict): Contains "vertices", "faces", "normals", and optionally "colors", "uvs".
        params (dict): Configuration options.
            - "include_normals" (bool): Whether to include normals.
            - "invert_normals" (bool): Whether to invert normals.
            - "include_colors" (bool): Whether to include vertex colors.
            - "include_uvs" (bool): Whether to include UVs.
            - "material_name" (str): Name of the material to reference.
            - "generate_mtl" (bool): Whether to generate a material file.
        filename (str): Output OBJ file name. Defaults to "output.obj".
    """
    vertices = data.get("vertices")
    faces = data.get("faces")
    normals = data.get("normals", None)
    colors = data.get("colors", None)
    uvs = data.get("uvs", None)

    material_name = params.get("material_name", "default")
    include_normals = params.get("include_normals", False)
    invert_normals = params.get("invert_normals", False)
    include_colors = params.get("include_colors", False)
    include_uvs = params.get("include_uvs", False)
    generate_mtl = params.get("generate_mtl", False)

    # Normalize and invert normals if required
    if normals is not None and include_normals:
        if invert_normals:
            normals = -normals

    # Write OBJ file
    with open(filename, "w") as f:
        if generate_mtl:
            f.write(f"mtllib {material_name}.mtl\n")
        f.write(f"usemtl {material_name}\n")

        # Write vertices
        for i, v in enumerate(vertices):
            if include_colors and colors is not None:
                color = colors[i]
                f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()} {color[0].item()} {color[1].item()} {color[2].item()}\n")
            else:
                f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")

        # Write UV coordinates
        if include_uvs and uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0].item()} {uv[1].item()}\n")

        # Write normals
        if include_normals and normals is not None:
            for n in normals:
                f.write(f"vn {n[0].item()} {n[1].item()} {n[2].item()}\n")

        # Write faces
        for face in faces:
            if include_normals and normals is not None:
                f.write(f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n")
            elif include_uvs and uvs is not None:
                f.write(f"f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1}\n")
            else:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    # Generate MTL file if requested
    if generate_mtl:
        export_mtl(material_name, f"{filename.split('.')[0]}.mtl")

def export_mtl(material_name="default", filename="default.mtl"):
    """
    Export a basic material file for OBJ.

    Args:
        material_name (str): Name of the material.
        filename (str): Output MTL file name. Defaults to "default.mtl".
    """
    with open(filename, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 0.1 0.1 0.1\n")  # Ambient color
        f.write("Kd 0.8 0.8 0.8\n")  # Diffuse color
        f.write("Ks 0.5 0.5 0.5\n")  # Specular color
        f.write("Ns 100.0\n")        # Shininess
        f.write("d 1.0\n")           # Transparency
        f.write("illum 2\n")         # Lighting model

def generate_sphere_field(grid_size: int, radius: float, center: Tuple[float, float, float]) -> torch.Tensor:
    """
    Generate a scalar field representing a sphere.

    Args:
        grid_size (int): Size of the grid in each dimension.
        radius (float): Radius of the sphere.
        center (tuple): Center of the sphere (x, y, z).

    Returns:
        torch.Tensor: Scalar field with values defining the sphere.
    """
    x = torch.linspace(0, grid_size - 1, grid_size)
    y = torch.linspace(0, grid_size - 1, grid_size)
    z = torch.linspace(0, grid_size - 1, grid_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    scalar_field = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    return scalar_field - radius**2  # Isovalue at 0 defines the sphere

def generate_vertex_positions(grid_size: int) -> torch.Tensor:
    """
    Generate vertex positions for the scalar field grid.

    Args:
        grid_size (int): Size of the grid in each dimension.

    Returns:
        torch.Tensor: Vertex positions in 3D space (Nx, Ny, Nz, 8, 3).
    """
    vertex_positions = torch.zeros((grid_size, grid_size, grid_size, 8, 3), dtype=torch.float32)
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                vertex_positions[x, y, z, :, :] = torch.tensor([
                    [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
                    [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
                ], dtype=torch.float32)
    return vertex_positions

# Parameters
grid_size = 64  # Increased grid size for better resolution
radius = 25
center = (32, 32, 32)  # Centered within the grid
scalar_field = generate_sphere_field(grid_size, radius, center)
vertex_positions = generate_vertex_positions(grid_size)

# Define multiple isovalues (shells)
isovalues = torch.tensor([0.0, 50.0, 100.0])  # Example isovalues for multiple shells

# Initialize marching cubes with multiple isovalues
marching_cubes = TetrahedralMarchingCubes(isovalues, tetrahedron_table)

# Extract shells
result_shells = marching_cubes.extract_shells(scalar_field, vertex_positions, tile_size=16)

# Validate meshes
marching_cubes.validate_meshes(result_shells)

# Export each shell to separate OBJ files
for idx, shell in enumerate(result_shells):
    data = {
        "vertices": shell["vertices"],
        "faces": shell["faces"],
        "normals": shell["normals"],
        "colors": None,  # Add colors if available
        "uvs": None      # Add UVs if available
    }

    params = {
        "include_normals": True,
        "invert_normals": False,
        "include_colors": False,
        "include_uvs": False,
        "material_name": f"default_material_{idx}",
        "generate_mtl": False
    }

    export_to_obj_with_options(data, params, filename=f"sphere_shell_{idx}.obj")
