import torch
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from isosurface3 import YoungManAlgorithm  # Ensure correct module path for your setup

# Define the scalar field for a sphere with sinusoidal ripples
def sinusoidal_sphere(x, y, z, frequency=5.0, amplitude=0.2, radius=1.0):
    """Scalar field for a sphere with sinusoidal ripples."""
    base_sphere = x**2 + y**2 + z**2 - radius**2
    ripples = amplitude * torch.sin(frequency * (torch.sqrt(x**2 + y**2 + z**2)))
    return base_sphere + ripples

# Initialize the Young Man Algorithm with icosahedral geometry
geometry = "icosahedron"
density = 10
domain_bounds = [
    (-2.0, 2.0),  # x range
    (-2.0, 2.0),  # y range
    (-2.0, 2.0),  # z range
]
isovalue = 0.0  # Define the isosurface level

# Create the Young Man Algorithm instance
processor = YoungManAlgorithm(geometry=geometry, density=density)

# Evaluate the scalar field using the algorithm
evaluation_result = processor.evaluate(
    scalar_function=lambda x, y, z: sinusoidal_sphere(x, y, z),
    domain_bounds=domain_bounds,
    isovalue=isovalue,
    gradient_normals=True  # Compute gradient-based normals
)

# Extract vertices for Delaunay triangulation
vertices = evaluation_result["vertices"].detach().cpu().numpy()

# Step 1: Apply Delaunay triangulation
tri = Delaunay(vertices)

# Step 2: Visualize the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Add the vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='blue', s=10, alpha=0.6, label='Vertices')

# Add the triangular faces
for simplex in tri.simplices:
    face = vertices[simplex]
    poly = Poly3DCollection([face], alpha=0.3, edgecolor='k')
    ax.add_collection3d(poly)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
