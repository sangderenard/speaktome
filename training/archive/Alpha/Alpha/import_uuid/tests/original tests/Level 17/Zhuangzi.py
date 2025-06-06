import sys
import math
import torch
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

#######################################################
# CONFIGURATION
#######################################################
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DT = 0.01
MAX_VELOCITY = 1.0   # "Relativistic" speed limit
RADIUS = 0.2         # Radius for collision checking
DOMAIN_BOUNDS = [(-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)]
# Boundary mode simplified to "reflect" for demonstration
BOUNDARY_MODE = "reflect"

# Mass field parameters
NUM_MASS_OBJECTS = 2   # Two spherical domains
MASS_OBJECT_RADII = [0.5, 0.7]
MASS_OBJECT_CENTERS = [[-1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]]
# Assign prime IDs to each mass object for detection
PRIME_IDS = [2, 3] # first object: prime 2, second object: prime 3

# Lambda chain for mass field piecewise definition
# For simplicity: Chain: increment -> double. 
# In practice, you would define a more complex chain.
def construct_lambda_chain(chain_str):
    operations = {
        "increment": lambda x: x + 1,
        "double": lambda x: x * 2,
    }
    funcs = chain_str.split("->")
    chain = [operations[f.strip()] for f in funcs]
    def combined(x):
        for fn in chain:
            x = fn(x)
        return x
    return combined

mass_field_chain = construct_lambda_chain("increment -> double")

#######################################################
# MASS FIELD FUNCTION USING PIECEWISE SPHERICAL DOMAINS
#######################################################
def mass_field_function(x, y, z):
    # For each mass object (sphere), check if inside radius.
    # If inside: multiply prime ID into a product. 
    # This product is used for binary decision logic (divisible checks).
    # After constructing the product, apply the lambda chain.

    product = torch.ones_like(x)
    for i, center in enumerate(MASS_OBJECT_CENTERS):
        cx, cy, cz = center
        dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        inside = dist_sq < MASS_OBJECT_RADII[i]**2
        product = torch.where(inside, product * PRIME_IDS[i], product)

    # Apply lambda chain to product
    product = product.to(torch.float32)
    result = mass_field_chain(product)
    return result

#######################################################
# GEOMETRY: Simple Tetrahedron
#######################################################
# A simple geometry: Tetrahedron
# Vertices:
tetra_vertices = torch.tensor([
    [1, 1, 1],
    [-1,-1, 1],
    [-1, 1,-1],
    [1, -1,-1]
], dtype=torch.float32)
# Centering:
centroid = tetra_vertices.mean(dim=0)
tetra_vertices -= centroid

# A single shuttle represented by this tetrahedron:
class Shuttle:
    def __init__(self, position=[0,0,0], velocity=[0,0,0]):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.alive = True

    def integrate(self, dt=DT):
        speed = self.velocity.norm()
        if speed > MAX_VELOCITY:
            self.velocity = self.velocity / speed * MAX_VELOCITY
        self.position += self.velocity * dt

#######################################################
# BOUNDARY CONDITIONS
#######################################################
def apply_boundaries(shuttle):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = DOMAIN_BOUNDS
    x, y, z = shuttle.position
    if x < xmin or x > xmax:
        # reflect
        shuttle.position[0] = max(min(x, xmax), xmin)
        shuttle.velocity[0] = -shuttle.velocity[0]
    if y < ymin or y > ymax:
        shuttle.position[1] = max(min(y, ymax), ymin)
        shuttle.velocity[1] = -shuttle.velocity[1]
    if z < zmin or z > zmax:
        shuttle.position[2] = max(min(z, zmax), zmin)
        shuttle.velocity[2] = -shuttle.velocity[2]
    return shuttle

#######################################################
# COLLISION & OVERLAP RESOLUTION
#######################################################
# If two mass objects overlap in a vertex region,
# we can detect by checking product divisibility by a prime.
# If product divisible by 2*3=6 means both objects present.
# If product divisible only by 2 means only first object inside, etc.
def detect_collisions(field_values):
    # field_values: result of mass_field_function for each vertex
    # If any vertex's field value indicates presence of multiple objects,
    # we apply a reflection force.
    # For demonstration: if divisible by 6 means overlap of both mass objects.
    overlap_mask = (field_values % (PRIME_IDS[0]*PRIME_IDS[1]) == 0) & (field_values > 1)
    return overlap_mask

def compute_reflection_force(shuttle_pos, vertices, overlap_mask):
    # If overlap_mask is True for any vertex, compute a reflection force
    # For simplicity: sum all vertex positions that overlap and reflect outward
    active_positions = vertices[overlap_mask]
    if active_positions.shape[0] == 0:
        return torch.zeros(3)

    centroid_overlap = active_positions.mean(dim=0)
    dir = (shuttle_pos - centroid_overlap)
    if dir.norm() > 1e-6:
        dir = dir / dir.norm()
    force = dir * 0.1  # arbitrary reflection strength
    return force

#######################################################
# OPENGL COMPUTE SHADER FOR BATCH PROCESS
#######################################################
# We'll define a simple shader that just returns the field_values for demonstration
simple_shader_code = """
#version 420 core
layout(local_size_x = 1) in;

layout(std430, binding=0) buffer Positions {
    vec4 shuttle_pos;
};
layout(std430, binding=1) buffer Vertices {
    vec4 vertices[]; 
};
layout(std430, binding=2) buffer FieldValues {
    float field_values[]; 
};

uniform uint num_vertices;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_vertices) return;

    // Just a placeholder: field_values already computed on CPU for now.
    // In a real scenario, you would run the mass_field_function on GPU or store precomputed values.
    // For demonstration, do nothing.
    field_values[idx] = field_values[idx]; 
}
"""

def compile_compute_shader(code):
    return compileProgram(compileShader(code, GL_COMPUTE_SHADER))

#######################################################
# MAIN SIMULATION LOOP
#######################################################
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF|OPENGL)
    pygame.display.set_caption("Mass Field Bitmask Vertex Demo")
    glClearColor(0,0,0,1)

    shuttle = Shuttle([0,0,0],[0.05,0.02,0.0])

    # Precompute vertex positions in world space = shuttle.position + tetra_vertices
    def get_world_vertices():
        return (tetra_vertices + shuttle.position).clone()

    # Just a single shuttle and geometry
    # We'll run CPU-side mass field evaluation each frame.
    # GPU usage is minimal here due to complexity, but shown as a placeholder.

    shader = compile_compute_shader(simple_shader_code)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False

        # Compute mass field for each vertex
        world_verts = get_world_vertices()
        x, y, z = world_verts[:,0], world_verts[:,1], world_verts[:,2]
        field_values = mass_field_function(x,y,z)

        # Detect overlap using prime logic
        overlap_mask = detect_collisions(field_values)

        # If overlap occurs, compute reflection force
        reflection_force = compute_reflection_force(shuttle.position, world_verts, overlap_mask)
        
        # Apply reflection force as acceleration step
        # mass=1.0 for simplicity
        shuttle.velocity += reflection_force * DT
        shuttle.integrate(DT)
        shuttle = apply_boundaries(shuttle)

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0,0,-5)

        # Draw shuttle vertices:
        glColor3f(0,1,0)
        glBegin(GL_POINTS)
        for v in get_world_vertices():
            glVertex3f(v[0].item(), v[1].item(), v[2].item())
        glEnd()

        # If overlap, draw a red sphere at centroid overlap
        if overlap_mask.any():
            overlap_positions = world_verts[overlap_mask]
            ov_cent = overlap_positions.mean(dim=0)
            glColor3f(1,0,0)
            glBegin(GL_LINES)
            glVertex3f(ov_cent[0], ov_cent[1], ov_cent[2])
            glVertex3f(ov_cent[0], ov_cent[1], ov_cent[2]+0.1)
            glEnd()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

