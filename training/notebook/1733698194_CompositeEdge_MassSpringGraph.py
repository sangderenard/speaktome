import sys
import torch
import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44, vector3

############################
# CompositeEdge Class
############################

class CompositeEdge:
    def __init__(self, i, j, rest_length, node_positions, device='cpu'):
        self.i = i
        self.j = j
        self.device = device

        # Physical properties
        self.rest_length = rest_length
        self.spring_constant = 500.0
        self.spring_slices = 10
        self.gas_volume_base = rest_length * 0.001
        self.gas_temperature = 300.0
        self.gas_moles = 0.01
        self.R = 8.314
        self.damper_rebound = 50.0
        self.damper_compression = 100.0
        self.rubber_threshold = rest_length * 0.7
        self.rubber_stiffness = 5000.0
        self.heat_capacity = 50.0

        self.damage_metric = 0.0

    def compute_forces(self, positions, velocities, dt=0.01):
        pi, pj = positions[self.i], positions[self.j]
        vi, vj = velocities[self.i], velocities[self.j]
        delta = pj - pi
        dist = delta.norm()
        direction = delta / (dist + 1e-6)
        extension = dist - self.rest_length

        # Spring force
        spring_force = torch.zeros(3, device=self.device)
        slice_length = extension / self.spring_slices
        for n in range(1, self.spring_slices + 1):
            spring_force += -self.spring_constant * slice_length * direction

        # Gas force
        current_volume = self.gas_volume_base * (dist / self.rest_length)
        pressure = (self.gas_moles * self.R * self.gas_temperature) / current_volume
        gas_force = -pressure * 0.0001 * direction

        # Damper force
        rel_vel = (vj - vi).dot(direction)
        damping = self.damper_rebound if rel_vel > 0 else self.damper_compression
        damper_force = -damping * rel_vel * direction

        # Rubber stopper force
        rubber_force = torch.zeros(3, device=self.device)
        if dist < self.rubber_threshold:
            penetration = self.rubber_threshold - dist
            rubber_force = penetration * self.rubber_stiffness * direction

        total_force = spring_force + gas_force + damper_force + rubber_force

        # Heat & damage
        energy_dissipated = damper_force.norm() * abs(rel_vel) * 0.01 + rubber_force.norm() * 0.01
        dT = energy_dissipated / self.heat_capacity
        self.gas_temperature += dT
        self.damage_metric += energy_dissipated * 0.1

        return -total_force, total_force

    def apply_membrane_collision(self, correction_energy):
        dT = correction_energy / self.heat_capacity
        self.gas_temperature += dT
        self.damage_metric += correction_energy * 0.1

############################
# MassSpringGraph Class
############################

class MassSpringGraph:
    def __init__(self, positions, edges, masses, device='cpu'):
        self.positions = positions.to(device)
        self.velocities = torch.zeros_like(self.positions)
        self.masses = masses.to(device)
        self.edges = []
        self.device = device

        for (i, j) in edges:
            rest_length = (positions[i] - positions[j]).norm().item()
            self.edges.append(CompositeEdge(i, j, rest_length, positions, device))

        self.gravity = torch.rand(3, device=device)
        self.membrane_z = 0.0

    def step(self, dt):
        forces = torch.zeros_like(self.positions)
        for edge in self.edges:
            fi, fj = edge.compute_forces(self.positions, self.velocities, dt)
            forces[edge.i] += fi
            forces[edge.j] += fj

        # Add gravity
        forces += self.masses.unsqueeze(1) * self.gravity

        # Update velocities and positions
        self.velocities += (forces / self.masses.unsqueeze(1)) * dt
        self.positions += self.velocities * dt

        # Handle membrane collisions
        for i in range(self.positions.shape[0]):
            if self.positions[i, 2] < self.membrane_z:
                penetration = self.membrane_z - self.positions[i, 2]
                correction_energy = 0.5 * 10000.0 * penetration ** 2
                self.positions[i, 2] = self.membrane_z
                self.velocities[i, 2] = 0.0
                for edge in self.edges:
                    if edge.i == i or edge.j == i:
                        edge.apply_membrane_collision(correction_energy)

        # Recenter positions around the center of mass
        center_of_mass = self.positions.mean(dim=0)
        self.positions -= center_of_mass

############################
# Modern OpenGL Setup
############################

vertex_shader_src = """
#version 420 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
uniform mat4 MVP;
out vec3 frag_color;
void main() {
    gl_Position = MVP * vec4(in_position, 1.0);
    frag_color = in_color;
}
"""

fragment_shader_src = """
#version 420 core
in vec3 frag_color;
out vec4 frag_output;
void main() {
    frag_output = vec4(frag_color, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check compilation errors
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode('utf-8'))
    return shader

def create_program(vertex_src, fragment_src):
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    # Check linking errors
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode('utf-8'))
    # Shaders can be detached and deleted after linking
    glDetachShader(prog, vs)
    glDetachShader(prog, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

############################
# Rendering Functions
############################

def init_opengl():
    # Request OpenGL 4.2 Core Profile if possible:
    # NOTE: Depending on your system, you may need additional setup.
    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    glViewport(0, 0, 800, 600)
    glEnable(GL_DEPTH_TEST)

    # Check the version
    version = glGetString(GL_VERSION)
    print("OpenGL version:", version.decode('utf-8'))

def update_buffers(system):
    # Extract line vertices and colors
    line_vertices = []
    line_colors = []

    for edge in system.edges:
        pi = system.positions[edge.i].cpu().numpy()
        pj = system.positions[edge.j].cpu().numpy()

        # Compute compression and heat for dynamic coloring
        current_length = np.linalg.norm(pj - pi)
        compression = abs((current_length - edge.rest_length) / edge.rest_length)
        heat = edge.gas_temperature / 1000.0  # Scale heat (arbitrary factor for visual clarity)
        #print(compression)
        # Map compression to red and heat to blue
        color = [min(1.0, compression*10), 0.0, min(1.0, heat.cpu())]

        line_vertices.extend(pi)
        line_colors.extend(color)
        line_vertices.extend(pj)
        line_colors.extend(color)

    line_vertices = np.array(line_vertices, dtype=np.float32)
    line_colors = np.array(line_colors, dtype=np.float32)

    # Update Line VBOs
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo)
    glBufferData(GL_ARRAY_BUFFER, line_vertices.nbytes, line_vertices, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glBufferData(GL_ARRAY_BUFFER, line_colors.nbytes, line_colors, GL_DYNAMIC_DRAW)


def draw_system(system, program, MVP):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Use the shader program
    glUseProgram(program)

    # Set the MVP uniform
    MVP_loc = glGetUniformLocation(program, "MVP")
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.astype(np.float32))

    # Draw lines
    glBindVertexArray(line_vao)
    # Each edge has 2 vertices, total edges = len(system.edges), so total line vertices = len(edges)*2
    glDrawArrays(GL_LINES, 0, len(system.edges)*2)

    # Draw points
    glBindVertexArray(point_vao)
    glPointSize(5.0)
    glDrawArrays(GL_POINTS, 0, system.positions.shape[0])

    pygame.display.flip()

def create_vao():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Vertex positions
    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Vertex colors
    vbo_col = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

    return vao, vbo_pos, vbo_col

############################
# Main Loop
############################
from compositegeometry import CompositeGeometry
def main():
    fps = 120
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Instantiate CompositeGeometry to generate dynamic shapes
    geometry_type = "cube"  # You can choose "cube", "tetrahedron", etc.
    density = 1.0  # Adjust density for scaling
    geometry = CompositeGeometry(geometry=geometry_type, device=device)
    
    # Fetch offsets and edges for the selected geometry
    offsets, _, edge_pairs, _, _ = geometry.configure_geometry(geometry=geometry_type, density=density)

    # Define positions and edges for MassSpringGraph
    positions = offsets + 10.5  # Offset for visualization
    edges = edge_pairs.cpu().tolist()  # Convert edges to a list of tuples
    
    masses = torch.ones(positions.shape[0], dtype=torch.float32, device=device) * 2000
    system = MassSpringGraph(positions, edges, masses, device=device)

    # OpenGL setup
    init_opengl()
    clock = pygame.time.Clock()
    program = create_program(vertex_shader_src, fragment_shader_src)

    global line_vao, line_vbo, color_vbo
    line_vao, line_vbo, color_vbo = create_vao()

    camera_offset = np.array([0.0, 0.0, 15.0], dtype=np.float32)

    # Main rendering loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        system.step(1/fps)  # Update spring simulation
        update_buffers(system)

        # Dynamic camera positioning
        center_of_mass = system.positions.mean(dim=0).cpu().numpy()
        eye = center_of_mass + camera_offset
        target = center_of_mass
        up = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        view = Matrix44.look_at(eye, target, up)
        projection = Matrix44.perspective_projection(60.0, 800 / 600, 0.01, 100.0)
        MVP = projection * view * Matrix44.identity()

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)

        MVP_loc = glGetUniformLocation(program, "MVP")
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, MVP.astype(np.float32))

        glBindVertexArray(line_vao)
        glDrawArrays(GL_LINES, 0, len(system.edges) * 2)

        pygame.display.flip()
        clock.tick(fps)

if __name__ == "__main__":
    main()