import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44
import torch_scatter
import pymunk
############################
# GNN-Based Force Propagation
############################

class SpringForceGNN(MessagePassing):
    def __init__(self):
        super(SpringForceGNN, self).__init__(aggr='add')  # Aggregate forces

    def forward(self, positions, velocities, edge_index, edge_attr):
        forces = self.propagate(edge_index, x=positions, velocities=velocities, edge_attr=edge_attr)
        return forces  # Aggregation happens automatically because aggr='add'

    def message(self, x_i, x_j, velocities_i, velocities_j, edge_attr):
        delta = x_j - x_i
        dist = torch.norm(delta, dim=1, keepdim=True)
        direction = delta / dist

        # Edge attributes
        rest_length = edge_attr[:, 0:1]
        spring_constant = edge_attr[:, 1:2]
        damper_ext = edge_attr[:, 2:3]  # Dampening for extension
        damper_cmp = edge_attr[:, 3:4]  # Dampening for compression
        gas_temperature = edge_attr[:, 4:5]
        min_extent = edge_attr[:, 5:6]  # Minimum allowable extent
        max_extent = edge_attr[:, 6:7]  # Maximum allowable extent
        rubber_stiffness = edge_attr[:, 7:8]
        steel_rigidity = edge_attr[:, 8:9]  # New: Extreme rigidity factor for boundaries

        # Hooke's Law - Spring Force
        extension = dist - rest_length
        spring_force = -spring_constant * extension * direction

        # Cylindrical gas pressure simulation
        volume = torch.pi * (rest_length ** 2) * dist
        pressure = gas_temperature / (volume + 1e-6)
        gas_force = pressure * direction

        # Directional Dampening Force
        rel_velocity = torch.sum((velocities_j - velocities_i) * direction, dim=1, keepdim=True)
        damper_force = -torch.where(rel_velocity > 0, damper_ext, damper_cmp) * rel_velocity * direction

        # Hard Stop Force at Min and Max Extents
        stop_force = torch.zeros_like(extension)
        velocity_damp = torch.zeros_like(extension).repeat(1, 3)  # Properly repeat tensor

        
        # Hard Stop Force at Min and Max Extents
        overshoot_max = torch.relu(dist - max_extent)  # Shape: [N, 1]
        overshoot_min = torch.relu(min_extent - dist)  # Shape: [N, 1]

        # Expand tensors to match direction shape [N, 3]
        overshoot_max_expanded = overshoot_max.expand(-1, 3)
        overshoot_min_expanded = overshoot_min.expand(-1, 3)
        steel_rigidity_expanded = steel_rigidity.expand(-1, 3)

        # Compute stop forces
        stop_force = -steel_rigidity_expanded * (overshoot_max_expanded ** 3) * direction
        stop_force += steel_rigidity_expanded * (overshoot_min_expanded ** 3) * direction

        print("rel_velocity shape:", rel_velocity.shape)  # Should be [N, 1]
        print("direction shape:", direction.shape)        # Should be [N, 3]
        print("velocity_damp shape:", velocity_damp.shape)  # Should be [N, 3]

        # Additional Velocity Dampening at Boundaries
        velocity_damp += torch.where(
            (dist >= max_extent) | (dist <= min_extent),
            -rel_velocity * direction,  # Complete damping at boundaries
            torch.zeros_like(rel_velocity)
        )

        # Total Force
        total_force = spring_force + gas_force + damper_force + stop_force + velocity_damp
        return total_force


############################
# PyG-Based MassSpringGraph
############################

class MassSpringGraph:
    def __init__(self, positions, edges, masses, space, device="cuda"):
        self.device = device
        self.positions = positions.to(device)
        self.velocities = torch.zeros_like(positions, device=device)
        self.masses = masses
                
        # Pymunk bodies and shapes
        self.bodies = []
        for i, pos in enumerate(self.positions):
            body = pymunk.Body(mass=1.0, moment=1.0)
            body.position = pos[0].item(), pos[1].item()
            shape = pymunk.Circle(body, radius=5)
            shape.filter = pymunk.ShapeFilter(group=1)  # Avoid collisions with other nodes
            space.add(body, shape)
            self.bodies.append(body)


        # Convert edges and edge features
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        rest_length = torch.norm(positions[edge_index[0]] - positions[edge_index[1]], dim=1) * .9
        spring_constant = torch.full_like(rest_length, 1.0)
        damper_ext = torch.full_like(rest_length, 1.0)
        damper_cmp = torch.full_like(rest_length, 1.0)  # Higher dampening on compression
        gas_temperature = torch.full_like(rest_length, 0.0)  # Default gas temperature
        max_extent = rest_length * 1.0  # Max allowable stretch
        min_extent = rest_length * 0.6
        rubber_stiffness = torch.full_like(rest_length, 0.0)
        steel_rigidity = torch.full_like(rest_length, 1000.0)

        self.edge_attr = torch.stack(
            [rest_length, spring_constant, damper_ext, damper_cmp, gas_temperature, min_extent, max_extent, rubber_stiffness, steel_rigidity], dim=1
        )
        self.graph = Data(x=self.positions, edge_index=edge_index, edge_attr=self.edge_attr)

        self.gnn = SpringForceGNN().to(device)

    def step(self, dt):
        # Sync Pymunk body positions to 3D positions (z=0)
        for i, body in enumerate(self.bodies):
            x, y = body.position
            self.positions[i] = torch.tensor([x, y, self.positions[i][2]], dtype=torch.float32, device=self.device)

        # Compute forces in the GNN
        forces = self.gnn(self.graph.x, self.velocities, self.graph.edge_index, self.graph.edge_attr)
        acceleration = forces / self.masses.unsqueeze(1)
        self.velocities += acceleration * dt
        self.graph.x += self.velocities * dt

        # Apply forces back to Pymunk bodies in 2D
        for i, body in enumerate(self.bodies):
            force_2d = forces[i, :2].detach().cpu().numpy()  # Extract only x, y components
            body.apply_impulse_at_local_point((force_2d[0] * dt, force_2d[1] * dt))

        # Recenter around center of mass
        center_of_mass = self.graph.x.mean(dim=0)
        self.graph.x -= center_of_mass




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

def create_program(vertex_src, fragment_src):
    def compile_shader(source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader).decode('utf-8'))
        return shader

    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    return program


def update_buffers(system):
    # Update vertex positions and colors dynamically
    line_vertices = []
    line_colors = []

    positions = system.graph.x.detach().cpu().numpy()
    edge_index = system.graph.edge_index.t().cpu().numpy()
    edge_attr = system.graph.edge_attr.detach().cpu().numpy()

    # Extract edge attributes
    rest_lengths = edge_attr[:, 0]  # Rest lengths
    spring_constants = edge_attr[:, 1]  # Spring constants
    dampers = edge_attr[:, 2]  # Damper values
    heat = edge_attr[:, 3]  # Heat values

    for idx, (i, j) in enumerate(edge_index):
        pi, pj = positions[i], positions[j]
        current_length = np.linalg.norm(pj - pi)

        # Line Colors
        compression_ratio = abs((current_length - rest_lengths[idx]) / rest_lengths[idx])
        spring_intensity = spring_constants[idx] / 1000.0  # Scale spring constants
        heat_intensity = heat[idx] / 1000.0  # Scale heat for visibility

        color = [
            min(1.0, compression_ratio * 5),  # Red: compression/stretch
            min(1.0, spring_intensity),       # Green: spring constant intensity
            min(1.0, heat_intensity)          # Blue: heat intensity
        ]

        line_vertices.extend(pi)
        line_colors.extend(color)
        line_vertices.extend(pj)
        line_colors.extend(color)

    line_vertices = np.array(line_vertices, dtype=np.float32)
    line_colors = np.array(line_colors, dtype=np.float32)

    # Update Line VBOs
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
    glBufferData(GL_ARRAY_BUFFER, line_vertices.nbytes, line_vertices, GL_DYNAMIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
    glBufferData(GL_ARRAY_BUFFER, line_colors.nbytes, line_colors, GL_DYNAMIC_DRAW)

############################
# Main Rendering Loop
############################
from compositegeometry import CompositeGeometry
def main():
    fps = 120
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pymunk Space
    space = pymunk.Space()
    space.gravity = (0, -5)  # Gravity in 2D downward

    # Instantiate CompositeGeometry to generate dynamic shapes
    geometry_type = "tetrahedron"  # Choose from "cube", "tetrahedron", etc.
    density = 1.0
    geometry = CompositeGeometry(geometry=geometry_type, device=device)

    # Fetch offsets and edges for the selected geometry
    offsets, _, edge_pairs, _, _ = geometry.configure_geometry(geometry=geometry_type, density=density)
    positions = offsets + 10.5  # Offset for visualization
    edges = edge_pairs.cpu().tolist()

    masses = torch.ones(positions.shape[0], dtype=torch.float32, device=device) * 20
    system = MassSpringGraph(positions, edges, masses, space, device=device)

    # Pygame and OpenGL Initialization
    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    program = create_program(vertex_shader_src, fragment_shader_src)
    global vbo_positions, vbo_colors
    vbo_positions, vbo_colors = glGenBuffers(2)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    MVP = Matrix44.perspective_projection(45.0, 800 / 600, 0.1, 100.0) * Matrix44.look_at([0, 0, 10], [0, 0, 0], [0, 1, 0])
    camera_offset = np.array([0.0, 0.0, 15.0], dtype=np.float32)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Step the Pymunk physics space
        space.step(1 / fps)

        # Update system with Pymunk positions
        system.step(1 / fps)
        update_buffers(system)
        # Dynamic camera positioning
        center_of_mass = system.positions.mean(dim=0).cpu().numpy()
        eye = center_of_mass + camera_offset
        target = center_of_mass
        up = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        view = Matrix44.look_at(eye, target, up)
        projection = Matrix44.perspective_projection(60.0, 800 / 600, 0.01, 100.0)
        MVP = projection * view * Matrix44.identity()

        # OpenGL Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        glUniformMatrix4fv(glGetUniformLocation(program, "MVP"), 1, GL_FALSE, MVP.astype(np.float32))
        glBindVertexArray(vao)
        glDrawArrays(GL_LINES, 0, len(system.graph.edge_index[0]) * 2)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    main()
