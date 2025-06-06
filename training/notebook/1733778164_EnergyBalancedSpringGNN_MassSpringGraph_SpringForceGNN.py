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
import argparse  # Added for configurator

############################
# GNN-Based Force Propagation
############################

class SpringForceGNN(MessagePassing):
    def __init__(self):
        super(SpringForceGNN, self).__init__(aggr='add')  # Aggregate forces

    def forward(self, positions, velocities, edge_index, edge_attr):
        self.profiles = []  # Store profiles for debugging
        forces = self.propagate(edge_index, x=positions, velocities=velocities, edge_attr=edge_attr)
        return forces  # Return forces only

    def message(self, x_i, x_j, velocities_i, velocities_j, edge_attr):
        delta = x_j - x_i
        dist = torch.norm(delta, dim=1, keepdim=True)
        direction = delta / (dist + 1e-8)

        # Edge attributes
        rest_length = edge_attr[:, 0:1]
        spring_constant = edge_attr[:, 1:2]
        damper_ext = edge_attr[:, 2:3]
        damper_cmp = edge_attr[:, 3:4]
        gas_temperature = edge_attr[:, 4:5]
        min_extent = edge_attr[:, 5:6]
        max_extent = edge_attr[:, 6:7]
        rubber_stiffness = edge_attr[:, 7:8]
        steel_rigidity = edge_attr[:, 8:9]

        # Hooke's Law - Spring Force
        extension = dist - rest_length
        spring_force = -spring_constant * extension * direction

        # Gas Pressure Force
        volume = torch.pi * (rest_length ** 2) * dist
        pressure = gas_temperature / (volume + 1e-6)
        gas_force = pressure * direction

        # Dampening Force
        rel_velocity = torch.sum((velocities_j - velocities_i) * direction, dim=1, keepdim=True)
        damper_force = -torch.where(rel_velocity > 0, damper_ext, damper_cmp) * rel_velocity * direction

        # Hard Stop Forces
        overshoot_max = torch.relu(dist - max_extent)
        overshoot_min = torch.relu(min_extent - dist)
        stop_force = (
            -steel_rigidity * (overshoot_max ** 3) * direction
            + steel_rigidity * (overshoot_min ** 3) * direction
        )

        # Tire Deflection for Compression
        tire_deflection = rubber_stiffness * torch.tanh(overshoot_min * 10)
        tire_force = tire_deflection * direction

        # Overextension Tearing Force
        tearing_decay = steel_rigidity * torch.exp(-0.5 * overshoot_max)
        tear_force = -tearing_decay * direction

        # Total Force
        total_force = spring_force + gas_force + damper_force + stop_force + tire_force + tear_force

        # Store profile information for debugging
        self.profiles.append({
            "travel": dist - rest_length,
            "spring_force": spring_force,
            "gas_force": gas_force,
            "damper_force": damper_force,
            "stop_force": stop_force,
            "tire_force": tire_force,
            "tear_force": tear_force,
        })

        return total_force


import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class EnergyBalancedSpringGNN(MessagePassing):
    def __init__(self):
        super(EnergyBalancedSpringGNN, self).__init__(aggr='add')  # Sum forces
        self.energy_profiles = []

    def forward(self, positions, velocities, edge_index, edge_attr, edge_masses, edge_moments):
        self.energy_profiles = []
        forces, energies = self.propagate(edge_index, x=positions, velocities=velocities, edge_attr=edge_attr,
                                          edge_mass=edge_masses, edge_moment=edge_moments)
        return forces, energies

    def message(self, x_i, x_j, velocities_i, velocities_j, edge_attr, edge_mass, edge_moment):
        delta = x_j - x_i
        dist = torch.norm(delta, dim=1, keepdim=True)
        direction = delta / (dist + 1e-8)

        # Edge parameters
        rest_length = edge_attr[:, 0:1]
        spring_constant = edge_attr[:, 1:2]
        damper_ext = edge_attr[:, 2:3]
        damper_cmp = edge_attr[:, 3:4]
        gas_temperature = edge_attr[:, 4:5]
        min_extent = edge_attr[:, 5:6]
        max_extent = edge_attr[:, 6:7]
        rubber_stiffness = edge_attr[:, 7:8]
        steel_rigidity = edge_attr[:, 8:9]

        # Hooke's Law - Spring Force
        extension = dist - rest_length
        spring_force = -spring_constant * extension * direction
        spring_energy = 0.5 * spring_constant * (extension ** 2)

        # Constants
        atmospheric_pressure = 101325.0  # Standard atmospheric pressure in Pascals
        minimum_radius = 0.05  # Minimum physical radius in meters
        minimum_volume = torch.pi * (minimum_radius ** 2) * min_extent  # Minimum gas volume

        # Current volume with clipping to ensure minimum volume
        current_volume = torch.pi * torch.clamp(rest_length ** 2, min=minimum_radius ** 2) * torch.clamp(dist, min=min_extent)
        internal_pressure = gas_temperature / (current_volume + 1e-6)  # Ideal gas law (P = nRT/V)

        # Net pressure difference: stops exerting force at max_extent
        net_pressure = torch.clamp(internal_pressure - atmospheric_pressure, min=0.0)

        # Gas force: stops exerting when extension stop (max_extent) occurs
        gas_force = torch.where(dist < max_extent, net_pressure * direction, torch.zeros_like(direction))

        # Gas Energy Contribution
        # E_gas = P * V (ideal gas energy: pressure-volume work)
        gas_energy = net_pressure * current_volume

        # Dampening Force
        rel_velocity = torch.sum((velocities_j - velocities_i) * direction, dim=1, keepdim=True)
        damper_force = -torch.where(rel_velocity > 0, damper_ext, damper_cmp) * rel_velocity * direction
        damping_energy_loss = torch.abs(rel_velocity) * torch.where(rel_velocity > 0, damper_ext, damper_cmp)

        # Hard Stops
        overshoot_max = torch.relu(dist - max_extent)
        overshoot_min = torch.relu(min_extent - dist)
        stop_force = (
            -steel_rigidity * (overshoot_max ** 3) * direction
            + steel_rigidity * (overshoot_min ** 3) * direction
        )
        stop_energy = 0.5 * steel_rigidity * ((overshoot_max ** 2) + (overshoot_min ** 2))

        # Edge Mass Contribution to Momentum
        edge_momentum = edge_mass * rel_velocity
        rotational_energy = 0.5 * edge_moment * (rel_velocity ** 2)

        # Total Energy
        total_energy = spring_energy + gas_energy + stop_energy - damping_energy_loss + rotational_energy

        # Total Force
        total_force = spring_force + gas_force + damper_force + stop_force

        # Append to energy profiles for debugging
        self.energy_profiles.append({
            "spring_energy": spring_energy.sum(),
            "gas_energy": gas_energy.sum(),
            "stop_energy": stop_energy.sum(),
            "damping_energy_loss": damping_energy_loss.sum(),
            "rotational_energy": rotational_energy.sum(),
            "total_energy": total_energy.sum(),
            "edge_momentum": edge_momentum.sum()
        })

        return -total_force, total_energy

    def aggregate(self, inputs, index, dim_size=None):
        forces, energies = inputs
        aggregated_forces = torch_scatter.scatter(forces, index, dim=0, reduce='sum')
        total_energy = torch_scatter.scatter(energies, index, dim=0, reduce='sum')
        return aggregated_forces, total_energy


############################
# PyG-Based MassSpringGraph
############################
def node_configuration_builder(config_type="particles", num_nodes=10, device="cuda"):
    """
    Build node configurations for testing based on the desired configuration type.

    Args:
        config_type (str): Type of configuration ('planets', 'particles', 'objects').
        num_nodes (int): Number of nodes to generate.
        device (str): Device to store tensors ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing positions, masses, charges, magnetic fields, 
              magnetic moments, and surface properties for plasma arc simulation.
    """
    positions = torch.randn((num_nodes, 3), device=device) * 10.0  # Random positions in space
    node_masses = torch.ones(num_nodes, device=device)  # Default masses
    node_charges = torch.zeros(num_nodes, device=device)  # Default neutral charge
    node_magnetic = torch.zeros((num_nodes, 3), device=device)  # Default no magnetic field
    node_magnetic_moment = torch.zeros((num_nodes, 3), device=device)  # Default no magnetic moment
    node_surface_quality = torch.zeros(num_nodes, device=device)  # Surface quality for plasma arcs

    if config_type == "planets":
        # Simulate massive planetary objects with gravity
        node_masses = torch.abs(torch.randn(num_nodes, device=device) * 5e24)  # Large masses
        node_charges = torch.zeros(num_nodes, device=device)  # Planets are electrically neutral
        node_magnetic = torch.randn((num_nodes, 3), device=device) * 1e-3  # Weak planetary magnetism
        node_magnetic_moment = node_magnetic * 1e2  # Proportional magnetic moment for planetary bodies
        node_surface_quality = torch.rand(num_nodes, device=device) * 0.1  # Low arc propensity for planets

    elif config_type == "particles":
        # Simulate fundamental particles with electric charge and magnetic fields
        node_masses = torch.abs(torch.randn(num_nodes, device=device) * 1e-27)  # Particle masses
        node_charges = torch.randn(num_nodes, device=device) * 1e-19  # Charges (Coulombs)
        node_magnetic = torch.randn((num_nodes, 3), device=device) * 1e-6  # Small magnetic vectors
        node_magnetic_moment = node_magnetic * 1e-4  # Proportional magnetic moment
        node_surface_quality = torch.rand(num_nodes, device=device) * 0.9 + 0.1  # High arc propensity for particles

    elif config_type == "objects":
        # Simulate everyday objects with moderate mass and optional magnetism
        node_masses = torch.abs(torch.randn(num_nodes, device=device) * 10.0)  # Moderate masses (kg)
        node_charges = torch.randn(num_nodes, device=device) * 1e-3  # Small charge distribution
        node_magnetic = torch.randn((num_nodes, 3), device=device) * 1e-3  # Optional magnetism
        node_magnetic_moment = node_magnetic * 1e0  # Moderate magnetic moment
        node_surface_quality = torch.rand(num_nodes, device=device) * 0.5 + 0.3  # Medium arc propensity

    else:
        raise ValueError("Invalid configuration type. Choose from 'planets', 'particles', or 'objects'.")

    # Optional property: Arc threshold based on charge and surface quality
    plasma_arc_threshold = node_surface_quality * torch.abs(node_charges) * 1e6

    return {
        "positions": positions,
        "node_masses": node_masses,
        "node_charges": node_charges,
        "node_magnetic": node_magnetic,
        "node_magnetic_moment": node_magnetic_moment,
        "node_surface_quality": node_surface_quality,
        "plasma_arc_threshold": plasma_arc_threshold
    }


class MassSpringGraph:
    def __init__(self, positions, edges, node_masses, edge_params, edge_masses, edge_moments, node_magnetic_moments, node_electrostatic_charge, device="cuda"):
        self.device = device
        self.positions = positions.to(device)
        self.velocities = torch.zeros_like(positions, device=device)
        self.node_masses = node_masses.to(device)
        self.edge_masses = edge_masses.to(device)
        self.edge_moments = edge_moments.to(device)
        self.node_magnetic = node_magnetic_moments.to(device)
        self.node_electrostatic_charge = node_electrostatic_charge.to(device)

        # Gravity: Constant downward force (adjust vector as needed)
        self.gravity = torch.tensor([0.0, -0.0, 0.0], dtype=torch.float32, device=device)

        # External field parameters (e.g., gravity well position and intensity)
        self.field_sources = [
            {"type": "gravity", "position": torch.tensor([0.0, 0.0, 0.0], device=self.device), "intensity": 10000.0},
            {"type": "electrostatic"},  # Uses self.node_charges
            {"type": "magnetic", "local_B": self.node_magnetic, "electric": torch.zeros_like(self.positions)}
        ]

        # Edge attributes
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        self.edge_attr = edge_params.to(device)

        self.graph = Data(x=self.positions, edge_index=edge_index, edge_attr=self.edge_attr)
        self.gnn = EnergyBalancedSpringGNN().to(device)
        #print("Edge Index Shape:", self.graph.edge_index.shape)
        #print("Edge Attr Shape:", self.edge_attr.shape)
        #print("Positions Shape:", self.positions.shape)

    def compute_gravity_force(self):
        """Compute gravitational force for each node."""
        return self.node_masses.unsqueeze(1) * self.gravity  # F = m * g

    def compute_field_forces(self):
        """
        Compute forces from all field sources:
        - Gravity wells (point sources)
        - Electrostatic forces
        - Magnetic forces (Lorentz Force: q * (E + v x B))
        """
        N = self.positions.shape[0]
        field_forces = torch.zeros_like(self.positions, device=self.device)

        for field in self.field_sources:
            field_type = field["type"]

            # Gravity Wells (point sources)
            if field_type == "gravity":
                source_pos = field["position"]  # Gravity well position
                intensity = field["intensity"]  # Intensity (gravitational constant)
                delta = self.positions - source_pos
                dist = torch.norm(delta, dim=1, keepdim=True) + 1e-6  # Avoid division by zero
                direction = delta / dist
                field_forces += -intensity * direction / (dist ** 2)  # Inverse-square gravity

            # Electrostatic Forces (Coulomb's Law)
            elif field_type == "electrostatic":
                k_e = 8.9875e9  # Coulomb constant
                charges = self.node_electrostatic_charge  # Node charges
                delta = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)  # Pairwise differences
                dist_sq = torch.sum(delta ** 2, dim=2) + 1e-6  # Avoid division by zero
                direction = delta / torch.sqrt(dist_sq).unsqueeze(2)
                charges_product = charges.unsqueeze(1) * charges.unsqueeze(0)
                electrostatic_forces = k_e * charges_product.unsqueeze(2) * direction / dist_sq.unsqueeze(2)
                field_forces += torch.sum(electrostatic_forces, dim=1)  # Sum pairwise forces

            # Magnetic Forces (Lorentz Force)
            elif field_type == "magnetic":
                local_B = field["local_B"]  # Local magnetic field vectors at each node (N, 3)
                E_field = field.get("electric", torch.zeros_like(self.positions))  # Optional electric field
                qv_cross_B = torch.cross(self.velocities, local_B, dim=1)  # q(v x B)
                lorentz_force = self.node_electrostatic_charge.unsqueeze(1) * (E_field + qv_cross_B)
                field_forces += lorentz_force

        return field_forces

    def step(self, dt):
        # Compute internal forces
        forces, energy_balance = self.gnn(
            self.graph.x, self.velocities, self.graph.edge_index, self.graph.edge_attr,
            self.edge_masses, self.edge_moments
        )

        # Add external forces: Gravity and field forces
        gravity_forces = self.compute_gravity_force()
        field_forces = self.compute_field_forces()
        total_external_forces = gravity_forces + field_forces

        # Combine forces
        total_forces = forces + total_external_forces

        # Node accelerations and momentum updates
        acceleration = total_forces / self.node_masses.unsqueeze(1)
        self.velocities += acceleration * dt
        self.positions += self.velocities * dt

        # Update graph positions
        self.graph.x = self.positions

        # Debugging external forces
        #print(f"Gravity Forces: {gravity_forces}")
        #print(f"Field Forces: {field_forces}")



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
        heat_intensity = heat[idx] / 10.0  # Scale heat for visibility

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
    # ============================
    # Configurator Integration Start
    # ============================
    parser = argparse.ArgumentParser(description="Mass-Spring System Simulator Configurator")
    parser.add_argument(
        '--node_type',
        type=str,
        default='particles',
        choices=['planets', 'particles', 'objects'],
        help="Type of nodes to simulate: 'planets', 'particles', or 'objects'. Default is 'particles'."
    )
    parser.add_argument(
        '--geometry_type',
        type=str,
        default='ray',
        choices=['ray', 'cube', 'tetrahedron'],
        help="Type of geometry to use: 'ray', 'cube', 'tetrahedron'. Default is 'ray'."
    )

    args = parser.parse_args()
    # ============================
    # Configurator Integration End
    # ============================

    fps = 240
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate CompositeGeometry to generate dynamic shapes
    geometry_type = args.geometry_type  # Set from configurator
    density = .20
    geometry = CompositeGeometry(geometry=geometry_type, device=device)

    # Fetch geometry offsets and edges
    offsets, _, edge_pairs, _, _ = geometry.configure_geometry(geometry=geometry_type, density=density)
    positions = offsets + 10.5  # Offset positions for visualization
    edges = edge_pairs

    # Node Masses and Edge Parameters
    node_masses = torch.ones(positions.shape[0], dtype=torch.float32, device=device) * 50.0
    edge_masses = torch.ones(len(edges), dtype=torch.float32, device=device) * 5.0
    edge_moments = torch.ones(len(edges), dtype=torch.float32, device=device) * 0.5

    # Edge Attributes (environmental and spring)
    rest_lengths = torch.norm(positions[edges[...,0]] - positions[edges[...,1]], dim=1) * 0.9
    spring_constants = torch.full_like(rest_lengths, 10.0)
    damping_ext = torch.full_like(rest_lengths, 10.0)
    damping_cmp = torch.full_like(rest_lengths, 20.0)
    gas_temp = torch.full_like(rest_lengths, 1000.0)
    max_extent = rest_lengths * 1.0
    min_extent = rest_lengths * 0.6
    rubber_stiffness = torch.full_like(rest_lengths, 10.0)
    steel_rigidity = torch.full_like(rest_lengths, 10.0)
    env_influence = torch.full_like(rest_lengths, 1.0)  # Placeholder for environmental influence
    #print(rest_lengths.shape)
    edge_params = torch.stack(
        [rest_lengths, spring_constants, damping_ext, damping_cmp, gas_temp,
         min_extent, max_extent, rubber_stiffness, steel_rigidity, env_influence], dim=1
    )
    #print(edge_params.shape)

    # ============================
    # Configurator Integration Start
    # ============================
    # Build node configurations based on node_type
    node_config = node_configuration_builder(
        config_type=args.node_type,
        num_nodes=offsets.shape[0],
        device=device
    )
    # Override positions if node_configuration_builder provides them
    positions = node_config["positions"] + 10.5  # Offset positions for visualization
    node_masses = node_config["node_masses"]
    node_charges = node_config["node_charges"]
    node_magnetic = node_config["node_magnetic"]
    node_magnetic_moment = node_config["node_magnetic_moment"]
    plasma_arc_threshold = node_config["plasma_arc_threshold"]

    # Optionally, adjust edges based on new positions
    # For simplicity, assuming edges remain the same
    # If geometry changes affect edges, regenerate edges here

    # ============================
    # Configurator Integration End
    # ============================

    # Mass-Spring Graph Initialization
    system = MassSpringGraph(
        positions=positions,
        edges=edges,
        node_masses=node_masses,
        edge_params=edge_params,
        edge_masses=edge_masses,
        edge_moments=edge_moments,
        node_magnetic_moments=node_magnetic_moment,
        node_electrostatic_charge=node_charges,  # Updated from node_charges
        device=device
    )

    # Pygame and OpenGL Initialization
    pygame.init()
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    # Compile Shader Program
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

    # Camera Setup
    camera_offset = np.array([0.0, 0.0, 15.0], dtype=np.float32)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Step Simulation
        system.step(1 / fps)

        # Update OpenGL Buffers
        update_buffers(system)

        # Dynamic Camera Position
        center_of_mass = system.positions.mean(dim=0).detach().cpu().numpy()
        eye = center_of_mass + camera_offset
        target = center_of_mass
        up = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        view = Matrix44.look_at(eye, target, up)
        projection = Matrix44.perspective_projection(60.0, 800 / 600, 0.01, 100.0)
        MVP = projection * view * Matrix44.identity()

        # Render System
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(program)
        glUniformMatrix4fv(glGetUniformLocation(program, "MVP"), 1, GL_FALSE, MVP.astype(np.float32))
        glBindVertexArray(vao)
        glDrawArrays(GL_LINES, 0, len(system.graph.edge_index[0]) * 2)
        pygame.display.flip()
        #clock.tick(fps)

if __name__ == "__main__":
    main()
