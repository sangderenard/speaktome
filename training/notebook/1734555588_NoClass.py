import pygame
from pygame.locals import QUIT
from pymunk.vec2d import Vec2d
import torch

# ============================
# Pygame and Projection Setup
# ============================

# Pygame initialization
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
NODE_RADIUS = 5  # Node size for visualization
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

precision = torch.float64

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Rigid System - Projected onto 2D")
clock = pygame.time.Clock()


# ============================
# Helper Functions
# ============================

import numpy as np

import torch

def vector(x, y, z=0):
    """Simple 3D vector as a torch tensor."""
    return torch.tensor([x, y, z], dtype=torch.float32)

def identity_matrix():
    """Returns a 3x3 identity matrix."""
    return torch.eye(3)

def rotate_vector(rotation_matrix, vec):
    """Applies a rotation matrix to a vector."""
    return torch.matmul(rotation_matrix, vec)

def cross_product(v1, v2):
    """Computes the cross product of two vectors."""
    return torch.cross(v1, v2)

def norm(v):
    """Euclidean norm of a vector."""
    return torch.norm(v)

def dot_product(v1, v2):
    """Dot product of two vectors."""
    return torch.dot(v1, v2)

def update_rotation(rotation, angular_velocity, dt):
    """
    Updates the rotation matrix using angular velocity.
    For simplicity, assumes small rotations using a skew-symmetric matrix.
    """
    skew = torch.tensor([
        [0, -angular_velocity[2], angular_velocity[1]],
        [angular_velocity[2], 0, -angular_velocity[0]],
        [-angular_velocity[1], angular_velocity[0], 0]
    ], dtype=precision)
    return rotation + torch.matmul(skew, rotation) * dt


def project_3d_to_2d(position):
    """
    Collapse the Z-dimension to project 3D points onto the XY plane.
    """
    return int(position[0]) + SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - int(position[1])


def draw_node(node):
    """Render a node projected onto 2D."""
    global_pos = project_3d_to_2d(node.position)
    pygame.draw.circle(screen, RED, global_pos, NODE_RADIUS)


def draw_edge(edge):
    """Render an edge as a line connecting two nodes projected onto 2D."""
    pos_a = edge.node_a.position + edge.node_a.attachment_points[edge.attach_a]
    pos_b = edge.node_b.position + edge.node_b.attachment_points[edge.attach_b]
    pygame.draw.line(screen, GREEN, project_3d_to_2d(pos_a), project_3d_to_2d(pos_b), 2)


class Node:
    def __init__(self, name, position, mass, inertia, attachment_points):
        """
        Parameters:
            name (str): Unique node name.
            position (vector): Global position of the node center.
            mass (float): Mass of the rigid body.
            inertia (tensor): Rotational inertia tensor.
            attachment_points (dict): {"point_name": local_coords}.
        """
        self.name = name
        self.position = position  # Global position
        self.velocity = vector(0, 0, 0)
        self.mass = mass
        self.inertia = inertia
        self.rotation = identity_matrix()  # Initial orientation
        self.angular_velocity = vector(0, 0, 0)
        self.attachment_points = attachment_points  # Map of attachment names to local coordinates

    def translate_force(self, force, point_name):
        """
        Translate a force applied at an attachment point into body translation and rotation.
        """
        local_point = self.attachment_points[point_name]
        lever_arm = rotate_vector(self.rotation, local_point)
        torque = cross_product(lever_arm, force)
        return torque
class Edge:
    def __init__(self, node_a, attach_a, node_b, attach_b, edge_type, parameters):
        """
        Parameters:
            node_a, node_b (Node): Connected nodes.
            attach_a, attach_b (str): Attachment points on the nodes.
            edge_type (str): Type of edge ('axial_spring', 'leaf_spring', 'dampener', 'pneumatic_sack', etc.).
            parameters (dict): Physical parameters specific to edge type.
        """
        self.node_a = node_a
        self.node_b = node_b
        self.attach_a = attach_a
        self.attach_b = attach_b
        self.edge_type = edge_type
        self.parameters = parameters  # Specific parameters for edge behavior
    def compute_bend_angle(load_type="end", support_type="start", offsets=None):
            # point types: 
            #  end: the load is applied at the destination of the edge
            #  center: the load is applied between the ends, defaulting to center but able to be offset
            #  start: the load is applied at the origin of the edge
            #  ends: the load is applied at beginning and end of the edge
            # load vs support types:
            #  not an especially important distinction except when trying to establish causality and transients
            #  when evaluating a force through an edge that has a bend angle, the direction of travel is from
            #  the load to the support, the transient to the stationary
            #
    def compute_forces(self):
        """
        Calculate forces and torques for the connected nodes based on edge type.
        """
        pos_a = self.node_a.position + self.node_a.attachment_points[self.attach_a]
        pos_b = self.node_b.position + self.node_b.attachment_points[self.attach_b]
        delta = pos_b - pos_a
        distance = norm(delta)
        direction = delta / (distance + 1e-6)

        if self.edge_type == "axial_spring":
            # Axial spring force
            rest_length = self.parameters["rest_length"]
            stiffness = self.parameters["stiffness"]
            damping = self.parameters["damping"]
            velocity_diff = self.node_b.velocity - self.node_a.velocity
            spring_force = -stiffness * (distance - rest_length) * direction
            damping_force = -damping * dot_product(velocity_diff, direction) * direction
            return spring_force + damping_force



        elif self.edge_type == "leaf_spring":
            # Leaf spring bending force (simplified)
            flex_limit = self.parameters["flex_limit"]
            bending_stiffness = self.parameters["stiffness"]
            angle = self.compute_bend_angle(load_type="center", support_type="ends")
            bending_force = -bending_stiffness * angle / flex_limit
            return bending_force * direction

        elif self.edge_type == "pneumatic_sack":
            # General gas pressure response
            volume = self.parameters["volume"]
            gas_mass = self.parameters["gas_mass"]
            temperature = self.parameters["temperature"]
            elasticity = self.parameters["elasticity"]
            pressure = gas_mass * temperature / volume
            compression_force = pressure * direction
            wall_stiffness = elasticity * max(0, (self.parameters["max_volume"] - volume))
            return compression_force - wall_stiffness * direction

        elif self.edge_type == "piston_chamber":
            # Special case: gas chamber constrained along the long axis
            volume = pi * self.parameters["radius"]**2 * distance  # Cylinder volume
            gas_mass = self.parameters["gas_mass"]
            temperature = self.parameters["temperature"]
            pressure = gas_mass * temperature / volume
            compression_axis = self.parameters["axis"]  # Fixed axis of compression
            return pressure * compression_axis  # Force acts along the defined axis

        elif self.edge_type == "bump_stop":
            # Energy absorption at threshold
            threshold = self.parameters["threshold"]
            if distance < threshold:
                return -self.parameters["stiffness"] * (threshold - distance) * direction
            return vector(0, 0, 0)

class RigidSystemGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # List of Node objects
        self.edges = edges  # List of Edge objects

    def step(self, dt):
        """
        Perform a simulation step.
        """
        forces = {node.name: vector(0, 0, 0) for node in self.nodes}

        # Compute all edge forces
        for edge in self.edges:
            force = edge.compute_forces()
            torque_a = edge.node_a.translate_force(force, edge.attach_a)
            torque_b = edge.node_b.translate_force(-force, edge.attach_b)

            # Accumulate forces and torques
            forces[edge.node_a.name] += force
            forces[edge.node_b.name] -= force

        # Update node positions and rotations
        for node in self.nodes:
            acceleration = forces[node.name] / node.mass
            node.velocity += acceleration * dt
            node.position += node.velocity * dt

            # Update angular motion (simplified)
            angular_accel = torque / node.inertia
            node.angular_velocity += angular_accel * dt
            node.rotation = update_rotation(node.rotation, node.angular_velocity * dt)
import numpy as np

class InertiaTensor:
    def __init__(self, mass, shape, dimensions):
        """
        Calculate the inertia tensor for various shapes.
        
        Parameters:
            mass (float): Mass of the rigid body.
            shape (str): Shape type ("sphere", "cylinder", "cuboid").
            dimensions (dict): Dimensions specific to the shape.
                - For "sphere": {"radius": r}
                - For "cylinder": {"radius": r, "height": h}
                - For "cuboid": {"width": w, "height": h, "depth": d}
        """
        self.mass = mass
        self.shape = shape
        self.dimensions = dimensions
        self.tensor = self._calculate_inertia()

    def _calculate_inertia(self):
        """Internal method to compute the inertia tensor."""
        if self.shape == "sphere":
            r = self.dimensions["radius"]
            I = (2 / 5) * self.mass * (r ** 2)
            return np.diag([I, I, I])

        elif self.shape == "cylinder":
            r = self.dimensions["radius"]
            h = self.dimensions["height"]
            Ix = Iy = (1 / 12) * self.mass * (3 * r**2 + h**2)
            Iz = (1 / 2) * self.mass * (r**2)
            return np.diag([Ix, Iy, Iz])

        elif self.shape == "cuboid":
            w = self.dimensions["width"]
            h = self.dimensions["height"]
            d = self.dimensions["depth"]
            Ix = (1 / 12) * self.mass * (h**2 + d**2)
            Iy = (1 / 12) * self.mass * (w**2 + d**2)
            Iz = (1 / 12) * self.mass * (w**2 + h**2)
            return np.diag([Ix, Iy, Iz])

        else:
            raise ValueError("Unsupported shape for inertia calculation.")

    def get_tensor(self):
        """Return the calculated inertia tensor."""
        return self.tensor

def main():
    # Scale factor and base position offset
    scale = 10  # Scale for conventional mm sizing
    base_offset = vector(0, 300)  # Center base offset for the system

    # Nodes (positions are relative offsets in mm * scale)
    tube_steel = Node("TubeSteel", base_offset + vector(0, -50), 2, identity_matrix(), {"center": vector(0, 0)})
    leaf_spring = Node("LeafSpring", base_offset + vector(0, 0), 5, identity_matrix(), {"center": vector(0, 0)})
    piston_left = Node("PistonLeft", base_offset + vector(-30, 50), 3, identity_matrix(), {"bottom": vector(0, 0)})
    piston_right = Node("PistonRight", base_offset + vector(30, 50), 3, identity_matrix(), {"bottom": vector(0, 0)})
    axle = Node("Axle", base_offset + vector(0, 100), 8, identity_matrix(), {"center": vector(0, 0)})
    wheel = Node("Wheel", base_offset + vector(0, 150), 10, identity_matrix(), {"center": vector(0, 0)})
    tire = Node("Tire", base_offset + vector(0, 155), 1, identity_matrix(), {"center": vector(0, 0)})

    # Edges (connections and their physical parameters)
    edges = [
        Edge(tube_steel, "center", leaf_spring, "center", "leaf_spring", {"stiffness": 500, "flex_limit": 5}),
        Edge(leaf_spring, "center", piston_left, "bottom", "axial_spring",
             {"rest_length": 60, "stiffness": 100, "damping": 20}),
        Edge(leaf_spring, "center", piston_right, "bottom", "axial_spring",
             {"rest_length": 60, "stiffness": 100, "damping": 20}),
        Edge(piston_left, "bottom", axle, "center", "piston_chamber",
             {"radius": 10, "gas_mass": 1, "temperature": 300}),
        Edge(piston_right, "bottom", axle, "center", "piston_chamber",
             {"radius": 10, "gas_mass": 1, "temperature": 300}),
        Edge(axle, "center", wheel, "center", "bearing_joint", {"stiffness": 200}),
        Edge(wheel, "center", tire, "center", "bump_stop", {"threshold": 5, "stiffness": 150})
    ]

    # Create the rigid system
    system = RigidSystemGraph(
        [tube_steel, leaf_spring, piston_left, piston_right, axle, wheel, tire], edges
    )

    # Main simulation loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        # Clear the screen
        screen.fill(WHITE)

        # Update and render the system
        system.step(dt=0.1)
        for edge in system.edges:
            draw_edge(edge)
        for node in system.nodes:
            draw_node(node)

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()



if __name__ == "__main__":
    main()
