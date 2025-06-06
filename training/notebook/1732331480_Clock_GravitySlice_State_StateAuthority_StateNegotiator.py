import pygame
import pymunk
import numpy as np
from pyquaternion import Quaternion
import time

# Constants
WIDTH, HEIGHT = 1200, 800  # Window size
FPS = 60
GRAVITY_MAGNITUDE = 981  # Pixels per second squared
SCALE = 2  # Scaling factor for rendering

# Define box rectangles for each simulation
CENTER_BOX_RECT = pygame.Rect(500, 300, 200, 200)  # Central box for aggregated state
SMALL_BOX_1 = pygame.Rect(50, 50, 200, 200)        # Top-left simulation (XY Plane)
SMALL_BOX_2 = pygame.Rect(50, 550, 200, 200)       # Bottom-left simulation (XZ Plane)
SMALL_BOX_3 = pygame.Rect(1050, 50, 200, 200)      # Top-right simulation (YZ Plane)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Three Orthogonal Physics Simulations")
clock = pygame.time.Clock()

# Helper Functions
def quaternion_average(quaternions, weights):
    """
    Compute the weighted average of quaternions.
    """
    # Ensure all quaternions are in the same hemisphere
    for i in range(len(quaternions)):
        if quaternions[i].scalar < 0:
            quaternions[i] = -quaternions[i]
    
    # Weighted sum
    q_sum = Quaternion(0, 0, 0, 0)
    for q, w in zip(quaternions, weights):
        q_sum += q * w
    return q_sum.normalised

def simulation_to_screen(sim_pos, box_rect, scale=SCALE):
    """
    Convert simulation coordinates to screen coordinates within a box.
    sim_pos: (x, y) tuple
    box_rect: pygame.Rect
    scale: scaling factor
    Returns: (screen_x, screen_y)
    """
    screen_x = box_rect.x + box_rect.width / 2 + sim_pos[0] * scale
    screen_y = box_rect.y + box_rect.height / 2 - sim_pos[1] * scale  # y inverted for Pygame
    return int(screen_x), int(screen_y)

# Classes
class State:
    def __init__(self, position, rotation, velocity):
        self.position = np.array(position, dtype=np.float64)  # [x, y]
        self.rotation = rotation  # Quaternion
        self.velocity = np.array(velocity, dtype=np.float64)  # [vx, vy]

class StateAuthority:
    def __init__(self, space, projection_plane, gravity_quaternion, position_offset=(0,0)):
        self.space = space
        self.projection_plane = projection_plane  # 'XY', 'XZ', 'YZ'
        self.gravity_quaternion = gravity_quaternion
        self.position_offset = position_offset
        self.create_objects()
        self.state = None  # To be updated each step

    def create_objects(self):
        # Constants for walls
        thickness = 10
        elasticity = 0.8
        friction = 0.5

        # Create a static body for the walls
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        static_body.position = self.position_offset

        # Define wall segments based on projection plane
        if self.projection_plane == 'XY':
            walls = [
                pymunk.Segment(static_body, (-50, -50), (50, -50), thickness),
                pymunk.Segment(static_body, (50, -50), (50, 50), thickness),
                pymunk.Segment(static_body, (50, 50), (-50, 50), thickness),
                pymunk.Segment(static_body, (-50, 50), (-50, -50), thickness),
            ]
        elif self.projection_plane == 'XZ':
            walls = [
                pymunk.Segment(static_body, (-50, -50), (50, -50), thickness),
                pymunk.Segment(static_body, (50, -50), (50, 50), thickness),
                pymunk.Segment(static_body, (50, 50), (-50, 50), thickness),
                pymunk.Segment(static_body, (-50, 50), (-50, -50), thickness),
            ]
        elif self.projection_plane == 'YZ':
            walls = [
                pymunk.Segment(static_body, (-50, -50), (50, -50), thickness),
                pymunk.Segment(static_body, (50, -50), (50, 50), thickness),
                pymunk.Segment(static_body, (50, 50), (-50, 50), thickness),
                pymunk.Segment(static_body, (-50, 50), (-50, -50), thickness),
            ]

        # Set properties for each wall
        for wall in walls:
            wall.elasticity = elasticity
            wall.friction = friction

        # Add the static body and walls to the space
        self.space.add(static_body, *walls)

        # Create a dynamic body for the circle
        mass = 1
        radius = 20
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        dynamic_body = pymunk.Body(mass, inertia)
        dynamic_body.position = self.position_offset

        # Create the circle shape
        circle_shape = pymunk.Circle(dynamic_body, radius)
        circle_shape.elasticity = elasticity
        circle_shape.friction = friction

        # Add the dynamic body and its shape to the space
        self.space.add(dynamic_body, circle_shape)

        # Reference to the dynamic body for later use
        self.circle_body = dynamic_body

    def apply_gravity(self):
        # Define the global gravity vector as a 3D vector
        global_gravity = np.array([0, -GRAVITY_MAGNITUDE, 0])

        # Rotate the gravity vector by the quaternion
        rotated_gravity = self.gravity_quaternion.rotate(global_gravity)

        # Project the rotated gravity onto the simulation's plane
        if self.projection_plane == 'XY':
            gravity_2d = (rotated_gravity[0], rotated_gravity[1])
        elif self.projection_plane == 'XZ':
            gravity_2d = (rotated_gravity[0], rotated_gravity[2])
        elif self.projection_plane == 'YZ':
            gravity_2d = (rotated_gravity[1], rotated_gravity[2])

        # Apply gravity to the Pymunk space
        self.space.gravity = gravity_2d

    def step(self, dt):
        self.apply_gravity()
        self.space.step(dt)
        self.update_state()

    def update_state(self):
        pos = self.circle_body.position + np.array(self.position_offset)
        vel = self.circle_body.velocity
        angle = self.circle_body.angle
        # Convert angle to degrees for Quaternion
        rotation = Quaternion(axis=[0,0,1], degrees=np.degrees(angle))
        self.state = State(pos, rotation, vel)

    def get_state(self):
        return self.state

class StateNegotiator:
    def __init__(self, authorities):
        self.authorities = authorities
        self.current_state = None

    def negotiate(self):
        positions = []
        rotations = []
        velocities = []
        weights = []

        for authority in self.authorities:
            state = authority.get_state()
            if state is None:
                continue
            # Weight based on velocity magnitude
            weight = np.linalg.norm(state.velocity) + 1  # +1 to avoid zero weight
            positions.append(state.position * weight)
            rotations.append(state.rotation)
            velocities.append(state.velocity * weight)
            weights.append(weight)

        if not weights:
            return

        total_weight = sum(weights)
        avg_position = sum(positions) / total_weight
        avg_velocity = sum(velocities) / total_weight
        avg_rotation = quaternion_average(rotations, weights)

        self.current_state = State(avg_position, avg_rotation, avg_velocity)

    def get_current_state(self):
        return self.current_state

class GravitySlice:
    def __init__(self, initial_quaternion):
        self.quaternion = initial_quaternion

    def adjust_orientation(self, axis, angle_deg):
        q = Quaternion(axis=axis, degrees=angle_deg)
        self.quaternion = q * self.quaternion
        self.quaternion = self.quaternion.normalised

class Clock:
    def __init__(self):
        self.last_time = time.time()

    def tick(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        return dt

# Initialize State Authorities with Orthogonal Projections
def create_authorities():
    authorities = []
    # Simulation 1: XY Plane
    space_xy = pymunk.Space()
    gravity_quaternion_xy = Quaternion()  # Identity quaternion
    authority_xy = StateAuthority(space_xy, 'XY', gravity_quaternion_xy, position_offset=(0,0))
    authorities.append(authority_xy)
    # Simulation 2: XZ Plane
    space_xz = pymunk.Space()
    gravity_quaternion_xz = Quaternion()  # Identity quaternion
    authority_xz = StateAuthority(space_xz, 'XZ', gravity_quaternion_xz, position_offset=(0,-1))
    authorities.append(authority_xz)
    # Simulation 3: YZ Plane
    space_yz = pymunk.Space()
    gravity_quaternion_yz = Quaternion()  # Identity quaternion
    authority_yz = StateAuthority(space_yz, 'YZ', gravity_quaternion_yz, position_offset=(0,0))
    authorities.append(authority_yz)
    return authorities

# Initialize Gravity Slices
gravity_slices = [
    GravitySlice(Quaternion()),  # Gravity slice 1
    GravitySlice(Quaternion()),  # Gravity slice 2
]

# Initialize Authorities and State Negotiator
authorities = create_authorities()
state_negotiator = StateNegotiator(authorities)

# Initialize Clock
global_clock = Clock()

# Function to Handle User Input
def handle_input(gravity_slices):
    keys = pygame.key.get_pressed()
    # Define rotation speed
    rotation_speed = 1  # Degrees per input

    # Gravity Slice 1 Controls (W, A, S, D)
    if keys[pygame.K_w]:
        gravity_slices[0].adjust_orientation([0,0,1], rotation_speed)  # Rotate upward around Z-axis
    if keys[pygame.K_s]:
        gravity_slices[0].adjust_orientation([0,0,1], -rotation_speed)  # Rotate downward around Z-axis
    if keys[pygame.K_a]:
        gravity_slices[0].adjust_orientation([0,0,1], rotation_speed)  # Rotate left around Z-axis
    if keys[pygame.K_d]:
        gravity_slices[0].adjust_orientation([0,0,1], -rotation_speed)  # Rotate right around Z-axis

    # Gravity Slice 2 Controls (Arrow Keys)
    if keys[pygame.K_UP]:
        gravity_slices[1].adjust_orientation([0,0,1], rotation_speed)  # Rotate upward around Z-axis
    if keys[pygame.K_DOWN]:
        gravity_slices[1].adjust_orientation([0,0,1], -rotation_speed)  # Rotate downward around Z-axis
    if keys[pygame.K_LEFT]:
        gravity_slices[1].adjust_orientation([0,0,1], rotation_speed)  # Rotate left around Z-axis
    if keys[pygame.K_RIGHT]:
        gravity_slices[1].adjust_orientation([0,0,1], -rotation_speed)  # Rotate right around Z-axis

# Function to Update Gravity Slices in Authorities
def update_authorities_gravity(authorities, gravity_slices):
    # Combine gravity slices by averaging their quaternions
    if len(gravity_slices) == 0:
        return
    combined_quaternion = quaternion_average([gs.quaternion for gs in gravity_slices],
                                            [1 for _ in gravity_slices])
    for authority in authorities:
        authority.gravity_quaternion = combined_quaternion

# Rendering Function
def render(screen, state_negotiator, authorities):
    screen.fill((255, 255, 255))  # White background

    # Draw Center Box
    pygame.draw.rect(screen, (0, 0, 0), CENTER_BOX_RECT, 2)

    # Draw StateNegotiator's current state in center box
    current_state = state_negotiator.get_current_state()
    if current_state:
        pos = current_state.position
        # Calculate the position to draw the circle in center box
        draw_x, draw_y = simulation_to_screen(pos, CENTER_BOX_RECT)
        pygame.draw.circle(screen, (255, 0, 0), (draw_x, draw_y), 15)
        
        # Rotate a 2D vector based on rotation angle
        rotated_vector = current_state.rotation.rotate([30, 0, 0])  # 3D vector
        rotated_vector_2d = rotated_vector[:2]  # Extract x and y
        end_pos = pos + rotated_vector_2d
        end_draw_x, end_draw_y = simulation_to_screen(end_pos, CENTER_BOX_RECT)
        
        # Draw the orientation line
        pygame.draw.line(screen, (0, 0, 255), 
                         (draw_x, draw_y),
                         (end_draw_x, end_draw_y), 2)

    # Define small boxes and their titles
    small_boxes = [
        (SMALL_BOX_1, "XY Plane Simulation"),
        (SMALL_BOX_2, "XZ Plane Simulation"),
        (SMALL_BOX_3, "YZ Plane Simulation")
    ]

    # For each StateAuthority, draw their simulation in their small box
    for authority, (box, title) in zip(authorities, small_boxes):
        # Draw box border
        pygame.draw.rect(screen, (0, 0, 0), box, 2)
        
        # Draw title
        font = pygame.font.SysFont(None, 24)
        text_surface = font.render(title, True, (0, 0, 0))
        screen.blit(text_surface, (box.x + 10, box.y + 10))
        
        # Draw all shapes in the authority's space
        for shape in authority.space.shapes:
            if isinstance(shape, pymunk.Segment):
                body = shape.body
                p1 = shape.a.rotated(body.angle) + body.position
                p2 = shape.b.rotated(body.angle) + body.position
                # Convert to screen coordinates
                screen_p1 = simulation_to_screen(p1, box)
                screen_p2 = simulation_to_screen(p2, box)
                pygame.draw.line(screen, (0, 0, 0), screen_p1, screen_p2, int(shape.radius))
            elif isinstance(shape, pymunk.Circle):
                body = shape.body
                pos = body.position
                radius = int(shape.radius * SCALE)
                screen_pos = simulation_to_screen(pos, box)
                pygame.draw.circle(screen, (0, 0, 255), screen_pos, radius)
                # Draw orientation line based on body angle
                angle = body.angle
                end_vector = [30 * np.cos(angle), 30 * np.sin(angle)]
                end_pos = pos + np.array(end_vector)
                screen_end_pos = simulation_to_screen(end_pos, box)
                pygame.draw.line(screen, (255, 0, 0), screen_pos, screen_end_pos, 2)

    pygame.display.flip()

# Main Function
def main():
    running = True
    while running:
        dt = global_clock.tick()  # Time since last tick in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        handle_input(gravity_slices)
        update_authorities_gravity(authorities, gravity_slices)

        # Step all authorities
        for authority in authorities:
            authority.step(dt)

        # Negotiate the state
        state_negotiator.negotiate()

        # Render the current state and simulations
        render(screen, state_negotiator, authorities)

        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
