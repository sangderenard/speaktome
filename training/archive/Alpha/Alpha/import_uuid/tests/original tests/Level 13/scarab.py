import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from pyquaternion import Quaternion
import time

# Constants
WIDTH, HEIGHT = 1200, 800  # Increased window size to accommodate multiple displays
CENTER_BOX_RECT = pygame.Rect(500, 300, 200, 200)  # Central box for aggregated state
SMALL_BOX_1 = pygame.Rect(50, 50, 200, 200)        # Top-left simulation
SMALL_BOX_2 = pygame.Rect(50, 550, 200, 200)       # Bottom-left simulation
SMALL_BOX_3 = pygame.Rect(1050, 50, 200, 200)      # Top-right simulation
FPS = 60
GRAVITY_MAGNITUDE = 981  # Pixels per second squared
SCALE = 2  # Scaling factor for rendering

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("State Negotiator Simulation")
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
        self.position = np.array(position, dtype=np.float64)
        self.rotation = rotation  # Quaternion
        self.velocity = np.array(velocity, dtype=np.float64)

class StateAuthority:
    def __init__(self, space, gravity_quaternion, position_offset=(0,0)):
        self.space = space
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

        # Define wall segments
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
        
        # Pymunk uses a 2D gravity vector, so extract the x and y components
        self.space.gravity = (rotated_gravity[0], rotated_gravity[1])
        
    def step(self, dt):
        self.apply_gravity()
        self.space.step(dt)
        self.update_state()

    def update_state(self):
        pos = self.circle_body.position + np.array(self.position_offset)
        vel = self.circle_body.velocity
        angle = self.circle_body.angle
        rotation = Quaternion(axis=[0,0,1], angle=np.degrees(angle))
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
            # Simple weighting: based on velocity magnitude
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

# Initialize State Authorities
def create_authorities():
    authorities = []
    for i in range(3):
        space = pymunk.Space()
        initial_quaternion = Quaternion()  # Identity quaternion
        position_offset = (0, 0)  # Centered
        authority = StateAuthority(space, initial_quaternion, position_offset)
        authorities.append(authority)
    return authorities

# Initialize Gravity Slices
gravity_slices = [
    GravitySlice(Quaternion()),  # Left gravity slice
    GravitySlice(Quaternion()),  # Right gravity slice
]

# Initialize Authorities and State Negotiator
authorities = create_authorities()
state_negotiator = StateNegotiator(authorities)

# Initialize Clock
global_clock = Clock()

# Function to Handle User Input
def handle_input(gravity_slices):
    keys = pygame.key.get_pressed()
    # Left Gravity Slice Controls (WASD)
    if keys[pygame.K_w]:
        gravity_slices[0].adjust_orientation([0,0,1], 1)  # Rotate upward
    if keys[pygame.K_s]:
        gravity_slices[0].adjust_orientation([0,0,1], -1)  # Rotate downward
    if keys[pygame.K_a]:
        gravity_slices[0].adjust_orientation([0,0,1], 1)  # Rotate left
    if keys[pygame.K_d]:
        gravity_slices[0].adjust_orientation([0,0,1], -1)  # Rotate right
    # Right Gravity Slice Controls (Arrow Keys)
    if keys[pygame.K_UP]:
        gravity_slices[1].adjust_orientation([0,0,1], 1)
    if keys[pygame.K_DOWN]:
        gravity_slices[1].adjust_orientation([0,0,1], -1)
    if keys[pygame.K_LEFT]:
        gravity_slices[1].adjust_orientation([0,0,1], 1)
    if keys[pygame.K_RIGHT]:
        gravity_slices[1].adjust_orientation([0,0,1], -1)

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
        draw_x = int(pos[0] + CENTER_BOX_RECT.centerx)
        draw_y = int(pos[1] + CENTER_BOX_RECT.centery)
        pygame.draw.circle(screen, (255, 0, 0), (draw_x, draw_y), 15)
        
        # Rotate a 3D vector and extract 2D components for the orientation line
        rotated_vector = current_state.rotation.rotate([30, 0, 0])  # 3D vector
        end_pos = pos + np.array([rotated_vector[0], rotated_vector[1]])  # Extract x and y
        end_draw_x = int(end_pos[0] + CENTER_BOX_RECT.centerx)
        end_draw_y = int(end_pos[1] + CENTER_BOX_RECT.centery)
        
        # Draw the orientation line
        pygame.draw.line(screen, (0, 0, 255), 
                         (draw_x, draw_y),
                         (end_draw_x, end_draw_y), 2)

    # Define small boxes
    small_boxes = [SMALL_BOX_1, SMALL_BOX_2, SMALL_BOX_3]

    # For each StateAuthority, draw their simulation in their small box
    for authority, box in zip(authorities, small_boxes):
        # Draw box border
        pygame.draw.rect(screen, (0, 0, 0), box, 2)

        # Draw all shapes in the authority's space
        for shape in authority.space.shapes:
            if isinstance(shape, pymunk.Segment):
                body = shape.body
                p1 = body.position + shape.a.rotated(body.angle)
                p2 = body.position + shape.b.rotated(body.angle)
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
        dt = global_clock.tick()
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
