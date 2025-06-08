import pygame
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 800  # Screen dimensions
FPS = 60
DT = 1 / FPS  # Time step
SCALE = 100  # Scaling for rendering

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physics Puzzle: Collaborative State")
clock = pygame.time.Clock()

# Helper Functions
def project_to_screen(pos_3d):
    """Project 3D coordinates into 2D screen space."""
    scale = SCALE / (pos_3d[2] + 1)  # Simple perspective scaling
    screen_x = WIDTH // 2 + int(pos_3d[0] * scale)
    screen_y = HEIGHT // 2 - int(pos_3d[1] * scale)  # Y is inverted
    return screen_x, screen_y

# Classes
class SliceState:
    def __init__(self, plane, offset):
        self.plane = plane  # 'XY', 'XZ', or 'YZ'
        self.position = np.zeros(2)  # 2D position in the slice plane
        self.velocity = np.zeros(2)  # 2D velocity
        self.acceleration = np.zeros(2)  # Calculated during updates
        self.offset = offset  # 3D offset for rendering

    def update(self, new_position):
        """Update position, velocity, and acceleration."""
        new_velocity = (new_position - self.position) / DT
        self.acceleration = (new_velocity - self.velocity) / DT
        self.velocity = new_velocity
        self.position = new_position

    def propose_change(self):
        """Apply a random force and attraction toward the slice origin."""
        # Random force
        random_force_2d = np.random.uniform(-50, 50, 2)

        # Attraction to origin
        displacement = -self.position  # Vector pointing to the origin
        distance = np.linalg.norm(displacement)
        if distance > 0:  # Avoid division by zero
            attractor_force = (displacement / distance) * (1 / (distance ** 2))
        else:
            attractor_force = np.zeros(2)

        # Combine forces
        total_force = random_force_2d + attractor_force

        # Update state with total force
        self.acceleration = total_force  # Assume unit mass
        new_velocity = self.velocity + self.acceleration * DT
        new_position = self.position + new_velocity * DT

        self.update(new_position)


    def to_3d(self):
        """Map 2D state to 3D coordinates."""
        if self.plane == 'XY':
            return np.array([self.position[0], self.position[1], 0])
        elif self.plane == 'XZ':
            return np.array([self.position[0], 0, self.position[1]])
        elif self.plane == 'YZ':
            return np.array([0, self.position[0], self.position[1]])

class GlobalState:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

    def compute_state(self, slice_states):
        """Calculate the global state by summing forces from slices."""
        total_force = np.zeros(3)
        for slice_state in slice_states:
            # Map 2D acceleration to 3D force
            acc_3d = np.zeros(3)
            if slice_state.plane == 'XY':
                acc_3d[:2] = slice_state.acceleration
            elif slice_state.plane == 'XZ':
                acc_3d[0] = slice_state.acceleration[0]
                acc_3d[2] = slice_state.acceleration[1]
            elif slice_state.plane == 'YZ':
                acc_3d[1:] = slice_state.acceleration

            total_force += acc_3d

        # Update global velocity and position based on total force
        self.velocity += total_force * DT
        self.position += self.velocity * DT

        # Propagate the global position back into slices
        for slice_state in slice_states:
            if slice_state.plane == 'XY':
                slice_state.position = self.position[:2]
            elif slice_state.plane == 'XZ':
                slice_state.position = self.position[[0, 2]]
            elif slice_state.plane == 'YZ':
                slice_state.position = self.position[1:]

            # Reset velocity and acceleration to avoid drift
            slice_state.velocity = np.zeros(2)
            slice_state.acceleration = np.zeros(2)

# Initialize slices and global state
slices = [
    SliceState('XY', offset=np.array([-2, -2, 0])),
    SliceState('XZ', offset=np.array([2, -2, 0])),
    SliceState('YZ', offset=np.array([0, 2, 0]))
]
global_state = GlobalState()

# Visualization
def render_3d_slices():
    """Render the slices and global state in a 3D perspective."""
    screen.fill((255, 255, 255))

    # Draw the global state
    global_screen_pos = project_to_screen(global_state.position)
    pygame.draw.circle(screen, (255, 0, 0), global_screen_pos, 10)

    # Draw each slice
    for slice_state in slices:
        # Slice plane center
        slice_center = global_state.position + slice_state.offset
        slice_center_screen = project_to_screen(slice_center)

        # Draw slice boundary
        pygame.draw.rect(screen, (0, 0, 0), 
                         (*slice_center_screen, 100, 100), 2)

        # Draw the slice's position
        pos_3d = slice_state.to_3d() + slice_state.offset
        pos_screen = project_to_screen(pos_3d)
        pygame.draw.circle(screen, (0, 0, 255), pos_screen, 5)

    pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Each slice proposes a change
    for slice_state in slices:
        slice_state.propose_change()

    # Compute the global state based on slice proposals
    global_state.compute_state(slices)

    # Render the updated positions
    render_3d_slices()

    clock.tick(FPS)

pygame.quit()
