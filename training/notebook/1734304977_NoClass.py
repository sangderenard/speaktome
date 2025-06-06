import torch
import pygame
import math
from matplotlib import pyplot as plt

# Pygame setup
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Rotating Points - Physics Simulation")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

# Font setup
FONT = pygame.font.SysFont("Arial", 20)

# Simulation Parameters
count = 10
RADIUS_1 = torch.rand(count) * 200 + 20
RADIUS_2 = torch.rand(count) * 200 + 20
LENGTH_L = torch.rand(count) * 200 + 20 + RADIUS_1 + RADIUS_2
FREQ_1 = torch.rand(count) * 0.5 + 0.002
FREQ_2 = torch.rand(count) * 0.5 + 0.002
FPS = 60

# Mass and velocity setup
masses = torch.ones(count) * 10.0  # Mass for vertices
velocities = torch.zeros((count, 3))  # Velocities for vertices
delta_t = 1 / FPS  # Time step for integration

# Helper functions
def compute_orbit(radius, angle, center_z):
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    z = center_z
    return x, y, z

def project_to_2d(x, y, z, center_x, center_y):
    screen_x = (center_x + x) % WIDTH
    screen_y = (center_y - z) % HEIGHT
    return int(screen_x), int(screen_y)

def apply_physics(p1, p2, velocities, masses, delta_t, length_l):
    """
    Apply physics: Force corrections adjust velocities to maintain rod constraints.
    """
    direction = p2 - p1
    distance = torch.norm(direction)
    
    # Compute the force to enforce rod constraint
    correction = (distance - length_l) / (distance + 1e-6) * direction
    force = correction / masses[:, None]  # Apply inverse mass
    
    # Adjust velocities based on forces
    velocities[:, 0:3] -= force * delta_t  # Newton's second law F = ma
    
    # Update positions with adjusted velocities
    p1 += velocities[:, 0:3] * delta_t
    p2 -= velocities[:, 0:3] * delta_t

    return p1, p2, velocities

def main():
    global physics_mode
    clock = pygame.time.Clock()
    running = True
    physics_mode = False  # Toggle physics mode with spacebar

    # Angles and positions
    angle_1 = torch.zeros(count)
    angle_2 = torch.zeros(count)

    Z_1 = torch.full((count,), -50.0)
    Z_2 = torch.full((count,), 50.0)

    p1 = torch.zeros((count, 3))
    p2 = torch.zeros((count, 3))

    while running:
        SCREEN.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                physics_mode = not physics_mode

        if not physics_mode:
            # Orbit mode: regular motion
            angle_1 += FREQ_1 * 2 * math.pi / FPS
            angle_2 -= FREQ_2 * 2 * math.pi / FPS
            p1[:, 0], p1[:, 1], p1[:, 2] = compute_orbit(RADIUS_1, angle_1, Z_1)
            p2[:, 0], p2[:, 1], p2[:, 2] = compute_orbit(RADIUS_2, angle_2, Z_2)
        else:
            # Physics mode: apply force communication and update positions
            p1, p2, velocities = apply_physics(p1, p2, velocities, masses, delta_t, LENGTH_L)

        # Draw points and connecting rods
        for i in range(count):
            p1_xz = project_to_2d(p1[i, 0], p1[i, 1], p1[i, 2], WIDTH // 2, CENTER_Y)
            p2_xz = project_to_2d(p2[i, 0], p2[i, 1], p2[i, 2], WIDTH // 2, CENTER_Y)
            pygame.draw.line(SCREEN, GREEN, p1_xz, p2_xz, 2)
            pygame.draw.circle(SCREEN, RED, p1_xz, 5)
            pygame.draw.circle(SCREEN, BLUE, p2_xz, 5)

        mode_text = FONT.render(f"Mode: {'Physics' if physics_mode else 'Orbit'}", True, YELLOW)
        SCREEN.blit(mode_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
