# visualization.py

import pygame
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Initialize Pygame
pygame.init()

# Define window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
WINDOW_TITLE = "Graph Network Visualization"

# Create the Pygame window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption(WINDOW_TITLE)

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
EDGE_COLOR = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)

# Define node properties
NODE_RADIUS = 10

# Define font for text
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 14)

# Clock to control the frame rate
clock = pygame.time.Clock()
FPS = 3  # Frames per second

# Define node box dimensions
NODE_BOX_WIDTH = 100
NODE_BOX_HEIGHT = 80

# ===========================
# 3. Helper Functions for Visualization
# ===========================

def draw_graph(screen, positions, node_states, error_magnitudes_with_bounds, error_magnitudes_without_bounds, simulators, include_bounds=True, cmap='jet', network=None):
    """
    Draws a grid of node status summaries on the Pygame screen.

    Args:
        screen (pygame.Surface): The Pygame screen surface.
        positions (dict): Dictionary mapping node indices to (x, y) positions.
        node_states (torch.Tensor): Tensor containing node state vectors.
        error_magnitudes_with_bounds (torch.Tensor): Error magnitudes including bound axes.
        error_magnitudes_without_bounds (torch.Tensor): Error magnitudes excluding bound axes.
        simulators (list): List of RandomWalkSimulator instances.
        include_bounds (bool): Whether to include bound axes in error magnitude calculation.
        cmap (str): Matplotlib colormap name for node coloring.
        network (PillbugNetwork): The top-level network instance.
    """
    # Compute background color based on total system temperature, total radiation, and connectivity
    if network:
        # Normalize values to [0, 1]
        temp_norm = min(network.total_temperature() / network.initial_total_temperature, 1.0)
        rad_norm = min(network.total_radiation() / network.initial_total_radiation, 1.0) if network.initial_total_radiation > 0 else 0
        conn_norm = min(network.connectivity / network.max_connectivity, 1.0)

        # Convert to RGB scale [0, 255]
        background_color = (
            np.clip(int(temp_norm * 255), 0, 255),
            np.clip(int(rad_norm * 255), 0, 255),
            np.clip(int(conn_norm * 255), 0, 255)
        )
    else:
        background_color = WHITE

    screen.fill(background_color)  # Fill screen with computed background color

    # Normalize error magnitudes for color mapping
    norm_with_bounds = mcolors.Normalize(vmin=error_magnitudes_with_bounds.min().item(),
                                         vmax=error_magnitudes_with_bounds.max().item())
    norm_without_bounds = mcolors.Normalize(vmin=error_magnitudes_without_bounds.min().item(),
                                            vmax=error_magnitudes_without_bounds.max().item())
    cmap = plt.get_cmap(cmap)

    for node_idx, pos in positions.items():
        # Determine colors based on error magnitudes
        error_with_bounds = error_magnitudes_with_bounds[node_idx].item()
        error_without_bounds = error_magnitudes_without_bounds[node_idx].item()

        color_with_bounds = cmap(norm_with_bounds(error_with_bounds))  # Returns RGBA
        color_without_bounds = cmap(norm_without_bounds(error_without_bounds))

        color_with_bounds = tuple(int(255 * c) for c in color_with_bounds[:3])  # Convert to RGB
        color_without_bounds = tuple(int(255 * c) for c in color_without_bounds[:3])

        # Draw split rectangle (box) for the node
        rect_left = pygame.Rect(pos[0], pos[1], NODE_BOX_WIDTH // 2, NODE_BOX_HEIGHT)
        rect_right = pygame.Rect(pos[0] + NODE_BOX_WIDTH // 2, pos[1], NODE_BOX_WIDTH // 2, NODE_BOX_HEIGHT)

        pygame.draw.rect(screen, color_with_bounds, rect_left)
        pygame.draw.rect(screen, color_without_bounds, rect_right)

        # Draw borders
        pygame.draw.rect(screen, BLACK, rect_left, 1)
        pygame.draw.rect(screen, BLACK, rect_right, 1)

        # Get number of bound and unbound axes
        sim = simulators[node_idx]
        bounds = sim.config.bounds
        num_bound_axes = int(torch.sum(bounds).item())
        num_unbound_axes = bounds.numel() - num_bound_axes

        # Render node ID, temperature, and bound/unbound axes
        text_id = FONT.render(f"Node {node_idx}", True, TEXT_COLOR)
        text_temp = FONT.render(f"Temp: {sim.temperature:.2f}", True, TEXT_COLOR)
        text_bounds = FONT.render(f"Bound: {num_bound_axes}", True, TEXT_COLOR)
        text_unbound = FONT.render(f"Unbound: {num_unbound_axes}", True, TEXT_COLOR)

        # Position text inside the box
        text_x = pos[0] + 5  # Small padding
        text_y = pos[1] + 5

        screen.blit(text_id, (text_x, text_y))
        screen.blit(text_temp, (text_x, text_y + 15))
        screen.blit(text_bounds, (text_x, text_y + 30))
        screen.blit(text_unbound, (text_x, text_y + 45))

        # Draw tiny circles representing acceleration of each feature
        acc = sim.acceleration
        num_features = acc.numel()
        circle_radius = 5
        circle_spacing = (NODE_BOX_WIDTH - 10) // num_features  # Spacing between circles

        for i in range(num_features):
            acc_value = acc[i].item()
            # Map acceleration value to color intensity
            acc_norm = min(abs(acc_value) / 10.0, 1.0)
            acc_color = (int(255 * acc_norm), 0, int(255 * (1 - acc_norm)))  # Red to blue gradient
            circle_x = pos[0] + 5 + i * circle_spacing
            circle_y = pos[1] + NODE_BOX_HEIGHT - 10
            pygame.draw.circle(screen, acc_color, (circle_x, circle_y), circle_radius)

    pygame.display.flip()  # Update the full display Surface to the screen

def generate_grid_positions(num_nodes, box_width, box_height, margin=50):
    """
    Generates grid positions for nodes.

    Args:
        num_nodes (int): Number of nodes.
        box_width (int): Width of each box.
        box_height (int): Height of each box.
        margin (int): Margin from the window edges.

    Returns:
        dict: Mapping from node index to (x, y) positions.
    """
    grid_cols = int(np.ceil(np.sqrt(num_nodes)))
    grid_rows = int(np.ceil(num_nodes / grid_cols))

    positions = {}
    node_idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if node_idx >= num_nodes:
                break
            x = margin + col * (box_width + margin)
            y = margin + row * (box_height + margin)
            positions[node_idx] = [x, y]
            node_idx += 1
    return positions

def compute_error_magnitudes(node_states, simulators, include_bounds=True):
    """
    Computes error magnitudes for all nodes, optionally including bound axes.

    Args:
        node_states (torch.Tensor): Tensor containing node state vectors.
        simulators (list): List of RandomWalkSimulator instances.
        include_bounds (bool): Whether to include bound axes in error magnitude calculation.

    Returns:
        torch.Tensor: Tensor containing error magnitudes.
    """
    error_magnitudes = []
    for node_idx, state in enumerate(node_states):
        sim = simulators[node_idx]
        bounds = sim.config.bounds
        if include_bounds:
            # Include all axes
            error = torch.sum(torch.abs(state))
        else:
            # Exclude bound axes
            unbound_axes = (bounds == 0)
            error = torch.sum(torch.abs(state * unbound_axes))
        error_magnitudes.append(error.item())
    return torch.tensor(error_magnitudes)
