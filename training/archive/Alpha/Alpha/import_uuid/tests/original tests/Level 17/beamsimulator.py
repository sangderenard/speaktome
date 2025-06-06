import pygame
import torch
from pygame.locals import QUIT, KEYDOWN, K_w, K_a, K_s, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT
import sys

import torch

class LeafSpringHelper:
    def __init__(self, E, G, geometry, material, k=5/6):
        """
        Initialize the beam helper.

        Parameters:
            E (float): Young's modulus (Pa).
            G (float): Shear modulus (Pa).
            geometry (dict): Geometric properties {type: "solid"|"hollow", shape: "rect"|"circle", dimensions}.
                - Example for solid rectangle: {"type": "solid", "shape": "rect", "width": w, "height": h}.
            material (dict): Beam material properties like density.
            k (float): Shear correction factor (default for rectangular cross-section).
        """
        self.E = E  # Young's modulus
        self.G = G  # Shear modulus
        self.k = k  # Shear correction factor
        self.geometry = geometry
        self.I, self.A = self._calculate_inertia_area(geometry)

    def _calculate_inertia_area(self, geometry):
        """Calculate moment of inertia and cross-sectional area."""
        if geometry["type"] == "solid":
            if geometry["shape"] == "rect":
                b, h = geometry["width"], geometry["height"]
                I = (b * h**3) / 12
                A = b * h
            elif geometry["shape"] == "circle":
                r = geometry["radius"]
                I = (torch.pi * r**4) / 4
                A = torch.pi * r**2
        elif geometry["type"] == "hollow":
            if geometry["shape"] == "rect":
                b_o, h_o, b_i, h_i = geometry["outer_width"], geometry["outer_height"], geometry["inner_width"], geometry["inner_height"]
                I = ((b_o * h_o**3) - (b_i * h_i**3)) / 12
                A = (b_o * h_o) - (b_i * h_i)
            elif geometry["shape"] == "circle":
                r_o, r_i = geometry["outer_radius"], geometry["inner_radius"]
                I = (torch.pi * (r_o**4 - r_i**4)) / 4
                A = torch.pi * (r_o**2 - r_i**2)
        else:
            raise ValueError("Unsupported geometry.")
        return I, A

    def compute_piecewise_deflection(self, forces, clamp_regions, x_positions):
        """
        Compute piecewise beam deflection.

        Parameters:
            forces (list): List of forces [(position, magnitude)].
            clamp_regions (list): List of (start, end) clamp regions.
            x_positions (tensor): Positions along the beam.

        Returns:
            tensor: Deflection values at x_positions.
        """
        deflection = torch.zeros_like(x_positions, dtype=torch.float64)
        kGA = self.k * self.G * self.A  # Shear stiffness

        for start, end in clamp_regions:
            region_mask = (x_positions >= start) & (x_positions <= end)
            x_region = x_positions[region_mask] - start

            for pos, F in forces:
                if start <= pos <= end:
                    local_force_pos = pos - start
                    shear_term = (F / kGA) * x_region
                    bending_term = (F / (6 * self.E * self.I)) * (x_region**3) - (F * local_force_pos / (2 * self.E * self.I)) * (x_region**2)
                    deflection[region_mask] += shear_term + bending_term
        return deflection

class Edge:
    def __init__(self, node_a, attach_a, node_b, attach_b, edge_type, parameters):
        self.node_a = node_a
        self.node_b = node_b
        self.attach_a = attach_a
        self.attach_b = attach_b
        self.edge_type = edge_type
        self.parameters = parameters

        if edge_type == "leaf_spring":
            self.helper = LeafSpringHelper(
                E=parameters["E"],
                G=parameters["G"],
                geometry=parameters["geometry"],
                material=parameters["material"]
            )

    def compute_bend_angle(self, load_type="center", support_type="ends", offsets=None):
        """Compute a bend angle based on loads."""
        if self.edge_type != "leaf_spring":
            raise NotImplementedError("Bend angle computation is for leaf spring edges only.")
        
        # Use simplified heuristic: angle proportional to max deflection
        forces = self.parameters["forces"]
        clamp_regions = self.parameters["clamp_regions"]
        x_positions = torch.linspace(0, self.parameters["length"], 100, dtype=torch.float64)

        deflection = self.helper.compute_piecewise_deflection(forces, clamp_regions, x_positions)
        max_deflection = deflection.abs().max()
        return torch.atan(max_deflection / self.parameters["length"])  # Small-angle approximation

    def compute_forces(self):
        """Evaluate and return forces for the leaf spring."""
        if self.edge_type == "leaf_spring":
            forces = self.parameters["forces"]
            clamp_regions = self.parameters["clamp_regions"]
            x_positions = torch.linspace(0, self.parameters["length"], 100, dtype=torch.float64)

            # Compute deflection as forces acting through the edge
            deflection = self.helper.compute_piecewise_deflection(forces, clamp_regions, x_positions)
            return deflection



# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SCREEN_CENTER_Y = SCREEN_HEIGHT // 2
BEAM_COLOR = (0, 100, 255)
BACKGROUND_COLOR = (20, 20, 20)
FPS = 60

# Pygame Setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Interactive Beam Deflection")
clock = pygame.time.Clock()

# Helper Functions
def render_deflected_beam(x_positions, deflections, beam_length, clamp_x, batch_size):
    """Render the deflected beam as small rectangles."""
    rect_width = SCREEN_WIDTH / beam_length / batch_size  # Scale for pixelated view

    for i, x in enumerate(x_positions):
        y_deflect = deflections[i].item() * 100  # Scale deflection for visibility
        rect_x = int(clamp_x + x * SCREEN_WIDTH / beam_length)
        rect_y = int(SCREEN_CENTER_Y + y_deflect)

        pygame.draw.rect(screen, BEAM_COLOR, (rect_x, rect_y, rect_width, 5))  # Small rectangles

def main():
    # Beam properties
    beam_length = 0.750  # 1 meter
    batch_size = 100  # Number of segments to evaluate
    x_positions = torch.linspace(0, beam_length, batch_size, dtype=torch.float64)

    # Initial beam characteristics
    parameters = {
        "E": 2.1e11,  # Young's modulus (Pa)
        "G": 8.1e10,  # Shear modulus (Pa)
        "geometry": {"type": "solid", "shape": "rect", "width": 0.01, "height": 0.005},
        "material": {"density": 7850},
        "forces": [[0.5, 500]],  # Force at middle with 500N
        "clamp_regions": [(0.0, 0.03)],  # Fixed clamp on the left
        "length": beam_length
    }

    leaf_spring_helper = LeafSpringHelper(
        E=parameters["E"],
        G=parameters["G"],
        geometry=parameters["geometry"],
        material=parameters["material"]
    )

    force_position = 0.5
    force_magnitude = 5000

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                # Adjust beam geometry
                if event.key == K_w:
                    parameters["geometry"]["height"] += 0.001  # Increase height
                if event.key == K_s and parameters["geometry"]["height"] > 0.001:
                    parameters["geometry"]["height"] -= 0.001  # Decrease height
                if event.key == K_d:
                    parameters["geometry"]["width"] += 0.001  # Increase width
                if event.key == K_a and parameters["geometry"]["width"] > 0.001:
                    parameters["geometry"]["width"] -= 0.001  # Decrease width
                
                # Adjust force position and magnitude
                if event.key == K_RIGHT and force_position < beam_length:
                    force_position += 0.05
                if event.key == K_LEFT and force_position > 0.0:
                    force_position -= 0.05
                if event.key == K_UP:
                    force_magnitude += 5000
                if event.key == K_DOWN and force_magnitude > 5000:
                    force_magnitude -= 5000

        # Update forces in the helper
        parameters["forces"] = [[force_position, force_magnitude]]
        leaf_spring_helper = LeafSpringHelper(
            E=parameters["E"],
            G=parameters["G"],
            geometry=parameters["geometry"],
            material=parameters["material"]
        )

        # Compute deflection
        forces = torch.tensor(parameters["forces"], dtype=torch.float64)
        clamp_regions = parameters["clamp_regions"]
        deflections = leaf_spring_helper.compute_piecewise_deflection(forces, clamp_regions, x_positions)
        print(f"deflections max: {torch.max(deflections)} forces max {torch.max(forces)}")
        # Pygame rendering
        screen.fill(BACKGROUND_COLOR)
        render_deflected_beam(x_positions, deflections, beam_length, clamp_x=50, batch_size=batch_size)

        # Display updates
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
