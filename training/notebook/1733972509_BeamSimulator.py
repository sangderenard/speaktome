import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from laplace import GridDomain, BuildLaplace3D, RectangularTransform
from scipy.sparse.linalg import cg  
import pygame
def scipy_to_torch_sparse(scipy_coo, device):
    values = torch.tensor(scipy_coo.data, dtype=torch.float64, device=device)
    indices = torch.tensor(np.vstack((scipy_coo.row, scipy_coo.col)), dtype=torch.long, device=device)
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)

class BeamSimulator:
    def __init__(self, length=1.0, width=0.1, height=0.1, resolution=(30, 10, 10), device='cpu'):
        self.length, self.width, self.height = length, width, height
        self.resolution = resolution
        self.device = device

        # Generate 3D grid
        transform = RectangularTransform(Lx=length, Ly=width, Lz=height, device=device)
        self.grid_u, self.grid_v, self.grid_w = transform.create_grid_mesh(*resolution)
        self.grid_domain = GridDomain(self.grid_u, self.grid_v, self.grid_w, transform=transform)

        # Laplace-Beltrami solver
        self.laplace_builder = BuildLaplace3D(grid_domain=self.grid_domain, resolution=resolution[0])
        _, self.laplace_tensor = self.laplace_builder.build_general_laplace(
            grid_u=self.grid_u, grid_v=self.grid_v, grid_w=self.grid_w, device=device
        )

        # Initialize state tensor
        self.state = torch.zeros(self.grid_u.shape, device=device, dtype=torch.float64)

    def evolve_state(self, time_steps=50, dt=0.01, freq=2.0, amplitude=0.1, 
                    boundary="neumann", steady_state=False, chaotic=False, noise_scale=0.01, damping=0.01):
        """
        Simulates the evolution of the beam deformation state using velocity-based time integration.

        Args:
            time_steps (int): Number of simulation steps (ignored for steady state).
            dt (float): Time step for simulation.
            freq (float): Frequency of the sinusoidal force.
            amplitude (float): Amplitude of the sinusoidal force.
            boundary (str): Boundary conditions: 'neumann' or 'dirichlet'.
            steady_state (bool): If True, directly compute steady-state solution.
            chaotic (bool): If True, introduce random noise at each step.
            noise_scale (float): Magnitude of the random noise when chaotic is True.
            damping (float): Damping coefficient to dissipate energy over time.

        Returns:
            list: List of states at each time step or the steady-state solution.
        """
        states = []
        center_u_idx = self.resolution[0] // 3  # Beam center index
        
        if steady_state:
            # Solve directly for steady-state: Laplacian(state) + forcing = 0
            #logger.info("Solving for steady state...")
            forcing = torch.zeros_like(self.state)
            forcing[center_u_idx, :, :] = amplitude  # Apply steady forcing
            
            forcing_flat = forcing.flatten()
            steady_state_flat, _ = cg(self.laplace_tensor, -forcing_flat.cpu().numpy())
            steady_state_flat = torch.tensor(steady_state_flat, device=self.device)

            self.state = steady_state_flat.reshape(self.state.shape)
            states.append(self.state.cpu().numpy())
            #logger.info("Steady-state solution computed.")
            return states

        # Time evolution setup
        #logger.info("Running time evolution with velocity integration...")
        velocity = torch.zeros_like(self.state, dtype=self.state.dtype, device=self.device)

        for t in range(time_steps):
            # Sinusoidal forcing at the beam center (replaced np with torch)
            sinusoidal_force = amplitude * torch.sin(2 * torch.pi * freq * torch.tensor([t]) * dt)
            self.state[center_u_idx, :, :] += sinusoidal_force
            
            # Add chaotic noise influence if enabled
            if chaotic:
                noise = noise_scale * torch.randn_like(self.state)
                self.state += noise

            # Apply boundary conditions
            if boundary == "dirichlet":
                self.state[0, :, :] = 0.0
                self.state[-1, :, :] = 0.0
            elif boundary == "neumann":
                self.state[0, :, :] = self.state[1, :, :]
                self.state[-1, :, :] = self.state[-2, :, :]

            # Solve Laplacian for acceleration
            state_flat = self.state.flatten()
            laplace_flat = torch.sparse.mm(scipy_to_torch_sparse(self.laplace_tensor, self.device), state_flat.unsqueeze(1))  # Sparse matmul
            acceleration = laplace_flat.reshape(self.state.shape)  # Reshape back to state shape

            # Update velocity and state using second-order integration
            velocity = velocity + dt * acceleration - damping * velocity  # Damping term added
            self.state = self.state + dt * velocity

            # Store current state for visualization
            states.append(self.state.cpu().numpy())

        return states




    def project_to_2d(self, state_3d):
        """
        Projects a 3D beam state into 2D by summing or averaging along the 'height' axis.
        """
        return state_3d.mean(axis=2)  # Average along the z-axis (height)


    def render_live(self, states, exaggeration=1):
        """
        Renders the beam state evolution in real-time using Pygame.
        """
        # Pygame setup
        pygame.init()
        screen_width, screen_height = 800, 800
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Beam State Evolution")
        clock = pygame.time.Clock()

        # Prepare scale factors
        max_value = np.max(states)  # Max value for normalization
        scale_x = screen_width / self.resolution[0]
        scale_y = screen_height / self.resolution[1]

        running = True
        frame = 0
        while running and frame < len(states):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear screen
            screen.fill((0, 0, 0))

            # Project 3D state to 2D and normalize for rendering
            state_2d = self.project_to_2d(states[frame])
            
            normalized_state = (state_2d / max_value * 255).clip(0, 255).astype(np.uint8)

            # Draw pixels
            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]):
                    color_value = normalized_state[i, j]
                    color = (color_value, color_value, 255 - color_value)  # Blue shades
                    pygame.draw.rect(
                        screen, color, 
                        pygame.Rect(i * scale_x, j * scale_y, scale_x, scale_y)
                    )

            pygame.display.flip()
            clock.tick(60)  # 60 FPS
            frame += 1

        pygame.quit()

    def visualize(self, states, exaggeration=1):
        """
        Animates the beam state evolution with:
        1. Heatmap of displacement.
        2. 3D exaggerated deflection.
        """
        fig = plt.figure(figsize=(12, 6))

        # Subplots for 2D heatmap and 3D deflection
        ax_heatmap = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

        # Initial 2D heatmap
        projection = self.project_to_2d(states[0])
        im = ax_heatmap.imshow(projection, extent=[0, self.length, 0, self.width],
                            origin='lower', cmap='viridis')
        ax_heatmap.set_title("Displacement Heatmap")
        plt.colorbar(im, ax=ax_heatmap)

        # Prepare meshgrid for 3D beam deflection
        X = np.linspace(0, self.length, self.resolution[0])
        Y = np.linspace(0, self.width, self.resolution[1])
        X, Y = np.meshgrid(X, Y)

        # Initialize 3D plot
        Z = np.zeros_like(X)
        surf = ax_3d.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
        ax_3d.set_title("Exaggerated Beam Deflection")
        #ax_3d.set_zlim(-0.5, 0.5)  # Static z-limits for exaggeration consistency

        def update(frame):
            # Update 2D heatmap
            projection = self.project_to_2d(states[frame])
            im.set_data(projection)
            ax_heatmap.set_title(f"Time Step: {frame}")

            # Update 3D beam deflection (with exaggeration)
            Z = states[frame].mean(axis=2) * exaggeration  # Average along height and exaggerate
            Z = Z.T  # Transpose to match X and Y dimensions
            ax_3d.clear()
            ax_3d.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
            ax_3d.set_title("Exaggerated Beam Deflection")
            #ax_3d.set_zlim(-0.5 * exaggeration, 0.5 * exaggeration)

            return [im]

        ani = FuncAnimation(fig, update, frames=len(states), interval=100, blit=False)
        plt.show()

if __name__ == "__main__":
    beam_sim = BeamSimulator(resolution=(20, 10, 10))
    states = beam_sim.evolve_state(steady_state=True, amplitude=0.1)
    beam_sim.visualize(states, exaggeration=1)

    states = beam_sim.evolve_state(time_steps=50, dt=0.01, chaotic=True, noise_scale=0.05, damping=0.02)
    beam_sim.visualize(states, exaggeration=1)

    states = beam_sim.evolve_state(time_steps=50, dt=0.01, freq=1.0, amplitude=0.05, damping=0.01)
    beam_sim.visualize(states, exaggeration=1)
