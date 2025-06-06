import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from laplace2 import GridDomain, BuildLaplace3D, RectangularTransform
from scipy.sparse.linalg import cg  

def scipy_to_torch_sparse(scipy_coo, device):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    values = torch.tensor(scipy_coo.data, dtype=torch.float64, device=device)
    indices = torch.tensor(np.vstack((scipy_coo.row, scipy_coo.col)), dtype=torch.long, device=device)
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)

class BeamSimulator:
    def __init__(self, length=1.0, width=1.0, height=1.0, resolution=(30, 10, 10), device='cpu'):
        self.length, self.width, self.height = length, width, height
        self.resolution = resolution
        self.device = device

        # Grid and Laplacian setup
        transform = RectangularTransform(Lx=length, Ly=width, Lz=height, device=device)
        self.grid_u, self.grid_v, self.grid_w = transform.create_grid_mesh(*resolution)
        self.grid_domain = GridDomain(self.grid_u, self.grid_v, self.grid_w, transform=transform)

        self.laplace_builder = BuildLaplace3D(grid_domain=self.grid_domain, resolution=resolution[0])
        _, self.laplace_tensor_sparse = self.laplace_builder.build_general_laplace(
            grid_u=self.grid_u, grid_v=self.grid_v, grid_w=self.grid_w, device=device
        )
        
        self.laplace_tensor_sparse = scipy_to_torch_sparse(self.laplace_tensor_sparse, self.device)

        # Beam state
        self.state = torch.zeros(self.grid_u.shape, device=device, dtype=torch.float64)
        self.velocity = torch.zeros_like(self.state)

    def apply_forces(self, forces):
        """Apply external forces as input tensor."""
        self.state += forces

    def evolve_state(self, forces_list, time_steps=50, dt=0.01, damping=0.01):
        """
        Evolves the beam state under spontaneous forces with Neumann boundaries.
        
        Args:
            forces_list (list of tensors): Sequence of force vectors applied at different time steps.
            time_steps (int): Number of time steps.
            dt (float): Time step.
            damping (float): Damping coefficient.
        """
        states = []

        for t in range(time_steps):
            # Apply forces for the current time step
            if t < len(forces_list):
                self.apply_forces(forces_list[t])

            # Solve Laplacian (acceleration) with Neumann boundaries
            state_flat = self.state.flatten().unsqueeze(1)
            acceleration_flat = torch.sparse.mm(self.laplace_tensor_sparse, state_flat)
            acceleration = acceleration_flat.reshape(self.state.shape)

            # Neumann boundaries: transmit forces at edges
            self.state[0, :, :] = self.state[1, :, :]
            self.state[-1, :, :] = self.state[-2, :, :]
            
            # Update velocity and state
            self.velocity += dt * acceleration - damping * self.velocity
            self.state += dt * self.velocity

            states.append(self.state.clone().cpu().numpy())

        return states

    def visualize_3d(self, states, interval=50):
        """Visualize 3D beam deformation as an animation."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(np.linspace(0, self.length, self.resolution[0]),
                           np.linspace(0, self.width, self.resolution[1]))

        def update(frame):
            ax.clear()
            Z = states[frame][:, :, self.resolution[2] // 2]  # Mid-plane slice
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_zlim(-0.05, 0.05)
            ax.set_title(f"Beam Deflection at Step {frame}")

        ani = FuncAnimation(fig, update, frames=len(states), interval=interval)
        plt.show()

if __name__ == "__main__":
    # Initialize the beam
    beam_sim = BeamSimulator(resolution=(30, 30, 30), device='cpu')

    # Define spontaneous forces
    forces_list = []
    for t in range(50):
        forces = torch.zeros_like(beam_sim.state)
        if t % 10 == 0:  # Every 10 steps, apply a force at random locations
            x, y, z = np.random.randint(5, 25, size=3)
            forces[x, y, z] = np.random.uniform(0.15, 2.1)
        forces_list.append(forces)

    # Simulate the beam deformation
    states = beam_sim.evolve_state(forces_list, time_steps=50, dt=0.01, damping=0.02)

    # Visualize results
    beam_sim.visualize_3d(states)
