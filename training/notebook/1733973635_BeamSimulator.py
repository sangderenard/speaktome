import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from laplace2 import GridDomain, BuildLaplace3D, RectangularTransform
from scipy.sparse.linalg import cg  
import pygame
def scipy_to_torch_sparse(scipy_coo, device):
    values = torch.tensor(scipy_coo.data, dtype=torch.float64, device=device)
    indices = torch.tensor(np.vstack((scipy_coo.row, scipy_coo.col)), dtype=torch.long, device=device)
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)

class BeamSimulator:
    def __init__(self, length=1.0, width=1.0, height=1.0, resolution=(30, 10, 10), device='cpu'):
        self.length, self.width, self.height = length, width, height
        self.resolution = resolution
        self.device = device

        # Generate 3D grid
        transform = RectangularTransform(Lx=length, Ly=width, Lz=height, device=device)
        self.grid_u, self.grid_v, self.grid_w = transform.create_grid_mesh(*resolution)
        self.grid_domain = GridDomain(self.grid_u, self.grid_v, self.grid_w, transform=transform)

        # Laplace-Beltrami solver
        self.laplace_builder = BuildLaplace3D(grid_domain=self.grid_domain, resolution=resolution[0])
        self.laplace_tensor_dense, self.laplace_tensor_sparse = self.laplace_builder.build_general_laplace(
            grid_u=self.grid_u, grid_v=self.grid_v, grid_w=self.grid_w, device=device
        )

        # Initialize state tensor
        self.state = torch.zeros(self.grid_u.shape, device=device, dtype=torch.float64)

    def evolve_state(self, time_steps=50, dt=0.01, freq=2.0, amplitude=0.1, 
                    boundary="neumann", steady_state=False, chaotic=False, noise_scale=0.01, damping=0.01):
        """
        Simulates the evolution of the beam deformation state using both dense and sparse Laplacians.

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
            tuple: Two lists of states (sparse version, dense version).
        """
        states_sparse = []
        states_dense = []
        center_u_idx = self.resolution[0] // 3  # Beam center index
        
        if steady_state:
            # Solve directly for steady-state: Laplacian(state) + forcing = 0
            forcing = torch.zeros_like(self.state)
            forcing[center_u_idx, :, :] = amplitude

            forcing_flat = forcing.flatten()

            # Sparse solver
            steady_state_sparse_flat, _ = cg(self.laplace_tensor_sparse, -forcing_flat.cpu().numpy())
            steady_state_sparse = torch.tensor(steady_state_sparse_flat, device=self.device).reshape(self.state.shape)

            # Dense solver
            steady_state_dense_flat = torch.linalg.solve(self.laplace_tensor_dense, -forcing_flat.unsqueeze(1))
            steady_state_dense = steady_state_dense_flat.reshape(self.state.shape)

            states_sparse.append(steady_state_sparse.cpu().numpy())
            states_dense.append(steady_state_dense.cpu().numpy())

            return states_sparse, states_dense

        # Time evolution setup
        velocity_sparse = torch.zeros_like(self.state, dtype=self.state.dtype, device=self.device)
        velocity_dense = torch.zeros_like(self.state, dtype=self.state.dtype, device=self.device)

        for t in range(time_steps):
            # Sinusoidal forcing
            sinusoidal_force = amplitude * torch.sin(2 * torch.pi * freq * torch.tensor([t]) * dt)
            self.state[center_u_idx, :, :] += sinusoidal_force

            # Add chaotic noise if enabled
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

            # Solve Laplacians for acceleration
            state_flat = self.state.flatten()

            # Sparse Laplacian
            laplace_sparse_flat = torch.sparse.mm(scipy_to_torch_sparse(self.laplace_tensor_sparse, self.device), state_flat.unsqueeze(1))
            acceleration_sparse = laplace_sparse_flat.reshape(self.state.shape)

            # Dense Laplacian
            laplace_dense_flat = torch.mm(self.laplace_tensor_dense, state_flat.unsqueeze(1))
            acceleration_dense = laplace_dense_flat.reshape(self.state.shape)

            # Integrate for sparse version
            velocity_sparse += dt * acceleration_sparse - damping * velocity_sparse
            state_sparse = self.state + dt * velocity_sparse

            # Integrate for dense version
            velocity_dense += dt * acceleration_dense - damping * velocity_dense
            state_dense = self.state + dt * velocity_dense

            # Store results
            states_sparse.append(state_sparse.cpu().numpy())
            states_dense.append(state_dense.cpu().numpy())

        return states_sparse, states_dense





    def project_to_2d(self, state_3d):
        """
        Projects a 3D beam state into 2D by summing or averaging along the 'height' axis.
        """
        return state_3d.mean(axis=2)  # Average along the z-axis (height)


    def visualize(self, states_sparse, states_dense):
        """
        Visualizes side-by-side comparisons of sparse and dense Laplacian outputs.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Select the final time step for comparison
        projection_sparse = self.project_to_2d(states_sparse[-1])
        projection_dense = self.project_to_2d(states_dense[-1])
        difference = projection_dense - projection_sparse

        # Sparse Result
        axes[0].imshow(projection_sparse, cmap='viridis', origin='lower', extent=[0, self.length, 0, self.width])
        axes[0].set_title("Sparse Laplacian Result")

        # Dense Result
        axes[1].imshow(projection_dense, cmap='viridis', origin='lower', extent=[0, self.length, 0, self.width])
        axes[1].set_title("Dense Laplacian Result")

        # Difference
        im = axes[2].imshow(difference, cmap='seismic', origin='lower', extent=[0, self.length, 0, self.width])
        axes[2].set_title("Difference (Dense - Sparse)")
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    beam_sim = BeamSimulator(resolution=(30, 30, 30))
    states_sparse, states_dense = beam_sim.evolve_state(steady_state=True, amplitude=0.1)
    beam_sim.visualize(states_sparse, states_dense)

    states_sparse, states_dense = beam_sim.evolve_state(time_steps=50, dt=0.01, chaotic=True, noise_scale=0.05, damping=0.02)
    beam_sim.visualize(states_sparse, states_dense)

    states_sparse, states_dense = beam_sim.evolve_state(time_steps=50, dt=0.01, freq=1.0, amplitude=0.05, damping=0.01)
    beam_sim.visualize(states_sparse, states_dense)
