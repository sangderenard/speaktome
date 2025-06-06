import torch
import numpy as np
import imageio
import logging

from laplace import (
    BuildLaplace3D,
    RectangularTransform,
    GridDomain,
    generate_grid,
    generate_full_meshgrid
)

class TemperatureSimulation3D:
    def __init__(self, Lx=4, Ly=4, Lz=1.0, resolution_u=20, resolution_v=20, resolution_w=20, device='cpu'):
        """
        Initializes the 3D temperature simulation.

        Args:
            Lx (float): Length in the x-direction.
            Ly (float): Length in the y-direction.
            Lz (float): Length in the z-direction (thickness).
            resolution_u (int): Grid resolution in the u-direction.
            resolution_v (int): Grid resolution in the v-direction.
            resolution_w (int): Grid resolution in the w-direction.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.resolution_u = resolution_u
        self.resolution_v = resolution_v
        self.resolution_w = resolution_w
        self.device = device

        # Initialize RectangularTransform for 3D
        self.transform = RectangularTransform(Lx=self.Lx, Ly=self.Ly, Lz=self.Lz, device=self.device)

        # Generate grid
        grid_u, grid_v, grid_w = self.transform.create_grid_mesh(self.resolution_u, self.resolution_v, self.resolution_w)
        print(f"Grid U shape: {grid_u.shape}")
        print(f"Grid V shape: {grid_v.shape}")
        print(f"Grid W shape: {grid_w.shape}")

        # Create GridDomain
        self.grid_domain = GridDomain(
            U=grid_u,
            V=grid_v,
            W=grid_w,
            transform=self.transform,
            coordinate_system='rectangular'
        )

        # Initialize BuildLaplace3D with Dirichlet boundary conditions
        boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')  # (u_min, u_max, v_min, v_max, w_min, w_max)

        self.solver = BuildLaplace3D(
            grid_domain=self.grid_domain,
            wave_speed=343,  # Arbitrary value
            precision=torch.float64,
            resolution=resolution_u,  # Assuming cubic grid; adjust if non-cubic
            metric_tensor_func=None,  # Use default Euclidean metric
            density_func=None,        # Uniform density
            tension_func=None,        # Uniform tension
            singularity_conditions=None,
            boundary_conditions=boundary_conditions,
            artificial_stability=1e-10
        )

        # Initialize temperature fields for Red, Green, Blue channels as 3D tensors
        shape = (self.resolution_u, self.resolution_v, self.resolution_w)
        self.T_r = torch.full(shape, 1300.0, dtype=torch.float64, device=self.device)
        self.T_g = torch.full(shape, 1300.0, dtype=torch.float64, device=self.device)
        self.T_b = torch.full(shape, 1300.0, dtype=torch.float64, device=self.device)

        # Initialize hotspots at the center slice (central w-index)
        center_u, center_v, center_w = self.resolution_u // 2, self.resolution_v // 2, self.resolution_w // 2
        self.T_r[center_u, center_v, center_w] = 5000.0
        self.T_g[center_u, center_v, center_w] = 5000.0
        self.T_b[center_u, center_v, center_w] = 5000.0

        # Simulation parameters
        self.time_steps = 100
        self.delta_t = 0.01
        self.thermal_diffusivity = 0.1

        # Step counter
        self.current_step = 0

        # Build Laplacian
        self.build_laplacian()

    def build_laplacian(self):
        """
        Builds the 3D Laplacian matrix using BuildLaplace3D.
        """
        boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')

        # Build the Laplacian
        laplacian_tensor, laplacian_sparse = self.solver.build_general_laplace(
            grid_u=self.grid_domain.U,
            grid_v=self.grid_domain.V,
            grid_w=self.grid_domain.W,
            boundary_conditions=boundary_conditions,
            grid_boundaries=(True, True, True, True, True, True),
            device=self.device,
            f=0.0  # No wave number term for this test
        )

        self.laplacian_tensor = laplacian_tensor  # Dense Laplacian
        self.laplacian_sparse = laplacian_sparse  # Sparse Laplacian

    def apply_gaussian_beam(self, T, center_u, center_v, center_w, amplitude, sigma_u, sigma_v, sigma_w):
        """
        Applies a Gaussian beam excitation to the temperature field.

        Args:
            T (torch.Tensor): Temperature field tensor.
            center_u (int): U-coordinate of the beam center.
            center_v (int): V-coordinate of the beam center.
            center_w (int): W-coordinate of the beam center.
            amplitude (float): Peak temperature increase.
            sigma_u (float): Standard deviation in the U direction.
            sigma_v (float): Standard deviation in the V direction.
            sigma_w (float): Standard deviation in the W direction.
        """
        u = torch.arange(0, self.resolution_u, device=self.device).unsqueeze(1).unsqueeze(2).repeat(1, self.resolution_v, self.resolution_w)
        v = torch.arange(0, self.resolution_v, device=self.device).unsqueeze(0).unsqueeze(2).repeat(self.resolution_u, 1, self.resolution_w)
        w = torch.arange(0, self.resolution_w, device=self.device).unsqueeze(0).unsqueeze(1).repeat(self.resolution_u, self.resolution_v, 1)
        gaussian = amplitude * torch.exp(-(((u - center_u)**2) / (2 * sigma_u**2) +
                                           ((v - center_v)**2) / (2 * sigma_v**2) +
                                           ((w - center_w)**2) / (2 * sigma_w**2)))
        T += gaussian

    def apply_bombardment(self, T, u, v, w, radius, intensity):
        """
        Applies a bombardment to a temperature field.

        Args:
            T (torch.Tensor): Temperature field tensor.
            u (int): U-coordinate of the bombardment center.
            v (int): V-coordinate of the bombardment center.
            w (int): W-coordinate of the bombardment center.
            radius (int): Radius of the affected region.
            intensity (float): Intensity of the bombardment.
        """
        u_min = max(0, u - radius)
        u_max = min(self.resolution_u, u + radius + 1)
        v_min = max(0, v - radius)
        v_max = min(self.resolution_v, v + radius + 1)
        w_min = max(0, w - radius)
        w_max = min(self.resolution_w, w + radius + 1)
        T[u_min:u_max, v_min:v_max, w_min:w_max] += intensity

    def update(self):
        """
        Advances the simulation by one time step.

        Returns:
            np.ndarray: Current RGB frame as a (resolution_u, resolution_v, 3) uint8 array.
        """
        # Apply the Laplacian to each temperature channel
        if self.laplacian_tensor is not None:
            # Dense Laplacian
            laplace_T_r = -self.laplacian_tensor @ self.T_r.flatten()
            laplace_T_g = -self.laplacian_tensor @ self.T_g.flatten()
            laplace_T_b = -self.laplacian_tensor @ self.T_b.flatten()
        else:
            # Sparse Laplacian (if dense is not available)
            # Convert sparse matrix to Torch sparse tensor
            indices = torch.tensor([self.laplacian_sparse.row, self.laplacian_sparse.col], dtype=torch.long)
            values = torch.tensor(self.laplacian_sparse.data, dtype=torch.float64)
            laplacian_sparse_torch = torch.sparse_coo_tensor(indices, values, size=(self.resolution_u * self.resolution_v * self.resolution_w, self.resolution_u * self.resolution_v * self.resolution_w)).to(self.device)
            laplace_T_r = torch.sparse.mm(laplacian_sparse_torch, self.T_r.flatten().unsqueeze(1)).squeeze(1)
            laplace_T_g = torch.sparse.mm(laplacian_sparse_torch, self.T_g.flatten().unsqueeze(1)).squeeze(1)
            laplace_T_b = torch.sparse.mm(laplacian_sparse_torch, self.T_b.flatten().unsqueeze(1)).squeeze(1)

        # Update temperature fields
        self.T_r += self.delta_t * self.thermal_diffusivity * laplace_T_r.view(self.resolution_u, self.resolution_v, self.resolution_w)
        self.T_g += self.delta_t * self.thermal_diffusivity * laplace_T_g.view(self.resolution_u, self.resolution_v, self.resolution_w)
        self.T_b += self.delta_t * self.thermal_diffusivity * laplace_T_b.view(self.resolution_u, self.resolution_v, self.resolution_w)

        # Example: Apply Gaussian beam every step (modify as needed)
        self.apply_gaussian_beam(
            self.T_r,
            self.resolution_u // 2,
            self.resolution_v // 2,
            self.resolution_w // 2,
            amplitude=1200,
            sigma_u=2,
            sigma_v=2,
            sigma_w=2
        )
        self.apply_gaussian_beam(
            self.T_g,
            self.resolution_u // 2,
            self.resolution_v // 2,
            self.resolution_w // 2,
            amplitude=1200,
            sigma_u=2,
            sigma_v=2,
            sigma_w=2
        )
        self.apply_gaussian_beam(
            self.T_b,
            self.resolution_u // 2,
            self.resolution_v // 2,
            self.resolution_w // 2,
            amplitude=1200,
            sigma_u=2,
            sigma_v=2,
            sigma_w=2
        )

        # Example: Apply random bombardment every step (modify as needed)
        u_rand = np.random.randint(0, self.resolution_u)
        v_rand = np.random.randint(0, self.resolution_v)
        w_rand = np.random.randint(0, self.resolution_w)
        bombardment_intensity = 100000.0
        bombardment_radius = 1
        self.apply_bombardment(self.T_r, u_rand, v_rand, w_rand, bombardment_radius, bombardment_intensity)
        self.apply_bombardment(self.T_g, u_rand, v_rand, w_rand, bombardment_radius, bombardment_intensity)
        self.apply_bombardment(self.T_b, u_rand, v_rand, w_rand, bombardment_radius, bombardment_intensity)

        # Normalize and convert to RGB
        rgb_frame = self.generate_rgb_frame()

        # Increment step counter
        self.current_step += 1

        return rgb_frame

    def generate_rgb_frame(self):
        """
        Normalizes temperature fields and stacks them into an RGB image by taking central slices.

        Returns:
            np.ndarray: RGB image as a (resolution_u, resolution_v, 3) uint8 array.
        """
        # Select central slice along the thickness (w) axis
        central_w = self.resolution_w // 2

        # Extract central slices
        T_r_slice = self.T_r[:, :, central_w]
        T_g_slice = self.T_g[:, :, central_w]
        T_b_slice = self.T_b[:, :, central_w]

        # Normalize each temperature field slice
        temp_min_r, temp_max_r = T_r_slice.min(), T_r_slice.max()
        temp_min_g, temp_max_g = T_g_slice.min(), T_g_slice.max()
        temp_min_b, temp_max_b = T_b_slice.min(), T_b_slice.max()

        norm_r = (T_r_slice - temp_min_r) / (temp_max_r - temp_min_r + 1e-8)
        norm_g = (T_g_slice - temp_min_g) / (temp_max_g - temp_min_g + 1e-8)
        norm_b = (T_b_slice - temp_min_b) / (temp_max_b - temp_min_b + 1e-8)

        # Scale to 0-255 and clamp
        temp_r = torch.clamp(norm_r * 255, 0, 255).cpu().numpy().astype(np.uint8)
        temp_g = torch.clamp(norm_g * 255, 0, 255).cpu().numpy().astype(np.uint8)
        temp_b = torch.clamp(norm_b * 255, 0, 255).cpu().numpy().astype(np.uint8)

        # Stack channels to create an RGB image
        rgb_image = np.stack((temp_r, temp_g, temp_b), axis=2)

        return rgb_image

    def visualize(self, rgb_frame, step):
        """
        Saves the RGB frame as an image file.

        Args:
            rgb_frame (np.ndarray): RGB image.
            step (int): Current simulation step.
        """
        filename = f"heat_step_{step:04d}.png"
        imageio.imwrite(filename, rgb_frame)
        print(f"Saved {filename}")

    def run_simulation(self):
        """
        Runs the temperature simulation and saves visualization frames.
        """
        for step in range(self.time_steps):
            rgb_frame = self.update()
            self.visualize(rgb_frame, step)
            if step % 10 == 0:
                print(f"Completed step {step}/{self.time_steps}")
        print("Simulation complete.")
if __name__ == "__main__":
    # Parameters
    N_u, N_v, N_w = 20, 20, 10  # Grid resolution with thickness
    Lx, Ly, Lz = 4.0, 4.0, 1.0  # Domain size with thickness

    # Initialize the simulation
    simulation = TemperatureSimulation3D(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        resolution_u=N_u,
        resolution_v=N_v,
        resolution_w=N_w,
        device='cpu'  # Change to 'cuda' if using GPU
    )

    # Run the simulation
    simulation.run_simulation()
