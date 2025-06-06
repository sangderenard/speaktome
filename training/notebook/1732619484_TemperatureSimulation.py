import torch
import numpy as np
import imageio

class TemperatureSimulation:
    def __init__(self, Lx=4, Ly=4, resolution_u=128, resolution_v=128, device='cuda'):
        """
        Initializes the temperature simulation.

        Args:
            Lx (float): Length in the x-direction.
            Ly (float): Length in the y-direction.
            resolution_u (int): Grid resolution in the u-direction.
            resolution_v (int): Grid resolution in the v-direction.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.Lx = Lx
        self.Ly = Ly
        self.resolution_u = resolution_u
        self.resolution_v = resolution_v
        self.device = device

        # Initialize RectangularTransform
        self.transform = RectangularTransform(Lx=self.Lx, Ly=self.Ly, device=self.device)

        # Generate grid
        self.U, self.V = self.transform.create_grid_mesh(self.resolution_u, self.resolution_v)

        # Compute geometry and metric
        geometry = self.transform.calculate_geometry(self.U, self.V)
        self.geometry = geometry

        # Create GridDomain
        self.grid_domain = GridDomain.generate_grid_domain(
            coordinate_system='toroidal',
            N_u=self.resolution_u,
            N_v=self.resolution_v,
            Lx=self.Lx,
            Ly=1,
            device=self.device
        )

        # Initialize BuildLaplace and Laplacian tensor
        self.solver = BuildLaplace(grid_domain=self.grid_domain)
        self.laplacian_tensor, _ = self.solver.build_general_laplace(
            grid_u=self.U,
            grid_v=self.V,
            metric_tensor_func=self.transform.metric_tensor_func,
            boundary_conditions=("neumann", "neumann", "neumann", "neumann"),
            device=self.device
        )

        # Initialize temperature fields for Red, Green, Blue channels
        self.T_r = torch.full((self.resolution_u, self.resolution_v), 1300.0, dtype=torch.float64, device=self.device)
        self.T_g = torch.full((self.resolution_u, self.resolution_v), 1300.0, dtype=torch.float64, device=self.device)
        self.T_b = torch.full((self.resolution_u, self.resolution_v), 1300.0, dtype=torch.float64, device=self.device)

        # Initialize hotspots at the center
        center_u, center_v = self.resolution_u // 2, self.resolution_v // 2
        self.T_r[center_u, center_v] = 5000.0
        self.T_g[center_u, center_v] = 5000.0
        self.T_b[center_u, center_v] = 5000.0

        # Simulation parameters
        self.time_steps = 100
        self.delta_t = 0.01
        self.thermal_diffusivity = 0.1

        # Step counter
        self.current_step = 0

    def apply_gaussian_beam(self, T, center_u, center_v, amplitude, sigma_u, sigma_v):
        """
        Applies a Gaussian beam excitation to the temperature field.

        Args:
            T (torch.Tensor): Temperature field tensor.
            center_u (int): U-coordinate of the beam center.
            center_v (int): V-coordinate of the beam center.
            amplitude (float): Peak temperature increase.
            sigma_u (float): Standard deviation in the U direction.
            sigma_v (float): Standard deviation in the V direction.
        """
        u = torch.arange(0, self.resolution_u, device=self.device).unsqueeze(1).repeat(1, self.resolution_v)
        v = torch.arange(0, self.resolution_v, device=self.device).unsqueeze(0).repeat(self.resolution_u, 1)
        gaussian = amplitude * torch.exp(-(((u - center_u)**2) / (2 * sigma_u**2) + ((v - center_v)**2) / (2 * sigma_v**2)))
        T += gaussian

    def apply_bombardment(self, T, u, v, radius, intensity):
        """
        Applies a bombardment to a temperature field.

        Args:
            T (torch.Tensor): Temperature field tensor.
            u (int): U-coordinate of the bombardment center.
            v (int): V-coordinate of the bombardment center.
            radius (int): Radius of the affected region.
            intensity (float): Intensity of the bombardment.
        """
        u_min = max(0, u - radius)
        u_max = min(self.resolution_u, u + radius + 1)
        v_min = max(0, v - radius)
        v_max = min(self.resolution_v, v + radius + 1)
        T[u_min:u_max, v_min:v_max] += intensity

    def update(self):
        """
        Advances the simulation by one time step.

        Returns:
            np.ndarray: Current RGB frame as a (resolution_u, resolution_v, 3) uint8 array.
        """
        # Compute geometry and metric if needed (omitted for brevity)

        # Update Red channel
        laplacian_T_r = self.laplacian_tensor @ self.T_r.flatten()
        self.T_r += self.delta_t * self.thermal_diffusivity * laplacian_T_r.view(self.resolution_u, self.resolution_v)

        # Update Green channel
        laplacian_T_g = self.laplacian_tensor @ self.T_g.flatten()
        self.T_g += self.delta_t * self.thermal_diffusivity * laplacian_T_g.view(self.resolution_u, self.resolution_v)

        # Update Blue channel
        laplacian_T_b = self.laplacian_tensor @ self.T_b.flatten()
        self.T_b += self.delta_t * self.thermal_diffusivity * laplacian_T_b.view(self.resolution_u, self.resolution_v)

        # Example: Apply Gaussian beam every step (modify as needed)
        self.apply_gaussian_beam(self.T_r, self.resolution_u // 2, self.resolution_v // 2, amplitude=1200, sigma_u=6, sigma_v=6)
        self.apply_gaussian_beam(self.T_g, self.resolution_u // 2, self.resolution_v // 2, amplitude=1200, sigma_u=6, sigma_v=6)
        self.apply_gaussian_beam(self.T_b, self.resolution_u // 2, self.resolution_v // 2, amplitude=1200, sigma_u=6, sigma_v=6)

        # Example: Apply random bombardment every step (modify as needed)
        u_rand = np.random.randint(0, self.resolution_u)
        v_rand = np.random.randint(0, self.resolution_v)
        bombardment_intensity = 100000.0
        bombardment_radius = 1
        self.apply_bombardment(self.T_r, u_rand, v_rand, bombardment_radius, bombardment_intensity)
        self.apply_bombardment(self.T_g, u_rand, v_rand, bombardment_radius, bombardment_intensity)
        self.apply_bombardment(self.T_b, u_rand, v_rand, bombardment_radius, bombardment_intensity)

        # Normalize and convert to RGB
        rgb_frame = self.generate_rgb_frame()

        # Increment step counter
        self.current_step += 1

        return rgb_frame

    def generate_rgb_frame(self):
        """
        Normalizes temperature fields and stacks them into an RGB image.

        Returns:
            np.ndarray: RGB image as a (resolution_u, resolution_v, 3) uint8 array.
        """
        # Normalize each temperature field
        temp_min_r, temp_max_r = self.T_r.min(), self.T_r.max()
        temp_min_g, temp_max_g = self.T_g.min(), self.T_g.max()
        temp_min_b, temp_max_b = self.T_b.min(), self.T_b.max()

        norm_r = (self.T_r - temp_min_r) / (temp_max_r - temp_min_r + 1e-8)
        norm_g = (self.T_g - temp_min_g) / (temp_max_g - temp_min_g + 1e-8)
        norm_b = (self.T_b - temp_min_b) / (temp_max_b - temp_min_b + 1e-8)

        # Scale to 0-255 and clamp
        temp_r = torch.clamp(norm_r * 255, 0, 255).cpu().numpy().astype(np.uint8)
        temp_g = torch.clamp(norm_g * 255, 0, 255).cpu().numpy().astype(np.uint8)
        temp_b = torch.clamp(norm_b * 255, 0, 255).cpu().numpy().astype(np.uint8)

        # Stack channels to create an RGB image
        rgb_image = np.stack((temp_r, temp_g, temp_b), axis=2)

        return rgb_image
    def get_blackbody_spectrum_texture(self, wavelengths=np.linspace(300, 800, 500)):
        """
        Generates a blackbody spectrum texture based on the average temperature field.

        Args:
            wavelengths (np.ndarray): Array of wavelengths in nanometers.

        Returns:
            np.ndarray: Spectral texture as a (len(wavelengths),) array.
        """
        # Average the temperature fields
        avg_T = (self.T_r + self.T_g + self.T_b) / 3.0  # Shape: (resolution_u, resolution_v)
        avg_T = avg_T.cpu().numpy()

        # Compute the average temperature across the grid
        overall_avg_T = avg_T.mean()

        # Constants for Planck's Law
        h = 6.62607015e-34  # Planck's constant (J·s)
        c = 3.0e8           # Speed of light (m/s)
        k = 1.380649e-23    # Boltzmann's constant (J/K)

        # Convert wavelengths from nm to meters
        wavelengths_m = wavelengths * 1e-9  # Shape: (num_wavelengths,)

        # Compute spectral radiance using Planck's Law
        exponent = (h * c) / (wavelengths_m * k * overall_avg_T)
        # Prevent overflow in the exponential
        exponent = np.clip(exponent, a_min=None, a_max=700)
        spectral_radiance = (2.0 * h * c**2) / (wavelengths_m**5 * (np.exp(exponent) - 1.0))

        # Normalize spectral radiance
        spectral_radiance /= spectral_radiance.max()

        return spectral_radiance

    def save_blackbody_texture(self, filename, wavelengths=np.linspace(300, 800, 500)):
        """
        Saves the blackbody spectrum texture to a file.

        Args:
            filename (str): Path to save the spectral texture.
            wavelengths (np.ndarray): Array of wavelengths in nanometers.
        """
        spectral_texture = self.get_blackbody_spectrum_texture(wavelengths)
        np.save(filename, spectral_texture)
