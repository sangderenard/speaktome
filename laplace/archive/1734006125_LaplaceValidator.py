import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg  
from laplace2 import GridDomain, BuildLaplace3D, RectangularTransform  # Your module

def scipy_to_torch_sparse(scipy_coo, device):
    values = torch.tensor(scipy_coo.data, dtype=torch.float64, device=device)
    indices = torch.tensor(np.vstack((scipy_coo.row, scipy_coo.col)), dtype=torch.long, device=device)
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)

class LaplaceValidator:
    def __init__(self, resolution=(50, 50, 50), device='cpu'):
        self.resolution = resolution
        self.device = device
        
    def known_solution_2d(self, X, Y):
        """Known 2D Laplace solution: sin(pi*x) * sin(pi*y)"""
        return np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    def known_solution_3d(self, X, Y, Z):
        """Known 3D Laplace solution: sin(pi*x) * sin(pi*y) * sin(pi*z)"""
        return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    
    def compute_laplace(self, u, laplace_tensor, dim=3):
        """
        Compute Laplacian of u using the given Laplace tensor.
        Args:
            u (ndarray): 2D or 3D input function.
            laplace_tensor (torch.sparse_coo_tensor): Precomputed sparse Laplace tensor.
            dim (int): Dimensionality of the input (2 or 3).
        Returns:
            ndarray: Laplacian of u.
        """
        if dim == 2:
            # Embed 2D solution into a 3D grid on the z=0 plane
            u_3d = np.zeros((u.shape[0], u.shape[1], self.resolution[2]))
            u_3d[:, :, 0] = u  # Place 2D u on z=0 plane
            u_flat = torch.tensor(u_3d.flatten(), dtype=torch.float64, device=self.device)
        else:
            # Directly flatten 3D input
            u_flat = torch.tensor(u.flatten(), dtype=torch.float64, device=self.device)
        
        # Apply the sparse Laplace tensor
        laplace_u_flat = torch.sparse.mm(laplace_tensor, u_flat.unsqueeze(1)).squeeze()
        laplace_u = laplace_u_flat.cpu().numpy().reshape(self.resolution)
        
        if dim == 2:
            # Return only the z=0 plane
            return laplace_u[:, :, 0]
        return laplace_u


    def validate(self):
        """Validate Laplace-Beltrami implementation."""
        # Generate grid
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        nx, ny, nz = self.resolution
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, Lz, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Known solutions
        u_2d_known = self.known_solution_2d(X[:, :, 0], Y[:, :, 0])
        u_3d_known = self.known_solution_3d(X, Y, Z)

        # Build Laplace-Beltrami operator
        transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=self.device)
        grid_u, grid_v, grid_w = transform.create_grid_mesh(nx, ny, nz)
        grid_domain = GridDomain(grid_u, grid_v, grid_w, transform=transform)
        laplace_builder = BuildLaplace3D(grid_domain=grid_domain, resolution=nx)
        _, laplace_tensor_sparse = laplace_builder.build_general_laplace(grid_u, grid_v, grid_w, device=self.device)
        laplace_tensor_sparse = scipy_to_torch_sparse(laplace_tensor_sparse, self.device)

        # Compute Laplacians
        u_2d_laplace = self.compute_laplace(u_2d_known, laplace_tensor_sparse, dim=2)
        u_3d_laplace = self.compute_laplace(u_3d_known, laplace_tensor_sparse, dim=3)

        # Plot results
        self.plot_results(u_2d_known, u_2d_laplace, "2D Known Solution")
        self.plot_results(u_3d_known[:, :, nz // 2], u_3d_laplace[:, :, nz // 2], "3D Known Solution (Midplane)")


    def plot_results(self, known, computed, title):
        """Plot known, computed, and difference."""
        difference = computed - known

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(known, cmap='viridis', origin='lower')
        axes[0].set_title(f"Known Solution ({title})")

        axes[1].imshow(computed, cmap='viridis', origin='lower')
        axes[1].set_title(f"Computed Solution ({title})")

        im = axes[2].imshow(difference, cmap='seismic', origin='lower')
        axes[2].set_title(f"Difference ({title})")
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    validator = LaplaceValidator(resolution=(50, 50, 50))
    validator.validate()
