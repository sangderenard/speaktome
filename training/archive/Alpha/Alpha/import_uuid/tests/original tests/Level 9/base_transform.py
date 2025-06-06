import torch
import random
class TransformHub:
    def __init__(self, uextent, vextent, grid_boundaries):
        self.uextent = uextent
        self.vextent = vextent
        self.grid_boundaries = grid_boundaries

    def calculate_geometry(self, U, V):
        """
        Compute coordinates, partials, normals, and metric tensor in one centralized function.
        
        Args:
            U, V (torch.Tensor): Grids for parameter space.

        Returns:
            dict: Dictionary containing coordinates, partials, normals, and metric tensors.
        """
        # Calculate coordinates and partial derivatives
        X, Y, Z, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, normals = self.compute_partials_and_normals(U, V)
        
        # Calculate metric tensor and its components
        g_ij, g_inv, det_g = self.metric_tensor_func(U, V, dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV)
        
        return {
            "coordinates": (X, Y, Z),
            "partials": (dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV),
            "normals": normals,
            "metric": (g_ij, g_inv, det_g),
            "frobenius_norm": self.compute_frobenius_norm(g_ij)
        }

    def compute_frobenius_norm(self, g_ij):
        """
        Compute the Frobenius norm of the metric tensor.

        Args:
            g_ij (torch.Tensor): Metric tensor with shape (..., 2, 2) for 2D surfaces.

        Returns:
            torch.Tensor: Frobenius norm of the metric tensor.
        """
        self.frobenius_norm = torch.sqrt(torch.sum(g_ij**2, dim=(-2, -1)))
        return self.frobenius_norm

    def compute_partials_and_normals(self, U, V, validate_normals=True, diagnostic_mode=False):
        # Ensure U and V require gradients for autograd
        U.requires_grad_(True)
        V.requires_grad_(True)

        # Forward pass: get transformed coordinates
        X, Y, Z = self.transform_spatial(U, V)

        if diagnostic_mode:
            print("U Grid:")
            print(U)
            print("V Grid:")
            print("Transformed Coordinates (X, Y, Z):")
            print("X:", X)
            print("Y:", Y)
            print("Z:", Z)

        # Calculate partial derivatives with respect to U
        dX_dU = torch.autograd.grad(X, U, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dY_dU = torch.autograd.grad(Y, U, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZ_dU = torch.autograd.grad(Z, U, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        # Calculate partial derivatives with respect to V
        dX_dV = torch.autograd.grad(X, V, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dY_dV = torch.autograd.grad(Y, V, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZ_dV = torch.autograd.grad(Z, V, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        target_shape = U.shape

        # Handle None values from autograd
        dX_dU = dX_dU if dX_dU is not None else torch.zeros(target_shape).to(U.device)
        dY_dU = dY_dU if dY_dU is not None else torch.zeros(target_shape).to(U.device)
        dZ_dU = dZ_dU if dZ_dU is not None else torch.zeros(target_shape).to(U.device)
        dX_dV = dX_dV if dX_dV is not None else torch.zeros(target_shape).to(V.device)
        dY_dV = dY_dV if dY_dV is not None else torch.zeros(target_shape).to(V.device)
        dZ_dV = dZ_dV if dZ_dV is not None else torch.zeros(target_shape).to(V.device)

        if diagnostic_mode:
            print("Partial Derivatives:")
            print("dX_dU:", dX_dU)
            print("dY_dU:", dY_dU)
            print("dZ_dU:", dZ_dU)
            print("dX_dV:", dX_dV)
            print("dY_dV:", dY_dV)
            print("dZ_dV:", dZ_dV)

        # Compute normals as cross-product of partial derivatives
        normals = torch.stack([
            dY_dU * dZ_dV - dZ_dU * dY_dV,
            dZ_dU * dX_dV - dX_dU * dZ_dV,
            dX_dU * dY_dV - dY_dU * dX_dV
        ], dim=-1)

        # Compute distances from the origin
        distances = torch.sqrt(X**2 + Y**2 + Z**2)
        
        # Select the top 10% farthest points
        top_10_percent_threshold = int(0.1 * distances.numel())
        top_10_percent_indices = torch.topk(distances.flatten(), top_10_percent_threshold).indices

        # Randomly sample 10% of the top 10% farthest points
        sample_size = max(1, int(0.1 * top_10_percent_threshold))  # Ensure at least one sample
        sample_indices = random.sample(top_10_percent_indices.tolist(), sample_size)

        # Conduct majority check based on sampled normals
        outward_votes = 0
        inward_votes = 0
        for idx in sample_indices:
            i, j = divmod(idx, target_shape[1])  # Convert flat index to 2D grid indices
            farthest_point = torch.tensor([X[i, j], Y[i, j], Z[i, j]], device=U.device, dtype=U.dtype)
            outward_reference_point = 1.01 * farthest_point  # 1% further outward
            
            # Directional check based on the sampled normal and reference point
            sample_normal = normals[i, j]
            direction_to_reference = outward_reference_point - farthest_point
            if torch.dot(sample_normal, direction_to_reference) > 0:
                outward_votes += 1
            else:
                inward_votes += 1

        # Conditionally invert normals based on majority vote
        if inward_votes > outward_votes:
            normals = -normals

        # Continue with normalization and validation
        norm_magnitudes = torch.norm(normals, dim=-1, keepdim=True)

        # Normalize normals, avoid division by zero for zero-magnitude normals
        normals = torch.where(norm_magnitudes > 1e-16, normals / norm_magnitudes, normals)

        # Identify zero-magnitude normals
        zero_norm_mask = norm_magnitudes.squeeze() < 1e-16  # Boolean mask for zero-magnitude normals

        if torch.any(zero_norm_mask):
            count_zero_normals = torch.sum(zero_norm_mask).item()  # Number of zero-magnitude normals
            print(f"{count_zero_normals} out of {normals.shape[0]*normals.shape[1]} zero-magnitude normals detected.")

            if diagnostic_mode:
                # Find the indices of the first zero-magnitude normal
                zero_indices = torch.nonzero(zero_norm_mask, as_tuple=True)
                first_zero_idx = (zero_indices[0][0].item(), zero_indices[1][0].item())

                print(f"First zero-magnitude normal at index: {first_zero_idx}")

                # Extract the partials contributing to this normal
                i, j = first_zero_idx

                partials = {
                    'dX_dU': dX_dU[i, j],
                    'dY_dU': dY_dU[i, j],
                    'dZ_dU': dZ_dU[i, j],
                    'dX_dV': dX_dV[i, j],
                    'dY_dV': dY_dV[i, j],
                    'dZ_dV': dZ_dV[i, j]
                }

                print("Partials at the first zero-magnitude normal:")
                for name, value in partials.items():
                    print(f"{name}[{i}, {j}] = {value}")

                # Stop execution until the issue is resolved
                print("Diagnostics complete. Exiting due to zero-magnitude normal.")
                exit()
            else:
                # Proceed to repair zero-magnitude normals if not in diagnostic mode
                print("Repairing zero-magnitude normals.")

                # Repair zero-magnitude normals by averaging surrounding normals
                zero_indices = torch.nonzero(zero_norm_mask, as_tuple=True)
                for idx in zip(*zero_indices):
                    # Collect neighboring normals
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = idx[0] + di, idx[1] + dj
                            if (0 <= ni < target_shape[0] and 0 <= nj < target_shape[1] and (di != 0 or dj != 0)):
                                neighbor_normal = normals[ni, nj]
                                neighbor_magnitude = norm_magnitudes[ni, nj]
                                if neighbor_magnitude > 1e-16:
                                    neighbors.append(neighbor_normal)
                    if neighbors:
                        avg_normal = torch.mean(torch.stack(neighbors), dim=0)
                        avg_normal_norm = torch.norm(avg_normal)
                        if avg_normal_norm > 1e-16:
                            normals[idx[0], idx[1]] = avg_normal / avg_normal_norm  # Normalize average
                        else:
                            print(f"Unable to repair normal at index {idx} due to zero magnitude of averaged normal.")
                    else:
                        print(f"No valid neighbors to repair normal at index {idx}.")

        if validate_normals:
            # Validation checks for the final normals
            if torch.any(torch.isnan(normals)):
                print("Validation failed: NaN values detected in normals.")
                exit()
            if not torch.all(torch.isfinite(normals)):
                print("Validation failed: Non-finite values detected in normals.")
                exit()
            if not torch.allclose(torch.norm(normals, dim=-1), torch.ones_like(norm_magnitudes.squeeze()), atol=1e-5):
                print("Validation failed: Normals are not unit length within tolerance after normalization.")
                exit()

            print("Validation passed: Normals are ideal.")

        return X, Y, Z, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, normals

                                            


    def get_or_compute_partials(self, U, V, dX_dU=None, dY_dU=None, dX_dV=None, dY_dV=None, dZ_dU=None, dZ_dV=None):
        """
        Helper to compute partials if they are not provided.
        
        Args:
            U, V (torch.Tensor): Parameter grids.
            dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV (torch.Tensor or None): Optional partials.
        
        Returns:
            Tuple[torch.Tensor]: Partial derivatives.
        """
        if all(partial is None for partial in [dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV]):
            _, _, _, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, _ = self.compute_partials_and_normals(U, V)
        return dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV

    def metric_tensor_func(self, U, V, dX_dU=None, dY_dU=None, dX_dV=None, dY_dV=None, dZ_dU=None, dZ_dV=None):
        """
        Enhanced metric tensor function for toroidal geometry, calculated adaptively using partial derivatives.
        
        Args:
            U, V (torch.Tensor): Grids for parameter space.
            dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV (torch.Tensor): Optional partial derivatives.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Metric tensor (g_ij), its inverse (g_inv), and determinant (det_g).
        """
        # Compute partial derivatives if not provided
        dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV = self.get_or_compute_partials(U, V, dX_dU, dY_dU, dX_dV, dY_dV, dZ_dU, dZ_dV)
        
        # Calculate metric tensor components directly from partial derivatives
        g_theta_theta = dX_dU**2 + dY_dU**2 + dZ_dU**2
        g_phi_phi = dX_dV**2 + dY_dV**2 + dZ_dV**2
        g_theta_phi = dX_dU * dX_dV + dY_dU * dY_dV + dZ_dU * dZ_dV

        # Construct metric tensor g_ij for the grid
        g_ij = torch.stack([
            torch.stack([g_theta_theta, g_theta_phi], dim=-1),
            torch.stack([g_theta_phi, g_phi_phi], dim=-1)
        ], dim=-2)

        # Determinant of the metric tensor
        det_g = g_theta_theta * g_phi_phi - g_theta_phi**2

        # Avoiding zero or near-zero determinant values by clamping
        det_g = torch.clamp(det_g, min=1e-6)

        # Compute inverse metric tensor g^ij
        g_inv = torch.stack([
            torch.stack([g_phi_phi / det_g, -g_theta_phi / det_g], dim=-1),
            torch.stack([-g_theta_phi / det_g, g_theta_theta / det_g], dim=-1)
        ], dim=-2)

        return g_ij, g_inv, det_g

    def transform_spatial(self, U, V):
        raise NotImplementedError("Subclasses must implement the transform_spatial method.")


from .base_metric_transform import MetricTransform
from .unpack_values import unpack_values

class Transform(TransformHub):
    def __init__(self, uextent, vextent, grid_boundaries):
        super().__init__(uextent, vextent, grid_boundaries)

    def get_transform_parameters(self):
        return (self.uextent, self.vextent), self.grid_boundaries

    def transform(self, U, V, use_metric=False):
        """
        Transform coordinates using either spatial or metric transformation.

        Args:
            U, V (torch.Tensor): Parameter grids.
            use_metric (bool): Whether to use the metric transformation.

        Returns:
            tuple: Transformed coordinates or metric data.
        """
        self.device = U.device
        geometry = self.calculate_geometry(U, V)
        return geometry["metric"] if use_metric else geometry["coordinates"]

    def convert_data_2d_to_3d(self, data_2d, use_metric=False):
        """
        Convert 2D parameter data to 3D coordinates and prepare for rendering.

        Args:
            data_2d (torch.Tensor): Stacked 2D data in U, V parameter space.
            use_metric (bool): Whether to use the metric transformation.

        Returns:
            Tuple of vertices, indices, normals, and data for OpenGL rendering.
        """
        resolution_u, resolution_v = data_2d.shape[-2], data_2d.shape[-1]
        # Generate U, V grid
        U, V = self.create_grid_mesh2(resolution_u, resolution_v)

        # Retrieve geometry (coordinates and optionally metric tensor)
        geometry_data = self.calculate_geometry(U, V)
        X, Y, Z = geometry_data["coordinates"]

        # Prepare vertices
        vertices = self.prepare_mesh_for_rendering(X, Y, Z)

        # Generate indices for triangulation (for OpenGL)
        indices = self.generate_triangle_indices(resolution_u, resolution_v)

        # Calculate normals for rendering
        normals = geometry_data["normals"]

        # Flatten data for rendering compatibility
        if data_2d.ndimension() == 2:
            data_3d = data_2d.flatten()
        else:
            data_3d = torch.stack([data_2d[i].flatten() for i in range(data_2d.shape[0])])

        return vertices, indices, normals, data_3d

    def create_grid_mesh2(self, resolution_u=100, resolution_v=100):
        if getattr(self, "autogrid", False):
            return self.obtain_autogrid()
        else:
            u_values = torch.linspace(0, self.uextent, resolution_u)
            v_values = torch.linspace(0, self.vextent, resolution_v)
            U, V = torch.meshgrid(u_values, v_values, indexing='ij')
            return U, V

    
    def create_grid_mesh(self, resolution_u, resolution_v):
        # Derive periodicity based on endpoint exclusion in grid boundaries
        periodic_u = not (self.grid_boundaries[0] and self.grid_boundaries[1])  # True if either endpoint is excluded for U
        periodic_v = not (self.grid_boundaries[2] and self.grid_boundaries[3])  # True if either endpoint is excluded for V
        from utils.gridhelper import generate_full_meshgrid
        # Use generate_full_meshgrid with inferred periodicity
        U_mesh, V_mesh, U_prime_mesh, V_prime_mesh = generate_full_meshgrid(
            N_u=resolution_u,
            L_u=self.uextent,
            N_v=resolution_v,
            L_v=self.vextent,
            periodic_u=periodic_u,
            periodic_v=periodic_v,
            device=self.device
        )
        return U_mesh, V_mesh
    def generate_triangle_indices(self, resolution_u, resolution_v):
        periodic_u = not (self.grid_boundaries[0] and self.grid_boundaries[1])  # True if either endpoint is excluded for U
        periodic_v = not (self.grid_boundaries[2] and self.grid_boundaries[3])  # True if either endpoint is excluded for V
        indices = []
        
        for u in range(resolution_u - 1):
            for v in range(resolution_v):
                # Add two vertices for each triangle strip
                indices.append(u * resolution_v + v)         # Current vertex
                indices.append((u + 1) * resolution_v + v)   # Vertex directly below in U direction
                
                # Connect the last vertex in v to the first if periodic_v
                if periodic_v and v == resolution_v - 1:
                    indices.append(u * resolution_v)              # Wrap to first column in current row
                    indices.append((u + 1) * resolution_v)        # Wrap to first column in next row

        # Connect the last row back to the first if periodic_u
        if periodic_u:
            for v in range(resolution_v):
                indices.append((resolution_u - 1) * resolution_v + v)   # Last row current column
                indices.append(v)                                       # First row current column
                
                # Handle periodicity in both dimensions at the corner
                if periodic_v and v == resolution_v - 1:
                    indices.append((resolution_u - 1) * resolution_v)   # Last row, first column
                    indices.append(0)                                   # First row, first column
                    
        return torch.tensor(indices, dtype=torch.int32)


    def prepare_mesh_for_rendering(self, X, Y, Z):
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        return torch.stack([X_flat, Y_flat, Z_flat], dim=-1)

    @classmethod
    def create_transform(cls, type_of_transform, **kwargs):
        metric_transform = MetricTransform.create_metric_transform(type_of_transform, **kwargs)
        if metric_transform is not None:
            return metric_transform

        from .regular_coordinate_transforms import (
            RectangularTransform, EllipticalTransform, HyperbolicTransform, ParabolicTransform,
            PolarTransform, ToroidalTransform, SphericalTransform
        )

        transform_map = {
            "rectangular": RectangularTransform,
            "elliptical": EllipticalTransform,
            "hyperbolic": HyperbolicTransform,
            "parabolic": ParabolicTransform,
            "polar": PolarTransform,
            "toroidal": ToroidalTransform,
            "spherical": SphericalTransform
        }

        if type_of_transform in transform_map:
            return transform_map[type_of_transform](**kwargs)
        else:
            raise ValueError(f"Unsupported transform type '{type_of_transform}'")


def main():
    import sys
    import pygame
    import torch
    from pygame.locals import DOUBLEBUF, OPENGL
    from openglrenderer.OGLRender import RendererConfig, TensorRenderer
    
    parent_dir = ".."
    sys.path.append(parent_dir)

    # Initialize Pygame and OpenGL context
    pygame.init()
    screen = pygame.display.set_mode((1024, 768), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Toroidal Transform Renderer")

    # Step 1: Initialize the Renderer
    config = RendererConfig(width=1024, height=768)
    renderer = TensorRenderer(config=config)

    # Step 2: Create a transform using the static class method
    transform_type = "toroidal_metric"  # Specify the type of transform
    toroidal_transform = Transform.create_transform(
        type_of_transform=transform_type,
        uextent=2 * torch.pi,    # Full range of theta (U axis)
        vextent=2 * torch.pi,    # Full range of phi (V axis)
        Lx = 4,
        Ly = 4,
        grid_boundaries=(True, False, True, False)  # Boundary conditions
    )

    # Step 3: Generate the U, V grid mesh for the transform
    resolution_u = 128  # Resolution along the u-axis
    resolution_v = 128  # Resolution along the v-axis
    U, V = toroidal_transform.create_grid_mesh(resolution_u=resolution_u, resolution_v=resolution_v)

    # Step 4: Generate spatial data (x, y, z) using the toroidal transform
    x_data, y_data, z_data = toroidal_transform.transform(U, V, use_metric=False)

    # Step 5: Flatten and combine the spatial data into a single tensor for rendering
    vertices = toroidal_transform.prepare_mesh_for_rendering(x_data.clone().detach(), y_data.clone().detach(), z_data.clone().detach())

    # Step 6: Create some color data (for example, using the Euclidean distance from origin)
    data_3d = torch.sqrt(x_data.clone().detach()**2 + y_data.clone().detach()**2 + z_data.clone().detach()**2).flatten()

    # Step 7: Render the scene using the OpenGL renderer
    renderer.render_still(vertices, data_3d)

    # Step 8: Enter the rendering loop
    pygame.event.set_grab(True)  # Lock the mouse to the window
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)  # Limit to 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        renderer.handle_input()  # Handle camera and input for user interaction

    # Quit Pygame when the loop exits
    pygame.quit()

# Call the main function when the script is executed
if __name__ == "__main__":
    main()
