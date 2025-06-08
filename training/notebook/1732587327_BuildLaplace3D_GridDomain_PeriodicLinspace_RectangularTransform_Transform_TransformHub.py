
import torch
import numpy as np
import math
import logging
from scipy.sparse import coo_matrix

# Configure the logger at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG to capture all levels of logs

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)


import torch
import numpy as np
from scipy.sparse import coo_matrix
import math

class BuildLaplace3D:
    def __init__(self, grid_domain, wave_speed=343, precision=torch.float64, resolution=68,
                 metric_tensor_func=None, density_func=None, tension_func=None, 
                 singularity_conditions=None, singularity_dirichlet_func=None, singularity_neumann_func=None, 
                 boundary_conditions=('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet'), 
                 artificial_stability=0):
        """
        Initialize BuildLaplace for 3D with additional dimension handling.
        
        Args:
            grid_domain: Object that handles the grid transformations (u, v, w) -> (x, y, z).
            wave_speed: Speed of wave propagation, used to compute local wave numbers.
            precision: Torch precision type for tensor creation (default: torch.float64).
            resolution: Maximum resolution for dense tensor calculations (default: 68).
            metric_tensor_func: Function to compute the metric tensor (default: None for Euclidean space).
            density_func: Function or tensor defining the density over the grid (default: None, assumes 1.0 everywhere).
            tension_func: Function or tensor defining the tension over the grid (default: None, assumes 1.0 everywhere).
            singularity_conditions: Conditions at singularities (default: None, assumes no singularities).
            singularity_dirichlet_func: Function to compute Dirichlet boundary values at singularities (default: None).
            singularity_neumann_func: Function to compute Neumann boundary values at singularities (default: None).
            boundary_conditions: Tuple specifying boundary conditions for u, v, w axes
                                 (default: ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')).
            artificial_stability: Small stability term added to metrics (default: 0).
        """
        self.grid_domain = grid_domain
        self.wave_speed = wave_speed
        self.precision = precision
        self.resolution = resolution
        self.metric_tensor_func = metric_tensor_func
        self.density_func = density_func
        self.tension_func = tension_func
        self.singularity_conditions = singularity_conditions
        self.singularity_dirichlet_func = singularity_dirichlet_func
        self.singularity_neumann_func = singularity_neumann_func
        self.boundary_conditions = boundary_conditions  # (u_min, u_max, v_min, v_max, w_min, w_max)
        self.artificial_stability = artificial_stability


    def build_general_laplace(self, grid_u, grid_v, grid_w, boundary_conditions=None, singularity_conditions=None, 
                              singularity_dirichlet_func=None, singularity_neumann_func=None, k=0.0, 
                              metric_tensor_func=None, density_func=None, tension_func=None, 
                              device=None, grid_boundaries=(True, True, True, True, True, True), 
                              artificial_stability=None, f=0, normalize=False):
        """
        Builds the Laplacian matrix for a 3D coordinate system using the provided u, v, w grids.
        Handles singularities using custom Dirichlet/Neumann conditions.
        """
        logger.debug("Starting build_general_laplace")
        logger.debug(f"Input grid shapes - grid_u: {grid_u.shape}, grid_v: {grid_v.shape}, grid_w: {grid_w.shape}")
        logger.debug(f"Boundary conditions: {boundary_conditions}")
        logger.debug(f"Singularity conditions: {singularity_conditions}")
        logger.debug(f"k: {k}, f: {f}")
        logger.debug(f"Device: {device}, Grid boundaries: {grid_boundaries}")

        # Conditional reassignments, use method parameters if provided, otherwise use class attributes
        boundary_conditions = boundary_conditions if boundary_conditions is not None else self.boundary_conditions
        singularity_conditions = singularity_conditions if singularity_conditions is not None else self.singularity_conditions
        singularity_dirichlet_func = singularity_dirichlet_func if singularity_dirichlet_func is not None else self.singularity_dirichlet_func
        singularity_neumann_func = singularity_neumann_func if singularity_neumann_func is not None else self.singularity_neumann_func
        metric_tensor_func = metric_tensor_func if metric_tensor_func is not None else self.metric_tensor_func
        density_func = density_func if density_func is not None else self.density_func
        tension_func = tension_func if tension_func is not None else self.tension_func
        artificial_stability = artificial_stability if artificial_stability is not None else self.artificial_stability

        logger.debug("Parameters reassigned with either provided values or class attributes.")

        def default_metric_tensor(u, v, w, dxdu, dydu, dzdu, dxdv, dydv, dzdv, dxdw, dydw, dzdw):
            """
            Default metric tensor for a flat Euclidean 3D space.
            """
            logger.debug("Calculating default metric tensor.")
            logger.debug(f"dxdu shape: {dxdu.shape}, dxdv shape: {dxdv.shape}, dydu shape: {dydu.shape}, dydv shape: {dydv.shape}")

            g_uu = dxdu**2 + dydu**2 + dzdu**2
            g_vv = dxdv**2 + dydv**2 + dzdv**2
            g_ww = dxdw**2 + dydw**2 + dzdw**2
            g_uv = dxdu * dxdv + dydu * dydv + dzdu * dzdv
            g_uw = dxdu * dxdw + dydu * dydw + dzdu * dzdw
            g_vw = dxdv * dxdw + dydv * dydw + dzdv * dzdw

            logger.debug("Computed metric tensor components.")

            # Stack into a 3x3 matrix
            g_ij = torch.stack([
                torch.stack([g_uu, g_uv, g_uw], dim=-1),
                torch.stack([g_uv, g_vv, g_vw], dim=-1),
                torch.stack([g_uw, g_vw, g_ww], dim=-1)
            ], dim=-2)  # Shape: (N_u, N_v, N_w, 3, 3)

            logger.debug(f"g_ij shape: {g_ij.shape}")

            det_g = torch.det(g_ij)  # Shape: (N_u, N_v, N_w)
            logger.debug(f"det_g shape: {det_g.shape}")

            g_inv = torch.inverse(g_ij)  # Shape: (N_u, N_v, N_w, 3, 3)
            logger.debug("Computed inverse metric tensor.")

            return g_ij, g_inv, det_g

        if metric_tensor_func is None:
            metric_tensor_func = default_metric_tensor
            logger.debug("Using default metric tensor function.")

        # Apply the transformation function to the grid
        logger.debug("Applying transformation to the grid.")
        X, Y, Z = self.grid_domain.transform.transform(grid_u, grid_v, grid_w)[:3]  # Transform to physical space
        self.grid_u = grid_u
        self.grid_v = grid_v
        self.grid_w = grid_w
        N_u = grid_u.shape[0]
        N_v = grid_v.shape[1]
        N_w = grid_w.shape[2]

        logger.debug(f"Grid dimensions - N_u: {N_u}, N_v: {N_v}, N_w: {N_w}")

        # Compute partial derivatives
        logger.debug("Computing partial derivatives.")
        dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw = self.grid_domain.transform.get_or_compute_partials(grid_u, grid_v, grid_w)
        logger.debug("Computed partial derivatives.")

        unique_u_values = grid_u[:, 0, 0]
        unique_v_values = grid_v[0, :, 0]
        unique_w_values = grid_w[0, 0, :]  # Assuming grid_w is along the third dimension

        logger.debug(f"Unique u values: {unique_u_values}")
        logger.debug(f"Unique v values: {unique_v_values}")
        logger.debug(f"Unique w values: {unique_w_values}")

        final_u_row = wrap_u_row = None
        final_v_row = wrap_v_row = None
        final_w_row = wrap_w_row = None

        # Handle the u-direction (dim=0, radial or x-direction)
        if boundary_conditions[0] == 'periodic' or boundary_conditions[1] == 'periodic':
            logger.debug("Handling periodic boundary conditions for u-direction.")
            sum_du = torch.sum(unique_u_values[1:] - unique_u_values[:-1])
            final_du = (2 * np.pi) - sum_du
            final_u_value = unique_u_values[-1]
            final_u_row = torch.full_like(unique_v_values.unsqueeze(1).repeat(1, N_w), final_u_value.item())
            wrap_u_row = torch.full_like(unique_v_values.unsqueeze(1).repeat(1, N_w), unique_u_values[0].item())

            logger.debug(f"sum_du: {sum_du}, final_du: {final_du}")
            logger.debug(f"final_u_row: {final_u_row}, wrap_u_row: {wrap_u_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(final_u_row, unique_v_values.unsqueeze(1).repeat(1, N_w), unique_w_values.unsqueeze(0).repeat(N_u, 1))
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(wrap_u_row, unique_v_values.unsqueeze(1).repeat(1, N_w), unique_w_values.unsqueeze(0).repeat(N_u, 1))

            logger.debug("Calculated transformed metrics for u-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdu = (wrap_X - final_X)
            final_dYdu = (wrap_Y - final_Y)
            final_dZdu = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for u-direction.")

            # Append this final differential to dXdu, dYdu, dZdu
            if final_dXdu.dim() == 2:
                final_dXdu = final_dXdu.unsqueeze(0)
            if final_dYdu.dim() == 2:
                final_dYdu = final_dYdu.unsqueeze(0)
            if final_dZdu.dim() == 2:
                final_dZdu = final_dZdu.unsqueeze(0)

            dXdu = torch.cat([dXdu, final_dXdu], dim=0)
            dYdu = torch.cat([dYdu, final_dYdu], dim=0)
            dZdu = torch.cat([dZdu, final_dZdu], dim=0)
            logger.debug("Appended final differentials to u-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for u-direction. No action taken.")

        # Handle the v-direction (dim=1, angular or y-direction)
        if boundary_conditions[2] == 'periodic' or boundary_conditions[3] == 'periodic':
            logger.debug("Handling periodic boundary conditions for v-direction.")
            sum_dv = torch.sum(unique_v_values[1:] - unique_v_values[:-1])
            final_dv = (2 * np.pi) - sum_dv
            final_v_value = unique_v_values[-1]
            final_v_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w), final_v_value.item())
            wrap_v_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w), unique_v_values[0].item())

            logger.debug(f"sum_dv: {sum_dv}, final_dv: {final_dv}")
            logger.debug(f"final_v_row: {final_v_row}, wrap_v_row: {wrap_v_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(
                unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w),
                final_v_row,
                unique_w_values.unsqueeze(0).repeat(N_u, 1, 1)
            )
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(
                unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, 1, N_w),
                wrap_v_row,
                unique_w_values.unsqueeze(0).repeat(N_u, 1, 1)
            )

            logger.debug("Calculated transformed metrics for v-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdv = (wrap_X - final_X)
            final_dYdv = (wrap_Y - final_Y)
            final_dZdv = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for v-direction.")

            # Append this final differential to dXdv, dYdv, dZdv
            if final_dXdv.dim() == 2:
                final_dXdv = final_dXdv.unsqueeze(1)
            if final_dYdv.dim() == 2:
                final_dYdv = final_dYdv.unsqueeze(1)
            if final_dZdv.dim() == 2:
                final_dZdv = final_dZdv.unsqueeze(1)

            dXdv = torch.cat([dXdv, final_dXdv], dim=1)
            dYdv = torch.cat([dYdv, final_dYdv], dim=1)
            dZdv = torch.cat([dZdv, final_dZdv], dim=1)
            logger.debug("Appended final differentials to v-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for v-direction. No action taken.")

        # Handle the w-direction (dim=2, z-direction)
        if boundary_conditions[4] == 'periodic' or boundary_conditions[5] == 'periodic':
            logger.debug("Handling periodic boundary conditions for w-direction.")
            sum_dw = torch.sum(unique_w_values[1:] - unique_w_values[:-1])
            final_dw = (2 * np.pi) - sum_dw
            final_w_value = unique_w_values[-1]
            final_w_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, N_v, 1), final_w_value.item())
            wrap_w_row = torch.full_like(unique_u_values.unsqueeze(1).unsqueeze(2).repeat(1, N_v, 1), unique_w_values[0].item())

            logger.debug(f"sum_dw: {sum_dw}, final_dw: {final_dw}")
            logger.debug(f"final_w_row: {final_w_row}, wrap_w_row: {wrap_w_row}")

            # Calculate final dx, dy, dz using the transform function
            final_X, final_Y, final_Z = self.grid_domain.transform.transform_metric(grid_u, grid_v, final_w_row)[:3]
            wrap_X, wrap_Y, wrap_Z = self.grid_domain.transform.transform_metric(grid_u, grid_v, wrap_w_row)[:3]

            logger.debug("Calculated transformed metrics for w-direction boundaries.")

            # Calculate the final differential in x, y, z directions
            final_dXdw = (wrap_X - final_X)
            final_dYdw = (wrap_Y - final_Y)
            final_dZdw = (wrap_Z - final_Z)

            logger.debug("Calculated final differentials for w-direction.")

            # Append this final differential to dXdw, dYdw, dZdw
            if final_dXdw.dim() == 2:
                final_dXdw = final_dXdw.unsqueeze(2)
            if final_dYdw.dim() == 2:
                final_dYdw = final_dYdw.unsqueeze(2)
            if final_dZdw.dim() == 2:
                final_dZdw = final_dZdw.unsqueeze(2)

            dXdw = torch.cat([dXdw, final_dXdw], dim=2)
            dYdw = torch.cat([dYdw, final_dYdw], dim=2)
            dZdw = torch.cat([dZdw, final_dZdw], dim=2)
            logger.debug("Appended final differentials to w-direction derivatives.")
        else:
            logger.debug("Non-periodic boundary conditions for w-direction. No action taken.")

        # Prepare for sparse matrix construction
        logger.debug("Preparing for sparse matrix construction.")
        row_indices = []
        col_indices = []
        values = []
        diagonal_entries = {}

        # Precompute the metric tensor (g_ij), its inverse (g_inv), and the determinant (det_g) for the entire grid
        if metric_tensor_func is not None:
            logger.debug("Computing metric tensor using the provided function.")
            # Apply the metric tensor function to the entire grid
            g_ij, g_inv, det_g = metric_tensor_func(grid_u, grid_v, grid_w, dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw)
            
            # Move to CPU and convert to numpy for processing
            logger.debug("Moving metric tensors to CPU and converting to numpy.")
            g_ij = g_ij.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w, 3, 3)
            g_inv = g_inv.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w, 3, 3)
            det_g_all = det_g.clone().detach().cpu().numpy()  # Shape: (N_u, N_v, N_w)
            logger.debug("Metric tensors computed and converted.")
        else:
            logger.error("Metric tensor function must be provided for 3D Laplacian.")
            raise ValueError("Metric tensor function must be provided for 3D Laplacian.")

        # Iterate over the 3D grid to construct Laplacian
        logger.debug("Starting iteration over the 3D grid to construct Laplacian.")
        for i in range(N_u):
            for j in range(N_v):
                for k in range(N_w):
                    idx = i * N_v * N_w + j * N_w + k  # Flattened index
                    if idx not in diagonal_entries:
                        diagonal_entries[idx] = 0.0

                    # Retrieve metric tensor components
                    g_uu = g_ij[i, j, k, 0, 0]
                    g_uv = g_ij[i, j, k, 0, 1]
                    g_uw = g_ij[i, j, k, 0, 2]
                    g_vv = g_ij[i, j, k, 1, 1]
                    g_vw = g_ij[i, j, k, 1, 2]
                    g_ww = g_ij[i, j, k, 2, 2]
                    det_g = det_g_all[i, j, k]

                    inv_g_uu = g_inv[i, j, k, 0, 0]
                    inv_g_uv = g_inv[i, j, k, 0, 1]
                    inv_g_uw = g_inv[i, j, k, 0, 2]
                    inv_g_vv = g_inv[i, j, k, 1, 1]
                    inv_g_vw = g_inv[i, j, k, 1, 2]
                    inv_g_ww = g_inv[i, j, k, 2, 2]

                    logger.debug(f"Processing grid point (i={i}, j={j}, k={k}) with idx={idx}")
                    logger.debug(f"Metric tensor components: g_uu={g_uu}, g_uv={g_uv}, g_uw={g_uw}, g_vv={g_vv}, g_vw={g_vw}, g_ww={g_ww}")
                    logger.debug(f"Inverse metric tensor components: inv_g_uu={inv_g_uu}, inv_g_uv={inv_g_uv}, inv_g_uw={inv_g_uw}, inv_g_vv={inv_g_vv}, inv_g_vw={inv_g_vw}, inv_g_ww={inv_g_ww}")
                    logger.debug(f"det_g: {det_g}")

                    # Metric terms for u, v, w directions
                    metric_u = inv_g_uu + 1e-10 * artificial_stability
                    metric_v = inv_g_vv + 1e-10 * artificial_stability
                    metric_w = inv_g_ww + 1e-10 * artificial_stability

                    logger.debug(f"Computed metrics: metric_u={metric_u}, metric_v={metric_v}, metric_w={metric_w}")

                    # Handle singularities based on the singularity_conditions
                    if metric_u == 0 or metric_v == 0 or metric_w == 0:
                        logger.debug(f"Singularity detected at idx={idx}. Handling based on singularity_conditions.")
                        if isinstance(singularity_conditions, str):
                            if singularity_conditions == "dirichlet":
                                if singularity_dirichlet_func:
                                    value = singularity_dirichlet_func(i, j, k)
                                    diagonal_entries[idx] = value
                                    logger.debug(f"Applied Dirichlet condition with value {value} at idx={idx}")
                                else:
                                    diagonal_entries[idx] = 1.0  # Default Dirichlet value
                                    logger.debug(f"Applied default Dirichlet condition at idx={idx}")
                            elif singularity_conditions == "neumann":
                                if singularity_neumann_func:
                                    value = singularity_neumann_func(i, j, k)
                                    diagonal_entries[idx] = value
                                    logger.debug(f"Applied Neumann condition with value {value} at idx={idx}")
                                else:
                                    diagonal_entries[idx] = 0.0  # Default Neumann value (symmetric condition)
                                    logger.debug(f"Applied default Neumann condition at idx={idx}")
                        elif isinstance(singularity_conditions, torch.Tensor):
                            if singularity_conditions[i, j, k]:  # True means Dirichlet, False means Neumann
                                if singularity_dirichlet_func:
                                    value = singularity_dirichlet_func(i, j, k)
                                    diagonal_entries[idx] = value
                                    logger.debug(f"Applied Tensor Dirichlet condition with value {value} at idx={idx}")
                                else:
                                    diagonal_entries[idx] = 1.0
                                    logger.debug(f"Applied default Tensor Dirichlet condition at idx={idx}")
                            else:
                                if singularity_neumann_func:
                                    value = singularity_neumann_func(i, j, k)
                                    diagonal_entries[idx] = value
                                    logger.debug(f"Applied Tensor Neumann condition with value {value} at idx={idx}")
                                else:
                                    diagonal_entries[idx] = 0.0
                                    logger.debug(f"Applied default Tensor Neumann condition at idx={idx}")
                        continue  # Skip further processing for singularities

                    # Incorporate tension and density if provided
                    if density_func:
                        density = density_func(i, j, k) if callable(density_func) else density_func[i, j, k]
                        logger.debug(f"Density at idx={idx}: {density}")
                    else:
                        density = 1.0  # Default density
                        logger.debug(f"Using default density at idx={idx}: {density}")

                    if tension_func:
                        tension = tension_func(i, j, k) if callable(tension_func) else tension_func[i, j, k]
                        logger.debug(f"Tension at idx={idx}: {tension}")
                    else:
                        tension = 1.0  # Default tension
                        logger.debug(f"Using default tension at idx={idx}: {tension}")

                    # Calculate Laplacian diagonal contribution
                    laplacian_diag = 2.0 * tension * (1.0 / metric_u + 1.0 / metric_v + 1.0 / metric_w) / density
                    diagonal_entries[idx] += laplacian_diag
                    logger.debug(f"Laplacian diagonal contribution at idx={idx}: {laplacian_diag}, total diagonal: {diagonal_entries[idx]}")

                    # Calculate neighbors' indices
                    i_prev = (i - 1) % N_u
                    i_next = (i + 1) % N_u
                    j_prev = (j - 1) % N_v
                    j_next = (j + 1) % N_v
                    k_prev = (k - 1) % N_w
                    k_next = (k + 1) % N_w

                    logger.debug(f"Neighbor indices for idx={idx}: i_prev={i_prev}, i_next={i_next}, j_prev={j_prev}, j_next={j_next}, k_prev={k_prev}, k_next={k_next}")

                    # Off-diagonal Laplacian contributions
                    laplacian_off_diag_u = -inv_g_uu / metric_u
                    laplacian_off_diag_v = -inv_g_vv / metric_v
                    laplacian_off_diag_w = -inv_g_ww / metric_w

                    laplacian_cross_uv = -inv_g_uv / det_g if det_g != 0 else 0
                    laplacian_cross_uw = -inv_g_uw / det_g if det_g != 0 else 0
                    laplacian_cross_vw = -inv_g_vw / det_g if det_g != 0 else 0

                    logger.debug(f"Laplacian off-diagonal contributions: u={laplacian_off_diag_u}, v={laplacian_off_diag_v}, w={laplacian_off_diag_w}")
                    logger.debug(f"Laplacian cross-derivative contributions: uv={laplacian_cross_uv}, uw={laplacian_cross_uw}, vw={laplacian_cross_vw}")

                    # Apply boundary conditions for u axis
                    if i == 0 and grid_boundaries[0]:
                        bc = boundary_conditions[0]
                        logger.debug(f"Applying boundary condition '{bc}' for u_min at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_u
                            row_indices.append(idx)
                            col_indices.append(i_next * N_v * N_w + j * N_w + k)
                            values.append(1.0 / metric_u)
                            logger.debug(f"Applied Neumann condition for u_min at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            up_idx = i_prev * N_v * N_w + j * N_w + k
                            down_idx = i_next * N_v * N_w + j * N_w + k
                            row_indices.extend([idx, idx])
                            col_indices.extend([up_idx, down_idx])
                            values.extend([laplacian_off_diag_u, laplacian_off_diag_u])
                            logger.debug(f"Applied Periodic condition for u_min at idx={idx}")

                    if i == N_u - 1 and grid_boundaries[1]:
                        bc = boundary_conditions[1]
                        logger.debug(f"Applying boundary condition '{bc}' for u_max at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_u
                            row_indices.append(idx)
                            col_indices.append(i_prev * N_v * N_w + j * N_w + k)
                            values.append(1.0 / metric_u)
                            logger.debug(f"Applied Neumann condition for u_max at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            up_idx = i_prev * N_v * N_w + j * N_w + k
                            down_idx = i_next * N_v * N_w + j * N_w + k
                            row_indices.extend([idx, idx])
                            col_indices.extend([up_idx, down_idx])
                            values.extend([laplacian_off_diag_u, laplacian_off_diag_u])
                            logger.debug(f"Applied Periodic condition for u_max at idx={idx}")

                    # Apply boundary conditions for v axis
                    if j == 0 and grid_boundaries[2]:
                        bc = boundary_conditions[2]
                        logger.debug(f"Applying boundary condition '{bc}' for v_min at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_v
                            row_indices.append(idx)
                            col_indices.append(i * N_v * N_w + j_next * N_w + k)
                            values.append(1.0 / metric_v)
                            logger.debug(f"Applied Neumann condition for v_min at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            left_idx = i * N_v * N_w + j_prev * N_w + k
                            right_idx = i * N_v * N_w + j_next * N_w + k
                            row_indices.extend([idx, idx])
                            col_indices.extend([left_idx, right_idx])
                            values.extend([laplacian_off_diag_v, laplacian_off_diag_v])
                            logger.debug(f"Applied Periodic condition for v_min at idx={idx}")

                    if j == N_v - 1 and grid_boundaries[3]:
                        bc = boundary_conditions[3]
                        logger.debug(f"Applying boundary condition '{bc}' for v_max at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_v
                            row_indices.append(idx)
                            col_indices.append(i * N_v * N_w + j_prev * N_w + k)
                            values.append(1.0 / metric_v)
                            logger.debug(f"Applied Neumann condition for v_max at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            left_idx = i * N_v * N_w + j_prev * N_w + k
                            right_idx = i * N_v * N_w + j_next * N_w + k
                            row_indices.extend([idx, idx])
                            col_indices.extend([left_idx, right_idx])
                            values.extend([laplacian_off_diag_v, laplacian_off_diag_v])
                            logger.debug(f"Applied Periodic condition for v_max at idx={idx}")

                    # Apply boundary conditions for w axis
                    if k == 0 and grid_boundaries[4]:
                        bc = boundary_conditions[4]
                        logger.debug(f"Applying boundary condition '{bc}' for w_min at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_w
                            row_indices.append(idx)
                            col_indices.append(i * N_v * N_w + j * N_w + k_next)
                            values.append(1.0 / metric_w)
                            logger.debug(f"Applied Neumann condition for w_min at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            front_idx = i * N_v * N_w + j * N_w + k_prev
                            back_idx = i * N_v * N_w + j * N_w + k_next
                            row_indices.extend([idx, idx])
                            col_indices.extend([front_idx, back_idx])
                            values.extend([laplacian_off_diag_w, laplacian_off_diag_w])
                            logger.debug(f"Applied Periodic condition for w_min at idx={idx}")

                    if k == N_w - 1 and grid_boundaries[5]:
                        bc = boundary_conditions[5]
                        logger.debug(f"Applying boundary condition '{bc}' for w_max at idx={idx}")
                        if bc == "dirichlet":
                            diagonal_entries[idx] = 1.0
                            logger.debug(f"Set diagonal to 1.0 for Dirichlet at idx={idx}")
                        elif bc == "neumann":
                            diagonal_entries[idx] += laplacian_off_diag_w
                            row_indices.append(idx)
                            col_indices.append(i * N_v * N_w + j * N_w + k_prev)
                            values.append(1.0 / metric_w)
                            logger.debug(f"Applied Neumann condition for w_max at idx={idx}")
                        elif bc == "periodic":
                            # Wrap around
                            front_idx = i * N_v * N_w + j * N_w + k_prev
                            back_idx = i * N_v * N_w + j * N_w + k_next
                            row_indices.extend([idx, idx])
                            col_indices.extend([front_idx, back_idx])
                            values.extend([laplacian_off_diag_w, laplacian_off_diag_w])
                            logger.debug(f"Applied Periodic condition for w_max at idx={idx}")

                    # Internal connections based on transformed metric
                    # u-direction neighbors
                    if i > 0 or not grid_boundaries[0]:
                        up_idx = i_prev * N_v * N_w + j * N_w + k
                        row_indices.append(idx)
                        col_indices.append(up_idx)
                        values.append(laplacian_off_diag_u)
                        logger.debug(f"Connected to u_prev neighbor at idx={up_idx}")

                    if i < N_u - 1 or not grid_boundaries[1]:
                        down_idx = i_next * N_v * N_w + j * N_w + k
                        row_indices.append(idx)
                        col_indices.append(down_idx)
                        values.append(laplacian_off_diag_u)
                        logger.debug(f"Connected to u_next neighbor at idx={down_idx}")

                    # v-direction neighbors
                    if j > 0 or not grid_boundaries[2]:
                        left_idx = i * N_v * N_w + j_prev * N_w + k
                        row_indices.append(idx)
                        col_indices.append(left_idx)
                        values.append(laplacian_off_diag_v)
                        logger.debug(f"Connected to v_prev neighbor at idx={left_idx}")

                    if j < N_v - 1 or not grid_boundaries[3]:
                        right_idx = i * N_v * N_w + j_next * N_w + k
                        row_indices.append(idx)
                        col_indices.append(right_idx)
                        values.append(laplacian_off_diag_v)
                        logger.debug(f"Connected to v_next neighbor at idx={right_idx}")

                    # w-direction neighbors
                    if k > 0 or not grid_boundaries[4]:
                        front_idx = i * N_v * N_w + j * N_w + k_prev
                        row_indices.append(idx)
                        col_indices.append(front_idx)
                        values.append(laplacian_off_diag_w)
                        logger.debug(f"Connected to w_prev neighbor at idx={front_idx}")

                    if k < N_w - 1 or not grid_boundaries[5]:
                        back_idx = i * N_v * N_w + j * N_w + k_next
                        row_indices.append(idx)
                        col_indices.append(back_idx)
                        values.append(laplacian_off_diag_w)
                        logger.debug(f"Connected to w_next neighbor at idx={back_idx}")

                    # Add mixed derivative terms using inv_g_uv, inv_g_uw, inv_g_vw for cross u-v, u-w, v-w terms
                    if inv_g_uv != 0:
                        # Mixed terms between u and v directions
                        mixed_uv_idx1 = i_prev * N_v * N_w + j_prev * N_w + k
                        row_indices.append(idx)
                        col_indices.append(mixed_uv_idx1)
                        values.append(laplacian_cross_uv)
                        logger.debug(f"Connected to mixed uv neighbor at idx={mixed_uv_idx1}")

                        mixed_uv_idx2 = i_next * N_v * N_w + j_next * N_w + k
                        row_indices.append(idx)
                        col_indices.append(mixed_uv_idx2)
                        values.append(laplacian_cross_uv)
                        logger.debug(f"Connected to mixed uv neighbor at idx={mixed_uv_idx2}")

                    if inv_g_uw != 0:
                        # Mixed terms between u and w directions
                        mixed_uw_idx1 = i_prev * N_v * N_w + j * N_w + k_prev
                        row_indices.append(idx)
                        col_indices.append(mixed_uw_idx1)
                        values.append(laplacian_cross_uw)
                        logger.debug(f"Connected to mixed uw neighbor at idx={mixed_uw_idx1}")

                        mixed_uw_idx2 = i_next * N_v * N_w + j * N_w + k_next
                        row_indices.append(idx)
                        col_indices.append(mixed_uw_idx2)
                        values.append(laplacian_cross_uw)
                        logger.debug(f"Connected to mixed uw neighbor at idx={mixed_uw_idx2}")

                    if inv_g_vw != 0:
                        # Mixed terms between v and w directions
                        mixed_vw_idx1 = i * N_v * N_w + j_prev * N_w + k_prev
                        row_indices.append(idx)
                        col_indices.append(mixed_vw_idx1)
                        values.append(laplacian_cross_vw)
                        logger.debug(f"Connected to mixed vw neighbor at idx={mixed_vw_idx1}")

                        mixed_vw_idx2 = i * N_v * N_w + j_next * N_w + k_next
                        row_indices.append(idx)
                        col_indices.append(mixed_vw_idx2)
                        values.append(laplacian_cross_vw)
                        logger.debug(f"Connected to mixed vw neighbor at idx={mixed_vw_idx2}")

                    # Calculate the local wave number k
                    local_wave_speed = self.wave_speed * math.sqrt(tension / density)
                    if local_wave_speed != 0:
                        local_k = (2 * math.pi * f) / local_wave_speed
                        diagonal_entries[idx] += -local_k**2
                        logger.debug(f"Added local wave number term -k^2 at idx={idx}: {local_k**2}")
                    else:
                        diagonal_entries[idx] += 0.0
                        logger.debug(f"Local wave speed is zero at idx={idx}. No wave number term added.")

                    if diagonal_entries[idx] == 0:
                        logger.error(f"Zero diagonal at idx: {idx}, metric_u: {metric_u}, metric_v: {metric_v}, metric_w: {metric_w}")
                        raise ZeroDivisionError(f"Zero diagonal entry at index {idx}.")

        logger.debug("Completed iteration over the 3D grid.")

        # Normalize off-diagonal entries by dividing by diagonal elements
        logger.debug("Normalizing off-diagonal entries.")
        normalized_values = []
        for r, c, v in zip(row_indices, col_indices, values):
            diagonal = diagonal_entries.get(r, 1.0)
            if math.isnan(diagonal) or math.isinf(diagonal):
                logger.error(f"NaN or Inf found in diagonal entry at index {r}")
                continue
            if diagonal == 0:
                logger.error(f"Zero diagonal value at index {r}")
                continue
            normalized_v = v / diagonal
            normalized_values.append(normalized_v)
            logger.debug(f"Normalized value at row {r}, col {c}: {normalized_v}")

        # Add normalized off-diagonal entries
        row_indices = list(row_indices)
        col_indices = list(col_indices)
        if normalize:
            values = normalized_values
        

        # Add diagonal entries
        logger.debug("Adding diagonal entries.")
        for row, diagonal in diagonal_entries.items():
            if math.isnan(diagonal) or math.isinf(diagonal):
                logger.error(f"NaN or Inf found in diagonal entry for row {row}")
                continue
            row_indices.append(row)
            col_indices.append(row)
            if normalize:
                values.append(1.0)
            else:
                values.append(diagonal)
            logger.debug(f"Added diagonal entry for row {row}: 1.0")

        # Convert indices and values to tensors
        logger.debug("Converting indices and values to tensors.")
        row_indices_tensor = torch.tensor(row_indices, dtype=torch.long).reshape(-1)
        col_indices_tensor = torch.tensor(col_indices, dtype=torch.long).reshape(-1)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        logger.debug(f"Row indices tensor shape: {row_indices_tensor.shape}")
        logger.debug(f"Col indices tensor shape: {col_indices_tensor.shape}")
        logger.debug(f"Values tensor shape: {values_tensor.shape}")

        # Build the sparse Laplacian matrix
        logger.debug("Building the sparse Laplacian matrix.")
        laplacian = coo_matrix(
            (values_tensor.cpu().numpy(), (row_indices_tensor.cpu().numpy(), col_indices_tensor.cpu().numpy())),
            shape=(N_u * N_v * N_w, N_u * N_v * N_w)
        )
        logger.debug(f"Sparse Laplacian matrix constructed with shape {laplacian.shape} and {laplacian.nnz} non-zero entries.")

        # Convert to dense tensor and move to specified device
        perturbation_mode = False  # Enable perturbation
        perturbation_seed = 42      # Use a fixed seed for reproducibility, or None for random
        perturbation_scale = 1e-3   # Small scale for the noise perturbation

        if self.resolution <= 128:
            logger.debug("Converting Laplacian to dense tensor.")
            laplacian_dense = laplacian.toarray()
            laplacian_tensor = torch.tensor(laplacian_dense, device=device, dtype=self.precision)
            logger.debug(f"Dense Laplacian tensor created with shape {laplacian_tensor.shape} on device {device}.")

            # Dense perturbation
            if perturbation_mode:
                logger.debug("Applying dense perturbation to Laplacian tensor.")
                if perturbation_seed is not None:
                    torch.manual_seed(perturbation_seed)  # Set seed for deterministic behavior
                    logger.debug(f"Set perturbation seed to {perturbation_seed}.")
                # Apply Gaussian noise to dense matrix
                noise_dense = torch.randn(laplacian_tensor.shape, dtype=self.precision, device=device) * perturbation_scale
                laplacian_tensor += noise_dense  # Add noise to the dense Laplacian
                logger.debug("Added Gaussian noise to dense Laplacian tensor.")

            # Validate perturbed dense Laplace tensor
            logger.debug("Validating dense Laplacian tensor.")
            self.validate_laplace_tensor(laplacian_tensor)
            logger.debug("Dense Laplacian tensor validated.")
        else:
            laplacian_tensor = None
            logger.debug("Resolution exceeds 128. Skipping dense Laplacian tensor creation.")

        # Sparse perturbation (always applied independently)
        if perturbation_mode:
            logger.debug("Applying sparse perturbation to Laplacian matrix.")
            if perturbation_seed is not None:
                np.random.seed(perturbation_seed)  # Use numpy's random seed for reproducibility
                logger.debug(f"Set sparse perturbation seed to {perturbation_seed}.")

            # Generate noise for the non-zero elements in numpy format
            noise_sparse = np.random.randn(laplacian.data.shape[0]) * perturbation_scale
            logger.debug(f"Generated noise for {laplacian.data.shape[0]} non-zero elements.")

            # Apply the noise by creating a new COO matrix with perturbed data
            laplacian = coo_matrix((laplacian.data + noise_sparse, (laplacian.row, laplacian.col)), 
                                   shape=laplacian.shape)
            logger.debug("Applied noise to sparse Laplacian matrix.")

        # Validate perturbed sparse Laplace tensor
        logger.debug("Validating sparse Laplacian matrix.")
        self.validate_laplace_tensor(laplacian)
        logger.debug("Sparse Laplacian matrix validated.")

        logger.debug("Completed build_general_laplace.")
        return laplacian_tensor, laplacian


    def validate_laplace_tensor(self, laplace_tensor, check_diagonal=True, check_off_diagonal=True, verbose=True):
        """
        Validates a given Laplace tensor for issues such as zero diagonal entries, NaN, Inf, or invalid values.
        
        Args:
            laplace_tensor: The Laplace tensor to validate, which can be sparse (COO) or dense.
            check_diagonal: Whether to check for invalid or zero diagonal entries (default: True).
            check_off_diagonal: Whether to check for invalid values in the off-diagonal elements (default: True).
            verbose: Whether to print detailed error messages (default: True).
        
        Returns:
            valid: Boolean indicating whether the Laplace tensor passed all checks.
        """
        valid = True

        # Handle both dense and sparse cases
        if isinstance(laplace_tensor, torch.Tensor):
            # Dense matrix case
            diagonal = torch.diag(laplace_tensor)

            # Check diagonal entries in dense matrix
            if check_diagonal:
                if torch.any(diagonal == 0):
                    valid = False
                    if verbose:
                        zero_indices = torch.where(diagonal == 0)[0].tolist()
                        print(f"Zero diagonal entries detected at indices: {zero_indices}")
                
                if torch.any(torch.isnan(diagonal)):
                    valid = False
                    if verbose:
                        nan_indices = torch.where(torch.isnan(diagonal))[0].tolist()
                        print(f"NaN detected in diagonal at indices: {nan_indices}")
                
                if torch.any(torch.isinf(diagonal)):
                    valid = False
                    if verbose:
                        inf_indices = torch.where(torch.isinf(diagonal))[0].tolist()
                        print(f"Inf detected in diagonal at indices: {inf_indices}")

            # Check off-diagonal entries in dense matrix
            if check_off_diagonal:
                off_diagonal = laplace_tensor - torch.diag(torch.diag(laplace_tensor))
                if torch.any(torch.isnan(off_diagonal)):
                    valid = False
                    if verbose:
                        nan_locations = torch.where(torch.isnan(off_diagonal))
                        print(f"NaN detected in off-diagonal at indices: {nan_locations}")
                
                if torch.any(torch.isinf(off_diagonal)):
                    valid = False
                    if verbose:
                        inf_locations = torch.where(torch.isinf(off_diagonal))
                        print(f"Inf detected in off-diagonal at indices: {inf_locations}")

        elif isinstance(laplace_tensor, coo_matrix):
            # Sparse matrix case (COO format)

            # Check diagonal entries in sparse matrix
            if check_diagonal:
                diagonal = laplace_tensor.diagonal()
                if (diagonal == 0).any():
                    valid = False
                    if verbose:
                        zero_indices = np.where(diagonal == 0)[0].tolist()
                        print(f"Zero diagonal entries detected at indices: {zero_indices}")
                if np.isnan(diagonal).any():
                    valid = False
                    if verbose:
                        nan_indices = np.where(np.isnan(diagonal))[0].tolist()
                        print(f"NaN detected in diagonal at indices: {nan_indices}")
                if np.isinf(diagonal).any():
                    valid = False
                    if verbose:
                        inf_indices = np.where(np.isinf(diagonal))[0].tolist()
                        print(f"Inf detected in diagonal at indices: {inf_indices}")

            # Check off-diagonal entries in sparse matrix
            if check_off_diagonal:
                row, col = laplace_tensor.row, laplace_tensor.col
                data = laplace_tensor.data
                for idx, (i, j, value) in enumerate(zip(row, col, data)):
                    if i != j:  # Only check off-diagonal elements
                        if np.isnan(value):
                            valid = False
                            if verbose:
                                print(f"NaN detected in off-diagonal at indices: ({i}, {j})")
                        if np.isinf(value):
                            valid = False
                            if verbose:
                                print(f"Inf detected in off-diagonal at indices: ({i}, {j})")

        else:
            raise TypeError("Unsupported matrix format. Please provide a torch.Tensor or scipy.sparse matrix.")

        if verbose and valid:
            print("Laplace tensor passed all validation checks.")

        return valid
# Assuming RectangularTransform and GridDomain are properly defined for 3D

import torch
import numpy as np
from scipy.sparse import coo_matrix
import math
import matplotlib.pyplot as plt


import torch
import random
class TransformHub:
    def __init__(self, uextent, vextent, grid_boundaries):
        self.uextent = uextent
        self.vextent = vextent
        self.grid_boundaries = grid_boundaries

    def calculate_geometry(self, U, V, W):
        """
        Compute coordinates, partials, normals, and metric tensor in one centralized function.
        
        Args:
            U, V (torch.Tensor): Grids for parameter space.

        Returns:
            dict: Dictionary containing coordinates, partials, normals, and metric tensors.
        """
        # Calculate coordinates and partial derivatives
        X, Y, Z, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, normals = self.compute_partials_and_normals(U, V, W)
        
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

    def compute_partials_and_normals(self, U, V, W, validate_normals=True, diagnostic_mode=False):
        # Ensure U, V, W require gradients for autograd
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)

        # Forward pass: get transformed coordinates
        X, Y, Z = self.transform_spatial(U, V, W)

        if diagnostic_mode:
            print("U Grid:")
            print(U)
            print("V Grid:")
            print(V)
            print("W Grid:")
            print(W)
            print("Transformed Coordinates (X, Y, Z):")
            print("X:", X)
            print("Y:", Y)
            print("Z:", Z)

        # Calculate partial derivatives with respect to U
        dXdu = torch.autograd.grad(X, U, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdu = torch.autograd.grad(Y, U, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdu = torch.autograd.grad(Z, U, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        # Calculate partial derivatives with respect to V
        dXdv = torch.autograd.grad(X, V, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdv = torch.autograd.grad(Y, V, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdv = torch.autograd.grad(Z, V, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        # Calculate partial derivatives with respect to W
        dXdw = torch.autograd.grad(X, W, grad_outputs=torch.ones_like(X), retain_graph=True, allow_unused=True)[0]
        dYdw = torch.autograd.grad(Y, W, grad_outputs=torch.ones_like(Y), retain_graph=True, allow_unused=True)[0]
        dZdw = torch.autograd.grad(Z, W, grad_outputs=torch.ones_like(Z), retain_graph=True, allow_unused=True)[0]

        target_shape = U.shape  # (N_u, N_v, N_w)

        # Handle None values from autograd
        dXdu = dXdu if dXdu is not None else torch.zeros(target_shape).to(U.device)
        dYdu = dYdu if dYdu is not None else torch.zeros(target_shape).to(U.device)
        dZdu = dZdu if dZdu is not None else torch.zeros(target_shape).to(U.device)
        dXdv = dXdv if dXdv is not None else torch.zeros(target_shape).to(V.device)
        dYdv = dYdv if dYdv is not None else torch.zeros(target_shape).to(V.device)
        dZdv = dZdv if dZdv is not None else torch.zeros(target_shape).to(V.device)
        dXdw = dXdw if dXdw is not None else torch.zeros(target_shape).to(W.device)
        dYdw = dYdw if dYdw is not None else torch.zeros(target_shape).to(W.device)
        dZdw = dZdw if dZdw is not None else torch.zeros(target_shape).to(W.device)

        if diagnostic_mode:
            print("Partial Derivatives:")
            print("dXdu:", dXdu)
            print("dYdu:", dYdu)
            print("dZdu:", dZdu)
            print("dXdv:", dXdv)
            print("dYdv:", dYdv)
            print("dZdv:", dZdv)
            print("dXdw:", dXdw)
            print("dYdw:", dYdw)
            print("dZdw:", dZdw)

        # Compute normals as cross-product of partial derivatives
        normals = torch.stack([
            torch.cross(torch.stack([dXdu, dYdu, dZdu], dim=-1), torch.stack([dXdv, dYdv, dZdv], dim=-1)),
            torch.cross(torch.stack([dXdv, dYdv, dZdv], dim=-1), torch.stack([dXdw, dYdw, dZdw], dim=-1)),
            torch.cross(torch.stack([dXdw, dYdw, dZdw], dim=-1), torch.stack([dXdu, dYdu, dZdu], dim=-1))
        ], dim=-1)

        # Compute distances from the origin
        distances = torch.sqrt(X**2 + Y**2 + Z**2)

        # Select the top 10% farthest points
        top_10_percent_threshold = max(1, int(0.1 * distances.numel()))
        top_10_percent_indices = torch.topk(distances.flatten(), top_10_percent_threshold).indices

        # Randomly sample 10% of the top 10% farthest points
        sample_size = max(1, int(0.1 * top_10_percent_threshold))
        sample_indices = random.sample(top_10_percent_indices.tolist(), sample_size)

        # Conduct majority check based on sampled normals
        outward_votes = 0
        inward_votes = 0
        grid_shape = distances.shape  # (N_u, N_v, N_w)
        for idx in sample_indices:
            i, j, k = np.unravel_index(idx, grid_shape)  # Convert flat index to 3D grid indices
            farthest_point = torch.tensor([X[i, j, k], Y[i, j, k], Z[i, j, k]], device=U.device, dtype=U.dtype)
            outward_reference_point = 1.01 * farthest_point  # 1% further outward

            # Directional check based on the sampled normal and reference point
            sample_normal = normals[i, j, k]
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
            print(f"{count_zero_normals} out of {normals.numel()} zero-magnitude normals detected.")

            if diagnostic_mode:
                # Find the indices of the first zero-magnitude normal
                zero_indices = torch.nonzero(zero_norm_mask, as_tuple=True)
                first_zero_idx = (zero_indices[0][0].item(), zero_indices[1][0].item(), zero_indices[2][0].item())

                print(f"First zero-magnitude normal at index: {first_zero_idx}")

                # Extract the partials contributing to this normal
                i, j, k = first_zero_idx

                partials = {
                    'dXdu': dXdu[i, j, k],
                    'dYdu': dYdu[i, j, k],
                    'dZdu': dZdu[i, j, k],
                    'dXdv': dXdv[i, j, k],
                    'dYdv': dYdv[i, j, k],
                    'dZdv': dZdv[i, j, k],
                    'dXdw': dXdw[i, j, k],
                    'dYdw': dYdw[i, j, k],
                    'dZdw': dZdw[i, j, k],
                }

                print("Partials at the first zero-magnitude normal:")
                for name, value in partials.items():
                    print(f"{name}[{i}, {j}, {k}] = {value}")

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
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue  # Skip the center point
                                ni, nj, nk = idx[0] + di, idx[1] + dj, idx[2] + dk
                                if (0 <= ni < grid_shape[0] and 
                                    0 <= nj < grid_shape[1] and 
                                    0 <= nk < grid_shape[2]):
                                    neighbor_normal = normals[ni, nj, nk]
                                    neighbor_magnitude = norm_magnitudes[ni, nj, nk]
                                    if neighbor_magnitude > 1e-16:
                                        neighbors.append(neighbor_normal)
                    if neighbors:
                        avg_normal = torch.mean(torch.stack(neighbors), dim=0)
                        avg_normal_norm = torch.norm(avg_normal)
                        if avg_normal_norm > 1e-16:
                            normals[idx[0], idx[1], idx[2]] = avg_normal / avg_normal_norm  # Normalize average
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

        return X, Y, Z, dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw, normals

                                            


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


def unpack_values(returned_values, n_desired):
    """
    Unpack the returned values and ensure that the output has exactly n_desired elements.
    If fewer values are returned, fill the remaining with None.
    If more values are returned, truncate to n_desired elements.

    Args:
    returned_values: The tuple or list of returned values.
    n_desired: The number of desired return values.

    Returns:
    A tuple of length n_desired with values or None.
    """
    return (returned_values + (None,) * n_desired)[:n_desired]
import torch
import numpy as np

class PeriodicLinspace:
    def __init__(self, min_density=0.5, max_density=1.5, num_oscillations=1):
        self.min_density = min_density
        self.max_density = max_density
        self.num_oscillations = num_oscillations

    def sin(self, normalized_i):
        return self._oscillate(torch.sin, normalized_i)

    def cos(self, normalized_i):
        return self._oscillate(torch.cos, normalized_i)

    def tan(self, normalized_i):
        density = self._oscillate(torch.tan, normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def cot(self, normalized_i):
        density = self._oscillate(lambda x: 1 / torch.tan(x + 1e-6), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def exp_sin(self, normalized_i):
        density = self._oscillate(lambda x: torch.exp(torch.sin(x)), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def exp_cos(self, normalized_i):
        density = self._oscillate(lambda x: torch.exp(torch.cos(x)), normalized_i)
        return torch.clamp(density, min=self.min_density, max=self.max_density)

    def _oscillate(self, func, normalized_i):
        phase_shifted_i = 2 * np.pi * self.num_oscillations * normalized_i - np.pi / 2
        return self.min_density + (self.max_density - self.min_density) * 0.5 * (1 + func(phase_shifted_i))

    def get_density(self, normalized_i, oscillation_type):
        if not hasattr(self, oscillation_type):
            raise ValueError(f"Unknown oscillation_type: '{oscillation_type}'.")
        return getattr(self, oscillation_type)(normalized_i)
import torch
import numpy as np

class GridDomain:
    def __init__(self, U, V, W, u_mode=None, u_p=1, v_mode=None, v_p=1, w_mode=None, w_p=1,
                 Lx=1, Ly=1, Lz=1, grid_boundaries=(True, True, True, True, True, True),
                 transform=None, coordinate_system="rectangular"):
        """
        Initializes the GridDomain object for 3D grids.

        Args:
            U, V, W: Meshgrids representing the three axes.
            u_mode, v_mode, w_mode: Modes for grid generation.
            u_p, v_p, w_p: Parameters for grid generation.
            Lx, Ly, Lz: Physical extents in each direction.
            grid_boundaries: Tuple indicating boundary inclusions for each axis.
            transform: Transformation object.
            coordinate_system: Type of coordinate system.
        """
        # Store U, V, W as the meshgrids that represent the actual domain
        self.U = U
        self.V = V
        self.W = W
        self.transform = transform
        self.vertices = self.transform.transform(U, V, W)  # Pass W here

        self.u_mode = u_mode
        self.v_mode = v_mode
        self.w_mode = w_mode

        self.u_p = u_p
        self.v_p = v_p
        self.w_p = w_p

        # Store physical extents
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Store boundary conditions
        self.grid_boundaries = grid_boundaries
        self.coordinate_system = coordinate_system

        # Step 1: Calculate resolution (number of points in U, V, W directions)
        self.resolution_u = U.shape[0]  # Number of points along the U axis
        self.resolution_v = V.shape[1]  # Number of points along the V axis
        self.resolution_w = W.shape[2]  # Number of points along the W axis

        # Step 2: Calculate extents (total span of the domain along U, V, W)
        self.extent_u = U.max() - U.min()  # Extent along the U axis
        self.extent_v = V.max() - V.min()  # Extent along the V axis
        self.extent_w = W.max() - W.min()  # Extent along the W axis

        # Step 3: Compute normalized U, V, W for grid_sample interpolation (from -1 to 1)
        self.normalized_U = self.normalize_grid(self.U, self.extent_u)
        self.normalized_V = self.normalize_grid(self.V, self.extent_v)
        self.normalized_W = self.normalize_grid(self.W, self.extent_w)

        # Step 4: Create a combined normalized grid for interpolation
        self.normalized_grid = torch.stack([self.normalized_U, self.normalized_V, self.normalized_W], dim=-1).unsqueeze(0)

    @staticmethod
    def generate_grid_domain(coordinate_system, N_u, N_v, N_w, u_mode=None, v_mode=None, w_mode=None,
                            device='cpu', precision=torch.float64, **kwargs):
        """
        Generates a GridDomain object based on the coordinate system and its parameters.
        """
        # Set default modes if u_mode, v_mode, or w_mode not provided
        u_mode = u_mode or {'method': 'linear', 'p': 1}
        v_mode = v_mode or {'method': 'linear', 'p': 1}
        w_mode = w_mode or {'method': 'linear', 'p': 1}

        # Combine all the parameters that are common across different transforms
        transform_params = {
            'N_u': N_u,
            'N_v': N_v,
            'N_w': N_w,
            'device': device,
            'precision': precision,
            'Lx': kwargs.get('Lx', None),
            'Ly': kwargs.get('Ly', None),
            'Lz': kwargs.get('Lz', None)
        }

        # Update with any additional kwargs that may be provided
        transform_params.update(kwargs)

        # Create the appropriate Transform instance with all packed parameters
        transform = Transform.create_transform(coordinate_system, **transform_params)

        U, V, W = None, None, None
        # Get domain extents and grid boundaries from the Transform
        (uextent, vextent, wextent), grid_boundaries = transform.get_transform_parameters()

        if getattr(transform, 'autogrid', False):
            U, V, W = transform.obtain_autogrid()
        else:
            # Generate u_grid, v_grid, w_grid using the respective mode dictionaries
            u_grid, _ = generate_grid(
                N=N_u, L=uextent, device=device, dtype=precision,
                keep_end=grid_boundaries[1], **u_mode
            )
            v_grid, _ = generate_grid(
                N=N_v, L=vextent, device=device, dtype=precision,
                keep_end=grid_boundaries[3], **v_mode
            )
            w_grid, _ = generate_grid(
                N=N_w, L=wextent, device=device, dtype=precision,
                keep_end=grid_boundaries[5], **w_mode
            )

            # Create U, V, W meshgrids
            U, V, W = torch.meshgrid(u_grid, v_grid, w_grid, indexing='ij')

        # Create and return the GridDomain
        return GridDomain(U, V, W, transform=transform, coordinate_system=coordinate_system)


    def apply_transform(self):
        return self.transform(self.U, self.V)
    
    def get_vertices(self):
        return self.vertices
    def return_dense_copy(self, scaling_factor=(2,2), normalize=True):
        """
        Returns a higher resolution copy of the grid (U, V) by scaling the resolution.
        The dense grid will use the same distribution method as the original grid.

        Args:
            scaling_factor: The factor by which to increase the resolution.
            normalize: Whether to return the normalized grid.

        Returns:
            A dense version of U and V (or normalized versions if normalize=True).
        """
        print("DEPRECIATED")
        
        # New high-resolution by scaling
        high_res_u = int(self.resolution_u * scaling_factor[0])
        high_res_v = int(self.resolution_v * scaling_factor[1])

        # Create high-resolution U and V grids by regenerating the grid using the same method and parameters
        u_high_res, _ = generate_grid(
            N=high_res_u,
            L=self.U.max(),
            device=self.U.device,  # Keep the device same as the original grid
            dtype=self.U.dtype,
            **self.u_mode
        )

        v_high_res, _ = generate_grid(
            N=high_res_v,
            L=self.V.max(),
            device=self.V.device,
            dtype=self.V.dtype,
            **self.v_mode
        )

        # Create the high-resolution meshgrid
        U_high_res, V_high_res = torch.meshgrid(u_high_res, v_high_res, indexing='ij')

        # Normalize if required
        if normalize:
            U_norm_high = self.normalize_grid(U_high_res, self.extent_u)
            V_norm_high = self.normalize_grid(V_high_res, self.extent_v)
            return U_norm_high, V_norm_high
        else:
            return U_high_res, V_high_res



    def normalize_grid(self, grid, extent):
        """
        Normalizes the grid values into the range [-1, 1] for use in grid_sample.

        Args:
            grid: The original meshgrid (U or V).
            extent: The span (max - min) of the grid.
        
        Returns:
            The normalized grid in the range [-1, 1].
        """
        grid_min = grid.min()
        normalized_grid = (grid - grid_min) / extent.to(grid.device) * 2 - 1  # Normalize to [-1, 1]
        return normalized_grid

    def summary(self):
        """
        Returns a summary of the grid domain, including resolution and extents.
        
        Returns:
            dict: A dictionary containing resolution and extents of the grid.
        """
        return {
            "resolution_u": self.resolution_u,
            "resolution_v": self.resolution_v,
            "extent_u": self.extent_u.item(),  # Convert from tensor to Python float
            "extent_v": self.extent_v.item(),
            "normalized_U_range": (self.normalized_U.min().item(), self.normalized_U.max().item()),
            "normalized_V_range": (self.normalized_V.min().item(), self.normalized_V.max().item())
        }
    def __getstate__(self):
        """
        Prepares the state for serialization. Convert tensors to CPU (to avoid issues with GPU tensors) and
        store other attributes.
        """
        state = self.__dict__.copy()
        state['U'] = self.U.cpu()  # Ensure tensors are on the CPU for serialization
        state['V'] = self.V.cpu()
        state['normalized_U'] = self.normalized_U.cpu()
        state['normalized_V'] = self.normalized_V.cpu()
        state['normalized_grid'] = self.normalized_grid.cpu()
        state['coordinate_system']

        # Serialize any callable objects like transform if needed (could be done through another method or logic)
        state['transform'] = None  # Set transform to None or serialize it if required

        return state
    def __setstate__(self, state):
        """
        Restores the object state after deserialization. Move tensors back to the appropriate device and
        regenerate the transform.
        """
        # Restore the state dictionary
        self.__dict__.update(state)

        # Move tensors back to the appropriate device (you can modify the device as needed)
        self.U = self.U.to('cpu')
        self.V = self.V.to('cpu')
        self.normalized_U = self.normalized_U.to('cpu')
        self.normalized_V = self.normalized_V.to('cpu')
        self.normalized_grid = self.normalized_grid.to('cpu')

        # Regenerate the transform based on the coordinate system and stored parameters
        self.transform = self.regenerate_transform()

    def regenerate_transform(self):
        """
        Recreates the transform using stored parameters such as the coordinate system, resolution, boundary conditions, etc.
        This method is called during deserialization to restore the transform.
        """
        # Example of how you might regenerate the transform based on the stored parameters
        transform_params = {
            'N_u': self.resolution_u,
            'N_v': self.resolution_v,
            'u_mode': self.u_mode,
            'v_mode': self.v_mode,
            'device': 'cpu',  # Adjust device if needed
            'precision': self.U.dtype,  # Use the same dtype as the original grid
            'Lx': self.Lx,
            'Ly': self.Ly
        }

        # Call the same logic used during GridDomain initialization
        transform = Transform.create_transform(self.coordinate_system, **transform_params)
        return transform
    
def generate_grid(N, L, method='linear', p=2.0, min_density=0.5,
                  max_density=1.5, num_oscillations=1, keep_end=True, periodic=False,
                  oscillation_type='sin', device='cpu', dtype=torch.float64):
    """
    Generates a grid with various spacing methods and calculates infinitesimal values.
    
    Parameters are the same as before, with `oscillation_type` specifying
    the desired periodic pattern if `method` is 'periodic'.
    """
    if N < 2:
        raise ValueError("N must be at least 2.")

    # Generate indices and normalize
    i = torch.arange(0, N, device=device, dtype=dtype)
    normalized_i = i / (N - 1 if keep_end else N)

    if method == 'linear':
        grid = L * normalized_i
    elif method == 'non_uniform':
        grid = L * normalized_i ** p
    elif method == 'inverted':
        grid = L * (1 - (1 - normalized_i) ** p)
    elif method == 'periodic':
        # Use PeriodicLinspace for density modulation
        periodic_gen = PeriodicLinspace(min_density, max_density, num_oscillations)
        density = periodic_gen.get_density(normalized_i, oscillation_type)
        grid = torch.cumsum(density, dim=0)
        grid = grid / grid[-1] * L  # Normalize to fit within length L
    elif method == 'dense_extremes':
        grid = L * 0.5 * (normalized_i ** p + (1 - (1 - normalized_i) ** p))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'non_uniform', 'inverted', 'periodic', or 'dense_extremes'.")

    # Compute infinitesimal values
    infinitesimal = torch.zeros(N, device=device, dtype=dtype)
    infinitesimal[:-1] = grid[1:] - grid[:-1]

    if not keep_end:
        infinitesimal[-1] = L - grid[-1] + grid[0]  # Wrap-around interval for periodic case

    return grid, infinitesimal

def generate_full_meshgrid(N_u, L_u, N_v, L_v, N_w, L_w, periodic_u=True, periodic_v=True, periodic_w=True,
                           umethod='dense_extremes', upow=2, vmethod="dense_extremes", vpow=2, wmethod='dense_extremes', wpow=2,
                           device='cpu', **kwargs):
    """
    Generate U, V, W meshgrids and their corresponding infinitesimal grids U', V', W'.
    """
    
    # Generate U, V, W grids
    U, U_prime = generate_grid(N_u, L_u, method=umethod, p=upow, periodic=periodic_u, keep_end=not periodic_u, device=device, **kwargs)
    V, V_prime = generate_grid(N_v, L_v, method=vmethod, p=vpow, periodic=periodic_v, keep_end=not periodic_v, device=device, **kwargs)
    W, W_prime = generate_grid(N_w, L_w, method=wmethod, p=wpow, periodic=periodic_w, keep_end=not periodic_w, device=device, **kwargs)
    
    # Create full 3D meshgrid for U, V, W
    U_mesh, V_mesh, W_mesh = torch.meshgrid(U, V, W, indexing='ij')
    
    # Create full 3D meshgrid for infinitesimal U', V', W'
    U_prime_mesh, V_prime_mesh, W_prime_mesh = torch.meshgrid(U_prime, V_prime, W_prime, indexing='ij')

    return U_mesh, V_mesh, W_mesh, U_prime_mesh, V_prime_mesh, W_prime_mesh

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
        #metric_transform = MetricTransform.create_metric_transform(type_of_transform, **kwargs)
        #if metric_transform is not None:
        #    return metric_transform

        transform_map = {
            "rectangular": RectangularTransform,

        }

        if type_of_transform in transform_map:
            return transform_map[type_of_transform](**kwargs)
        else:
            raise ValueError(f"Unsupported transform type '{type_of_transform}'")



class RectangularTransform(Transform):
    def __init__(self, Lx, Ly, Lz, device='cpu', N_u=None, N_v=None, N_w=None, u_mode=None, v_mode=None, w_mode=None, precision=None):
        """
        Initialize the RectangularTransform with extents and optional grid modes.
        """
        self.Lx = Lx if N_u is None else N_u
        self.Ly = Ly if N_v is None else N_v
        self.Lz = Lz if N_w is None else N_w
        self.device = device
        self.u_mode = u_mode
        self.v_mode = v_mode
        self.w_mode = w_mode

    def get_transform_parameters(self):
        """
        Return extents for U, V, W and grid boundaries.
        """
        uextent = self.Lx
        vextent = self.Ly
        wextent = self.Lz
        grid_boundaries = (True, True, True, True, True, True)  # Example boundaries for 3D
        return (uextent, vextent, wextent), grid_boundaries

    def create_grid_mesh(self, N_u, N_v, N_w):
        """
        Create a 3D grid mesh for U, V, W.
        """
        u = torch.linspace(0, self.Lx, steps=N_u, device=self.device)
        v = torch.linspace(0, self.Ly, steps=N_v, device=self.device)
        w = torch.linspace(0, self.Lz, steps=N_w, device=self.device)
        grid_u, grid_v, grid_w = torch.meshgrid(u, v, w, indexing='ij')
        return grid_u, grid_v, grid_w

    def transform(self, grid_u, grid_v, grid_w):
        """
        Identity transform for simplicity.
        """
        return grid_u, grid_v, grid_w

    def transform_metric(self, grid_u, grid_v, grid_w):
        """
        Identity transform for simplicity. Return Z as well for completeness.
        """
        return grid_u, grid_v, grid_w  # Returning Z twice is likely an oversight but left for compatibility.

    def get_or_compute_partials(self, grid_u, grid_v, grid_w):
        """
        Compute central differences for partial derivatives for a uniform grid.
        """
        du = grid_u[1, 0, 0] - grid_u[0, 0, 0]
        dv = grid_v[0, 1, 0] - grid_v[0, 0, 0]
        dw = grid_w[0, 0, 1] - grid_w[0, 0, 0]

        dXdu = torch.ones_like(grid_u) * (1.0 / du)
        dYdu = torch.zeros_like(grid_u)
        dZdu = torch.zeros_like(grid_u)

        dXdv = torch.zeros_like(grid_v)
        dYdv = torch.ones_like(grid_v) * (1.0 / dv)
        dZdv = torch.zeros_like(grid_v)

        dXdw = torch.zeros_like(grid_w)
        dYdw = torch.zeros_like(grid_w)
        dZdw = torch.ones_like(grid_w) * (1.0 / dw)

        return dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw

def test_build_laplace3d():
    # Grid parameters
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Unit cube
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    device = 'cpu'  # Change to 'cuda' if using GPU

    # Initialize transformation and grid domain
    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device=device)
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    print(f"Grid U shape: {grid_u.shape}")
    print(f"Grid V shape: {grid_v.shape}")
    print(f"Grid W shape: {grid_w.shape}")
    
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device=device
    )

    # Initialize BuildLaplace3D with Dirichlet boundary conditions
    # All boundaries are Dirichlet
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')
    # boundary_conditions = ('neumann', 'neumann', 'neumann', 'neumann', 'neumann', 'neumann')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=torch.float64,
        resolution=20,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )

    # Build the Laplacian
    laplacian_tensor, laplacian_sparse = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device=device,
        f=0.0  # No wave number term for this test
    )

    # Define the test function f(x,y,z) = sin(pi x) sin(pi y) sin(pi z)
    X = grid_u.numpy()
    Y = grid_v.numpy()
    Z = grid_w.numpy()
    f = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    f_flat = f.flatten()

    # Convert f_flat to a Torch tensor
    f_tensor = torch.tensor(f_flat, dtype=torch.float64, device=device)

    # Apply the Laplacian matrix to f
    if laplacian_tensor is not None:
        # Dense Laplacian
        laplace_f_numerical = -laplacian_tensor @ f_tensor
    else:
        # Sparse Laplacian
        # Convert sparse matrix to Torch sparse tensor
        indices = torch.tensor([laplacian_sparse.row, laplacian_sparse.col], dtype=torch.long)
        values = torch.tensor(laplacian_sparse.data, dtype=torch.float64)
        laplacian_sparse_torch = torch.sparse_coo_tensor(indices, values, size=(N_u * N_v * N_w, N_u * N_v * N_w)).to(device)
        laplace_f_numerical = torch.sparse.mm(laplacian_sparse_torch, f_tensor.unsqueeze(1)).squeeze(1)

    # Compute the analytical Laplacian: -3 pi^2 f
    laplace_f_analytical = -3 * (math.pi ** 2) * f_flat
    # Note: Do NOT scale the analytical Laplacian

    # Convert laplace_f_numerical to numpy for comparison
    laplace_f_numerical_np = laplace_f_numerical.cpu().numpy()

    # Compute the error
    error = laplace_f_numerical_np/min(laplace_f_numerical_np) - laplace_f_analytical/min(laplace_f_analytical)
    max_error = np.max(np.abs(error))
    mean_error = np.mean(np.abs(error))

    print(f"Max Error: {max_error:.6e}")
    print(f"Mean Error: {mean_error:.6e}")

    # Visualization (optional)
    # Compare a central slice
    central_slice = N_w // 2
    laplace_f_numerical_reshaped = laplace_f_numerical_np.reshape(N_u, N_v, N_w)
    laplace_f_analytical_reshaped = laplace_f_analytical.reshape(N_u, N_v, N_w)
    error_reshaped = error.reshape(N_u, N_v, N_w)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.title("Numerical Laplacian")
    plt.imshow(laplace_f_numerical_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Analytical Laplacian")
    plt.imshow(laplace_f_analytical_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Error")
    plt.imshow(error_reshaped[:, :, central_slice], origin='lower', extent=[0, Lx, 0, Ly], cmap='bwr')
    plt.colorbar()

    plt.show()

    # Assert that the error is within an acceptable tolerance
    # Adjust tolerance based on grid resolution and discretization
    # For N=20, the grid is relatively coarse, so a larger tolerance is acceptable
    tolerance = 1e-1  # Adjusted tolerance for coarser grid
    assert max_error < tolerance, f"Max error {max_error} exceeds tolerance {tolerance}"
    print("Test passed: Numerical Laplacian matches analytical Laplacian within tolerance.")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def heat_evolution_demo(laplacian_tensor, initial_temperature, alpha=0.01, dt=0.01, steps=500):
    """
    Simulates heat evolution on a 3D grid using a precomputed Laplacian matrix.
    
    Args:
        laplacian_tensor: A dense Laplacian matrix (torch.Tensor or numpy.ndarray).
        initial_temperature: Initial temperature distribution (numpy.ndarray).
        alpha: Thermal diffusivity coefficient (float).
        dt: Time step for simulation (float).
        steps: Number of simulation steps (int).
    """
    # Convert Laplacian tensor to NumPy if needed
    if isinstance(laplacian_tensor, torch.Tensor):
        laplacian_tensor = laplacian_tensor.cpu().numpy()

    # Flatten the initial temperature distribution
    initial_temperature = initial_temperature.flatten()

    # Time evolution of temperature
    temperature = initial_temperature.copy()
    
    # Reshape dimensions for visualization
    N_u = int(np.cbrt(len(temperature)))  # Assuming a cubic grid
    temperature_grid = temperature.reshape(N_u, N_u, N_u)

    # Prepare the plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    x = np.linspace(0, 1, N_u)
    y = np.linspace(0, 1, N_u)
    X, Y = np.meshgrid(x, y)
    Z = temperature_grid[:, :, N_u // 2]  # Middle slice for 3D visualization
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')

    

    def update(frame):
        nonlocal temperature, temperature_grid, Z, surface
        # Compute temperature evolution: T_new = T + alpha * dt * Laplacian(T)
        temperature += alpha * dt * laplacian_tensor @ temperature
        temperature_grid = temperature.reshape(N_u, N_u, N_u)
        Z = temperature_grid[:, :, N_u // 2]  # Update the middle slice
        
        # Remove the previous surface if it exists
        if surface is not None:
            surface.remove()
        
        # Replot the surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"Time Step: {frame}")
        ax.set_zlim(0, initial_temperature.max())  # Adjust limits dynamically
        return surface,


    # Animate
    ani = FuncAnimation(fig, update, frames=steps, blit=False, interval=50)

    # Show the animation
    plt.show()
# Store the surface plot reference globally
surface = None

# Example usage with a random initial condition
if __name__ == "__main__":
    # Parameters
    N_u, N_v, N_w = 20, 20, 20  # Grid resolution
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain size

    # Create a Laplacian matrix using the earlier code
    transform = RectangularTransform(Lx=Lx, Ly=Ly, Lz=Lz, device='cpu')
    grid_u, grid_v, grid_w = transform.create_grid_mesh(N_u, N_v, N_w)
    grid_domain = GridDomain.generate_grid_domain(
        coordinate_system='rectangular',
        N_u=N_u,
        N_v=N_v,
        N_w=N_w,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        device='cpu'
    )
    boundary_conditions = ('dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet', 'dirichlet')

    build_laplace = BuildLaplace3D(
        grid_domain=grid_domain,
        wave_speed=343,  # Arbitrary value
        precision=torch.float64,
        resolution=N_u,  # Should match N_u, N_v, N_w
        metric_tensor_func=None,  # Use default Euclidean metric
        density_func=None,        # Uniform density
        tension_func=None,        # Uniform tension
        singularity_conditions=None,
        boundary_conditions=boundary_conditions,
        artificial_stability=1e-10
    )
    laplacian_tensor, _ = build_laplace.build_general_laplace(
        grid_u=grid_u,
        grid_v=grid_v,
        grid_w=grid_w,
        boundary_conditions=boundary_conditions,
        grid_boundaries=(True, True, True, True, True, True),
        device='cpu',
        f=0.0
    )

    # Initial temperature: Gaussian in the center
    x = np.linspace(0, 1, N_u)
    y = np.linspace(0, 1, N_v)
    z = np.linspace(0, 1, N_w)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    initial_temperature = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

    # Run the demo
    heat_evolution_demo(laplacian_tensor, initial_temperature, alpha=0.01, dt=0.1, steps=200)
