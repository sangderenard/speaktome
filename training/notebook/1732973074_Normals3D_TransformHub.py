import torch

class Normals3D:
    def __init__(self, partials):
        """
        Initialize with precomputed partial derivatives.

        Args:
            partials (dict): Dictionary containing partial derivatives:
                - dX_dU, dY_dU, dZ_dU
                - dX_dV, dY_dV, dZ_dV
                - dX_dW, dY_dW, dZ_dW
        """
        self.partials = partials

    def compute_surface_normals(self, primary_axis="U", secondary_axis="V"):
        """
        Compute surface normals for a parametric surface defined by two axes.

        Args:
            primary_axis (str): One axis defining the surface ("U", "V", or "W").
            secondary_axis (str): The other axis defining the surface ("U", "V", or "W").

        Returns:
            torch.Tensor: Normalized surface normals for the given parametric axes.
        """
        if primary_axis == secondary_axis:
            raise ValueError("Primary and secondary axes must be different.")

        # Get partial derivatives for the chosen axes
        dX1 = self.partials[f"dX_d{primary_axis}"]
        dY1 = self.partials[f"dY_d{primary_axis}"]
        dZ1 = self.partials[f"dZ_d{primary_axis}"]

        dX2 = self.partials[f"dX_d{secondary_axis}"]
        dY2 = self.partials[f"dY_d{secondary_axis}"]
        dZ2 = self.partials[f"dZ_d{secondary_axis}"]

        # Compute cross product
        normals = torch.cross(
            torch.stack([dX1, dY1, dZ1], dim=-1),
            torch.stack([dX2, dY2, dZ2], dim=-1),
            dim=-1
        )
        return self._normalize(normals)

    def compute_volume_normals(self):
        """
        Compute volumetric normals using all three parametric axes.

        Returns:
            torch.Tensor: Normalized volumetric normals for each grid point.
        """
        # Compute normals using three cross products: dU x dV, dV x dW, and dW x dU
        normals_dU_dV = torch.cross(
            torch.stack([self.partials["dX_dU"], self.partials["dY_dU"], self.partials["dZ_dU"]], dim=-1),
            torch.stack([self.partials["dX_dV"], self.partials["dY_dV"], self.partials["dZ_dV"]], dim=-1),
            dim=-1
        )

        normals_dV_dW = torch.cross(
            torch.stack([self.partials["dX_dV"], self.partials["dY_dV"], self.partials["dZ_dV"]], dim=-1),
            torch.stack([self.partials["dX_dW"], self.partials["dY_dW"], self.partials["dZ_dW"]], dim=-1),
            dim=-1
        )

        normals_dW_dU = torch.cross(
            torch.stack([self.partials["dX_dW"], self.partials["dY_dW"], self.partials["dZ_dW"]], dim=-1),
            torch.stack([self.partials["dX_dU"], self.partials["dY_dU"], self.partials["dZ_dU"]], dim=-1),
            dim=-1
        )

        # Combine volumetric normals (sum of contributions from all faces)
        combined_normals = normals_dU_dV + normals_dV_dW + normals_dW_dU
        return self._normalize(combined_normals)

    @staticmethod
    def _normalize(vectors):
        """
        Normalize a batch of vectors to unit length.

        Args:
            vectors (torch.Tensor): Input vectors of shape (..., 3).

        Returns:
            torch.Tensor: Normalized vectors of the same shape.
        """
        magnitudes = torch.norm(vectors, dim=-1, keepdim=True)
        return torch.where(magnitudes > 1e-16, vectors / magnitudes, torch.zeros_like(vectors))

class TransformHub:
    def __init__(self, transform_function):
        """
        Initialize with a function that maps (U, V, W) to (X, Y, Z).
        
        Args:
            transform_function: Callable that takes (U, V, W) tensors and outputs (X, Y, Z).
        """
        self.transform_function = transform_function

    def calculate_geometry(self, U, V, W):
        """
        Compute coordinates, partials, normals, and metric tensor for a given parametric domain.

        Args:
            U, V, W: Torch tensors representing the parametric grid.

        Returns:
            dict: A dictionary containing:
                - "coordinates": (X, Y, Z)
                - "partials": Partial derivatives with respect to U, V, W.
                - "normals": Normals for slices, shells, and volumes.
                - "metric": Metric tensor for the geometry.
        """
        # Ensure gradients are enabled only if needed
        U.requires_grad_(not U.requires_grad)
        V.requires_grad_(not V.requires_grad)
        W.requires_grad_(not W.requires_grad)

        # Transform parametric domain to 3D coordinates
        X, Y, Z = self.transform_function(U, V, W)

        # Compute partial derivatives
        dX_dU, dY_dU, dZ_dU = self._compute_grad(X, Y, Z, U)
        dX_dV, dY_dV, dZ_dV = self._compute_grad(X, Y, Z, V)
        dX_dW, dY_dW, dZ_dW = self._compute_grad(X, Y, Z, W)

        # Stack partial derivatives for later use
        partials = {
            "dX_dU": dX_dU, "dY_dU": dY_dU, "dZ_dU": dZ_dU,
            "dX_dV": dX_dV, "dY_dV": dY_dV, "dZ_dV": dZ_dV,
            "dX_dW": dX_dW, "dY_dW": dY_dW, "dZ_dW": dZ_dW,
        }

        # Compute normals and metric tensor
        normals = Normals3D(partials=partials)
        metric = self._compute_3d_metric_tensor(partials)

        return {
            "coordinates": (X, Y, Z),
            "partials": partials,
            "normals": normals,
            "metric": metric,
        }

    def extract_slice(self, fixed_axis, fixed_value, U, V, W):
        """
        Extract a 2D slice by fixing one parameter (U, V, or W).

        Args:
            fixed_axis (str): The axis to fix ("U", "V", or "W").
            fixed_value (float): The value to fix the parameter to.
            U, V, W: Parametric grids.

        Returns:
            dict: Geometry and normals for the extracted slice.
        """
        if fixed_axis == "U":
            U = torch.full_like(U, fixed_value)
        elif fixed_axis == "V":
            V = torch.full_like(V, fixed_value)
        elif fixed_axis == "W":
            W = torch.full_like(W, fixed_value)
        else:
            raise ValueError(f"Invalid fixed_axis: {fixed_axis}. Must be 'U', 'V', or 'W'.")

        # Recalculate geometry for the fixed parameter
        return self.calculate_geometry(U, V, W)

    def extract_shell(self, isosurface_function, U, V, W):
        """
        Extract a shell (isosurface) defined by a scalar function.

        Args:
            isosurface_function (callable): Function f(U, V, W) -> scalar.
            U, V, W: Parametric grids.

        Returns:
            dict: Geometry, partials, and normals for the isosurface.
        """
        # Compute scalar field
        scalar_field = isosurface_function(U, V, W)

        # Placeholder: Actual implementation requires advanced isosurface extraction (e.g., marching cubes)
        raise NotImplementedError("Isosurface extraction requires specialized methods (e.g., marching cubes).")

    def project_to_plane(self, projection_axis, U, V, W):
        """
        Project the parametric geometry onto a plane by collapsing along one axis.

        Args:
            projection_axis (str): The axis to collapse ("U", "V", or "W").
            U, V, W: Parametric grids.

        Returns:
            torch.Tensor: Projected 2D coordinates.
        """
        X, Y, Z = self.transform_function(U, V, W)

        if projection_axis == "U":
            return torch.stack([Y, Z], dim=-1)
        elif projection_axis == "V":
            return torch.stack([X, Z], dim=-1)
        elif projection_axis == "W":
            return torch.stack([X, Y], dim=-1)
        else:
            raise ValueError(f"Invalid projection_axis: {projection_axis}. Must be 'U', 'V', or 'W'.")

    def _compute_grad(self, X, Y, Z, param):
        """
        Compute gradients of (X, Y, Z) with respect to a given parametric input.
        """
        dX = self._safe_grad(X, param)
        dY = self._safe_grad(Y, param)
        dZ = self._safe_grad(Z, param)
        return dX, dY, dZ

    def _safe_grad(self, tensor, param):
        """
        Compute gradient safely, replacing None with zeros if gradient computation fails.
        """
        grad = torch.autograd.grad(tensor, param, grad_outputs=torch.ones_like(tensor), retain_graph=True, allow_unused=True)[0]
        return grad if grad is not None else torch.zeros_like(param)

    def _compute_3d_metric_tensor(self, partials):
        """
        Compute the 3D metric tensor and its determinant.
        """
        g_uu = partials["dX_dU"]**2 + partials["dY_dU"]**2 + partials["dZ_dU"]**2
        g_vv = partials["dX_dV"]**2 + partials["dY_dV"]**2 + partials["dZ_dV"]**2
        g_ww = partials["dX_dW"]**2 + partials["dY_dW"]**2 + partials["dZ_dW"]**2

        g_uv = partials["dX_dU"] * partials["dX_dV"] + partials["dY_dU"] * partials["dY_dV"] + partials["dZ_dU"] * partials["dZ_dV"]
        g_uw = partials["dX_dU"] * partials["dX_dW"] + partials["dY_dU"] * partials["dY_dW"] + partials["dZ_dU"] * partials["dZ_dW"]
        g_vw = partials["dX_dV"] * partials["dX_dW"] + partials["dY_dV"] * partials["dY_dW"] + partials["dZ_dV"] * partials["dZ_dW"]

        g_ij = torch.stack([
            torch.stack([g_uu, g_uv, g_uw], dim=-1),
            torch.stack([g_uv, g_vv, g_vw], dim=-1),
            torch.stack([g_uw, g_vw, g_ww], dim=-1),
        ], dim=-2)

        det_g = torch.det(g_ij)
        return {"metric_tensor": g_ij, "determinant": det_g}

# Define a parametric transform (e.g., twisted torus)
def twisted_torus(U, V, W, R=2.0, r=1.0):
    twist = W * 2 * torch.pi
    X = (R + r * torch.cos(V)) * torch.cos(U + twist)
    Y = (R + r * torch.cos(V)) * torch.sin(U + twist)
    Z = r * torch.sin(V)
    return X, Y, Z

# Initialize TransformHub
transform_hub = TransformHub(transform_function=twisted_torus)

# Generate parametric grid
U = torch.linspace(0, 2 * torch.pi, 50)
V = torch.linspace(0, 2 * torch.pi, 50)
W = torch.linspace(0, 1, 20)
U, V, W = torch.meshgrid(U, V, W, indexing='ij')

# Calculate geometry
geometry = transform_hub.calculate_geometry(U, V, W)

# Access features
coordinates = geometry["coordinates"]
partials = geometry["partials"]
normals = geometry["normals"]
metric = geometry["metric"]

print(coordinates, partials, normals, metric)


