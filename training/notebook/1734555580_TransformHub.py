import torch
import random
import numpy as np

from hodge_star_builder import HodgeStarBuilder
from face_map_generator import FaceMapGenerator
from volume_map_generator import VolumeMapGenerator  # Placeholder for future volume logic


class TransformHub:
    def __init__(self, uextent, vextent, grid_boundaries, device="cpu"):
        self.uextent = uextent
        self.vextent = vextent
        self.grid_boundaries = grid_boundaries
        self.device = device
        self.hodge_builder = HodgeStarBuilder(device=device)

    def calculate_geometry(self, U, V, W, edge_index=None, detect_faces=False, detect_volumes=False):
        """
        Compute geometry (coordinates, metric tensors, normals) and primary DEC operators.
        Optionally detect faces and volumes for advanced DEC computations.
        """
        # Compute coordinates and metric-related data
        X, Y, Z, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dX_dW, dY_dW, dZ_dW, normals = \
            self.compute_partials_and_normals(U, V, W)
        g_ij, g_inv, det_g = self.metric_tensor_func(
            U, V, W, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dX_dW, dY_dW, dZ_dW
        )

        geometry = {
            "coordinates": (X, Y, Z),
            "normals": normals,
            "metric": {"tensor": g_ij, "inverse": g_inv, "determinant": det_g}
        }

        if edge_index is not None:
            network_profile = self.build_network_profile(edge_index, X, Y, Z)
            d_operators = self.build_d_operators(edge_index, network_profile)
            vertices = torch.stack([X, Y, Z], dim=1)  # (N, 3) vertex positions

            # Base Hodge star: without faces or volumes, can still do vertex and edge ops
            hodge_stars = self.hodge_builder.build_basic_hodge_star(vertices, edge_index)

            geometry["DEC"] = {
                "network_profile": network_profile,
                "d_operators": d_operators,
                "hodge_stars": hodge_stars
            }

            # Optional: Enhanced detection of faces
            if detect_faces:
                face_generator = FaceMapGenerator(vertices, edge_index, device=self.device)
                face_map = face_generator.generate_face_map()
                geometry["DEC"]["faces"] = face_map

                # Update Hodge stars now that we have faces
                hodge_stars = self.hodge_builder.build_full_hodge_star(vertices, edge_index, face_map)
                geometry["DEC"]["hodge_stars"] = hodge_stars

            # Optional: Enhanced detection of volumes (3-simplices)
            if detect_volumes:
                volume_generator = VolumeMapGenerator(vertices, edge_index, geometry["DEC"].get("faces", None), device=self.device)
                volume_map = volume_generator.generate_volume_map()
                geometry["DEC"]["volumes"] = volume_map
                # Extend Hodge stars further with volume information if needed
                # hodge_stars = self.hodge_builder.build_3d_hodge_star(vertices, edge_index, face_map, volume_map)
                # geometry["DEC"]["hodge_stars"] = hodge_stars

        return geometry

    def build_network_profile(self, edge_index, X, Y, Z):
        source, target = edge_index[:, 0], edge_index[:, 1]
        edge_lengths = torch.sqrt((X[source] - X[target])**2 +
                                  (Y[source] - Y[target])**2 +
                                  (Z[source] - Z[target])**2)
        num_vertices = X.numel()
        return {
            "edge_index": edge_index,
            "edge_lengths": edge_lengths,
            "num_vertices": num_vertices,
            "num_edges": edge_index.shape[0]
        }

    def build_d_operators(self, edge_index, network_profile):
        num_vertices = network_profile["num_vertices"]
        num_edges = network_profile["num_edges"]

        d0 = torch.zeros((num_edges, num_vertices), device=self.device)
        for i, (src, tgt) in enumerate(edge_index):
            d0[i, src] = -1
            d0[i, tgt] = 1

        # d1 placeholder if we have faces defined
        # This can be constructed similarly if face maps are available
        d1 = None

        return {"d0": d0, "d1": d1}

    def metric_tensor_func(self, U, V, W,
                           dX_dU, dY_dU, dZ_dU,
                           dX_dV, dY_dV, dZ_dV,
                           dX_dW, dY_dW, dZ_dW):
        # Compute metric tensor g_ij, its inverse, and determinant from partial derivatives
        # Similar to the original implementation
        g_uu = dX_dU**2 + dY_dU**2 + dZ_dU**2
        g_vv = dX_dV**2 + dY_dV**2 + dZ_dV**2
        g_ww = dX_dW**2 + dY_dW**2 + dZ_dW**2
        g_uv = dX_dU*dX_dV + dY_dU*dY_dV + dZ_dU*dZ_dV
        g_uw = dX_dU*dX_dW + dY_dU*dY_dW + dZ_dU*dZ_dW
        g_vw = dX_dV*dX_dW + dY_dV*dY_dW + dZ_dV*dZ_dW

        g_ij = torch.stack([
            torch.stack([g_uu, g_uv, g_uw], dim=-1),
            torch.stack([g_uv, g_vv, g_vw], dim=-1),
            torch.stack([g_uw, g_vw, g_ww], dim=-1)
        ], dim=-2)

        det_g = (g_uu * (g_vv * g_ww - g_vw**2)
                 - g_uv * (g_uv * g_ww - g_vw * g_uw)
                 + g_uw * (g_uv * g_vw - g_vv * g_uw))

        det_g = torch.clamp(det_g, min=1e-16)

        # Inverse metric
        g_inv = torch.zeros_like(g_ij)
        g_inv[..., 0, 0] = (g_vv*g_ww - g_vw**2) / det_g
        g_inv[..., 0, 1] = (g_uw*g_vw - g_uv*g_ww) / det_g
        g_inv[..., 0, 2] = (g_uv*g_vw - g_uw*g_vv) / det_g
        g_inv[..., 1, 0] = g_inv[..., 0, 1]
        g_inv[..., 1, 1] = (g_uu*g_ww - g_uw**2) / det_g
        g_inv[..., 1, 2] = (g_uw*g_uv - g_uu*g_vw) / det_g
        g_inv[..., 2, 0] = g_inv[..., 0, 2]
        g_inv[..., 2, 1] = g_inv[..., 1, 2]
        g_inv[..., 2, 2] = (g_uu*g_vv - g_uv**2) / det_g

        return g_ij, g_inv, det_g

    def transform_spatial(self, U, V, W):
        # Must be overridden by subclass
        raise NotImplementedError("Subclasses must implement transform_spatial().")

    def compute_partials_and_normals(self, U, V, W, validate_normals=True, diagnostic_mode=False):
        U.requires_grad_(True)
        V.requires_grad_(True)
        W.requires_grad_(True)

        X, Y, Z = self.transform_spatial(U, V, W)

        # Compute partial derivatives
        onesX = torch.ones_like(X)
        dXdu = torch.autograd.grad(X, U, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dYdu = torch.autograd.grad(Y, U, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dZdu = torch.autograd.grad(Z, U, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]

        dXdv = torch.autograd.grad(X, V, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dYdv = torch.autograd.grad(Y, V, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dZdv = torch.autograd.grad(Z, V, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]

        dXdw = torch.autograd.grad(X, W, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dYdw = torch.autograd.grad(Y, W, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]
        dZdw = torch.autograd.grad(Z, W, grad_outputs=onesX, retain_graph=True, allow_unused=True)[0]

        # Handle None returns from autograd
        def safe(tensor, shape, device):
            return tensor if tensor is not None else torch.zeros(shape, device=device)

        shape = U.shape
        dXdu = safe(dXdu, shape, U.device)
        dYdu = safe(dYdu, shape, U.device)
        dZdu = safe(dZdu, shape, U.device)
        dXdv = safe(dXdv, shape, U.device)
        dYdv = safe(dYdv, shape, U.device)
        dZdv = safe(dZdv, shape, U.device)
        dXdw = safe(dXdw, shape, U.device)
        dYdw = safe(dYdw, shape, U.device)
        dZdw = safe(dZdw, shape, U.device)

        # Compute normals (3D surface normal logic may need refinement; for now, just returning partial-based normals)
        # For a 3D parameterization, you may want to return 3 normals corresponding to local parameter directions.
        # Here, we can form normals by cross products of pairs of partials.
        # For demonstration, let's just pick dU and dV partials to form one normal:
        normal_vec = torch.cross(torch.stack([dXdu, dYdu, dZdu], dim=-1),
                                 torch.stack([dXdv, dYdv, dZdv], dim=-1), dim=-1)
        norm_magnitudes = torch.norm(normal_vec, dim=-1, keepdim=True)
        normals = torch.where(norm_magnitudes > 1e-16, normal_vec/norm_magnitudes, normal_vec)

        if validate_normals:
            if torch.any(torch.isnan(normals)):
                raise ValueError("NaN detected in normals.")
            # Additional validation checks can be added here.

        return X, Y, Z, dXdu, dYdu, dZdu, dXdv, dYdv, dZdv, dXdw, dYdw, dZdw, normals

    def get_or_compute_partials(self, U, V, W):
        _, _, _, dX_dU, dY_dU, dZ_dU, dX_dV, dY_dV, dZ_dV, dXdw, dYdw, dZdw, _ = self.compute_partials_and_normals(U, V, W)
        return dX_dU, dY_d_U, dZ_dU, dX_dV, dY_dV, dZ_dV, dXdw, dYdw, dZdw
