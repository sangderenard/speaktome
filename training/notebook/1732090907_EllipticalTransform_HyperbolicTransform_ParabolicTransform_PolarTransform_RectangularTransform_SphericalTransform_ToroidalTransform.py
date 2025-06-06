from .base_transform import Transform
import torch

class PolarTransform(Transform):
    def __init__(self, Lx, **kwargs):
        super().__init__(Lx / 2, 2 * torch.pi, (True, True, True, False))
        self.Lx = Lx

    def get_domain_extents(self):
        uextent = self.Lx / 2  # Radial extent
        vextent = 2 * torch.pi  # Theta wraps around
        grid_boundaries = (True, True, True, False)
        return uextent, vextent, grid_boundaries

    def transform_metric(self, U=None, V=None):
        return self.transform_spatial(U, V)

    def transform_spatial(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V

        X = U * torch.cos(V)
        Y = U * torch.sin(V)
        Z = U * V * 0.0
        return X, Y, Z

    def evaluate(self, u_coord, v_coord):
        X = u_coord * torch.cos(v_coord)
        Y = u_coord * torch.sin(v_coord)
        return X, Y
    

class ToroidalTransform(Transform):
    def __init__(self, Lx, Ly, **kwargs):
        super().__init__(2 * torch.pi, 2 * torch.pi, (True, False, True, False))  # Both angles are periodic
        self.Lx = self.R_major = Lx
        self.Ly = self.R_minor = Ly

    @staticmethod
    def get_domain_extents():
        uextent = 2 * torch.pi  # Theta wraps around
        vextent = 2 * torch.pi  # Phi wraps around
        grid_boundaries = (True, False, True, False)
        return uextent, vextent, grid_boundaries

    def transform_spatial(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        X = (self.R_major + self.R_minor * torch.cos(U)) * torch.cos(V)
        Y = (self.R_major + self.R_minor * torch.cos(U)) * torch.sin(V)
        Z = self.R_minor * torch.sin(U)
        return X, Y, Z  # Return Z as well for higher dimensionality cases

    def transform_metric(self, U, V):
        # In toroidal geometry, we need the metric-based transformation
        return self.transform_spatial(U, V)  # The default toroidal transform already incorporates the necessary structure

    def evaluate(self, u_coord, v_coord):
        X = (self.R_major + self.R_minor * torch.cos(u_coord)) * torch.cos(v_coord)
        Y = (self.R_major + self.R_minor * torch.cos(u_coord)) * torch.sin(v_coord)
        Z = self.R_minor * torch.sin(u_coord)
        return X, Y, Z



class SphericalTransform(Transform):
    def __init__(self, radius=1.0, **kwargs):
        super().__init__(torch.pi, 2 * torch.pi, (True, True, True, False))  # Theta is from 0 to pi, Phi wraps around
        self.radius = radius  # The radius of the sphere

    @staticmethod
    def get_domain_extents():
        uextent = torch.pi  # Theta extends from 0 to pi
        vextent = 2 * torch.pi  # Phi wraps around
        grid_boundaries = (True, True, True, False) 
        return uextent, vextent, grid_boundaries

    def transform_spatial(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        X = self.radius * torch.sin(U) * torch.cos(V)
        Y = self.radius * torch.sin(U) * torch.sin(V)
        Z = self.radius * torch.cos(U)
        return X, Y, Z

    def transform_metric(self, U, V):
        # For the spherical geometry, the metric-based transformation
        return self.transform_spatial(U, V)  # Spherical geometry is inherently captured in the spatial transform

    def evaluate(self, u_coord, v_coord):
        X = self.radius * torch.sin(u_coord) * torch.cos(v_coord)
        Y = self.radius * torch.sin(u_coord) * torch.sin(v_coord)
        Z = self.radius * torch.cos(u_coord)
        return X, Y, Z



class RectangularTransform(Transform):
    def __init__(self, Lx, Ly, **kwargs):
        super().__init__(Lx, Ly, (True, True, True, True))
        self.Lx = Lx
        self.Ly = Ly
    
    def get_domain_extents(self):
        uextent = self.Lx
        vextent = self.Ly
        grid_boundaries = (True, True, True, True)  # Rectangular keeps both U and V ends
        return uextent, vextent, grid_boundaries
    
    def transform_spatial(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        Z = U * V * 0
        return U, V, Z

    def evaluate(self, u_coord, v_coord):
        return u_coord, v_coord

class EllipticalTransform(Transform):
    def __init__(self, Lx, Ly, **kwargs):
        super().__init__(Lx / 2, 2 * torch.pi, (True, True, True, False))  # Polar-like, doesn't keep theta end
        self.Lx = Lx
        self.Ly = Ly

    def get_domain_extents(self):
        uextent = self.Lx / 2
        vextent = 2 * torch.pi  # Theta wraps around
        grid_boundaries = (True, True, True, False)
        return uextent, vextent, grid_boundaries

    def transform_spatial(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        X = U * torch.cos(V)
        Y = U * torch.sin(V) * (self.Lx / self.Ly)
        Z = U * V * 0
        return X, Y, Z
    
    def transform_metric(self, U=None, V=None):
        return self.transform_spatial(U, V)[:2]

    def evaluate(self, u_coord, v_coord):
        X = u_coord * torch.cos(v_coord)
        Y = u_coord * torch.sin(v_coord) * (self.Lx / self.Ly)
        return X, Y

class HyperbolicTransform(Transform):
    def __init__(self, Lx, Ly, **kwargs):
        super().__init__(Lx, Ly, (True, True, True, True))
        self.Lx = Lx
        self.Ly = Ly

    def get_domain_extents(self):
        uextent = self.Lx
        vextent = self.Ly
        grid_boundaries = (True, True, True, True)
        return uextent, vextent, grid_boundaries

    def transform(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        X = torch.cosh(U) * torch.cos(V)
        Y = torch.sinh(U) * torch.sin(V)
        return X, Y

    def evaluate(self, u_coord, v_coord):
        X = torch.cosh(u_coord) * torch.cos(v_coord)
        Y = torch.sinh(u_coord) * torch.sin(v_coord)
        return X, Y

class ParabolicTransform(Transform):
    def __init__(self, Lx, Ly, **kwargs):
        super().__init__(Lx, Ly, (True, True, True, True))
        self.Lx = Lx
        self.Ly = Ly

    def get_domain_extents(self):
        uextent = self.Lx
        vextent = self.Ly
        grid_boundaries = (True, True, True, True)
        return uextent, vextent, grid_boundaries

    def transform(self, U=None, V=None):
        U = U if U is not None else self.U
        V = V if V is not None else self.V
        X = U * U + V
        Y = U + V * V
        return X, Y

    def evaluate(self, u_coord, v_coord):
        X = u_coord * u_coord + v_coord
        Y = u_coord + v_coord * v_coord
        return X, Y
