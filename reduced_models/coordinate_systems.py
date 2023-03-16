import dolfin as df
from typing import Protocol, Callable 


class CoordinateSystem(Protocol):
    @property
    def stiffness_form(self) -> Callable[[df.Expression, df.Expression, df.Expression], df.Form]:
        """Returns a bilinearform for -\nabla \cdot m \nabla u 
           with homogeneous Neumann boundary conditions in the 
           current coordiante system."""

    @property
    def mass_form(self) -> Callable[[df.Expression, df.Expression], df.Form]:
        """Returns a bilinearform for the mass matrix in the
           current coordiante system."""


class SphericalCoordinateSystem:
    def __init__(self, mesh):
        self.mesh = mesh
    
    @property
    def r(self):
        return df.SpatialCoordinate(self.mesh)

    def diff_r(self, u):
        '''Derivative in r direction.'''
        return df.Dx(u, 0)
    
    @property
    def stiffness_form(self):
        def a(m, u, v):
            '''Radial symmetric spherical laplacian with a mobility term.'''
            return self.r**2 * m * self.diff_r(u) * self.diff_r(v) * df.Constant(4 * df.pi) * df.dx
        return a

    @property
    def mass_form(self):
        def m(u, v):
            '''Creates Radial symmetric spherical functional.'''
            return df.Constant(4 * df.pi) * self.r**2 * u * v * df.dx
        return m
        

class CylindricalCoordinateSystem:
    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def r(self):
        return df.SpatialCoordinate(self.mesh)[0]

    @property
    def z(self):
        return df.SpatialCoordinate(self.mesh)[1]

    def diff_r(self, u):
        '''Derivative in r direction.'''
        return df.Dx(u, 0)

    def diff_z(self, u):
        '''Derivative in z direction.'''
        return df.Dx(u, 1)

    @property
    def stiffness_form(self):
        def a(m, u, v):
            '''Radial symmetric spherical laplacian with a mobility term.'''
            return m * (self.diff_r(u) * self.diff_r(v) + self.diff_z(u) * self.diff_z(v)) * self.r * df.Constant(2 * df.pi) * df.dx
        return a

    @property
    def mass_form(self):
        def m(u, v):
            '''Creates Radial symmetric spherical functional.'''
            return df.Constant(2 * df.pi) * self.r * u * v * df.dx
        return m


class CartesianCoordinateSystem:
    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def stiffness_form(self):
        def a(m, u, v):
            '''Radial symmetric spherical laplacian with a mobility term.'''
            return m * df.inner(df.grad(u), df.grad(v)) * df.dx
        return a

    @property
    def mass_form(self):
        def m(u, v):
            '''Creates Radial symmetric spherical functional.'''
            return df.inner(u, v) * df.dx(domain=self.mesh)
        return m