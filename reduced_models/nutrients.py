import os
import time
import numpy as np
import dolfin as df
import petsc4py as p4py
from dataclasses import dataclass
import vasculature_io as vio
from integrate_2d_surface import integrate_2d_surface 
from initial_conditions import refine_locally


class NutrientSolver:
    def __init__(self, mesh, kappa) -> None:
        self.function_space = V = df.FunctionSpace(mesh, 'P', 1)

        self.f_2d = df.Function(V, name='rhs')

        self.kappa = kappa
    
    def assemble_2d_source_terms(self, surface_triangles, accuracy):
        num_triangles = int(len(surface_triangles) / 3)
        surface_values = 1 * np.ones(num_triangles)

        tic = time.perf_counter()
        self.indices_list, self.values_list = integrate_2d_surface(self.function_space, surface_triangles, surface_values, accuracy)
        toc = time.perf_counter()
        print(f"needed {toc - tic:0.4f} seconds to assemble the 2d surface integrals")

        f_vec = self.f_2d.vector().vec()
        f_vec.setValues(self.indices_list, self.values_list, p4py.PETSc.InsertMode.ADD)
        f_vec.assemble()

    def save_2d_source_terms(self, path):
        with df.XDMFFile(path) as f:
            f.write_checkpoint(self.f_2d, 'nutrients', 0, df.XDMFFile.Encoding.HDF5, True)

    def load_2d_source_terms(self, path):
        with df.XDMFFile(path) as f:
            f.read_checkpoint(self.f_2d, 'nutrients')

    def zero_2d_source_terms(self):
        self.f_2d.interpolate(df.Constant(0))
    
    def solve(self, alpha, beta, f):
        u = df.TrialFunction(self.function_space)
        v = df.TestFunction(self.function_space)
        a = self.kappa * df.inner(df.grad(u), df.grad(v)) * df.dx + alpha * u * v * df.dx
        bcs = []

        A = df.assemble(a)
        b = df.assemble(f * v * df.dx)
        df.as_backend_type(b).axpy(beta, self.f_2d.vector())

        for bc in bcs:
            bc.apply(A)
            bc.apply(b)

        u = df.Function(self.function_space, name='nutrients')

        solver = df.PETScKrylovSolver('cg')
        df.PETScOptions.set('pc_type', 'hypre')
        df.PETScOptions.set('pc_hypre_type', 'boomeramg')
        solver.set_operator(A)
        solver.set_from_options()

        it = solver.solve(u.vector(), b)
        print(f'solving nutrients needed {it} iterations')

        return u

@dataclass
class ParameterNutrients:
    kappa_v:float = 1e-1
    kappa_i:float = 1e-5
    alpha_H:float = 1.
    alpha_P:float = 4.
    beta:float = 1e-1
    xi_va:float = 0
    eta_vi:float = 1.0
    eta_iv:float = 0.0


@dataclass
class ResultNutrientSolver:
    solution: df.Function
    nut_i: df.Function
    nut_v: df.Function


class NutrientDoubleContinuumSolver:
    def __init__(self, mesh, parameter: ParameterNutrients, save_at: str) -> None:
        P1   = df.FiniteElement("Lagrange", df.tetrahedron, 1)
        self.function_space = W = df.FunctionSpace(mesh, df.MixedElement([P1, P1]))
        df.FunctionSpace(mesh, 'P', 1)

        self.f_2d = df.Function(W, name='rhs')

        self.parameter = parameter

        self.NUT_0 = df.Function(W, name='nutrients')

        self.NUT = df.Function(W, name='nutrients')

        self.file_NUT_v = df.File(os.path.join(save_at, 'nutrients_v.pvd'))
        self.file_NUT_i = df.File(os.path.join(save_at, 'nutrients_i.pvd'))
    
    @property
    def NUT_i_0(self):
        return self.NUT_0.sub(1)

    @property
    def NUT_v_0(self):
        return self.NUT_0.sub(0)

    @property
    def NUT_i(self):
        return self.NUT.sub(1)

    @property
    def NUT_v(self):
        return self.NUT.sub(0)
    
    def write(self, t):
        self.file_NUT_v << (self.NUT_v, t) 
        self.file_NUT_i << (self.NUT_i, t) 
    
    def reassign(self):
        self.NUT_0.assign(self.NUT)
    
    def assemble_2d_source_terms(self, surface_triangles, accuracy):
        num_triangles = int(len(surface_triangles) / 3)
        surface_values = 1 * np.ones(num_triangles)

        tic = time.perf_counter()
        self.indices_list, self.values_list = integrate_2d_surface(self.function_space.sub(0), surface_triangles, surface_values, accuracy)
        toc = time.perf_counter()
        print(f"needed {toc - tic:0.4f} seconds to assemble the 2d surface integrals")

        f_vec = self.f_2d.vector().vec()
        f_vec.setValues(self.indices_list, self.values_list, p4py.PETSc.InsertMode.ADD)
        f_vec.assemble()

    def solve(self, PRO_0):
        phi_v, phi_i = df.TrialFunctions(self.function_space)
        psi_v, psi_i = df.TestFunctions(self.function_space)

        kappa_v = df.Constant(self.parameter.kappa_v)
        kappa_i = df.Constant(self.parameter.kappa_i)
        alpha_H = df.Constant(self.parameter.alpha_H)
        alpha_P = df.Constant(self.parameter.alpha_P)
        beta = self.parameter.beta
        xi_va = df.Constant(self.parameter.xi_va)
        eta_vi = df.Constant(self.parameter.eta_vi)
        eta_iv = df.Constant(self.parameter.eta_iv)

        def a(k, u, v):
            return k * df.inner(df.grad(u), df.grad(v)) * df.dx

        def m(u, v):
            return u * v * df.dx

        F = 0
        F += a(kappa_v, phi_v, psi_v )
        F += m(xi_va* phi_v, psi_v)
        F += m(eta_vi* phi_v, psi_v)
        F += m(-eta_iv* phi_i, psi_v)

        F += a(kappa_i, phi_i, psi_i )
        F += m(-eta_vi* phi_v, psi_i )
        F += m(eta_iv* phi_i, psi_i )
        F += m(alpha_H * (1 - PRO_0)* phi_i, psi_i ) # TODO: cutoff!
        F += m(alpha_P * PRO_0, phi_i* psi_i) # TODO: cutoff!

        # additional terms for rhs
        F += m(df.Constant(0), psi_i) 
        F += m(df.Constant(0), psi_v) 

        bcs = []

        A = df.assemble(df.lhs(F))
        b = df.assemble(df.rhs(F))
        df.as_backend_type(b).axpy(beta, self.f_2d.vector())

        for bc in bcs:
            bc.apply(A)
            bc.apply(b)

        solver = df.PETScKrylovSolver('minres')
        df.PETScOptions.set('pc_type', 'hypre')
        df.PETScOptions.set('pc_hypre_type', 'boomeramg')
        solver.set_operator(A)
        solver.set_from_options()

        it = solver.solve(self.NUT.vector(), b)
        print(f'solving nutrients needed {it} iterations')

        return ResultNutrientSolver(solution=self.NUT, nut_i=self.NUT.sub(1), nut_v=self.NUT.sub(0))
    
    def get_nutrient_i_mass(self):
        return df.assemble(self.NUT.sub(1) * df.dx)

    def get_nutrient_v_mass(self):
        return df.assemble(self.NUT.sub(0) * df.dx)


def _demo_nutrients():
    M = 4

    vasculature = vio.read_default_vanilla()
    mesh = df.Mesh()
    with df.XDMFFile('data/tetrahedral-meshes/lung.xdmf') as f:
        f.read(mesh)
    print(f'tets: {mesh.num_entities_global(3)}')

    alpha = df.Constant(0.05)
    beta = 1
    kappa = df.Constant(4.)
    solver = NutrientSolver(mesh, kappa)
    solver.assemble_2d_source_terms(vasculature, M)
    solver.save_2d_source_terms('tmp/nutrients_checkpoint_test.xdmf')
    solver.zero_2d_source_terms()
    solver.load_2d_source_terms('tmp/nutrients_checkpoint_test.xdmf')
    u = solver.solve(alpha=alpha, beta=beta, f=df.Constant(0))

    print(u.vector().norm('l2'))

    file = df.File('output/solution.pvd')
    file << u
    pass


def _demo_double_continuum():
    M = 4

    vasculature = vio.read_default_vanilla()
    vasculature[:] /= 1000 # convert mm to m
    mesh = df.Mesh()
    with df.XDMFFile('data/tetrahedral-meshes/lung.xdmf') as f:
        f.read(mesh)
    mesh.coordinates()[:] /= 1000
    radius = 0.0068 
    midpoint = df.Point(-87.5e-3, -96.9e-3, -254.3e-3)
    mesh = refine_locally(mesh, midpoint, 8 * radius, 2)
    mesh = refine_locally(mesh, midpoint, 1.5 * radius, 1)
    mesh = refine_locally(mesh, midpoint, 1.25 * radius, 2)
    print(f'tets: {mesh.num_entities_global(3)}')

    mx = '(x[0] - mx) * (x[0] - mx)'
    my = '(x[1] - my) * (x[1] - my)'
    mz = '(x[2] - mz) * (x[2] - mz)'

    midpoint = df.Point(-87.5e-3, -96.9e-3, -254.3e-3)
    PRO_0 = df.Expression(
        f'1. / (exp( a * (sqrt({mx} + {my} + {mz})-r)) + 1)', 
        r=0.005, a=2000, 
        mx=midpoint[0], my=midpoint[1], mz=midpoint[2],
        degree=2
    ) 

    solver = NutrientDoubleContinuumSolver(
        mesh, 
        parameter=ParameterNutrients(
            kappa_v=0.001, 
            ), 
            save_at='output_nutrients')
    solver.assemble_2d_source_terms(vasculature, M)
    res = solver.solve(PRO_0=PRO_0)

    print(res.solution.vector().norm('l2'))

    solver.write(0)


if __name__ == '__main__':
    _demo_double_continuum()
    print('finished')
