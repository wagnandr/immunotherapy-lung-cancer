import time
import numpy as np
import dolfin as df
import petsc4py as p4py
import reduced_models.vasculature_io as vio
from integrate_2d_surface import integrate_2d_surface, integrate_2d_surface_mat


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

        tic = time.perf_counter()
        self.mat_row_indices, self.mat_col_indices, self.mat_values = integrate_2d_surface_mat(self.function_space, surface_triangles, surface_values, accuracy)
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
        print(len(self.mat_row_indices), len(self.mat_col_indices), len(self.mat_values))
        print(np.array(self.mat_row_indices).max(), np.array(self.mat_col_indices).max(), np.array(self.mat_values).max())
        A_mat = df.as_backend_type(A).mat()
        #a = A_mat.getValue(61255, 61255)
        print('min', np.array(self.mat_values).min())
        print('ind: ', self.mat_row_indices[0], self.mat_col_indices[0])
        for (row,col,val) in zip(self.mat_row_indices, self.mat_col_indices, self.mat_values):
            #if row == 61225 and col == 61225:
            #    a += val
            A_mat.setValue(row, col, beta * val, p4py.PETSc.InsertMode.ADD)
        A_mat.assemble()
        #print('end', a, A_mat.getValue(61255, 61255))
            
        #df.as_backend_type(A).mat().setValuesRCV(self.mat_row_indices, self.mat_col_indices, self.mat_values, p4py.PETSc.InsertMode.ADD)

        for bc in bcs:
            bc.apply(A)
            bc.apply(b)

        u = df.Function(self.function_space, name='nutrients')

        solver = df.PETScKrylovSolver('cg')
        df.PETScOptions.set('ksp_atol', 1e-16)
        df.PETScOptions.set('ksp_rtol', 1e-8)
        df.PETScOptions.set('pc_type', 'hypre')
        df.PETScOptions.set('pc_hypre_type', 'boomeramg')
        solver.set_operator(A)
        solver.set_from_options()

        it = solver.solve(u.vector(), b)
        print(f'solving nutrients needed {it} iterations')

        return u


if __name__ == '__main__':
    N = 64 
    M = 4

    vasculature = vio.read_default_vanilla()
    #mesh: df.Mesh = vio.get_bounding_cube(vasculature, N)
    mesh = df.Mesh()
    with df.XDMFFile('data/tetrahedral-meshes/lung.xdmf') as f:
        f.read(mesh)
    mesh = df.refine(mesh)
    mesh = df.refine(mesh)
    mesh = df.refine(mesh)
    # mesh: df.Mesh = vio.get_bounding_cube(vasculature, N)
    print(f'tets: {mesh.num_entities_global(3)}')

    alpha = df.Constant(0.05)
    beta = 1 
    kappa = df.Constant(0.01)
    solver = NutrientSolver(mesh, kappa)
    solver.assemble_2d_source_terms(vasculature, M)
    solver.save_2d_source_terms('tmp/nutrients_checkpoint_test.xdmf')
    solver.zero_2d_source_terms()
    solver.load_2d_source_terms('tmp/nutrients_checkpoint_test.xdmf')
    u = solver.solve(alpha=alpha, beta=beta, f=df.Constant(0))

    print(u.vector().norm('l2'))

    file = df.File('output/solution.pvd')
    file << u
