import time
import math
import os
import dolfin as df
from petsc4py import PETSc
from time_therapy_evaluator import TimeTherapyEvaluator
from medicine_solver import MedicineSolver
from parameters import TumorModelParameters
from coordinate_systems import CoordinateSystem


def FUN_potential_im(x, y, xi):
    return xi*3*x + xi*3*y


def FUN_potential_ex(x, y, xi):
    return xi*x*(4*x**2-6*x-1) + xi*y*(4*y**2-6*y-1)


def FUN_positive(x):
    #value is set to 0 if negative and to 1 if larger than 1
    y = df.conditional(df.ge(x,1.0),1.0,x) #if x is larger than 1, set to 1
    z = df.conditional(df.ge(y,0.0),y,0.0) #if y is less than 0, then set to 0
    return z


def FUN_mobility(x, y, xi):
    ''' mobility prefactor, quadratic logistic funciton with x growth and bound by 1-y with prefactor xi'''
    return xi * ( FUN_positive(x) * (1-FUN_positive(y)) )**2
    #return xi * ( x * (1-y) )**2
    #return xi 
    #return xi * FUN_positive(x) * (1-FUN_positive(y))


def FUN_growth(x, y, xi, landa, gomp_reg):
    return xi * FUN_positive(x)**landa * df.ln((1+gomp_reg)/(FUN_positive(y)+gomp_reg))

    
def FUN_sigmoid(value, threshold=0.5, width=30):
    return 1 / (1 + df.exp(- width * (value - threshold) ) )


def FUN_HEAVYSIDE(value, width=30):
    return 1 / (1 + df.exp(- 30 * (value) ) )


class SolverP:
    def __init__(
        self, 
        coordinate_system: CoordinateSystem, 
        initial_guess: df.Expression,
        nutrients: df.Expression,
        dt: float, 
        params: TumorModelParameters, 
        time_therapy,
        save_at:str='output',
        nonlinear:bool=False
    ):
        self.coordinate_system = coordinate_system
        self.initial_guess = initial_guess
        self.nutrients = nutrients
        self.dt = dt
        self.t = 0
        self.iter = 0
        self.file_prohyp = df.File(os.path.join(save_at, 'prohyp.pvd'))
        self.nonlinear = nonlinear

        self.setup(params, time_therapy, initial_guess)
    
    def setup(
        self, 
        params: TumorModelParameters, 
        time_therapy: TimeTherapyEvaluator, 
        initial_guess: df.Function=None
    ):
        self.params = params

        if initial_guess is None:
            initial_guess = self.PROMUP
        
        self.medicine_solver = MedicineSolver(params, time_therapy, self.dt, 1./24)

        # proliferative parameters
        self.mob_P = mob_P = df.Constant(params.mob_P)
        self.pot_P = pot_P = df.Constant(params.pot_P)
        self.chi_P = chi_P = df.Constant(params.chi_P) 
        self.eps_P = eps_P = df.Constant(params.eps_P) 
        self.pro_P = pro_P = df.Constant(params.pro_P) 
        self.deg_P = deg_P = df.Constant(params.deg_P) 
        self.gomp_reg = gomp_reg = params.gomp_reg

        # medicinal parameters:
        self.MED_hal = MED_hal = df.Constant(params.MED_hal) 
        self.MED_eff = MED_eff = df.Constant(params.MED_eff) 

        # Utility operators to keep the bilinear forms concise:
        a = self.coordinate_system.stiffness_form
        m = self.coordinate_system.mass_form
        
        # Function spaces for Cahn Hilliard and Nutrients:
        P1   = df.FiniteElement("Lagrange", self.coordinate_system.mesh.ufl_cell(), 1)
        W = df.FunctionSpace(self.coordinate_system.mesh, df.MixedElement([P1, P1]))

        # Trial and test functions
        PRO, MUP = df.TrialFunctions(W)
        T_PRO, T_MUP = df.TestFunctions(W)

        # Current and previous time steps:
        PROMUP   = df.Function(W, name='proliferative')
        PROMUP_0 = df.Function(W, name='proliferative')
        PRO_0, _ = df.split(PROMUP_0)
        if self.nonlinear:
            PRO, MUP = df.split(PROMUP)

        TUM_0 = PRO_0

        #NUT = df.Constant(1)
        NUT = self.nutrients

        self.MED_0 = MED_0 = df.Constant(0) 
        self.MED = 0

        weak_PRO  = (
            m(PRO, T_PRO) 
            - m(PRO_0, T_PRO) 
            + self.dt * a(FUN_mobility(PRO_0, TUM_0, mob_P), MUP, T_PRO)
            - self.dt * m(NUT * FUN_growth(PRO_0, TUM_0, pro_P, params.landa, gomp_reg), T_PRO) 
            + self.dt * m(MED_eff * 0.7 / 80 * (MED_0 * PRO_0)/(MED_0+MED_hal), T_PRO)
            + self.dt * m(deg_P * FUN_positive(PRO_0), T_PRO)
        ) 

        if self.nonlinear:
            c = df.variable(PRO)
            psi_i = pot_P * ( (2 * c - 1)**4)
            dpsidc_i = df.diff(psi_i, c)
            c = df.variable(PRO_0)
            psi_e = pot_P * (- 2 * (2 * c - 1)**2)
            dpsidc_e = df.diff(psi_e, c)

        weak_MUP  = (
            m(MUP, T_MUP) 
            - a(eps_P**2, PRO, T_MUP)
            + chi_P * m(NUT, T_MUP)
        )

        if self.nonlinear:
            weak_MUP += (
                - m(dpsidc_i, T_MUP) 
                - m(dpsidc_e, T_MUP) 
            )
        else:
            weak_MUP += (
                - m(FUN_potential_im(PRO, TUM_0, pot_P), T_MUP) 
                - m(FUN_potential_ex(PRO_0, TUM_0, pot_P), T_MUP) 
            )

        self.weak_PROMUP = weak_PRO + weak_MUP

        PROMUP.interpolate(initial_guess)

        self.PROMUP = PROMUP
        self.PROMUP_0 = PROMUP_0

    def next(self):
        self.PROMUP_0.assign(self.PROMUP)
        self.MED_0.assign(self.MED)
        self.medicine_solver.reassign()

        df.set_log_active(False)
        if self.nonlinear:
            df.solve(self.weak_PROMUP == 0, self.PROMUP, solver_parameters={'newton_solver': {'relative_tolerance': 1e-8}})
        else:
            df.solve(df.lhs(self.weak_PROMUP) == df.rhs(self.weak_PROMUP), self.PROMUP, solver_parameters={'linear_solver': 'lu'})
        df.set_log_active(True)

        self.MED = self.medicine_solver.solve(self.t)

        self.t += self.dt
    
    @property
    def next_t(self):
        return self.t + self.dt

    def write(self):
        self.file_prohyp << (self.PROMUP.sub(0), self.t)

    def get_proliferative_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0), df.Constant(1)))

    def get_necrotic_mass(self):
        return 0 

    def get_nutrient_v_mass(self):
        return 0 

    def get_nutrient_i_mass(self):
        return 0 

    def get_tumor_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0), df.Constant(1)))

    def get_tumor_mass_threshold(self, threshold):
        phi_now = self.PROMUP.sub(0)
        cond = df.conditional(df.ge(phi_now, threshold), df.Constant(1), df.Constant(0))
        return df.assemble(self.coordinate_system.mass_form(cond, df.Constant(1)))

    def get_tumor_mass_sigmoid(self, threshold):
        phi_now = self.PROMUP.sub(0)
        sigmoid = 1 / (1 + df.exp(- 30 * (phi_now - threshold) ) )
        return df.assemble(self.coordinate_system.mass_form(sigmoid, df.Constant(1)))

    def get_medicine_mass(self):
        return self.MED

    def get_nutrient_mass(self):
        raise RuntimeError("not implemented")


class StupidCSVFile:
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.iter = 0
        self.data = []
        with open(self.filepath, 'w') as f:
            pass

    def write(self, u, t):
        V = u.function_space()
        mesh = V.mesh()
        data = []
        if self.iter == 0:
            coords = mesh.coordinates().transpose().tolist()[0]
            data.append([-1.] + coords)
        nodal_values = u.compute_vertex_values(mesh)
        data.append([t] + nodal_values.tolist())
        import numpy as np
        with open(self.filepath, 'a') as f:
            f.write('\n')
            np.savetxt(f, np.array(data))
        self.iter += 1


def create_xdmf_file(filepath, name):
    return df.XDMFFile(os.path.join(filepath, f'{name}.xdmf'))


def create_stupid_csv_file(filepath, name):
    return df.StupidCSVFile(os.path.join(filepath, f'{name}.csv'))



class SolverTumorPN:
    def __init__(
        self, 
        coordinate_system: CoordinateSystem, 
        initial_guess: df.Expression,
        dt: float, 
        params: TumorModelParameters, 
        save_at:str='output'
    ):
        self.coordinate_system = coordinate_system
        self.initial_guess = initial_guess
        self.dt = dt

        # file_factory = create_stupid_csv_file
        file_factory = create_xdmf_file

        self.file_prohyp = file_factory(save_at, 'prohyp')
        self.file_nec = file_factory(save_at, 'nec')
        self.file_tumor = file_factory(save_at, 'tumor')

        self.parameter = params

        self.verbose = False

        self._setup_spaces_and_functions()

        if initial_guess is not None:
            self.PROMUP.interpolate(initial_guess)
        else:
            self.PROMUP.interpolate(df.Constant((0, 0)))
        
        self.NEC.interpolate(df.Constant(0))
    
    def _setup_spaces_and_functions(self):
        P1 = df.FiniteElement("Lagrange", self.coordinate_system.mesh.ufl_cell(), 1)

        self.W = df.FunctionSpace(self.coordinate_system.mesh, df.MixedElement([P1, P1]))
        self.V = df.FunctionSpace(self.coordinate_system.mesh, P1)

        self.function_space_proliferative = self.W 
        self.function_space_necrotic = self.V

        self.PROMUP   = df.Function(self.W, name='proliferative')
        self.PROMUP_0 = df.Function(self.W, name='proliferative')
        self.NEC = df.Function(self.V, name='necrotic')
        self.NEC_0 = df.Function(self.V, name='necrotic')
        self.PRO_0, _ = df.split(self.PROMUP_0)

    def reassign(self):
        self.PROMUP_0.assign(self.PROMUP)
        self.NEC_0.assign(self.NEC)
    
    def _setup_forms(self, MED_0, NUT_0):
        # Trial and test functions
        PRO, MUP = df.TrialFunctions(self.W)
        T_PRO, T_MUP = df.TestFunctions(self.W)
        NEC = df.TrialFunction(self.V)
        T_NEC = df.TestFunction(self.V)

        # Current and previous time steps:
        PRO_0, _ = df.split(self.PROMUP_0)
        NEC_0 = self.NEC_0

        TUM_0 = PRO_0 + NEC_0

        # Utility operators to keep the bilinear forms concise:
        a = self.coordinate_system.stiffness_form
        m = self.coordinate_system.mass_form

        # proliferative parameters
        mob_P = df.Constant(self.parameter.mob_P)
        pot_P = df.Constant(self.parameter.pot_P)
        chi_P = df.Constant(self.parameter.chi_P) 
        eps_P = df.Constant(self.parameter.eps_P) 
        pro_P = df.Constant(self.parameter.pro_P) 
        deg_P = df.Constant(self.parameter.deg_P) 
        landa_HN = df.Constant(self.parameter.landa_HN) 
        sigma_HN = df.Constant(self.parameter.sigma_HN) 
        gomp_reg = self.parameter.gomp_reg

        MED_hal = df.Constant(self.parameter.MED_hal) 
        MED_eff = df.Constant(self.parameter.MED_eff) 

        weak_PRO  = (
            m(PRO, T_MUP) 
            - m(PRO_0, T_MUP) 
            + self.dt * a(FUN_mobility(PRO_0, TUM_0, mob_P), MUP, T_MUP)
            - self.dt * m(NUT_0 * FUN_growth(PRO_0, TUM_0, pro_P, self.parameter.landa, gomp_reg), T_MUP) 
            + self.dt * m(MED_eff * 0.7 / 80 * (MED_0 * PRO_0)/(MED_0+MED_hal), T_MUP)
            + self.dt * m(deg_P * FUN_positive(PRO_0), T_MUP)
            + self.dt * landa_HN * m(FUN_HEAVYSIDE(sigma_HN - NUT_0) * PRO_0, T_MUP)
        ) 

        weak_MUP  = (
            m(MUP, T_PRO) 
            - a(eps_P**2, PRO, T_PRO)
            + chi_P * m(NUT_0, T_PRO)
        )

        weak_MUP  += (
            - m(FUN_potential_im(PRO, TUM_0, pot_P), T_PRO) 
            - m(FUN_potential_ex(PRO_0, TUM_0, pot_P), T_PRO) 
        )

        self.weak_NEC = (
            m(NEC, T_NEC)
            - m(NEC_0, T_NEC)
            - self.dt * landa_HN * m(FUN_HEAVYSIDE(sigma_HN - NUT_0) * PRO_0, T_NEC)
        )

        self.weak_PROMUP = weak_PRO + weak_MUP

    def solve(self, MED_0, NUT_0):
        self._setup_forms(MED_0=MED_0, NUT_0=NUT_0)
    
        df.set_log_active(False)
        solver_parameters = {'linear_solver': 'lu'}
        df.solve(df.lhs(self.weak_PROMUP) == df.rhs(self.weak_PROMUP), self.PROMUP, solver_parameters=solver_parameters)
        # self._solve_proliferative()
        # solve necrotic+nutrients:
        #solver_parameters ={'linear_solver': 'lu'}
        df.solve(df.lhs(self.weak_NEC) == df.rhs(self.weak_NEC), self.NEC)
        df.set_log_active(True)
    
    def _setup_solver(self):
        phi, mu = df.TrialFunctions(self.W)
        psi, nu = df.TestFunctions(self.W)

        mob_P = self.parameter.mob_P
        pot_P = self.parameter.pot_P
        tau = self.dt
        eps_P = self.parameter.eps_P

        #mobility = FUN_mobility(FUN_positive(self.PRO_0), FUN_positive(self.PRO_0 + self.NEC_0), mob_P)
        mobility = FUN_mobility(self.PRO_0, self.PRO_0 + self.NEC_0, mob_P)
        # mobility = df.Constant(1.) 

        a = self.coordinate_system.stiffness_form
        m = self.coordinate_system.mass_form

        p_form = 0
        p_form += df.Constant(math.sqrt( tau * mob_P )) * m(mu, nu)
        p_form += a(mobility * df.Constant( tau ), mu, nu)

        p_form += df.Constant(6. * pot_P / math.sqrt( mob_P * tau )) * m(phi, psi)
        p_form += a(df.Constant(math.pow(eps_P,2)), phi, psi)

        self.P = df.assemble(p_form)

        ksp = PETSc.KSP().create()
        df.PETScOptions.clear()
        #df.PETScOptions.set('ksp_view')
        #df.PETScOptions.set('ksp_monitor_true_residual')
        #df.PETScOptions.set('ksp_monitor')
        df.PETScOptions.set('ksp_initial_guess_nonzero')
        df.PETScOptions.set('ksp_type', 'minres')
        #df.PETScOptions.set('ksp_rtol', '1e-12')
        #df.PETScOptions.set('ksp_atol', '1e-12')
        df.PETScOptions.set('pc_type', 'fieldsplit')
        df.PETScOptions.set('pc_fieldsplit_type', 'additive')
        df.PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
        #df.PETScOptions.set('fieldsplit_0_ksp_monitor')
        #df.PETScOptions.set('fieldsplit_0_ksp_type', 'cg')
        df.PETScOptions.set('fieldsplit_0_pc_type', 'gamg')
        #df.PETScOptions.set('fieldsplit_0_pc_type', 'hypre')
        #df.PETScOptions.set('fieldsplit_0_pc_hypre_type', 'boomeramg')
        #df.PETScOptions.set('fieldsplit_0_pc_type', 'none')
        #df.PETScOptions.set('fieldsplit_0_pc_type', 'hypre')
        #df.PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
        #df.PETScOptions.set('fieldsplit_1_ksp_type', 'cg')
        df.PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
        #df.PETScOptions.set('fieldsplit_1_pc_type', 'jacobi')
        #df.PETScOptions.set('fieldsplit_1_pc_type', 'gamg')
        df.PETScOptions.set('fieldsplit_1_pc_type', 'hypre')
        df.PETScOptions.set('fieldsplit_1_pc_hypre_type', 'boomeramg')
        #df.PETScOptions.set('fieldsplit_1_ksp_monitor')
        #df.PETScOptions.set('fieldsplit_1_pc_type', 'jacobi')
        #df.PETScOptions.set('fieldsplit_1_pc_type', 'hypre')
        ksp.setFromOptions()

        pc = ksp.getPC()
        is0 = PETSc.IS().createGeneral(self.W.sub(0).dofmap().dofs())
        is1 = PETSc.IS().createGeneral(self.W.sub(1).dofmap().dofs())
        fields = [('0', is0), ('1', is1)]
        pc.setFieldSplitIS(*fields)

        self.solver = solver = df.PETScKrylovSolver(ksp)
        solver.set_reuse_preconditioner(True)

    def _solve_proliferative(self):
        a = df.lhs(self.weak_PROMUP)
        l = df.rhs(self.weak_PROMUP)

        tic = time.perf_counter()
        A = df.assemble(a)
        b = df.assemble(l)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds to setup the system")

        tic = time.perf_counter()
        self._setup_solver()
        self.solver.set_operators(A, self.P)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds to setup the preconditioner")

        tic = time.perf_counter()
        num_linear_iter = self.solver.solve(self.PROMUP.vector(), b)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds ({num_linear_iter} minres iterations) to solve the system")

    def write(self, t):
        self.file_prohyp.write(self.PROMUP.sub(0), t)
        self.file_nec.write(self.NEC, t)
        TUM = df.Function(self.V, name='tumor')
        TUM.interpolate(self.PROMUP.sub(0))
        TUM.vector()[:] += self.NEC.vector()[:]
        self.file_tumor.write(TUM, t)

    def get_tumor_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0)+self.NEC, df.Constant(1)))

    def get_tumor_mass_threshold(self, threshold: float):
        phi_now = self.PROMUP.sub(0)
        cond = df.conditional(df.ge(phi_now+self.NEC, threshold), df.Constant(1), df.Constant(0))
        return df.assemble(self.coordinate_system.mass_form(cond, df.Constant(1)))

    def get_tumor_mass_sigmoid(self, threshold: float):
        phi_now = self.PROMUP.sub(0)
        sigmoid = FUN_sigmoid(phi_now+self.NEC, threshold=threshold, width=30)
        form = self.coordinate_system.mass_form(sigmoid, df.Constant(1))
        return df.assemble(form)

    def get_proliferative_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0), df.Constant(1)))

    def get_necrotic_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.NEC, df.Constant(1)))



class SolverNutrients:
    def __init__(
        self, 
        coordinate_system: CoordinateSystem, 
        nutrients: df.Expression,
        params: TumorModelParameters, 
        save_at:str='output',
    ):
        self.coordinate_system = coordinate_system
        self.nutrients = nutrients

        # file_factory = create_stupid_csv_file
        file_factory = create_xdmf_file

        self.file_nut = file_factory(save_at, 'nut')

        self.parameter = params

        self._setup_spaces_and_functions()

    def _setup_spaces_and_functions(self):
        # Function spaces for Cahn Hilliard and Nutrients:
        P1   = df.FiniteElement("Lagrange", self.coordinate_system.mesh.ufl_cell(), 1)
        self.V = df.FunctionSpace(self.coordinate_system.mesh, P1)

        self.NUT_i = df.Function(self.V, name='nutrient')
        self.NUT_i_0 = df.Function(self.V, name='nutrient') 

    def reassign(self):
        self.NUT_i_0.assign(self.NUT_i)

    def setup_forms(self, PRO_0, NEC_0):
        # proliferative parameters
        source_NUT = df.Constant(self.parameter.source_NUT)
        # source_NUT = df.Expression(f'0.2 * (x[0]+r)/(2*r) + 0.5', r=0.01, degree=1)
        kappa = df.Constant(self.parameter.kappa)
        alpha_healthy = df.Constant(self.parameter.alpha_healthy)
        alpha_tumor = df.Constant(self.parameter.alpha_tumor)
        kappa = df.Constant(self.parameter.kappa)

        # Utility operators to keep the bilinear forms concise:
        a = self.coordinate_system.stiffness_form
        m = self.coordinate_system.mass_form

        # Trial and test functions
        NUT = df.TrialFunction(self.V)
        T_NUT = df.TestFunction(self.V)

        alpha = alpha_healthy * FUN_positive(1 - NEC_0 - PRO_0) + alpha_tumor * FUN_positive(PRO_0)

        self.weak_NUT = (
            a(kappa, NUT, T_NUT)
            + m(alpha * NUT, T_NUT)
            - m(source_NUT, T_NUT)
        )

    def solve(self, PRO_0, NEC_0):
        self.setup_forms(PRO_0=PRO_0, NEC_0=NEC_0)
        df.set_log_active(False)
        df.solve(df.lhs(self.weak_NUT) == df.rhs(self.weak_NUT), self.NUT_i)
        df.set_log_active(True)

    def write(self, t):
        self.file_nut.write(self.NUT_i, t)

    def get_nutrient_i_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.NUT_i, df.Constant(1)))

    def get_nutrient_v_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.parameter.source_NUT, df.Constant(1)))


class SolverPN:
    def __init__(
        self, 
        nutrient_solver: SolverNutrients,
        tumor_solver: SolverTumorPN,
        medicine_solver: MedicineSolver,
        dt: float
    ):
        self.dt = dt
        self.t = 0
        self.iter = 0
        self.verbose = False 

        self.nutrient_solver = nutrient_solver
        self.tumor_solver = tumor_solver
        self.medicine_solver = medicine_solver 

        # solve initial guess
        self.nutrient_solver.solve(PRO_0=self.tumor_solver.PRO_0, NEC_0=self.tumor_solver.NEC_0)

    def next(self):
        self.nutrient_solver.reassign()
        self.tumor_solver.reassign()
        self.medicine_solver.reassign()

        tic = time.perf_counter()
        self.nutrient_solver.solve(PRO_0=self.tumor_solver.PRO_0, NEC_0=self.tumor_solver.NEC_0)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds to solve nutrient model")

        tic = time.perf_counter()
        self.tumor_solver.solve(MED_0=df.Constant(self.medicine_solver.MED_0), NUT_0=self.nutrient_solver.NUT_i_0)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds to solve tumor model")

        tic = time.perf_counter()
        self.medicine_solver.solve(self.t)
        toc = time.perf_counter()
        if self.verbose:
            print(f"needed {toc - tic:0.4f} seconds to solve medicine model")

        self.t += self.dt
    
    @property
    def next_t(self):
        return self.t + self.dt

    def write(self):
        self.nutrient_solver.write(self.t)
        self.tumor_solver.write(self.t)

    def get_tumor_mass(self):
        return self.tumor_solver.get_tumor_mass() 

    def get_tumor_mass_threshold(self, threshold: float):
        return self.tumor_solver.get_tumor_mass_threshold(threshold)

    def get_tumor_mass_sigmoid(self, threshold: float):
        return self.tumor_solver.get_tumor_mass_sigmoid(threshold) 

    def get_medicine_mass(self):
        return self.medicine_solver.MED

    def get_nutrient_i_mass(self):
        return self.nutrient_solver.get_nutrient_i_mass()

    def get_nutrient_v_mass(self):
        return self.nutrient_solver.get_nutrient_v_mass()

    def get_proliferative_mass(self):
        return self.tumor_solver.get_proliferative_mass()

    def get_necrotic_mass(self):
        return self.tumor_solver.get_necrotic_mass() 


class SolverPNOld:
    def __init__(
        self, 
        coordinate_system: CoordinateSystem, 
        initial_guess: df.Expression,
        nutrients: df.Expression,
        dt: float, 
        params: TumorModelParameters, 
        time_therapy,
        save_at:str='output',
        nonlinear:bool=False,
    ):
        self.coordinate_system = coordinate_system
        self.initial_guess = initial_guess
        self.nutrients = nutrients
        self.dt = dt
        self.t = 0
        self.iter = 0

        self.nonlinear = nonlinear

        self.file_prohyp = df.File(os.path.join(save_at, 'prohyp.pvd'))
        self.file_nut = df.File(os.path.join(save_at, 'nut.pvd'))
        self.file_nec = df.File(os.path.join(save_at, 'nec.pvd'))
        self.file_tumor = df.File(os.path.join(save_at, 'tumor.pvd'))

        self.setup(params, time_therapy, initial_guess)
    
    def setup(self, params, time_therapy, initial_guess=None):
        self.params = params

        if initial_guess is None:
            initial_guess = self.PROMUP
        
        self.medicine_solver = MedicineSolver(params, time_therapy, self.dt, 1./24)

        # proliferative parameters
        self.mob_P = mob_P = df.Constant(params.mob_P)
        self.pot_P = pot_P = df.Constant(params.pot_P)
        self.chi_P = chi_P = df.Constant(params.chi_P) 
        self.eps_P = eps_P = df.Constant(params.eps_P) 
        self.pro_P = pro_P = df.Constant(params.pro_P) 
        self.deg_P = deg_P = df.Constant(params.deg_P) 
        self.landa_HN = landa_HN = df.Constant(params.landa_HN) 
        self.sigma_HN = sigma_HN = df.Constant(params.sigma_HN) 
        self.gomp_reg = gomp_reg = params.gomp_reg
        self.source_NUT = source_NUT = df.Constant(params.source_NUT)
        self.kappa = kappa = df.Constant(params.kappa)

        # medicinal parameters:
        self.MED_hal = MED_hal = df.Constant(params.MED_hal) 
        self.MED_eff = MED_eff = df.Constant(params.MED_eff) 

        # Utility operators to keep the bilinear forms concise:
        a = self.coordinate_system.stiffness_form
        m = self.coordinate_system.mass_form
        
        # Function spaces for Cahn Hilliard and Nutrients:
        P1   = df.FiniteElement("Lagrange", self.coordinate_system.mesh.ufl_cell(), 1)
        W = df.FunctionSpace(self.coordinate_system.mesh, df.MixedElement([P1, P1]))
        self.V = V = df.FunctionSpace(self.coordinate_system.mesh, P1)

        # Trial and test functions
        PRO, MUP = df.TrialFunctions(W)
        T_PRO, T_MUP = df.TestFunctions(W)
        NEC = df.TrialFunction(V)
        T_NEC = df.TestFunction(V)
        NUT = df.TrialFunction(V)
        T_NUT = df.TestFunction(V)

        # Current and previous time steps:
        PROMUP   = df.Function(W, name='proliferative')
        PROMUP_0 = df.Function(W, name='proliferative')
        NEC_0 = df.Function(V, name='necrotic')
        if self.nonlinear:
            PRO, MUP = df.split(PROMUP)
        PRO_0, _ = df.split(PROMUP_0)
        NUT_0 = df.Function(V, name='necrotic')

        TUM_0 = PRO_0 + NEC_0

        self.MED_0 = MED_0 = df.Constant(0) 
        self.MED = 0 

        weak_PRO  = (
            m(PRO, T_PRO) 
            - m(PRO_0, T_PRO) 
            + self.dt * a(FUN_mobility(PRO_0, TUM_0, mob_P), MUP, T_PRO)
            - self.dt * m(NUT_0 * FUN_growth(PRO_0, TUM_0, pro_P, params.landa, gomp_reg), T_PRO) 
            + self.dt * m(MED_eff * 0.7 / 80 * (MED_0 * PRO_0)/(MED_0+MED_hal), T_PRO)
            + self.dt * m(deg_P * FUN_positive(PRO_0), T_PRO)
            + self.dt * landa_HN * m(FUN_HEAVYSIDE(sigma_HN - NUT_0) * PRO_0, T_PRO)
        ) 


        weak_MUP  = (
            m(MUP, T_MUP) 
            - a(eps_P**2, PRO, T_MUP)
            + chi_P * m(NUT_0, T_MUP)
        )

        if self.nonlinear:
            c = df.variable(PRO)
            psi_i = pot_P * ( (2 * c - 1)**4)
            dpsidc_i = df.diff(psi_i, c)
            c = df.variable(PRO_0)
            psi_e = pot_P * (- 2 * (2 * c - 1)**2)
            dpsidc_e = df.diff(psi_e, c)

            weak_MUP  += (
                - m(dpsidc_i, T_MUP) 
                - m(dpsidc_e, T_MUP) 
            )
        else:
            weak_MUP  += (
                - m(FUN_potential_im(PRO, TUM_0, pot_P), T_MUP) 
                - m(FUN_potential_ex(PRO_0, TUM_0, pot_P), T_MUP) 
            )

        self.weak_PROMUP = weak_PRO + weak_MUP

        self.weak_NEC = (
            m(NEC, T_NEC)
            - m(NEC_0, T_NEC)
            - self.dt * landa_HN * m(FUN_HEAVYSIDE(sigma_HN - NUT_0) * PRO_0, T_NEC)
        )

        alpha = params.alpha_healthy * FUN_positive(1 - PRO_0) + params.alpha_tumor * FUN_positive(PRO_0)

        self.weak_NUT = (
            a(kappa, NUT, T_NUT)
            + m(alpha * NUT, T_NUT)
            - m(source_NUT, T_NUT)
        )

        PROMUP.interpolate(initial_guess)

        self.PROMUP = PROMUP
        self.PROMUP_0 = PROMUP_0

        self.NEC = df.Function(V, name='necrotic') 
        self.NEC_0 = NEC_0

        self.NUT = df.Function(V, name='nutrient')
        self.NUT_0 = NUT_0 

        # initial_guess nutrients
        df.solve(df.lhs(self.weak_NUT) == df.rhs(self.weak_NUT), self.NUT, solver_parameters={'linear_solver': 'lu'})

    def next(self):
        self.PROMUP_0.assign(self.PROMUP)
        self.NEC_0.assign(self.NEC)
        self.NUT_0.assign(self.NUT)
        self.MED_0.assign(self.MED)

        df.set_log_active(False)
        # solve tumor:
        if self.nonlinear:
            solver_parameters = {'newton_solver': {'relative_tolerance': 1e-10, 'absolute_tolerance': 1e-10}}
            df.solve(self.weak_PROMUP == 0, self.PROMUP, solver_parameters=solver_parameters)
        else:
            solver_parameters = {'linear_solver': 'lu'}
            df.solve(df.lhs(self.weak_PROMUP) == df.rhs(self.weak_PROMUP), self.PROMUP, solver_parameters=solver_parameters)
        # solve necrotic+nutrients:
        solver_parameters ={'linear_solver': 'lu'}
        df.solve(df.lhs(self.weak_NEC) == df.rhs(self.weak_NEC), self.NEC, solver_parameters=solver_parameters)
        df.solve(df.lhs(self.weak_NUT) == df.rhs(self.weak_NUT), self.NUT, solver_parameters=solver_parameters)
        df.set_log_active(True)

        self.MED = self.medicine_solver.solve(self.MED, self.t)

        self.t += self.dt
    
    @property
    def next_t(self):
        return self.t + self.dt

    def write(self):
        self.file_prohyp << (self.PROMUP.sub(0), self.t)
        self.file_nut << (self.NUT, self.t)
        self.file_nec << (self.NEC, self.t)
        TUM = df.Function(self.V, name='tumor')
        TUM.interpolate(self.PROMUP.sub(0))
        TUM.vector()[:] += self.NEC.vector()[:]
        self.file_tumor << (TUM, self.t)

    def get_tumor_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0)+self.NEC, df.Constant(1)))

    def get_proliferative_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.PROMUP.sub(0), df.Constant(1)))

    def get_necrotic_mass(self):
        return df.assemble(self.coordinate_system.mass_form(self.NEC, df.Constant(1)))

    def get_tumor_mass_threshold(self, threshold: float):
        phi_now = self.PROMUP.sub(0)
        cond = df.conditional(df.ge(phi_now+self.NEC, threshold), df.Constant(1), df.Constant(0))
        return df.assemble(self.coordinate_system.mass_form(cond, df.Constant(1)))

    def get_tumor_mass_sigmoid(self, threshold: float):
        phi_now = self.PROMUP.sub(0)
        sigmoid = FUN_sigmoid(phi_now+self.NEC, threshold=threshold, width=30)
        form = self.coordinate_system.mass_form(sigmoid, df.Constant(1))
        return df.assemble(form)

    def get_medicine_mass(self):
        return self.MED

    def get_nutrient_mass(self):
        raise RuntimeError("not implemented")


def create_system_P_i(
    coordinate_system: CoordinateSystem, 
    initial_guess: df.Expression,
    nutrients: df.Expression,
    dt: float, 
    params: TumorModelParameters, 
    time_therapy,
    save_at:str='output',
) -> SolverP:
    """Creates a system just with proliferative cells"""
    return SolverP(
        coordinate_system=coordinate_system, 
        initial_guess=initial_guess,
        nutrients=nutrients,
        dt=dt, 
        params=params, 
        time_therapy=time_therapy,
        save_at=save_at)


def create_system_PN_i(
    coordinate_system: CoordinateSystem, 
    initial_guess: df.Expression,
    nutrients: df.Expression,
    dt: float, 
    params: TumorModelParameters, 
    time_therapy,
    save_at:str='output',
) -> SolverPN:
    nutrient_solver = SolverNutrients(
        coordinate_system,
        nutrients,
        params,
        save_at)
    
    tumor_solver = SolverTumorPN(
        coordinate_system,
        initial_guess,
        dt,
        params,
        save_at)

    medicine_solver = MedicineSolver(
        params, 
        time_therapy, 
        dt, 
        1./24./32.)

    system_solver = SolverPN(
        tumor_solver=tumor_solver,
        nutrient_solver=nutrient_solver,
        medicine_solver=medicine_solver,
        dt=dt)
    
    return system_solver
