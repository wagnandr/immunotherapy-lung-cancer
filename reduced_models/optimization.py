from dataclasses import dataclass
import numpy as np
from tumor_io.experimental_data import load_experimental_data, PATH_PATIENT_2
from parameters import TherapyParameters, NumericalParameters, OutputParameters, TumorModelParameters
import scipy as sp
from scipy import optimize as opt
from run_spherical import SimulationRunner
from solvers import SolverP, SolverPN, create_system_PN_i, create_system_P_i
from enum import Enum


@dataclass(frozen=True)
class ResidualFunctionalResult:
    energy: float 
    energy_least_squares: float 
    energy_tumor_average: float 
    errors: np.ndarray 
    matched_time_points: np.ndarray
    average_tumor_mass: float

    
def match_times(time1, time2):
    matched_indices_time1 = []
    matched_indices_time2 = []
    for idx1, t1 in enumerate(time1):
        idx_candidates = [idx2 for idx2, t2 in enumerate(time2) if np.abs(t1 - t2) < 1e-9]
        if len(idx_candidates) != 0:
            idx2 = idx_candidates[0]
            matched_indices_time1.append(idx1)
            matched_indices_time2.append(idx2)
    return matched_indices_time1, matched_indices_time2


def time_averaged_L1_norm(t_list, p_list):
    # trapezoidal rule
    avg_tumor_mass = 0.5*(p_list[0:-1] + p_list[1:]) * (t_list[1:] - t_list[0:-1])
    avg_tumor_mass = avg_tumor_mass.sum() / (t_list[-1] - t_list[0])
    return avg_tumor_mass


class ResidualFunctional:
    def __init__(self, time_offset, experimental_data):
        self.exp_data = experimental_data 
        self.weights = np.array([1] * len(self.exp_data.t))
        self.integral_weight = 1e-3
        self.time_offset = time_offset
    
    def error(self, data):
        sim_t_list = data['time'] + self.time_offset 
        sim_p_vis_list = data['tumor_sigmoid'] * 1e6
        sim_p_list = data['tumor'] * 1e6

        self.weights = self.weights / self.weights.sum()

        sim_avg_tumor_mass = time_averaged_L1_norm(sim_t_list, sim_p_list)

        experiment_indices, simulation_indices = match_times(self.exp_data.t, sim_t_list)

        errors = sim_p_vis_list[simulation_indices] - self.exp_data.volumes[experiment_indices]
        error_least_squares = np.dot(errors**2, self.weights[experiment_indices])

        energy = error_least_squares + self.integral_weight * sim_avg_tumor_mass 

        return ResidualFunctionalResult(
            energy=100*energy,
            energy_least_squares=error_least_squares,
            energy_tumor_average=self.integral_weight * sim_avg_tumor_mass,
            errors=errors,
            matched_time_points=self.exp_data.t[experiment_indices],
            average_tumor_mass=sim_avg_tumor_mass
        )


class ParameterIndex(Enum):
    pro_P = 0
    MED_eff = 1
    landa = 2
    landa_HN = 3
    r_tumor_init = 4


class ParameterManager:
    def __init__(self):
        # constraints:
        lower_bounds = []
        upper_bounds = []
        # growth
        lower_bounds.append(1.)
        upper_bounds.append(40.)
        # immunotherapy 
        lower_bounds.append(1.)
        upper_bounds.append(15.)
        # lambda 
        lower_bounds.append(1.)
        upper_bounds.append(2.)
        # exchange necrotic hypoxic
        lower_bounds.append(1e-2)
        upper_bounds.append(1e-1)
        # r tumor init 
        lower_bounds.append(0.006074216961365892)
        upper_bounds.append(0.006074216961365892)

        self.global_lower_bounds = np.array(lower_bounds)
        self.global_upper_bounds = np.array(upper_bounds)

        self.global_x00 = np.array([1., 1., 2., 1e-2, 0.0060742169631])
        self.global_rescale = np.array([6.87e-3, 0.499, 1., 1., 1.])

        self.global_dim = len(self.global_x00)

        self.active_parameters = np.array([True] * self.global_dim, dtype=np.bool8)
    
    def deactivate_all_parameters(self):
        for i in range(len(self.active_parameters)):
            self.active_parameters[i] = False 
    
    def set(self, param: ParameterIndex, lower=None, upper=None, start=None, active=None, scale=None):
        if lower is not None:
            self.global_lower_bounds[param.value] = lower
        if upper is not None:
            self.global_upper_bounds[param.value] = upper 
        if start is not None:
            self.global_x00[param.value] = start 
        if active is not None:
            self.active_parameters[param.value] = active
        if scale is not None:
            self.global_rescale[param.value] = scale

    @property
    def map_inactive_to_global(self):
        return np.arange(0,len(self.active_parameters))[~self.active_parameters]

    @property
    def map_active_to_global(self):
        return np.arange(0,len(self.active_parameters))[self.active_parameters]
    
    @property
    def x00(self):
        return self.restrict_vector_to_active(self.global_x00)

    @property
    def lower_bounds(self):
        return self.restrict_vector_to_active(self.global_lower_bounds)

    @property
    def upper_bounds(self):
        return self.restrict_vector_to_active(self.global_upper_bounds)

    def extend_and_scale_vector(self, vector):
        return self.extend_vector(vector) * self.global_rescale

    def extend_vector(self, vector):
        extension = np.zeros(self.global_dim)
        extension[self.map_inactive_to_global] = self.global_x00[self.map_inactive_to_global]
        extension[self.map_active_to_global] = vector[:] 
        return extension
    
    def restrict_vector_to_active(self, vector):
        return vector[self.map_active_to_global] 


class OptimizationFunctional:
    def __init__(self, 
        simulation_runner: SimulationRunner,
        parameter_manager: ParameterManager,
        residual: ResidualFunctional,
        verbose: bool = False
    ) -> None:
        self.simulation_runner = simulation_runner 
        self.parameter_manager = parameter_manager 
        self.residual = residual
        self.num_evaluations = 0
        self.verbose = verbose
	
    def __call__(self, vec, *args):
        return self.evaluate(vec).energy 
    
    def evaluate(self, vec):
        data = self.run(vec)
        residual_result = self.residual.error(data)
        self.num_evaluations += 1
        if self.verbose:
            print(f'{residual_result}')
            vec_extension = self.parameter_manager.extend_vector(vec)
            print(f'vec = {vec_extension},\t\t energy = {residual_result.energy} ({self.num_evaluations} evaluations)')
        return residual_result
    
    def run(self, vec):
        vec = self.parameter_manager.extend_and_scale_vector(vec)
        self.simulation_runner.tumor_model_parameters.pro_P = vec[0] 
        self.simulation_runner.tumor_model_parameters.MED_eff = vec[1] 
        self.simulation_runner.tumor_model_parameters.landa = vec[2] 
        self.simulation_runner.tumor_model_parameters.landa_HN = vec[3] 
        self.simulation_runner.numerical_parameters.r_tumor_init = vec[4] 
        return self.simulation_runner.run()


def fast_plots(results, energies, vectors):
    from matplotlib import pyplot as plt 

    vectors = [np.array([1.,1.,2.])] + vectors
    scale = np.array([6.87e-3, 0.499, 1])
    scaled_vectors = [res*scale for res in vectors]
    pro_p = [res[0] for res in scaled_vectors]
    med = [res[1] for res in scaled_vectors]
    landa = [res[2] for res in scaled_vectors]
    energy_lsq = [res.energy_least_squares for res in results]
    energy_tavg = [res.energy_tumor_average for res in results]

    fig, axes = plt.subplots(4,1,sharex=True)
    it = np.arange(0, len(energies)+1)

    axes[0].plot(it[:-1], energies, '-x')
    axes[0].plot(it[:-1], energy_lsq, '-x')
    axes[0].plot(it[:-1], energy_tavg, '-x')
    axes[0].grid(True)
    axes[0].set_ylabel('energy')

    axes[1].plot(it, pro_p, '-x')
    axes[1].grid(True)
    axes[1].set_ylabel('pro_p')

    axes[2].plot(it, med, '-x')
    axes[2].grid(True)
    axes[2].set_ylabel('med')

    axes[3].plot(it, landa, '-x')
    axes[3].grid(True)
    axes[3].set_ylabel('$\lambda$')

    axes[-1].set_xlabel('iteration')
    plt.tight_layout()
    plt.show()


class Save:
    def __init__(
        self, 
        optimizer: OptimizationFunctional, 
        parameter_manager: ParameterManager
    ) -> None:
        self.optimizer = optimizer
        self.vectors = []
        self.energies = []
        self.results = []
        self.parameter_manager = parameter_manager
	
    def __call__(self, vec):
        result = self.optimizer.evaluate(vec)
        vec = self.parameter_manager.extend_vector(vec)
        energy = result.energy
        self.vectors.append(vec)
        self.energies.append(energy)
        self.results.append(result)
        #fast_plots(self.results, self.energies, self.vectors)
        print(f'results = {self.results}')
        print(f'vectors = {self.vectors}')
        print(f'energies = {self.energies}')


class SimpleLBFGSBasedOptimizer:
    def __init__(
        self, 
        simulation_runner, 
        time_offset,
        experimental_data 
    ) -> None:
        self.simulation_runner = simulation_runner

        self.parameter_manager = ParameterManager() 

        self.residual = ResidualFunctional(
            time_offset=time_offset,
            experimental_data=experimental_data
        )

        self.to_optimize = OptimizationFunctional(
            simulation_runner=simulation_runner,
            parameter_manager=self.parameter_manager,
            residual=self.residual
        )

        self.saver = Save(self.to_optimize, self.parameter_manager)

    def create_optimization_functional(self):
        def fun():
            pass
        return fun
    
    def run(self):
        bounds = sp.optimize.Bounds(self.parameter_manager.lower_bounds, self.parameter_manager.upper_bounds, True) 
        x0 = self.parameter_manager.x00
        print(f'! {x0}')
        return opt.minimize(
            fun=self.to_optimize,
            method = 'L-BFGS-B',
            x0=x0,
            options={
                #'disp': 1, 
                'eps': 1e-6,
                'maxiter': 100,
                #'gtol': 1e-9,
                #'ftol': 1e-9,
                'maxcor': 1,
                'iprint': 100,
                'maxls': 100
            },
            bounds=bounds,
            callback=self.saver
        )


def run_optimization_spherical_coordinates_patient1():
    from tumor_io.plotting import plot_comparative_patient1

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655,
        drug_application_interval=14
    )
    num_params = NumericalParameters(
        dt=1.,
        final_time=200,
        num_elements=125*4,
        r_max=0.04,
        r_tumor_init=0.006074216961365892,
        med_init = 0
    )
    model_params = TumorModelParameters()
    out_params = OutputParameters(enabled=False)

    runner = SimulationRunner(
        numerical_parameters=num_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        #solver_type=create_system_PN_i
        solver_type=create_system_P_i
    )

    experimental_data = load_experimental_data()

    optimizer = SimpleLBFGSBasedOptimizer(
        simulation_runner=runner,
        time_offset=176,
        experimental_data=experimental_data
    )
    optimizer.residual.integral_weight = 4e-2

    optimizer.parameter_manager.upper_bounds[0] = 60
    optimizer.parameter_manager.upper_bounds[1] = 30
    optimizer.parameter_manager.upper_bounds[2] = 2 
    optimizer.parameter_manager.upper_bounds[3] = 1

    optimizer.parameter_manager.set(ParameterIndex.pro_P, 1e-1, 100, 5., True, 6.87e-3)
    optimizer.parameter_manager.set(ParameterIndex.MED_eff, 1e-2, 20, 1.*0.6, True, 0.499)
    optimizer.parameter_manager.set(ParameterIndex.landa, 1, 2, 2, False, 1)
    optimizer.parameter_manager.set(ParameterIndex.landa_HN, 1e-2, 1e3, 0.0, False, 1)
    optimizer.parameter_manager.set(ParameterIndex.r_tumor_init, 0.002, 0.01, 0.006074, False, 1)

    #optimizer.parameter_manager.global_x00 = np.array([55., 15., 2., 1e-1])

    # deactivate lambda
    optimizer.parameter_manager.active_parameters[2] = False

    weights = optimizer.residual.weights
    weights[5] = 40 
    weights[6] = 10 
    weights[7] = 10 
    weights[9] = 2.5
    weights[11] = 10.

    result = optimizer.run()

    print(f'final result: {result.x}')

    plot_comparative_patient1([result])

    data = optimizer.to_optimize.run(result.x)


def run_optimization_spherical_coordinates_patient2():
    from tumor_io.plotting import plot_comparative_patient2

    q3w = TherapyParameters(
        start_therapy_day=14,
        end_therapy_day=557,
        drug_application_interval=7*3
    )

    q6w = TherapyParameters(
        start_therapy_day=557,
        end_therapy_day=10000,
        drug_application_interval=7*6
    )

    therapy_params = [q3w, q6w]

    num_params = NumericalParameters(
        dt=1.,
        final_time=1400,
        num_elements=125*8,
        r_max=0.04,
        r_tumor_init=0.0036,
        med_init = 0
    )
    model_params = TumorModelParameters()
    out_params = OutputParameters(enabled=False)

    runner = SimulationRunner(
        numerical_parameters=num_params,
        tumor_model_parameters=model_params,
        therapy_parameters=therapy_params,
        output_parameters=out_params,
        solver_type=create_system_PN_i
    )

    experimental_data = load_experimental_data(path=PATH_PATIENT_2)
    experimental_data.t = np.append(experimental_data.t, [1350])
    experimental_data.volumes = np.append(experimental_data.volumes, [10])

    optimizer = SimpleLBFGSBasedOptimizer(
        simulation_runner=runner,
        time_offset=176,
        experimental_data=experimental_data
    )
    optimizer.residual.integral_weight = 1e-16

    weights = optimizer.residual.weights
    weights[0] = 0
    # weights[-1] = 100000
    weights[-1] = 10

    optimizer.parameter_manager.upper_bounds[0] = 60
    optimizer.parameter_manager.upper_bounds[1] = 30
    optimizer.parameter_manager.upper_bounds[2] = 2 
    optimizer.parameter_manager.upper_bounds[3] = 1

    optimizer.parameter_manager.set(ParameterIndex.pro_P, 1e-1, 2, 0.55, True, 6.87e-3)
    optimizer.parameter_manager.set(ParameterIndex.MED_eff, 1e-2, 2, 0.7, True, 0.499)
    optimizer.parameter_manager.set(ParameterIndex.landa, 1, 2, 2, False, 1)
    optimizer.parameter_manager.set(ParameterIndex.landa_HN, 1e-2, 1e3, 20., True, 1)
    optimizer.parameter_manager.set(ParameterIndex.r_tumor_init, 0.002, 0.01, 0.004, True, 1)

    result = optimizer.run()

    print(f'final result: {result.x}')

    plot_comparative_patient2([result])

    data = optimizer.to_optimize.run(result.x)


if __name__ == '__main__':
    run_optimization_spherical_coordinates_patient1()
    # run_optimization_spherical_coordinates_patient2()