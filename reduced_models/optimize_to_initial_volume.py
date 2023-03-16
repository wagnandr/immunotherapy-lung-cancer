from run_spherical import SimulationRunner
from parameters import TherapyParameters, NumericalParameters, TumorModelParameters, OutputParameters
from optimization import SimpleLBFGSBasedOptimizer
from tumor_io.experimental_data import (load_experimental_data, PATH_PATIENT_1, PATH_PATIENT_2)
from run_spherical import SolverP


def run_optimization_initial_volume(path_to_data):
    rt_lower = 0 
    rt_upper = 0.007

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655,
        drug_application_interval=14
    )
    num_params = NumericalParameters(
        dt=1.,
        final_time=1,
        num_elements=125,
        r_max=0.02,
        r_tumor_init=rt_lower
    )
    model_params = TumorModelParameters()
    out_params = OutputParameters(enabled=False)

    runner = SimulationRunner(
        numerical_parameters=num_params,
        tumor_model_parameters=model_params,
        therapy_parameters=therapy_params,
        output_parameters=out_params,
        solver_type=SolverP
    )

    experimental_data = load_experimental_data(path=path_to_data)

    optimizer = SimpleLBFGSBasedOptimizer(
        simulation_runner=runner,
        time_offset=0,
        experimental_data=experimental_data
    )

    data = runner.run()
    err = optimizer.to_optimize.residual.error(data).errors[0]
    print (err)

    while abs(err) > 1e-12:
        r = 0.5 * (rt_lower + rt_upper)
        num_params.r_tumor_init = r
        data = runner.run()
        err = optimizer.to_optimize.residual.error(data).errors[0]
        print(rt_upper, rt_lower, r, err)

        if err > 0:
            rt_upper = r
        else:
            rt_lower = r

    print(f'final result: {r}')


if __name__ == '__main__':
    run_optimization_initial_volume(PATH_PATIENT_2)
