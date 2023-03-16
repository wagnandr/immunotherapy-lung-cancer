import numpy as np
from parameters import (
    TherapyParameters, 
    NumericalParameters, 
    TumorModelParameters, 
    OutputParameters)
from run_spherical import SimulationRunner
from solvers import SolverP, SolverPN
from tumor_io.plotting import plot
from tumor_io.experimental_data import load_experimental_data, ExperimentalData
from optimization import SimpleLBFGSBasedOptimizer, ParameterIndex


def draw_samples(data, interval_size=30):
    mask = [i % interval_size == 0 and i != 0 for i in range(data['time'].size)]
    for k in data.keys():
        data[k] = data[k][mask]
    return data


def data_to_experimental_data(data):
    return ExperimentalData(
        t=data['time'],
        volumes=data['tumor_sigmoid'] * 1e6,
        recist=None,
        who=None,
        dates=None,
        date_labels=None
    )


def run_optimization_synthetical():
    from tumor_io.plotting import plot_comparative_patient1

    interval_size = 30
    pro_P_active = True 
    pro_P_syn = 3. 
    pro_P_scale = 6.87e-2

    med_eff_active = True 
    med_eff_syn = 4.
    med_eff_scale = 1.

    r_tumor_init_active = False
    r_tumor_init_syn = 0.006074216961365892
    r_tumor_init_scale = 1.

    solver_type = SolverPN

    therapy_params = TherapyParameters(
        #start_therapy_day=146,
        start_therapy_day=0,
        end_therapy_day=1655,
        drug_application_interval=14
    )
    num_params = NumericalParameters(
        dt=0.5,
        final_time=100,
        num_elements=125*4,
        r_max=0.04,
        r_tumor_init=r_tumor_init_syn*r_tumor_init_scale,
        med_init = 0
    )
    model_params = TumorModelParameters(
        pro_P=pro_P_syn * pro_P_scale,
        MED_eff=med_eff_syn * med_eff_scale
    )
    out_params = OutputParameters(enabled=False)

    runner = SimulationRunner(
        numerical_parameters=num_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        solver_type=solver_type
    )

    data_syn = runner.run()
    data_syn = draw_samples(data_syn, interval_size=interval_size)
    experimental_data = data_to_experimental_data(data_syn)

    runner = SimulationRunner(
        numerical_parameters=num_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        solver_type=solver_type
    )
    optimizer = SimpleLBFGSBasedOptimizer(
        simulation_runner=runner,
        time_offset=0,
        experimental_data=experimental_data,
    )
    optimizer.residual.integral_weight = 0
    optimizer.to_optimize.verbose = True

    optimizer.parameter_manager.deactivate_all_parameters()

    if pro_P_active:
        pro_P_lower = pro_P_syn * 0.01 
        pro_P_upper = pro_P_syn * 100
        pro_P_0 = pro_P_syn * 0.1
        optimizer.parameter_manager.set(ParameterIndex.pro_P, lower=pro_P_lower, upper=pro_P_upper, start=pro_P_0, scale=pro_P_scale, active=True)
    else:
        optimizer.parameter_manager.set(ParameterIndex.pro_P, start=pro_P_syn, scale=pro_P_scale, active=False)

    if med_eff_active:
        med_eff_lower = med_eff_syn * 0.01 
        med_eff_upper = med_eff_syn * 100
        med_eff_0 = med_eff_syn * 0.2
        optimizer.parameter_manager.set(ParameterIndex.MED_eff, lower=med_eff_lower, upper=med_eff_upper, start=med_eff_0, scale=med_eff_scale, active=True)
    else:
        optimizer.parameter_manager.set(ParameterIndex.MED_eff, start=model_params.MED_eff, scale=1., active=False)
        
    if r_tumor_init_active:
        r_tumor_init_lower = r_tumor_init_syn * 0.4 
        r_tumor_init_upper = r_tumor_init_syn * 2 
        r_tumor_init_0 = r_tumor_init_syn * 0.8
        optimizer.parameter_manager.set(ParameterIndex.r_tumor_init, lower=r_tumor_init_lower, upper=r_tumor_init_upper, start=r_tumor_init_0, scale=r_tumor_init_scale, active=True)
    else:
        optimizer.parameter_manager.set(ParameterIndex.r_tumor_init, start=r_tumor_init_syn, scale=r_tumor_init_scale, active=False)


    optimizer.parameter_manager.set(ParameterIndex.landa, start=model_params.landa, scale=1., active=False)
    optimizer.parameter_manager.set(ParameterIndex.landa_HN, start=model_params.landa_HN, scale=1., active=False)

    x00 = np.array(optimizer.parameter_manager.x00.tolist())

    if True:
        result = optimizer.run()
        data = optimizer.to_optimize.run(result.x)
        data00 = optimizer.to_optimize.run(x00)

        plot([data, data00, data_syn], ['end', 'initial', 'synthetic'])

    if False:
        model_params = TumorModelParameters(
            pro_P= 0.1 * pro_P_syn * pro_P_scale 
        )
        runner = SimulationRunner(
            numerical_parameters=num_params,
            tumor_model_parameters=model_params,
            therapy_parameters=[therapy_params],
            output_parameters=out_params,
            #solver_type=SolverPN
            solver_type=SolverP
        )

        data = runner.run()
        plot([data, data_syn], ['data0', 'syn'])



run_optimization_synthetical()