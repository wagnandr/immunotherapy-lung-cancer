import argparse
import numpy as np
from coordinate_systems import SphericalCoordinateSystem
from solvers import (
    create_system_PN_i,
)
from parameters import (
    TherapyParameters, 
    TumorModelParameters, 
    NumericalParameters, 
    OutputParameters
)
from create_therapy_schedule import create_therapy_schedule
from tumor_io.plotting import (
    plot_comparative_patient1, 
)
from tumor_io.experimental_data import load_experimental_data 
from typing import List
from spherical_simulation_runner import SimulationRunner


def run_patient_1(dt, final_time, dosage, vec=None, verbose=True, output_enabled=True):
    if vec is None:
        vec = np.array([52.5, 15 * 0.7, 2.])

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655-176,
        drug_application_interval=7*2
    )

    numerical_params = NumericalParameters(
        #dt=1/24.,
        dt=dt,
        #final_time=2400,
        #final_time=2400,
        final_time=final_time,
        num_elements=125*4,
        r_max=0.04,
        r_tumor_init=0.006074216961365892
    )

    scaling = np.array([6.87e-3, 0.499, 1])

    parameters = vec * scaling

    pro_P = parameters[0]
    med = parameters[1]
    landa = parameters[2]

    model_params = TumorModelParameters(
        pro_P=pro_P,
        landa=landa,
        MED_eff=med,
        landa_HN=0.1,
        # landa_HN=0.0,
        #MED_dos=0.240 / 8.
        MED_dos=0.240 * dosage 
        # MED_dos=0.240 / 8.
    )

    out_params = OutputParameters(dt_out=1, save_at=f'output/spherical_patient_1/dt={dt}/ft={final_time}/dos={dosage}', enabled=output_enabled)

    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        solver_type=create_system_PN_i,
    ).run(verbose=verbose)

    return data


def run_patient_1_parametrized(dt, final_time, model_params: TumorModelParameters, verbose=True, output_enabled=True):
    dosage = 1.
    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655-176,
        drug_application_interval=7*2
    )

    numerical_params = NumericalParameters(
        #dt=1/24.,
        dt=dt,
        #final_time=2400,
        #final_time=2400,
        final_time=final_time,
        num_elements=125*4,
        r_max=0.04,
        r_tumor_init=0.006074216961365892
    )

    out_params = OutputParameters(dt_out=1, save_at=f'output', enabled=output_enabled)

    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        solver_type=create_system_PN_i,
    ).run(verbose=verbose)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt', type=float, default=1./24.)
    parser.add_argument('--final-time', type=float, default=2400)
    parser.add_argument('--dosage', type=float, default=1.)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    data = run_patient_1(dt=args.dt, final_time=args.final_time, dosage=args.dosage)

    if args.show: 
        plot_comparative_patient1([data])
