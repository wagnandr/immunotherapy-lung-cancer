import numpy as np
import argparse
from solvers import (
    create_system_PN_i,
)
from parameters import (
    TherapyParameters, 
    TumorModelParameters, 
    NumericalParameters, 
    OutputParameters
)
from tumor_io.plotting import (
    plot_comparative_patient2
)
from spherical_simulation_runner import SimulationRunner


def run_patient_2(dt: float, final_time: float, therapy_type: int, dosage=1., vec=None):
    if vec is None:
        vec = np.array([0.55, 1.0, 1., 20., 0.0045])
        vec = np.array([0.55, 2., 1., 20., 0.0045])
        vec = np.array([0.55, 1.5, 1., 20., 0.0045])
        vec = np.array([0.55, 1.25, 1., 20., 0.0045]) # maybe
        vec = np.array([0.55, 1.1, 1., 20., 0.0045]) # maybe
        #vec = np.array([0.625, 1.1, 1., 20., 0.0045]) # maybe

    scaling = np.array([6.87e-3, 0.499, 1, 1, 1])

    parameters = vec * scaling

    pro_P = parameters[0]
    med = parameters[1]
    landa = parameters[2]
    landa_HN = parameters[3]
    r_tumor_init = parameters[4]

    med_init = 1.2e17 /2 * dosage
    start_therapy = 14

    if therapy_type == 1:
        q3w = TherapyParameters(
            start_therapy_day=start_therapy,
            end_therapy_day=557,
            drug_application_interval=7*3
        )
        
        q6w = TherapyParameters(
            start_therapy_day=557,
            end_therapy_day=1104,
            drug_application_interval=7*6
        )

        therapy_params = [ q3w, q6w ]
    elif therapy_type == 2:
        q3w = TherapyParameters(
            start_therapy_day=start_therapy,
            end_therapy_day=557,
            drug_application_interval=7*3
        )
        
        q6w = TherapyParameters(
            start_therapy_day=557,
            end_therapy_day=11040,
            drug_application_interval=7*6
        )

        therapy_params = [q3w, q6w]
    elif therapy_type == 3:
        q3w = TherapyParameters(
            start_therapy_day=start_therapy,
            end_therapy_day=10000,
            drug_application_interval=7*3
        )
        therapy_params = [ q3w, ]
    else:
        raise RuntimeError("unknown therapy type")

    numerical_params = NumericalParameters(
        dt=dt,
        final_time=final_time,
        num_elements=125 * 8,
        r_max=0.04,
        r_tumor_init=r_tumor_init,
        med_init=med_init
    )

    model_params = TumorModelParameters(
        pro_P=pro_P,
        landa=landa,
        MED_eff=med,
        landa_HN=landa_HN,
        MED_mas=143600,
        MED_dos=0.2 * dosage,
        MED_tim=22.
    )

    out_params = OutputParameters(dt_out=1, save_at=f'output/spherical_patient_2/dt={dt}/ft={final_time}/tt={therapy_type}/dos={dosage}')


    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=therapy_params,
        output_parameters=out_params,
        solver_type=create_system_PN_i
    ).run()

    plot_comparative_patient2([data])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt', type=float, default=1./24.)
    parser.add_argument('--final-time', type=float, default=1800)
    parser.add_argument('--therapy-type', type=int, choices=[1, 2, 3])
    parser.add_argument('--dosage', type=float, default=1.)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    data = run_patient_2(dt=args.dt, final_time=args.final_time, therapy_type=args.therapy_type, dosage=args.dosage)

    if args.show: 
        run_patient_2(final_time=args.final_time, therapy_type=int(args.therapy_type), dt=1./24.)