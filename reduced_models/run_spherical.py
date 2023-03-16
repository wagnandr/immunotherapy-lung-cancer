import os
import dolfin as df
import numpy as np
from tqdm import tqdm
from coordinate_systems import SphericalCoordinateSystem
from solvers import (
    create_system_P_i,
    create_system_PN_i,
    TimeTherapyEvaluator, 
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
    plot_comparative_patient2
)
from tumor_io.experimental_data import load_experimental_data 
from typing import List
from utils import (
    QoILogger, 
    generate_figure_repository_path
)


class SimulationRunner:
    def __init__(
        self, 
        numerical_parameters: NumericalParameters,
        tumor_model_parameters: TumorModelParameters,
        therapy_parameters: List[TherapyParameters],
        output_parameters: OutputParameters,
        solver_type 
    ):
        self.numerical_parameters = numerical_parameters
        self.tumor_model_parameters = tumor_model_parameters
        self.therapy_parameters = therapy_parameters
        self.output_parameters = output_parameters
        self.solver_type = solver_type
    
    def run(self):
        # print('running', self.tumor_model_parameters)
        initial_guess = df.Expression(('1. / (exp( a * (x[0]-r) ) + 1)', '0'), r=self.numerical_parameters.r_tumor_init, a=2000, degree=2)

        nutrient_value = self.tumor_model_parameters.nutrient_value
        nutrients = df.Constant(nutrient_value)

        start_therapy, length_therapy = create_therapy_schedule(self.therapy_parameters)

        mesh = df.IntervalMesh(self.numerical_parameters.num_elements, 0, self.numerical_parameters.r_max)

        coordinates = SphericalCoordinateSystem(mesh)

        time_therapy = TimeTherapyEvaluator(start=start_therapy, length=length_therapy)

        solver = self.solver_type(coordinates, initial_guess, nutrients, self.numerical_parameters.dt, self.tumor_model_parameters, time_therapy, self.output_parameters.save_at)
        solver.medicine_solver.MED = self.numerical_parameters.med_init

        solver.write()

        qoi_logger = QoILogger(threshold=0.3)
        qoi_logger.add(solver)

        qoi_directory = generate_figure_repository_path(self.output_parameters.save_at)
        print(f'QoI directory is {qoi_directory}')

        num_time_steps = int(self.numerical_parameters.final_time/self.numerical_parameters.dt)
        out_frequency = int(round(self.output_parameters.dt_out/self.numerical_parameters.dt))

        for i in tqdm(range(num_time_steps), total=int(num_time_steps)):
            solver.next()

            qoi_logger.add(solver)

            if self.output_parameters.enabled and i % out_frequency == 0:
                qoi_logger.write(filepath=os.path.join(qoi_directory, "info.txt"))
                solver.write()
        
        return qoi_logger.get_dict()

def run_patient_1(vec=None):
    if vec is None:
        vec = np.array([55., 15 * 0.6, 2.])

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655-176,
        drug_application_interval=7*2
    )

    numerical_params = NumericalParameters(
        dt=1/24.,
        #dt=1.,
        #final_time=2400,
        final_time=2400,
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
        #MED_dos=0.240 / 8.
        MED_dos=0.240 # * 0.24
        # MED_dos=0.240 / 8.
    )

    out_params = OutputParameters(dt_out=1, save_at='output_spherical_9')

    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        solver_type=create_system_PN_i
    ).run()

    plot_comparative_patient1([data])


def run_patient_2(vec=None):
    if vec is None:
        vec = np.array([0.55, 1.0, 1., 20., 0.0045])
        vec = np.array([0.55, 2., 1., 20., 0.0045])
        vec = np.array([0.55, 1.5, 1., 20., 0.0045])
        vec = np.array([0.55, 1.25, 1., 20., 0.0045]) # maybe
        vec = np.array([0.55, 1.1, 1., 20., 0.0045]) # maybe

    scaling = np.array([6.87e-3, 0.499, 1, 1, 1])

    parameters = vec * scaling

    pro_P = parameters[0]
    med = parameters[1]
    landa = parameters[2]
    landa_HN = parameters[3]
    r_tumor_init = parameters[4]

    #med_init = 1.52e17
    med_init = 5.9e16 
    start_therapy = 14

    therapy_type = 3

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
        #dt=1./24.,
        dt=1.,
        final_time=1800,
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
        #MED_dos=0.2 * 100000000,
        MED_dos=0.2 * 2,
        MED_tim=22.
    )

    out_params = OutputParameters(dt_out=1, save_at=f'output_spherical_much_new_{therapy_type}')


    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=therapy_params,
        output_parameters=out_params,
        solver_type=create_system_PN_i
        #solver_type=create_system_P_i
    ).run()

    plot_comparative_patient2([data])


if __name__ == '__main__':
    # run_patient_2()
    run_patient_1()