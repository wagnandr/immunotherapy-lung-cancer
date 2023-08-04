import numpy as np
from coordinate_systems import SphericalCoordinateSystem
from solvers import (
    create_system_P_i, create_system_PN_i
)
from parameters import (
    TherapyParameters, 
    TumorModelParameters, 
    NumericalParameters, 
    OutputParameters
)
from create_therapy_schedule import create_therapy_schedule
from typing import List
import dolfin as df
from tqdm import tqdm
from coordinate_systems import SphericalCoordinateSystem
from solvers import (
    TimeTherapyEvaluator, 
)
from parameters import (
    TherapyParameters, 
    TumorModelParameters, 
    NumericalParameters, 
    OutputParameters
)
from create_therapy_schedule import create_therapy_schedule
from typing import List
from utils import QoILogger


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

        num_time_steps = int(self.numerical_parameters.final_time/self.numerical_parameters.dt)

        min_mass = max_mass = solver.get_tumor_mass_threshold(0.3)

        for i in tqdm(range(num_time_steps), total=int(num_time_steps), disable=True):
            solver.next()

            qoi_logger.add(solver)

            mass = solver.get_tumor_mass_threshold(0.3)

            min_mass = min(min_mass, mass)
            max_mass = max(max_mass, mass)

            if (max_mass > 2 * min_mass):
                return qoi_logger.get_dict()


def run_patient_1(dt, final_time, pro_P, landa):

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655-176,
        drug_application_interval=7*2
    )

    numerical_params = NumericalParameters(
        dt=dt,
        final_time=final_time,
        num_elements=125*4,
        r_max=0.04,
        r_tumor_init=0.006074216961365892
    )

    med = 15 * 0.6

    model_params = TumorModelParameters(
        pro_P=pro_P,
        landa=landa,
        MED_eff=med,
        landa_HN=0.050,
        sigma_HN=0.2,
    )

    out_params = OutputParameters(dt_out=1, save_at=f'output/tmp')

    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params,
        #solver_type=create_system_P_i
        solver_type=create_system_PN_i
    ).run()

    return data


def run_optimization(goal_dd):
    list_landas = np.linspace(1., 2., 10).tolist()
    list_pro_P = []

    for landa in list_landas:

        pro_P_upper = 100000 
        pro_P_lower = 6.87e-3 * 1. * 1.25
        current_dd = 0
        while (np.abs(goal_dd - current_dd) > 1./24.):
            pro_P = 0.5 * (pro_P_upper + pro_P_lower)
            data = run_patient_1(dt=1./2., final_time=2000, pro_P=pro_P, landa=landa)
            current_dd = data['time'][-1]
            if current_dd < goal_dd:
                pro_P_upper = pro_P
            else:
                pro_P_lower = pro_P
            print(f'> T = {current_dd}, lower={pro_P_lower}, upper={pro_P_upper}, diff={goal_dd - current_dd}')
        
        list_pro_P.append(pro_P)
        print(f'landas = {list_landas[:len(list_pro_P)]}')
        print(f'pro_P = {list_pro_P}')


def run_direct_line():
    list_landas = [1.0, 1.1111111111111112, 1.2222222222222223, 1.3333333333333333, 1.4444444444444444, 1.5555555555555556, 1.6666666666666665, 1.7777777777777777, 1.8888888888888888, 2.0]
    list_pro_P = [0.013570075346314117, 0.020368729557172626, 0.02972852062828606, 0.04118378731233534, 0.05301158299423987, 0.063535527183651, 0.07284875213003252, 0.08160318357963114, 0.08979882153244688, 0.09799445948526264]

    list_landas, list_pro_P = map(np.array, [list_landas, list_pro_P])
    pro_P_start = list_pro_P[0]
    pro_P_end = list_pro_P[-1]

    list_pro_P = pro_P_end * (list_landas - list_landas[0]) + pro_P_start * (list_landas[-1] - list_landas)

    list_tdt = []

    for landa,pro_P in zip(list_landas, list_pro_P):
        print(landa, pro_P)
        data = run_patient_1(dt=1./2., final_time=2000, pro_P=pro_P, landa=landa)
        current_tdt = data['time'][-1]
        list_tdt.append(current_tdt)
        print(f'tdt {list_tdt}')


def run_regression_line():
    list_landas = [1.0, 1.1111111111111112, 1.2222222222222223, 1.3333333333333333, 1.4444444444444444, 1.5555555555555556, 1.6666666666666665, 1.7777777777777777, 1.8888888888888888, 2.0]
    list_pro_P = [0.013570075346314117, 0.020368729557172626, 0.02972852062828606, 0.04118378731233534, 0.05301158299423987, 0.063535527183651, 0.07284875213003252, 0.08160318357963114, 0.08979882153244688, 0.09799445948526264]

    list_landas, list_pro_P = map(np.array, [list_landas, list_pro_P])
    list_landas_log, list_pro_P_log = map(np.log, [list_landas, list_pro_P])

    from scipy.stats import linregress
    a = linregress(list_landas_log, list_pro_P_log)

    print(a)

    #list_pro_P = list_landas * a.slope + a.intercept
    list_pro_P = np.exp(np.log(list_landas) * a.slope + a.intercept)

    list_tdt = []

    for landa,pro_P in zip(list_landas, list_pro_P):
        print(landa, pro_P)
        data = run_patient_1(dt=1./2., final_time=2000, pro_P=pro_P, landa=landa)
        current_tdt = data['time'][-1]
        list_tdt.append(current_tdt)
        print(f'tdt {list_tdt}')


if __name__ == '__main__':
    #run_regression_line()
    run_optimization(goal_dd=100)