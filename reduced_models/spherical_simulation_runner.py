
import os
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
    
    def run(self, verbose=True):
        qoi_directory = generate_figure_repository_path(self.output_parameters.save_at) if self.output_parameters.enabled else None
        print(f'QoI directory is {qoi_directory}')

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
        out_frequency = int(round(self.output_parameters.dt_out/self.numerical_parameters.dt))

        for i in tqdm(range(num_time_steps), total=int(num_time_steps), disable=(not verbose)):
            solver.next()

            qoi_logger.add(solver)

            if self.output_parameters.enabled and i % out_frequency == 0:
                qoi_logger.write(filepath=os.path.join(qoi_directory, "info.txt"))
                solver.write()
        
        return qoi_logger.get_dict()