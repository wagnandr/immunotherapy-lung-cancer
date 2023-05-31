import os
import sys 
import shutil
import numpy as np
import dolfin as df
from tqdm import tqdm
from coordinate_systems import CartesianCoordinateSystem
from solvers import TimeTherapyEvaluator, SolverPN, SolverTumorPN, MedicineSolver
from parameters import TherapyParameters, TumorModelParameters, NumericalParameters, OutputParameters
from tumor_io.plotting import (
    plot_comparative_patient1,
)
from create_therapy_schedule import create_therapy_schedule
from tumor_io.experimental_data import load_experimental_data 
from typing import List
from initial_conditions import refine_locally, ExpressionSmoothCircle
import vasculature_io as vio
from nutrients import NutrientDoubleContinuumSolver, ParameterNutrients
from utils import (
    QoILogger,
    generate_figure_repository_path
)


def refined_lung_mesh():
    mesh = df.Mesh()
    with df.XDMFFile('data/tetrahedral-meshes/lung.xdmf') as f:
        f.read(mesh)
    mesh.coordinates()[:] /= 1000
    print(f'tets: {mesh.num_entities_global(3)}, vertices: {mesh.num_entities_global(0)}')

    radius = 0.0068 
    midpoint = df.Point(-87.5e-3, -96.9e-3, -254.3e-3)
    mesh = refine_locally(mesh, midpoint, 8 * radius, 2)
    mesh = refine_locally(mesh, midpoint, 1.5 * radius, 1)
    mesh = refine_locally(mesh, midpoint, 1.45 * radius, 2)
    return mesh


class SimulationRunner:
    def __init__(
        self, 
        numerical_parameters: NumericalParameters,
        tumor_model_parameters: TumorModelParameters,
        therapy_parameters: List[TherapyParameters],
        output_parameters: OutputParameters
    ):
        self.numerical_parameters = numerical_parameters
        self.tumor_model_parameters = tumor_model_parameters
        self.therapy_parameters = therapy_parameters
        self.output_parameters = output_parameters
    
    def run(self):

        midpoint = df.Point(-87.5e-3, -96.9e-3, -254.3e-3)
        r = self.numerical_parameters.r_tumor_init
        initial_guess = ExpressionSmoothCircle(midpoint=midpoint, r=r, a=8000, degree=2)

        start_therapy, length_therapy = create_therapy_schedule(self.therapy_parameters)

        mesh = refined_lung_mesh()

        coordinates = CartesianCoordinateSystem(mesh)

        time_therapy = TimeTherapyEvaluator(start=start_therapy, length=length_therapy)

        # Simple nutrient solver with a very simplistic nutrient model
        # nutrient_solver = SolverNutrients(
        #     coordinates,
        #     nutrients,
        #     self.tumor_model_parameters,
        #     self.output_parameters.save_at)

        vasculature = vio.read_default_vanilla(convert_mm_to_m=True)
        nutrient_solver = NutrientDoubleContinuumSolver(
            mesh=mesh,
            parameter=ParameterNutrients(kappa_v=0.001),
            save_at=self.output_parameters.save_at) 

        nutrient_solver.assemble_2d_source_terms(vasculature, accuracy=4)

        tumor_solver = SolverTumorPN(
            coordinates,
            initial_guess,
            self.numerical_parameters.dt,
            self.tumor_model_parameters,
            self.output_parameters.save_at)

        medicine_solver = MedicineSolver(
            self.tumor_model_parameters, 
            time_therapy, 
            self.numerical_parameters.dt, 
            1./24./32.)

        solver = SolverPN(
            tumor_solver=tumor_solver,
            nutrient_solver=nutrient_solver,
            medicine_solver=medicine_solver,
            dt=self.numerical_parameters.dt)

        solver.medicine_solver.MED = self.numerical_parameters.med_init

        solver.write()

        qoi_logger = QoILogger(threshold=0.3)
        qoi_logger.add(solver)

        qoi_directory = generate_figure_repository_path(self.output_parameters.save_at)
        print(f'QoI directory is {qoi_directory}')

        num_time_steps = int(self.numerical_parameters.final_time/self.numerical_parameters.dt)
        out_frequency = int(round(self.output_parameters.dt_out/self.numerical_parameters.dt))

        for i in tqdm(range(1, num_time_steps+1), total=int(num_time_steps)):
            exit()
            print('before solve')
            solver.next()
            exit()

            qoi_logger.add(solver)


            if self.output_parameters.enabled:
                if self.output_parameters.enabled and i % out_frequency == 0:
                    qoi_logger.write(filepath=os.path.join(qoi_directory, "info.txt"))
                    solver.write()
        
        return qoi_logger.get_dict()


def run_patient_1():
    vec = np.array([55.*1.5, 15*0.6, 2.])
    vec = np.array([55.*1.4, 15*0.6, 2.])

    therapy_params = TherapyParameters(
        start_therapy_day=146,
        end_therapy_day=1655-176,
        drug_application_interval=14
    )

    numerical_params = NumericalParameters(
        dt=1/24.,
        #dt=1.,
        final_time=2400,
        # final_time=100,
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
        landa_HN=0.025,
        sigma_HN=0.2
    )

    out_params = OutputParameters(dt_out=10, save_at='output_3d_lung')

    data = SimulationRunner(
        numerical_parameters=numerical_params,
        tumor_model_parameters=model_params,
        therapy_parameters=[therapy_params],
        output_parameters=out_params
    ).run()
    print('finished')

    #plot_comparative_patient1([data])


if __name__ == '__main__':
    run_patient_1()