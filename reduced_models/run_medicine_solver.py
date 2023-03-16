import math
import time
from parameters import TumorModelParameters 
from time_therapy_evaluator import TimeTherapyEvaluator
from medicine_solver import MedicineSolver
from create_therapy_schedule import create_therapy_schedule
from parameters import TherapyParameters


if __name__  == '__main__':

    params = TumorModelParameters(
        MED_mas=146000,         #Schlicke2021: 143597.3811 [g/mol] M_tau
        MED_hal=1.009e16,       #Schlicke2021: 1.01e16 [mol/vol] phi_tau^50
        MED_tim=22,             #Schlicke2021: 26.7 [days]
        MED_dos=0.200,          #Schlicke2021: 0.24 [g]
        MED_eff=0.088,          #Schlicke2021: 0.499 [1/day]
        MED_avo=6.022140857e23, #Schlicke2021: 6.022140857e23
    )

    dt = 1.
    dt_immuno = 1. / 24. / 32.

    therapy_interval = 3*7.
    therapy_application_length = 1./24. 

    therapy_start, therapy_length = create_therapy_schedule([TherapyParameters(0, 500, therapy_interval, therapy_application_length)]) 
    time_therapy = TimeTherapyEvaluator(therapy_start, therapy_length)
    #time_therapy = lambda x: [1]*len(x)
    solver_medicine = MedicineSolver(params, time_therapy, dt, dt_immuno)

    med = 0
    medicine_list = [med]

    num_time_steps = 256 

    med_static = (params.MED_avo * params.MED_dos * params.MED_tim) / (params.MED_mas * math.log(2)) * (therapy_application_length / therapy_interval)

    start = time.time()
    for it_t in range(num_time_steps):
        t_prev = it_t*dt

        solver_medicine.reassign()
        solver_medicine.solve(t_prev)

        medicine_list.append(solver_medicine.MED)
    elapsed = time.time() - start
    print(f'elapsed = {elapsed}')

    from matplotlib import pyplot as plt
    plt.plot(medicine_list)
    plt.axhline(y=med_static, color='r', linestyle='-')
    plt.grid(True)
    plt.show()