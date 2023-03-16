import os
import dolfin as df
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from coordinate_systems import CylindricalCoordinateSystem
from solvers import Solver, TimeTherapyEvaluator


def run(
    N, 
    r_max, 
    z_max,
    final_time,
    dt, 
    dt_out,
    params,
    start_therapy,
    length_therapy,
    save_at
):
    initial_guess = df.Expression(('1. / (exp( a * (sqrt(x[0]*x[0]+x[1]*x[1]) -r) ) + 1)', '0'), r=0.002, a=8000, degree=2)

    nutrients = df.Expression('1 + a * x[1]', a=0, degree=1)

    z_max = r_max
    mesh = df.RectangleMesh(df.Point(0, -z_max), df.Point(r_max, z_max), N, 2*N)

    coordinates = CylindricalCoordinateSystem(mesh)

    solver = Solver(coordinates, initial_guess, nutrients, dt, params, save_at)

    solver.write()

    # preiterations for steady state without growth:
    # we set the growth to zero to just refine the interface
    print('Start preiterations')
    solver.pro_P.assign(0)              # deactivate tumor growth
    for i in tqdm(range(round(50/dt)), total=int(round(50/dt))):
        solver.next()                   # advance in time
    solver.pro_P.assign(params.pro_P)   # reactivate tumor growth
    solver.t = 0                        # start at the beginning

    time_therapy = TimeTherapyEvaluator(start=start_therapy, length=length_therapy)

    # start the real iterations:
    initial_tumor_mass = solver.get_tumor_mass()
    t_start = solver.t

    vector_time = [t_start]
    vector_tumor_mass = [initial_tumor_mass]
    vector_medicine_mass = [solver.get_medicine_mass()]

    for i in tqdm(range(int(final_time/dt)), total=int(final_time/dt)):
        # activate immunotherapy:
        solver.eff_ON.c = time_therapy(solver.next_t)

        # calculate next time step:
        solver.next()

        current_tumor_mass = solver.get_tumor_mass()

        vector_time.append(solver.t)
        vector_tumor_mass.append(current_tumor_mass)
        vector_medicine_mass.append(solver.get_medicine_mass())
        np.savetxt(os.path.join(save_at, "info.txt"), np.transpose([vector_time, vector_tumor_mass, vector_medicine_mass]), fmt='%s')

        # write results to files:
        if i % int(round(dt_out/dt)) == 0:
            print(f'{solver.t-t_start}: mass = {current_tumor_mass}, growth {current_tumor_mass/initial_tumor_mass * 100} %')

            solver.write()


if __name__ == '__main__':
    #N = 250 
    N = 125 
    r_max = 0.004

    params = SimpleNamespace()

    dt              = 1 
    final_time      = 600

    # parameters proliferative cells:
    params.mob_P   = 0.0005
    params.pot_P = 1       # changed!
    params.chi_P = 0.0005  # changed!
    #params.chi_P = 0.0005  # changed!
    params.eps_P = 0.0001  # changed!
    params.pro_P   = 0.3   # changed!
    params.gomp_reg = 0.1

    # parameters medical:
    params.MED_mas = 146000    #Schlicke2021: 143597.3811 [g/mol] M_tau
    params.MED_hal = 1.012e16        #Schlicke2021: 1.01e16 [mol/vol] phi_tau^50
    params.MED_tim = 22           #Schlicke2021: 26.7 [days]
    params.MED_dos = 0.200           #Schlicke2021: 0.24 [g]
    #params.MED_eff = 0.067          #Schlicke2021: 0.499 [1/day]
    params.MED_eff = 2 # changed!
    params.MED_avo = 6.022140857e23 #Schlicke2021: 6.022140857e23

    start_therapy   = list(100 + np.array([23,44,65,86,107,128,149,170,191,212,233,254,275,296,317,338,359,380,401,422,443,464,485])) #Start of the therapies
    print(start_therapy)
    # length_therapy  = 1/24                 #Length of the each immunotherapy is 1/2-1hr
    length_therapy  = 1 # changed!

    save_at = 'output_cylindrical'

    run(N, r_max, r_max, final_time, dt, 7, params, start_therapy, length_therapy, save_at)