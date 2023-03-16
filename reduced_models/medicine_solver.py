import math
from typing import Callable
from parameters import TherapyParameters 
import numpy as np


class MedicineSolver:
    def __init__(
        self, 
        params: TherapyParameters, 
        time_therapy: Callable[[float], float], 
        dt_large: float, 
        dt_small: float
    ):
        self.params = params
        self.time_therapy = time_therapy
        self.dt = dt_small
        self.num_steps = int(round(dt_large / dt_small))
        self.MED = 0
        self.MED_0 = 0
    
    def reassign(self):
        self.MED_0 = self.MED

    def solve(self, t: float):
        self.MED = self.MED_0 
        time_points = t + np.arange(self.num_steps+1) * self.dt
        eff_on = self.time_therapy(time_points)

        MED_tim = self.params.MED_tim
        MED_avo = self.params.MED_avo
        MED_mas = self.params.MED_mas
        MED_dos = self.params.MED_dos

        med_prev = self.MED

        for i in range(self.num_steps):
            med_next = med_prev
            med_next += self.dt * 0.5* MED_avo/MED_mas * MED_dos * eff_on[i]
            med_next += self.dt * 0.5* MED_avo/MED_mas * MED_dos * eff_on[i+1]
            med_next += - self.dt * 0.5 * med_prev * math.log(2) / MED_tim
            med_next /= (1 + self.dt * 0.5 * math.log(2)/MED_tim) 
            self.MED = med_prev = med_next
            t += self.dt
