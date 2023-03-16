from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass(frozen=False)
class TherapyParameters:
    start_therapy_day: int
    end_therapy_day: int
    drug_application_interval: int
    length_therapy:float = 1./24.


@dataclass(frozen=False)
class TumorModelParameters:
    mob_P:float = 5e-4 
    chi_P:float = 0.0000
    eps_P:float = 5e-4
    pro_P:float = 30.
    pot_P:float = 2. 
    deg_P:float = 0. 
    gomp_reg:float = 0.1
    nutrient_value:float = 0.975
    landa:float = 2

    landa_HN:float = 1e-2 
    sigma_HN:float = 0.2 
    source_NUT:float = 1.
    alpha_healthy:float = 1. 
    alpha_tumor:float = 4 
    kappa:float = 1e-5 

    # parameters medical:
    MED_mas:float = 146000
    MED_hal:float = 1.012e16
    MED_tim:float = 26.7
    MED_dos:float = 0.240
    MED_eff:float = 4.
    MED_avo:float = 6.022140857e23


@dataclass(frozen=False)
class NumericalParameters:
    dt: float
    final_time: float
    num_elements:int = 625 
    r_max:float = 0.1
    r_tumor_init:float = 0.006
    med_init: float = 0.


@dataclass(frozen=False)
class OutputParameters:
    dt_out: float = 1.
    save_at: str = 'output'
    enabled:bool = True
