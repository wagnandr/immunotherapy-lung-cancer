import numpy as np
from typing import List
from parameters import TherapyParameters


def create_therapy_schedule_single_therapy(params: TherapyParameters):
    start_therapy = params.drug_application_interval * np.arange(0,200) + params.start_therapy_day
    start_therapy = [t for t in list(start_therapy) if t < params.end_therapy_day]
    length_therapy = [params.length_therapy for t in list(start_therapy)]
    return start_therapy, length_therapy


def create_therapy_schedule(params_list: List[TherapyParameters]):
    full_therapy_plan_start = []
    full_therapy_plan_length = []
    for params in params_list:
        start_therapy, length_therapy = create_therapy_schedule_single_therapy(params)
        full_therapy_plan_start += start_therapy
        full_therapy_plan_length += length_therapy
    return full_therapy_plan_start, full_therapy_plan_length 