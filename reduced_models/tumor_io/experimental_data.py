import os
import pathlib
from dataclasses import dataclass
import datetime
from typing import List
import numpy as np


@dataclass
class ExperimentalData:
    volumes: List[float]
    recist: List[float]
    who: List[float]
    t: List[float]
    dates: List[datetime.date]
    date_labels: List[str]


def absolute_path_from_relative(*relative_path) -> str:
    current_dir = pathlib.Path(__file__).parent.resolve()
    path = os.path.join(current_dir, *relative_path)
    return path


PATH_PATIENT_1 = absolute_path_from_relative('..', '..', 'data/tumor/pirmin-tumor-pat1.csv')


PATH_PATIENT_2 = absolute_path_from_relative('..', '..', 'data/tumor/pirmin-tumor-pat2.csv')


def load_experimental_data(t_start=0, offset_rows=1, filter_by_volume=True, path=None):
    if path is None:
        path = PATH_PATIENT_1 

    data = np.genfromtxt(path, delimiter=',', filling_values=np.NaN)

    t = data[offset_rows:,0]
    volume = data[offset_rows:,1]
    recist = data[offset_rows:,2]
    who = data[offset_rows:,3]

    if filter_by_volume:
        mask = ~np.isnan(volume)
        t = t[mask]
        volume = volume[mask]
        recist = recist[mask]
        who = who[mask]

    date_labels = [str(tt) for tt in t]

    t_start_index = len([ x for x in t if x < t_start])

    volume = volume[t_start_index:]
    recist = recist[t_start_index:]
    who = who[t_start_index:]
    t = np.array(t)[t_start_index:] - t[t_start_index]
    date_labels = date_labels[t_start_index:]

    return ExperimentalData(volume, recist, who, t, [], date_labels)
