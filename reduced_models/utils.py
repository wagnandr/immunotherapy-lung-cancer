import os
import sys
import shutil
import numpy as np
import dolfin as df


def generate_figure_repository_path(base_directory):
    ite = 1
    save_at = os.path.join(base_directory, str(ite))
    while os.path.exists(save_at):
        ite = ite + 1
        save_at = os.path.join(base_directory, str(ite))
    df.MPI.barrier(df.MPI.comm_world)
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(f'mkdirs {save_at}')
        os.makedirs(save_at, exist_ok=True)   
        shutil.copyfile(sys.argv[0],  os.path.join(save_at, 'run.py'))
    return save_at


class QoILogger:
    def __init__(self, threshold=0.3) -> None:
        self.threshold = threshold 
        self.time = []
        self.tumor_mass = []
        self.tumor_mass_visible = []
        self.tumor_mass_sigmoid = []
        self.proliferative_mass = []
        self.necrotic_mass = []
        self.medicine_mass = []
        self.nutrient_v_mass = []
        self.nutrient_i_mass = []
    
    def add(self, solver):
        self.time.append(solver.t)
        self.tumor_mass.append(solver.get_tumor_mass())
        self.tumor_mass_visible.append(solver.get_tumor_mass_threshold(threshold=self.threshold))
        self.tumor_mass_sigmoid.append(solver.get_tumor_mass_sigmoid(threshold=self.threshold))
        self.proliferative_mass.append(solver.get_proliferative_mass())
        self.necrotic_mass.append(solver.get_necrotic_mass())
        self.medicine_mass.append(solver.get_medicine_mass())
        self.nutrient_v_mass.append(solver.get_nutrient_v_mass())
        self.nutrient_i_mass.append(solver.get_nutrient_i_mass())
    
    def write(self, filepath):
        np.savetxt(filepath, np.transpose([
            self.time, 
            self.tumor_mass, 
            self.proliferative_mass,
            self.necrotic_mass,
            self.tumor_mass_visible,
            self.medicine_mass, 
            self.nutrient_v_mass,
            self.nutrient_i_mass
        ]), fmt='%s')
    
    def read(self, filepath):
        data = np.loadtxt(filepath)
        self.time = data[:,0]
        self.tumor_mass = data[:,1]
        self.proliferative_mass = data[:,2]
        self.necrotic_mass = data[:,3]
        self.tumor_mass_visible = data[:,4]
        self.medicine_mass = data[:,5]
        self.nutrient_v_mass = data[:,6]
        self.nutrient_i_mass = data[:,7]
    
    def get_dict(self):
        return {
            'time': np.array(self.time),
            'tumor': np.array(self.tumor_mass),
            'tumor_visible': np.array(self.tumor_mass_visible),
            'tumor_sigmoid': np.array(self.tumor_mass_sigmoid),
            'medicine': np.array(self.medicine_mass),
            'nutrient': np.array(self.nutrient_i_mass),
        }
