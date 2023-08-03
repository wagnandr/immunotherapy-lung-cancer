'''
Script for executing several dosages for patient 1 in parallel.
'''

import multiprocessing
import subprocess

    
def mywork(dosage):
    print(f'Started (dosage {dosage})')
    cmd = ['python', 'reduced_models/run_spherical_patient_1.py', 
           '--dt', f'{1./24.}', 
           '--final-time', f'{2400}', 
           '--dosage', f'{dosage}']
    res = subprocess.run(cmd, shell=False, capture_output=False)
    print(res)
    print(f'Finished (dosage {dosage})')

    
if __name__ == '__main__':
    pool = multiprocessing.Pool(6)
    r = pool.map_async(mywork, [1., 0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.1, 0.075, 0.05])
    r.wait()