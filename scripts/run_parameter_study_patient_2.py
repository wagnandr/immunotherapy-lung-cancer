'''
Script for executing several dosages and therapy types for patient 2 in parallel.
'''

import multiprocessing
import subprocess

    
def mywork(arg):
    therapy_type, dosage = arg
    print(f'Started (Therapy type {therapy_type})')
    cmd = ['python', 'reduced_models/run_spherical_patient_2.py', 
           '--dt', f'{1./24.}', 
           '--final-time', f'{2400*2}', 
           '--therapy-type', f'{therapy_type}',
           '--dosage', f'{dosage}'
    ]
    res = subprocess.run(cmd, shell=False, capture_output=False)
    print(res)
    print(f'Finished (Therapy type {therapy_type})')
    

if __name__ == '__main__':
    pool = multiprocessing.Pool(6)
    args = list(zip([1,2,3]*3, [1.]*3 + [2.]*3 + [10.]*3))
    r = pool.map_async(mywork, args)
    r.wait()
