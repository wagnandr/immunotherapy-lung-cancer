# 3D Simulation for lung cancer under an immunotherapeutic treatment

## Project structure
data/tumor: Contains the volume data from our patients.
data/tetrahedral: The 3D finite-element meshes.
data/Segmentation_Domains: The 2D surface manifolds of the nutrient sources.
reduced_models: Contains the python code of the project.

## Dependencies
- fenics stack (2019.1.0)
- numpy
- scipy
- matplotlib
- PyWavefront (1.3.3)

## Execute code
Run the code, for instance, via
```
OMP_NUM_THREADS=1 mpirun -np 6 python -u reduced_models/run_3d_lung_simulation.py
```
