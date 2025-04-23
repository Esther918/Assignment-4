# Assignment-4 Part 1
* Start with FEniCSX  
```
module load miniconda
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
```
Run the example (this is an example from: https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html  see linearelasticity_code.ipynb)  
```
python linear_elasticity.py
```
# Assignment-4 Part 2
* Part A  
  Analytical solution: Maximum deflection at x=5, u =−0.0265625  
* Part B  
  h refinement:
  p refinement:
* Part C  
Reasons that FEniCSX fail:  
1. The mesh is invalid

