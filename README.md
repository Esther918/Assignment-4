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
python linearelasticity.py
```
# Assignment-4 Part 2
* Part A
```
python linear_elasticity.py
```
Comparison:
Analytical tip displacement: -8.758929e-01 mm
Numerical tip displacement: -7.966060e-01 mm
Relative error: 9.05%
  
* Part B
```
python partB.py
``` 
* Part C
```
python fail.py
```
Reasons that FEniCSX fail:  
1. The mesh is invalid

