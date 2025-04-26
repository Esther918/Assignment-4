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
Analytical tip displacement: -8.758929e-01  
Numerical tip displacement: -7.966060e-01   
  
* Part B
```
python partB.py
```
=== h-refinement Study ===    
Nx=4: DOFs=90, Error=-3.66%  
Nx=8: DOFs=170, Error=-1.42%  
Nx=16: DOFs=330, Error=-0.59%  
Nx=32: DOFs=650, Error=-0.27%  
Nx=64: DOFs=1290, Error=-0.13%  
=== p-refinement Study ===  
p=1: DOFs=54, Error=-93.18%  
p=2: DOFs=170, Error=-1.42%  
p=3: DOFs=350, Error=-0.62%  
p=4: DOFs=594, Error=-0.37%  

* Part C
```
python fail.py
```
Relative error: 10.45%  
Reasons that FEniCSX fail:  
1. Improper mesh size
2. Free boundary condition

