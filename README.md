# Assignment-4
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
