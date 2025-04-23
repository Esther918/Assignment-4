from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import pyvista

# Define coordinates of 5 points
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
], dtype=np.float64)

# Define triangles
cells = np.array([
    [0, 1, 4],
    [1, 2, 4],
    [2, 3, 4],
    [3, 0, 4],
], dtype=np.int32)

try:
    domain = mesh.create_mesh(MPI.COMM_WORLD, cells, points, mesh.CellType.triangle)
except Exception as e:
    print("Failed to create FEniCSx mesh:", e)

vtk_cells = np.hstack([
    np.full((len(cells), 1), 3, dtype=np.int32), 
    cells
]).flatten()

# VTK cell type for triangle is 5
cell_types = np.full(len(cells), 5, dtype=np.uint8)

try:
    grid = pyvista.UnstructuredGrid(vtk_cells, cell_types, points.astype(np.float32))
    grid.plot(show_edges=True, color="lightblue", line_width=5)
except Exception as e:
    print("PyVista grid creation failed:", e)
