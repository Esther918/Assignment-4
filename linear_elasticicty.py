# Linear elasticity beam
import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
from mpi4py import MPI
from dolfinx import fem, io, geometry, mesh, plot
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.mesh import create_rectangle, CellType, locate_entities
import matplotlib.pyplot as plt
import pyvista

# Mesh paremeters
length, height = 50.0, 1.0
Nx, Ny = 200, 4
# Create mesh
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

# Function space
dim = domain.topology.dim
# print(f"Mesh topology dimension d={dim}.")
degree = 2
V = fem.functionspace(domain, ("P", degree, (2,)))
u_sol = fem.Function(V, name="Displacement")

# Material parameters
E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

# Kinematics
def epsilon(v):
    return sym(grad(v))

def sigma(v):
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)

# Body force
rho = 7.85e-6 
g = 9810
f = fem.Constant(domain, np.array([0.0, -rho * g]))  

# Variational formulation
u = TrialFunction(V)
v = TestFunction(V)
dx = Measure("dx", domain=domain)
a = inner(sigma(u), epsilon(v)) * dx
L = inner(f, v) * dx

# Boundary conditions
def left(x):
    return np.isclose(x[0], 0)
left_dofs = fem.locate_dofs_geometrical(V, left)
bcs = [fem.dirichletbc(np.zeros(2), left_dofs, V)]
# Free at right end

# Solve
problem = fem.petsc.LinearProblem(
    a, L, u=u_sol, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve()

# # Visualization
# pyvista.start_xvfb()
# plotter = pyvista.Plotter(off_screen=True)
# topology, cells, geometry = plot.vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(topology, cells, geometry)
# # Add displacement to the grid
# u_values = np.zeros((geometry.shape[0], 3))
# u_values[:, :2] = u_sol.x.array.reshape(-1, 2)
# grid["u"] = u_values
# grid.set_active_vectors("u")

# # Displacement
# warped = grid.warp_by_vector("u", factor=1e5) 
# plotter.add_mesh(warped, show_edges=True)
# plotter.view_xy()
# plotter.screenshot("beam_deflection.png", transparent_background=False)
# plotter.close()  
# print("Image saved")

# Analytical solution
q = -rho * g * height
I = height**3 / 12   
D_analytical = float(q * length**4 / (8 * E * I))

# Numerical solution
def get_tip_displacement(u_sol, domain, tip_coords):
    tip_point = np.array([tip_coords[0], tip_coords[1], 0.0])
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, tip_point.reshape(1, -1))
    colliding_cells = compute_colliding_cells(domain, cell_candidates, tip_point.reshape(1, -1))
    if len(colliding_cells.links(0)) > 0:
        cell = colliding_cells.links(0)[0]
        u_tip = u_sol.eval(tip_point, [cell])
        if hasattr(u_tip, 'shape'):
            if u_tip.shape == (1, 2):
                return float(u_tip[0, 1]) 
            elif u_tip.shape == (2,):
                return float(u_tip[1])
        return float(u_tip) 
    all_points = domain.geometry.x[:, :2]
    distances = np.linalg.norm(all_points - tip_point[:2], axis=1)
    nearest_idx = np.argmin(distances)
    return u_sol.x.array.reshape(-1, 2)[nearest_idx, 1]

u_tip_numerical = get_tip_displacement(u_sol, domain, [length, height/2])

E_effective = E_effective = E / ( 1 - nu ** 2 )
delta_th_corrected = float( q * length**4 / ( 8 * E_effective * I ) )

print(f"\nComparison:")
print(f"Analytical tip displacement: {D_analytical:.6e}")
print(f"Numerical tip displacement: {u_tip_numerical:.6e}")
print(f"Relative error: {abs((u_tip_numerical - D_analytical)/D_analytical):.2%}")

# # Plot displacement comparison
# plt.figure(figsize=(10, 6))
# plt.plot(x_vals, [u_y_analytical(x) for x in x_vals], 
#          label="Analytical", linewidth=2)
# plt.plot(x_vals, u_y_numerical, '--', label="Numerical", linewidth=2)
# plt.xlabel("Beam length (mm)", fontsize=12)
# plt.ylabel("Vertical displacement (mm)", fontsize=12)
# plt.title("Beam Deflection under Body Force", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("body_force_tip_displacement_comparison.png", dpi=300)
# plt.show()
