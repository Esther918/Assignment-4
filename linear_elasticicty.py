# Linear elasticity beam
import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
from mpi4py import MPI
from dolfinx import fem, io, geometry
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Mesh
length, height = 10.0, 1.0
Nx, Ny = 50, 5
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)
dim = domain.topology.dim
# print(f"Mesh topology dimension d={dim}.")

# Function space
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
print("mu (UFL):\n", mu)
print("epsilon (UFL):\n", epsilon(u_sol))
print("sigma (UFL):\n", sigma(u_sol))
# Body force
rho = 2e-3
g = 9.81
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

def right(x):
    return np.isclose(x[0], length)

left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)
bcs = [
    fem.dirichletbc(np.zeros(2), left_dofs, V),
    fem.dirichletbc(np.zeros(2), right_dofs, V),
]

# Solve
problem = fem.petsc.LinearProblem(
    a, L, u=u_sol, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve()

# vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
# vtk.write_function(u_sol)
# vtk.close()
with io.VTKFile(domain.comm, "linear_elasticity_fixed.pvd", "w") as vtk:
    vtk.write_function(u_sol)

# Compare with analytical solution
x_vals = np.linspace(0, length, 100)
u_y_analytical = lambda x: 4.25e-5 * x**2 * (x - length)**2
u_y_numerical = []

# Evaluate u_y at height/2
V_sub, _ = V.sub(1).collapse()
u_y = fem.Function(V_sub)
for x in x_vals:
    point = np.array([[x, height / 2]])
    try:
        u_y.x.array[:] = u_sol.sub(1).eval(point, domain)[0]
        u_y_numerical.append(u_y.x.array[0])
    except:
        u_y_numerical.append(np.nan)

u_y_numerical = np.nan_to_num(u_y_numerical, nan=0.0)

# Compute L2 error
u_y_num = np.array(u_y_numerical)
# print(u_y_num)
u_y_ana = u_y_analytical(x_vals)
# print(u_y_ana)
error_L2 = np.sqrt(np.trapz((u_y_num - u_y_ana)**2, x_vals)) / np.sqrt(np.trapz(u_y_ana**2, x_vals))
print(f"Relative L2 error: {error_L2:.6e}")

plt.plot(x_vals, u_y_ana, label="Analytical")
plt.plot(x_vals, u_y_num, '--', label="Numerical")
plt.xlabel("x")
plt.ylabel("u_y")
plt.legend()
plt.savefig("displacement_comparison.png")
plt.close()