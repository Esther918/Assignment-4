import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
from mpi4py import MPI
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType
import matplotlib.pyplot as plt

# Example from Part A
# Problem parameters
length, height = 10.0, 1.0
E_val = 210e3
nu_val = 0.3
rho = 2e-3
g = 9.81

# Analytical solution
def u_y_analytical(x):
    return 4.25e-5 * x**2 * (x - length)**2

# Function to solve the elasticity problem for a given mesh and degree
def solve_elasticity(Nx, Ny, degree):
    # Mesh
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([length, height])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )

    # Function space
    V = fem.functionspace(domain, ("P", degree, (2,)))

    # Material parameters
    E = fem.Constant(domain, E_val)
    nu = fem.Constant(domain, nu_val)
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)

    # Kinematics
    def epsilon(v):
        return sym(grad(v))

    def sigma(v):
        return lmbda * tr(epsilon(v)) * Identity(2) + 2 * mu * epsilon(v)

    # Body force
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
    u_sol = fem.Function(V, name="Displacement")
    problem = fem.petsc.LinearProblem(
        a, L, u=u_sol, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()

    # Compute L2 error
    x_vals = np.linspace(0, length, 100)
    u_y_numerical = []
    V_sub, _ = V.sub(1).collapse()
    u_y = fem.Function(V_sub)
    for x in x_vals:
        point = np.array([[x, height / 2]])
        try:
            u_y.x.array[:] = u_sol.sub(1).eval(point, domain)[0]
            u_y_numerical.append(u_y.x.array[0])
        except:
            u_y_numerical.append(np.nan)
    # u_y_numerical = np.nan_to_num(u_y_numerical, nan=0.0)
    u_y_num = np.array(u_y_numerical)
    u_y_ana = u_y_analytical(x_vals)
    error_L2 = np.sqrt(np.trapz((u_y_num - u_y_ana)**2, x_vals)) / np.sqrt(np.trapz(u_y_ana**2, x_vals))

    # Compute element size h (approximate)
    h_x = length / Nx
    h_y = height / Ny
    h = np.sqrt(h_x**2 + h_y**2)

    return error_L2, h, u_sol, domain

# h-refinement study
h_refinement = {
    "Nx": [25, 50, 100, 200],
    "Ny": [3, 5, 10, 20],
    "degree": 2
}
errors_h = []
h_vals = []
for Nx, Ny in zip(h_refinement["Nx"], h_refinement["Ny"]):
    error, h, u_sol, domain = solve_elasticity(Nx, Ny, h_refinement["degree"])
    errors_h.append(error)
    h_vals.append(h)
    # Save solution for the finest mesh
    if Nx == 200:
        with io.VTKFile(domain.comm, f"elasticity_h_Nx{Nx}_p{h_refinement['degree']}.pvd", "w") as vtk:
            vtk.write_function(u_sol)

# p-refinement study
p_refinement = {
    "Nx": 50,
    "Ny": 5,
    "degrees": [1, 2, 3, 4]
}
errors_p = []
p_vals = []
for degree in p_refinement["degrees"]:
    error, h, u_sol, domain = solve_elasticity(p_refinement["Nx"], p_refinement["Ny"], degree)
    errors_p.append(error)
    p_vals.append(degree)
    # Save solution for the highest degree
    if degree == 4:
        with io.VTKFile(domain.comm, f"elasticity_p{degree}_Nx{p_refinement['Nx']}.pvd", "w") as vtk:
            vtk.write_function(u_sol)

# Plot convergence
plt.figure(figsize=(10, 6))

# h-refinement plot
plt.subplot(1, 2, 1)
plt.loglog(h_vals, errors_h, 'o-', label=f'p={h_refinement["degree"]}')
# Reference slope for O(h^2)
h_ref = np.array(h_vals)
error_ref = errors_h[0] * (h_ref / h_vals[0])**2
plt.loglog(h_ref, error_ref, '--', label='O(h^2)')
plt.xlabel('Element size h')
plt.ylabel('Relative L2 error')
plt.title('h-refinement')
plt.legend()
plt.grid(True)

# p-refinement plot
plt.subplot(1, 2, 2)
plt.semilogy(p_vals, errors_p, 'o-', label=f'Nx={p_refinement["Nx"]}, Ny={p_refinement["Ny"]}')
plt.xlabel('Polynomial degree p')
plt.ylabel('Relative L2 error')
plt.title('p-refinement')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("convergence_study.png")
plt.close()

# Print results
print("h-refinement results:")
for Nx, Ny, h, error in zip(h_refinement["Nx"], h_refinement["Ny"], h_vals, errors_h):
    print(f"Nx={Nx}, Ny={Ny}, h={h:.4e}, L2 error={error:.4e}")

print("\np-refinement results:")
for p, error in zip(p_refinement["degrees"], errors_p):
    print(f"p={p}, L2 error={error:.4e}")