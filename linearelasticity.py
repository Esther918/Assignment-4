import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
'''
This script solves a linear elasticity problem for a 3D beam clamped at one end and subjected to a body force, which is a gravity-like load in this test.
It computes the displacement field and von Mises stress distribution, then visualizes the results using PyVista.
'''

# Pysical parameters
'''
Length of the beam: L = 1 [m]
Width of the beam: W = 0.2 [m]
shear modulus: mu = 1 [Pa]
Density: rho = 1 [kg/m^3]
Aspect ratio: delta = W / L 
Scaling factor: gamma = 0.4 * delta**2
Parameter controlling lambda: beta = 1.25 
lambda_ = beta [Pa]
Effective gravity: g = gamma [m/s^2]
'''
L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Mesh generation
"""
Create a 3D box mesh of the beam:
Dimensions: L * W * W
Discretization: 20 * 6 * 6 hexahedral elements
MPI.COMM_WORLD: Parallel communicator for distributed mesh
"""
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
# Vector function space for displacement
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# Boundary conditions
"""
Identify the clamped boundary 
"""
def clamped_boundary(x):
    # x=0 plane
    return np.isclose(x[0], 0)
# Get boundary facets for application of BCs
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
# Dirichlet BC: Zero displacement on clamped boundary
u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# Traction boundary condition (zero)
T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)

# Constitutive equations
"""Compute the strain tensor"""
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
"""Compute the stress tensor using linear elasticity"""
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Variational formulation
u = ufl.TrialFunction(V) # Unknown displacement field
v = ufl.TestFunction(V) # Test function
# Body force 
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g))) # In negative z-direction
# Stiffness matrix
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
# Load factor
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

# Solve the linear system
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# Displacement solution
uh = problem.solve()

# Visualization of Displacement
pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3)) # Add displacement field to the grid
actor_0 = p.add_mesh(grid, style="wireframe", color="k") # Visualize original mesh as wireframe
# Create and visualize deformed configuration
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()

# Show the plot
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")

# Save results to file
with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)

# Stress computation and visualization
'''Compute von Mises stress'''
s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))
# Interpolate stress onto DG0 space
V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)
# Visualize stress
warped.cell_data["VonMises"] = stresses.x.petsc_vec.array
warped.set_active_scalars("VonMises")
p = pyvista.Plotter()
p.add_mesh(warped)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    stress_figure = p.screenshot(f"stresses.png")