import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.mesh import create_rectangle, CellType
import matplotlib.pyplot as plt

# Example from part A
# Analytical solution
length, height = 50.0, 1.0 
rho = 7.85e-6  
g = 9810 
E_val = 210e3
nu_val = 0.3
E_effective = E_val / (1 - nu_val**2)
q = -rho * g * height  
I = height**3 / 12   
D_analytical = q * length**4 / (8 * E_effective * I) 

def solve_beam_problem(Nx, Ny, degree=2):
    # Create mesh
    domain = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([length, height])],
        [Nx, Ny],
        cell_type=CellType.quadrilateral,
    )
    # Function space
    V = fem.functionspace(domain, ("P", degree, (2,)))
    u_sol = fem.Function(V, name="Displacement")
    # Material parameters
    E = fem.Constant(domain, E_val)
    nu = fem.Constant(domain, nu_val)
    mu = fem.Constant(domain, E_val / (2 * (1 + nu_val)))
    lmbda = fem.Constant(domain, E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val)))
    
    # Kinematics
    def epsilon(v):
        return sym(grad(v))
    def sigma(v):
        return lmbda * tr(epsilon(v)) * Identity(2) + 2 * mu * epsilon(v)
    
    # Variational formulation
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx", domain=domain)
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(fem.Constant(domain, np.array([0.0, q])), v) * dx
    
    # Boundary conditions
    def left(x):
        return np.isclose(x[0], 0)
    left_dofs = fem.locate_dofs_geometrical(V, left)
    bcs = [fem.dirichletbc(np.zeros(2), left_dofs, V)]
    # Free at right end
    
    # Solve
    problem = fem.petsc.LinearProblem(
        a, L, u=u_sol, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    problem.solve()
    
    # Evaluate tip displacement
    def get_tip_displacement():
        tip_point = np.array([length, height/2, 0.0])
        tree = bb_tree(domain, domain.topology.dim)
        cell_candidates = compute_collisions_points(tree, tip_point.reshape(1, -1))
        colliding_cells = compute_colliding_cells(domain, cell_candidates, tip_point.reshape(1, -1))
        
        if len(colliding_cells.links(0)) > 0:
            cell = colliding_cells.links(0)[0]
            u_tip = u_sol.eval(tip_point, [cell])
            if hasattr(u_tip, 'shape'):
                return float(u_tip[0,1] if u_tip.shape == (1,2) else u_tip[1])
            return float(u_tip)
        
        points = domain.geometry.x[:, :2]
        distances = np.linalg.norm(points - tip_point[:2], axis=1)
        return u_sol.x.array.reshape(-1, 2)[np.argmin(distances), 1]
    
    return get_tip_displacement(), domain, V

# h-refinement study
h_refinement_levels = [(4, 4), (8, 4), (16, 8), (32, 8), (64, 16)]
mesh_sizes = [length / h[0] for h in h_refinement_levels]  
errors_h = []

print("\nh-refinement Study")
for h_level in h_refinement_levels:
    Nx, Ny = h_level
    max_disp, _, _ = solve_beam_problem(Nx, Ny, 2)
    error = abs(max_disp - D_analytical)
    errors_h.append(error)
    print(f"Nx={Nx}, Ny={Ny}: Displacement = {max_disp:.6e}, Error = {error:.6e}")

# Log-log plot for h-refinement
plt.figure(figsize=(10, 6))
plt.loglog(mesh_sizes, errors_h, marker='o', label="Error")
plt.xlabel("Element size", fontsize=14)
plt.ylabel("Absolute error in tip displacement", fontsize=14)
plt.title("h-refinement Convergence", fontsize=16)
plt.grid(True, which='both')
plt.legend()
plt.savefig("h_refinement_error.png", dpi=300)
plt.show()

# p-refinement study
fixed_degree = 2 
h_refinement_levels = [(4, 4), (8, 4), (16, 8), (32, 8), (64, 16)]  
mesh_sizes = [length / h[0] for h in h_refinement_levels]  
errors_p = []
max_disp_p = []
print("\np-refinement Study")
for h_level in h_refinement_levels:
    Nx, Ny = h_level
    max_disp, _, _ = solve_beam_problem(Nx, Ny, fixed_degree)
    error = abs(max_disp - D_analytical)
    errors_p.append(error)
    max_disp_p.append(max_disp)
    print(f"Nx={Nx}, Ny={Ny}: Displacement = {max_disp:.6e}, Error = {error:.6e}")

plt.figure(figsize=(10, 6))
plt.loglog(mesh_sizes, errors_p, marker='s', markersize=8, linewidth=2.5, label="Error")
for i, (h, disp, err) in enumerate(zip(mesh_sizes, max_disp_p, errors_p)):
    plt.annotate(f"Disp: {disp:.6f}\nErr: {err:.6e}", 
                 (h, err), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add table with numerical results
table_data = [["Nx", "Ny", "Displacement", "Error"]] + [
    [h_refinement_levels[i][0], h_refinement_levels[i][1], f"{disp:.6f}", f"{err:.6e}"] 
    for i, (disp, err) in enumerate(zip(max_disp_p, errors_p))
]
table = plt.table(cellText=table_data, 
                  loc='upper right', 
                  cellLoc='center', 
                  bbox=[0.65, 0.45, 0.33, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.xlabel("Element size", fontsize=14)
plt.ylabel("Absolute error in tip displacement", fontsize=14)
plt.title("p-refinement Convergence", fontsize=16)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("p_refinement_error.png", dpi=300, bbox_inches='tight')
plt.show()