####################################################################
# Temporal adaptivity for the exponential growth ODE               #
# with backward Euler in time (dG(0)) for primal  problem          #
# and dG(1) discretization for dual problem                        #
#                                                                  #
# Author: Julian Roth                                              #
# Year:   2023                                                     #
####################################################################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
import sys

class TemporalMesh:
    def __init__(self, t0, T, Δt):
        self.t0 = t0
        self.T = T
        self.Δt = Δt
        self.n_elements = int((T - t0) / Δt)
        self.create_mesh()

    def create_mesh(self):
        self.mesh = [np.array([self.t0, self.t0 + self.Δt])]
        for i in range(1, self.n_elements):
            self.mesh.append(self.mesh[i - 1] + self.Δt)
    
    def plot_mesh(self, title="Temporal Mesh", savepath=None):
        plt.clf()
        plt.title(title)
        plt.plot([self.mesh[0][0], self.mesh[-1][1]], [0., 0.], color="black")
        plt.plot([self.t0, self.t0], [-0.1, 0.1], color="black")
        for i in range(len(self.mesh)):
            plt.plot([self.mesh[i][1], self.mesh[i][1]], [-0.1, 0.1], color="black")
        # remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        # set x and y limits
        plt.xlim([self.t0-self.Δt, self.T+self.Δt])
        plt.ylim([-1., 1.])
        # remove all spines
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, bbox_inches='tight')

    def refine(self, refine_flags=None):
        if refine_flags is None:
            refine_flags = [True for i in range(self.n_elements)]

        new_mesh = []
        for flag, element in zip(refine_flags, self.mesh):
            if flag:
                midpoint = (element[0] + element[1]) / 2.
                new_mesh.append(np.array([element[0], midpoint]))
                new_mesh.append(np.array([midpoint, element[1]]))
            else:
                new_mesh.append(element)
        self.mesh = new_mesh
        self.n_elements = len(self.mesh)

    def print(self):
        for i, element in enumerate(self.mesh):
            print(f"I_{i} = ({element[0]}, {element[1]})")

def solve_primal(mesh):
    # solve primal problem
    u = [1.] # u_0 = 1
    for element in mesh:
        # dG(0) approximation of ∂_t u - u = 0:
        #      (1 - Δt) * u_n = u_{n-1}
        # <=>  u_n = u_{n-1} / (1 - Δt)
        Δt = element[1] - element[0]
        u.append(u[-1] / (1. - Δt))
    return u

def compute_goal_functional(mesh, primal_solutions, goal_functional):
    # compute goal functional
    J = 0.
    if goal_functional == "end_time":
        # J(u) = u(T)
        J += primal_solutions[-1]
    elif goal_functional == "time_integral":
        # J(u) = ∫_0^T u(t) dt
        for i, element in enumerate(mesh):
            Δt = element[1] - element[0]
            J += Δt * primal_solutions[i+1]
    return J

def solve_dual(mesh, primal_solutions, goal_functional):
    # solve dual problem
    z = []
    if goal_functional == "end_time":
        # z(T) = 1
        z.append(np.array([1., 1.]))
    elif goal_functional == "time_integral":
        # z(T) = 0
        z.append(np.array([0., 0.]))

    for element in reversed(mesh):
        Δt = element[1] - element[0]
        # dG(1) approximation of -∂_t z - z = 0:
        # [(0.5, -0.5)   (-k/3, -k/6)] (z_m(t_{m-1}))    (0)
        # [(0.5,  0.5) + (-k/6, -k/3)] (z_m(t_m))      = (z_{m+1}(t_m))

        system_matrix = np.array(
            [
                [0.5, -0.5], 
                [0.5, 0.5]
            ]
        ) - Δt / 6. * np.array(
            [
                [2., 1.],
                [1., 2.]
            ]
        )
        rhs = np.array(
            [
                0.,
                z[-1][0]
            ]
        )
        if goal_functional == "time_integral":
            # dG(0) approximation of -∂_t z - z = 1:
            # [(0.5, -0.5)   (-k/3, -k/6)] (z_m(t_{m-1}))    (0)              (k/2)
            # [(0.5,  0.5) + (-k/6, -k/3)] (z_m(t_m))      = (z_{m+1}(t_m)) + (k/2)
            rhs += Δt / 2.
        z.append(np.linalg.solve(system_matrix, rhs))   

    return z[::-1]

def plot_solutions(mesh, primal_solutions, dual_solutions, goal_functional_type):
    # plot primal and dual solutions
    plt.clf()
    plt.title("Primal and dual solutions")
    # plot temporal mesh
    plt.plot([mesh.mesh[0][0], mesh.mesh[-1][1]], [0., 0.], color="black")
    plt.plot([mesh.t0, mesh.t0], [-0.1, 0.1], color="black")
    for i in range(len(mesh.mesh)):
        plt.plot([mesh.mesh[i][1], mesh.mesh[i][1]], [-0.1, 0.1], color="black")
    
    # plot primal solutions
    X = []
    for element in mesh.mesh:
        X.append(element[0])
        X.append(element[1])
        X.append(element[1])
    Y = []
    for i in range(len(mesh.mesh)):
        Y.append(primal_solutions[i+1])
        Y.append(primal_solutions[i+1])
        Y.append(np.inf)
    plt.plot(X, Y, color="blue", label="primal (FE)")
    X_true = np.linspace(mesh.t0, mesh.T, 1000)
    Y_true = np.exp(X_true)
    plt.plot(X_true, Y_true, color="blue", linestyle="dashed", label="primal (true)")

    # plot dual solutions
    Y = []
    for i in range(len(mesh.mesh)):
        Y.append(dual_solutions[i][0])
        Y.append(dual_solutions[i][1])
        Y.append(np.inf)
    plt.plot(X, Y, color="red", label="dual (FE)")
    if goal_functional_type == "end_time":
        Y_true = np.exp(-X_true+1.)
    elif goal_functional_type == "time_integral":
        Y_true = np.exp(-X_true+1.) - 1.
    plt.plot(X_true, Y_true, color="red", linestyle="dashed", label="dual (true)")
    plt.legend()
    plt.xlim([mesh.t0-mesh.Δt, mesh.T+mesh.Δt])
    plt.ylim([-1., 4.])
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()

def compute_error_estimator(mesh, primal_solutions, dual_solutions):
    # compute error estimator for dG(0) primal and dG(1) dual solutions
    values = np.zeros(mesh.n_elements)

    for i, element in enumerate(mesh.mesh):
        Δt = element[1] - element[0]
        u = primal_solutions[i+1]
        u_n = primal_solutions[i]

        Z = dual_solutions[i]
    
        # z_k: dG(1) solution on the temporal element I_m = (t_{m-1}, t_m)
        def z_fine(t):
            assert t >= element[0] and t <= element[1]
            return Z[0] + (Z[1] - Z[0]) * (t - element[0]) / Δt
            
        # i_k z_k: constant interpolation of dG(1) solution on the temporal element I_m = (t_{m-1}, t_m)
        def z_coarse(t):
            assert t >= element[0] and t <= element[1]
            return Z[0]
        
        # primal residual based error estimator:
        #   J(u) - J(u_k) ≈ η := ρ(u_k)(z_k - i_k z_k)
        #      ∘ η: error estimator
        #      ∘ ρ: residual of the space-time formulation of the primal problem
        #      ∘ u_k: low-order in time primal solution (here: dG(0) / Backward Euler; in general: dG(r))
        #      ∘ z_k: high-order in time dual solution (here: dG(1); in general: dG(r+1))
        #      ∘ i_k z_k: interpolated low-order in time dual solution (here: dG(0); in general: dG(r))

        # jump term: -(u_k^{m} - u_k^{m-1}) * (I_k z_k - z_k)^+_{m-1}
        values[i] -= (u-u_n) * (z_fine(element[0]) - z_coarse(element[0]))

        # mass term: (u_k^m, I_k z_k - z_k)           [trapezoidal rule for temporal integral]
        for w_q, t_q in zip([Δt / 2., Δt / 2.], [element[0], element[1]]): 
            values[i] += w_q * u * (z_fine(t_q) - z_coarse(t_q))

    # copy error estimator on 1st temporal element to 0th temporal element
    values[0] = values[1]
    return values

def compute_analytical_error_estimator(mesh, primal_solutions, goal_functional):
    from scipy.integrate import quad
    
    # compute error estimator for dG(0) primal and analytical dual solutions
    values = np.zeros(mesh.n_elements)

    z = lambda t: 0.
    if goal_functional == "end_time":
        z = lambda t: np.exp(-t+1.)
    elif goal_functional == "time_integral":
        z = lambda t: np.exp(-t+1.) - 1.

    for i, element in enumerate(mesh.mesh):
        Δt = element[1] - element[0]
        u = primal_solutions[i+1]
        u_n = primal_solutions[i]
        
        # primal residual based error estimator:
        #   J(u) - J(u_k) ≈ η := ρ(u_k)(z)
        #      ∘ η: error estimator
        #      ∘ ρ: residual of the space-time formulation of the primal problem
        #      ∘ u_k: low-order in time primal solution (here: dG(0) / Backward Euler; in general: dG(r))
        #      ∘ z: analytical dual solution

        # jump term: - (u_k^{m} - u_k^{m-1}) * z^+_{m-1}
        jump = -(u-u_n) * z(element[0])

        # mass term: (u_k^m, z)           [Simpson's rule for temporal integral]
        # integral = 0.
        # for w_q, t_q in zip([Δt / 6., 2. * Δt / 3., Δt / 6.], [element[0], (element[0] + element[1]) / 2., element[1]]): 
        #     integral += w_q * u * z(t_q)

        # integrate (u_k^m, z) = u_k^m * (1, z) using scipy.integrate.quad
        integral = u * quad(z, element[0], element[1])[0]

        # for debugging (end time goal functional with uniform refinement):
        # jump = -np.power(1. / (1. - Δt), i) * (Δt / (1. - Δt)) * np.exp(1. - i * Δt)
        # integral = np.power(1. / (1. - Δt), i+1) * (np.exp(1. - i * Δt) - np.exp(1. - (i+1) * Δt))

        values[i] = integral + jump

    return values

if __name__ == "__main__":
    # get refinement type from cli
    if len(sys.argv) != 4 or sys.argv[1] not in ["uniform", "adaptive"] or sys.argv[2] not in ["end_time", "time_integral"]:
        print("Usage: python3 main.py <refinement_type> <goal_functional> <analytical_dual>")
        print("  refinement_type: uniform, adaptive")
        print("  goal_functional: end_time, time_integral")
        print("  analytical_dual: True, False")
        quit()
    refinement_type = sys.argv[1]
    goal_functional_type = sys.argv[2]
    ANALYTICAL_DUAL = (sys.argv[3] == "True")
    print(f"Refinement type: {refinement_type}")
    print(f"Goal functional: {goal_functional_type}")
    print(f"Analytical dual: {ANALYTICAL_DUAL}")

    # hyperparameters
    ERROR_TOL = 1e-14 # stopping criterion for DWR loop
    MAX_DWR_ITERATIONS = 1
    if refinement_type == "uniform":
        MAX_DWR_ITERATIONS = 19
    elif refinement_type == "adaptive":
        MAX_DWR_ITERATIONS = 43
    PLOT_ESTIMATOR = False #True
    PLOT_SOLUTIONS = False #True
    mesh = TemporalMesh(
        t0 = 0.0, # start time
        T = 1.0,  # end time
        Δt = 0.1  # time step size
    )

    convergence_table = {}

    for iteration_dwr in range(1, MAX_DWR_ITERATIONS+1):
        print(f"\nDWR ITERATION {iteration_dwr}:")
        print("================\n")

        print("Solve primal problem:")
        primal_solutions = solve_primal(mesh.mesh)

        print("Compute goal functional:")
        goal_functional = compute_goal_functional(mesh.mesh, primal_solutions, goal_functional_type)
        J_reference = np.nan
        if goal_functional_type == "end_time":
            J_reference = np.exp(1.)
        elif goal_functional_type == "time_integral":
            J_reference = np.exp(1.) - 1.
        true_error = J_reference - goal_functional
        print(f"  DoFs (n_k):    {mesh.n_elements}")
        print(f"  J(u):          {J_reference:.8e}")
        print(f"  J(u_k):        {goal_functional:.8e}")
        print(f"  J(u) - J(u_k): {true_error:.8e}")

        print("Solve dual problem:")
        dual_solutions = solve_dual(mesh.mesh, primal_solutions, goal_functional_type)

        if PLOT_SOLUTIONS:
            plot_solutions(mesh, primal_solutions, dual_solutions, goal_functional_type)

        print("Compute error estimator:")
        error_estimator = None
        if not ANALYTICAL_DUAL:
            # use dG(0) dual solution for error estimator
            error_estimator = compute_error_estimator(mesh, primal_solutions, dual_solutions)
        else:
            # use analytical dual solution for error estimator to check correctness
            error_estimator = compute_analytical_error_estimator(mesh, primal_solutions, goal_functional_type)

        # get the start and end time values for each temporal element
        start_times = np.array([mesh.mesh[i][0] for i in range(mesh.n_elements)])
        end_times = np.array([mesh.mesh[i][1] for i in range(mesh.n_elements)])

        if PLOT_ESTIMATOR:
            # plot the error estimator in a bar chart
            plt.bar(0.5*(start_times+end_times), error_estimator, width=end_times-start_times, align='center', edgecolor='black')
            plt.title("Error estimator")
            plt.xlabel("Temporal element midpoint")
            plt.ylabel("Error estimate")
            plt.show()

        estimated_error = np.sum(error_estimator)
        effectivity_index = true_error / estimated_error
        convergence_table[mesh.n_elements] = {
            "J(u_k)": goal_functional, 
            "J(u) - J(u_k)": true_error, 
            "η_k": estimated_error,
            "effectivity index": effectivity_index
        }
        print(f"  η_k: {estimated_error}")
        print(f"  effectivity index: {effectivity_index:.8e}")

        if refinement_type == "uniform":
            # uniform refinement in time
            mesh.refine()
        elif refinement_type == "adaptive":
            # TODO: use relative stopping criterion
            if np.abs(np.sum(error_estimator)) > ERROR_TOL:
                print("Mark temporal elements for refinement")
                # create a list of boolean values which indicate whether a temporal element should be refined or not
                # mark the cells responsible for 50 % of the total error for refinement
                total_abs_error = np.sum(np.abs(error_estimator))
                
                # sort the absolute error estimator in descending order including the index
                sorted_indices = np.argsort(np.abs(error_estimator))[::-1]
                sorted_abs_error = np.abs(error_estimator[sorted_indices])
                
                # get temporal elements which are responsible for 50 % of the total error
                refine_flags = [False for i in range(mesh.n_elements)]
                sum_abs_error = 0.
                for i in range(mesh.n_elements):
                    sum_abs_error += sorted_abs_error[i]
                    refine_flags[sorted_indices[i]] = True
                    if sum_abs_error >= 0.5 * total_abs_error:
                        break

                if PLOT_ESTIMATOR:
                    # plot the error estimator in a bar chart and use a different color for the elements that should be refined
                    plt.bar(0.5*(start_times+end_times), error_estimator, width=end_times-start_times, align='center', edgecolor='black', color=np.array(["blue" if flag else "orange" for flag in refine_flags]))
                    plt.title("Error estimator")
                    plt.xlabel("Temporal element midpoint")
                    plt.ylabel("Error estimate")
                    plt.show()

                print("Refine temporal mesh")
                mesh.refine(refine_flags=refine_flags)
            else:
                print(f"Temporal adaptivity finished! (estimated error = {np.abs(np.sum(error_estimator))} < {ERROR_TOL})")
                break


    # print convergence table as tabulate
    for tablefmt in ["simple", "latex"]:
        table = tabulate([[row[0], *row[1].values()] for row in convergence_table.items()], headers=["#DoFs", "$J(u_k)$", "$J(u) - J(u_k)$", "$\eta_k$", "$I_{eff}$"], tablefmt=tablefmt)
        print(table)
    
    # convert python scientific to latex scientific notation
    def science_format(num):
        num = num.split("e")
        return fr"${num[0]} \cdot 10^{{{int(num[1])}}}$"
    
    print("\nConvergence table:")
    print("==================\n")
    print("DoFs (n_k) | J(u_k) | J(u) - J(u_k) | η_k | effectivity index")
    print("-------------------------------------------------------------")
    for key, value in convergence_table.items():
        _true_error = science_format(f"{value['J(u) - J(u_k)']:.4e}")
        _est_error = science_format(f"{value['η_k']:.4e}")
        print(f"    {key:,} & {value['J(u_k)']:.8f} & {_true_error} & {_est_error} & {value['effectivity index']:.6f} \\\\")


    # plot convergence table
    plt.clf()
    plt.title("Convergence table")
    plt.xlabel("DoFs")
    plt.ylabel("True error")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(list(convergence_table.keys()), [abs(value["J(u) - J(u_k)"]) for value in convergence_table.values()], color="blue", marker="o")
    plt.show()

    # plot effectivity index
    plt.clf()
    plt.title("Effectivity index")
    plt.xlabel("DoFs")
    plt.ylabel("Effectivity index")
    plt.xscale("log")
    plt.plot(list(convergence_table.keys()), [value["effectivity index"] for value in convergence_table.values()], color="blue", marker="o")
    plt.show()

    print("\nConvergence plot:")
    for dof in convergence_table.keys():
        print(f"({dof},{abs(convergence_table[dof]['J(u) - J(u_k)'])})", end="")
    print("\n")

    print("Effectivity index:")
    for dof in convergence_table.keys():
        print(f"({dof},{convergence_table[dof]['effectivity index']})", end="")
    print("")
    
