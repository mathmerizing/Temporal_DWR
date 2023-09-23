####################################################################
# Temporal adaptivity for the exponential growth ODE               #
# with backward Euler in time (dG(0)) for primal and dual problem  #
#                                                                  #
# Author: Julian Roth                                              #
# Year:   2023                                                     #
####################################################################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        #      (1 - Δt) * u_n = u_{n-1
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
        z.append(1.)
    elif goal_functional == "time_integral":
        # z(T) = 0
        z.append(0.)
    for element in reversed(mesh):
        # dG(0) approximation of ∂_t z + z = 0:
        #      z_n = (1 + Δt) z_{n+1} 
        # NOTE: for goal functional J(u) = ∫_0^T u(t) dt, we have to add Δt to z_n since we have a RHS in the dual problem
        Δt = element[1] - element[0]
        z.append(z[-1] * (1. + Δt))
        if goal_functional == "time_integral":
            z[-1] += Δt
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
        Y.append(dual_solutions[i])
        Y.append(dual_solutions[i])
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

if __name__ == "__main__":
    # get refinement type from cli
    if len(sys.argv) != 3 or sys.argv[1] not in ["uniform", "adaptive"] or sys.argv[2] not in ["end_time", "time_integral"]:
        print("Usage: python3 main.py <refinement_type> <goal_functional>")
        print("  refinement_type: uniform, adaptive")
        print("  goal_functional: end_time, time_integral")
        quit()
    refinement_type = sys.argv[1]
    goal_functional_type = sys.argv[2]
    print(f"Refinement type: {refinement_type}")
    print(f"Goal functional: {goal_functional_type}")

    # hyperparameters
    ERROR_TOL = 1e-14 # stopping criterion for DWR loop
    MAX_DWR_ITERATIONS = 5 #10 #15 # 25
    PLOT_ESTIMATOR = True #False
    PLOT_SOLUTIONS = True #False
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
        print(f"  n_k:           {mesh.n_elements}")
        print(f"  J(u_k):        {goal_functional:.8e}")
        print(f"  J(u) - J(u_k): {true_error:.8e}")

        print("Solve dual problem:")
        dual_solutions = solve_dual(mesh.mesh, primal_solutions, goal_functional_type)

        if PLOT_SOLUTIONS:
            plot_solutions(mesh, primal_solutions, dual_solutions, goal_functional_type)

        print("Compute error estimator:")
        
        # # TODO

        # uniform refinement
        mesh.refine()
