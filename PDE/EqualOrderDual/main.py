####################################################################
# Temporal adaptivity for the heat equation                        #
# with backward Euler in time (dG(0)) for primal and dual problem  #
#                                                                  #
# Author: Julian Roth                                              #
# Year:   2023                                                     #
####################################################################
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from tabulate import tabulate
import sys

set_log_active(False) # turn off FEniCS logging

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
        
# indicator function for upper half of domain
class IndicatorUpperHalf(UserExpression):
    def eval_cell(self, value, x, _):
        if (x[1] > 0.5):
            value[0] = 1.
        else:
            value[0] = 0.

    def value_shape(self):
        return ()  # scalar function
    
# indicator function for first quadrant of domain
class IndicatorFirstQuadrant(UserExpression):
    def eval_cell(self, value, x, _):
        if (x[0] > 0.5 and x[1] > 0.5):
            value[0] = 1.
        else:
            value[0] = 0.

    def value_shape(self):
        return ()  # scalar function
    
# indicator function for third quadrant of domain
class IndicatorThirdQuadrant(UserExpression):
    def eval_cell(self, value, x, _):
        if (x[0] < 0.5 and x[1] < 0.5):
            value[0] = 1.
        else:
            value[0] = 0.

    def value_shape(self):
        return ()  # scalar function

class SpatialFE:
    def __init__(self):
        self.mesh = UnitSquareMesh(50, 50) #20, 20)
        self.V = FunctionSpace(self.mesh, 'P', 1) # linear FE in space
        self.bc = DirichletBC(self.V, Constant(0.), lambda _, on_boundary: on_boundary) # homogeneous Dirichlet BC everywhere
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.mass_form = self.u * self.v * dx
        self.laplace_form = inner(grad(self.u), grad(self.v)) * dx

        # assemble matrices and convert to scipy.sparse matrices
        self.mass_matrix =  scipy.sparse.csr_matrix(
            as_backend_type(
                assemble(self.mass_form)
            )
            .mat()
            .getValuesCSR()[::-1],
            shape=(self.V.dim(), self.V.dim()),
        )
        self.laplace_matrix =  scipy.sparse.csr_matrix(
            as_backend_type(
                assemble(self.laplace_form)
            )
            .mat()
            .getValuesCSR()[::-1],
            shape=(self.V.dim(), self.V.dim()),
        )

        self.indicator_upper_half = IndicatorUpperHalf() # indicator function for upper half of domain
        self.indicator_first_quadrant = IndicatorFirstQuadrant() # indicator function for first quadrant of domain
        self.indicator_third_quadrant = IndicatorThirdQuadrant() # indicator function for third quadrant of domain

        self.goal_functional_vector = np.array(assemble(self.v * self.indicator_upper_half * dx))
        self.first_quadrant_vector = np.array(assemble(self.v * self.indicator_first_quadrant * dx))
        self.third_quadrant_vector = np.array(assemble(self.v * self.indicator_third_quadrant * dx))

        self.boundary_dof_vector = np.zeros((self.V.dim(),))
        for i, val in self.bc.get_boundary_values().items():
            self.boundary_dof_vector[i] = 1.0

        # store matrices and its factorizations for each time step size for faster solve
        self.system_matrix = {}
        self.solve_factorized = {}

    def solve_primal(self, temporal_mesh):
        solutions = []

        # initial condition
        u_0 = np.zeros((self.V.dim(),))
        # u_n: solution from last time step
        u_n = u_0.copy()
        # solution on current time step
        u = np.zeros((self.V.dim(),))

        # store initial condition as numpy array
        solutions.append(u_n.copy())

        # for each temporal element:
        #    solve forward in time with backward Euler
        for i, temporal_element in enumerate(tqdm(temporal_mesh)):
            # print(f"Solve primal on I_{i} = ({temporal_element[0]}, {temporal_element[1]})")

            Δt = temporal_element[1] - temporal_element[0]
            for dt in self.solve_factorized.keys():
                if np.abs(dt - Δt) < 1e-8:
                    Δt = dt
                    break
            else:
                print(f"Factorize matrix for Δt = {Δt}")
                # store system matrix with enforced homogeneous Dirichlet BC
                self.system_matrix[Δt] = (
                    (
                        self.mass_matrix + Δt*self.laplace_matrix
                    ).multiply((1.0 - self.boundary_dof_vector).reshape(-1, 1)) + scipy.sparse.diags(self.boundary_dof_vector)
                ).tocsc()

                # factorize system matrix
                self.solve_factorized[Δt] = scipy.sparse.linalg.factorized(
                    self.system_matrix[Δt]
                )

            rhs_vector = self.mass_matrix.dot(u_n)
            if temporal_element[1] <= 0.5:
                rhs_vector += Δt * self.first_quadrant_vector
            elif temporal_element[0] >= 1. and temporal_element[1] <= 1.5:
                rhs_vector += Δt * self.third_quadrant_vector

            # apply homogeneous Dirichlet BC to right hand side
            rhs_vector = rhs_vector * (1.0 - self.boundary_dof_vector)

            u = self.solve_factorized[Δt](rhs_vector)

            # store solution as numpy array
            solutions.append(u.copy())

            # U = Function(self.V)
            # U.vector()[:] = u
            # c = plot(U)
            # plt.colorbar(c)
            # plot(self.mesh)
            # plt.show()

            u_n = u.copy()
            
        return solutions
    
    def solve_dual(self, temporal_mesh, primal_solutions):
        # NOTE: primal_solutions is only used for nonlinear PDEs or nonlinear goal functionals

        solutions = []

        # initial condition
        z_M = np.zeros((self.V.dim(),))
        # z_n: solution from next time step
        z_n = z_M.copy()
        # solution on current time step
        z = np.zeros((self.V.dim(),))

        # store initial condition as numpy array
        solutions.append(z_n.copy())

        # for each temporal element:
        #    solve backward in time with backward Euler
        for i, temporal_element in tqdm(list(enumerate(temporal_mesh))[::-1]):
            # print(f"Solve dual on I_{i} = ({temporal_element[0]}, {temporal_element[1]})")
            
            Δt = temporal_element[1] - temporal_element[0]
            for dt in self.solve_factorized.keys():
                if np.abs(dt - Δt) < 1e-8:
                    Δt = dt
                    break
            else:
                raise "Factorized matrix for Δt not found! (This should not happen!)"

            rhs_vector = self.mass_matrix.dot(z_n) + Δt * self.goal_functional_vector

            # apply homogeneous Dirichlet BC to right hand side
            rhs_vector = rhs_vector * (1.0 - self.boundary_dof_vector)

            z = self.solve_factorized[Δt](rhs_vector)

            # store solution as numpy array
            solutions.append(z.copy())

            # c = plot(z)
            # plt.colorbar(c)
            # plt.show()

            z_n = z.copy()
        
        return solutions[::-1] # sort solutions from t = t0 to t = T
    
    def compute_goal_functional(self, temporal_mesh, primal_solutions):
        value = 0.

        for temporal_element, solution in tqdm(list(zip(temporal_mesh, primal_solutions[1:]))):
            Δt = temporal_element[1] - temporal_element[0]
            value += Δt * np.dot(solution, self.goal_functional_vector)

        return value
    
    def compute_error_estimator(self, temporal_mesh, primal_solutions, dual_solutions):
        values = np.zeros(len(temporal_mesh))

        u = np.zeros((self.V.dim(),))   # u^{n+1}
        u_n = np.zeros((self.V.dim(),)) # u^n
        z = np.zeros((self.V.dim(),))   # z^{n+1}
        z_n = np.zeros((self.V.dim(),)) # z^n
        for i, temporal_element in enumerate(tqdm(temporal_mesh)):
            Δt = temporal_element[1] - temporal_element[0]
            u = primal_solutions[i+1]
            u_n = primal_solutions[i]

            z = dual_solutions[i]
            if i > 0:
                z_n = dual_solutions[i-1]
            else:
                z_n = dual_solutions[i]
            
            # I_k z_k: for dG(0) this is a linear interpolation between z_k(t_{m-1}) and z_k(t_m) on the temporal element I_m = (t_{m-1}, t_m)
            def z_fine(t):
                assert t >= temporal_element[0] and t <= temporal_element[1]
                return z_n + (z - z_n) * (t - temporal_element[0]) / Δt
                
            # z_k: for dG(0) this is a constant function on the temporal element I_m = (t_{m-1}, t_m)
            def z_coarse(t):
                assert t >= temporal_element[0] and t <= temporal_element[1]
                return z
                
            # primal residual based error estimator:
            #   J(u) - J(u_k) ≈ η := ρ(u_k)(I_k z_k - z_k)
            #      ∘ η: error estimator
            #      ∘ ρ: residual of the space-time formulation of the primal problem
            #      ∘ u_k, z_k: low-order in time primal/dual solution (here: dG(0) / Backward Euler; in general: dG(r))
            #      ∘ I_k: temporal reconstruction of a higher-order in time solution, i.e. I_k u_k is a dG(r+1) solution which is interpolated from the dG(r) solution u_k
            #             for dG(0): I_k u_k is a linear interpolation of (t_{m-1}, u_k(t_{m-1})) and (t_m, u_k(t_m)) on the temporal element I_m = (t_{m-1}, t_m)
            #             for dG(1): I_k u_k can be implemented as a patch-wise interpolation of u_k from to neighboring temporal elements I_{m-1} = (t_{m-2}, t_{m-1}) and I_m = (t_{m-1}, t_m) onto \tilde{I}_m = (t_{m-2}, t_m) and then interpreting this (r+1)-degree polynomial as a dG(r+1) solution on I_{m-1} and I_m
            #
            # More conretely for dG(0):
            #    I_k z_k = z_k(t_{m-1}) + (z_k(t_m) - z_k(t_{m-1})) * (t - t_{m-1}) / (t_m - t_{m-1})   for t ∈ I_m = (t_{m-1}, t_m)
            # And consequently:
            #    I_k z_k - z_k = (z_k(t_m) - z_k(t_{m-1})) * (t - t_{m-1}) / (t_m - t_{m-1})   for t ∈ I_m = (t_{m-1}, t_m)
            # For the evaluation of the primal residual, we need to exactly integrate constant functions in time => trapezoidal rule is enough for quadrature of the temporal integral
            # The time derivative of u_k is zero because u_k is a dG(0) solution.

            # assemble individual terms of residual and apply boundary conditions
            residual_jump = -self.mass_matrix.dot(u - u_n) * (1.0 - self.boundary_dof_vector)
            residual_laplace = - 0.5 * Δt * self.laplace_matrix.dot(u) * (1.0 - self.boundary_dof_vector)
            # NOTE: Here the assembly of the RHS term is easy because the RHS is element-wise constant in time
            residual_rhs = np.zeros((self.V.dim(),))
            if temporal_element[1] <= 0.5:
                residual_rhs = Δt * self.first_quadrant_vector * (1.0 - self.boundary_dof_vector)
            elif temporal_element[0] >= 1. and temporal_element[1] <= 1.5:
                residual_rhs = Δt * self.third_quadrant_vector * (1.0 - self.boundary_dof_vector)

           # multiply residual parts with dual solution
            values[i] += np.dot(residual_jump, z_fine(temporal_element[0]) - z_coarse(temporal_element[0]))
            values[i] += np.dot(residual_laplace, z_fine(temporal_element[0]) - z_coarse(temporal_element[0]))
            values[i] += np.dot(residual_laplace, z_fine(temporal_element[1]) - z_coarse(temporal_element[1]))
            values[i] += 0.5 * np.dot(residual_rhs, z_fine(temporal_element[0]) - z_coarse(temporal_element[0]))
            values[i] += 0.5 * np.dot(residual_rhs, z_fine(temporal_element[1]) - z_coarse(temporal_element[1]))

        return values

if __name__ == "__main__":
    # get refinement type from cli
    if len(sys.argv) != 2 or sys.argv[1] not in ["uniform", "adaptive"]:
        print("Usage: python3 main.py <refinement_type>")
        print("  refinement_type: uniform, adaptive")
        quit()
    refinement_type = sys.argv[1]
    print(f"Refinement type: {refinement_type}")

    # hyperparameters
    ERROR_TOL = 1e-14 # stopping criterion for DWR loop
    MAX_DWR_ITERATIONS = 1
    if refinement_type == "uniform":
        MAX_DWR_ITERATIONS = 11
    elif refinement_type == "adaptive":
        MAX_DWR_ITERATIONS = 26
    PLOT_ESTIMATOR = False # True
    temporal_mesh = TemporalMesh(
        t0 = 0.0, # start time 
        T = 2.0, # end time
        Δt = 0.125 # initial uniform time step size
    )
    spatial_fe = SpatialFE()

    convergence_table = {}
    
    # plot the spatial mesh
    # plot(spatial_fe.mesh)
    # plt.title("Spatial Mesh")
    # plt.show()

    # # testing adaptive refinement: refine last element
    # for i in range(5):
    #     temporal_mesh.refine(refine_flags=[False]*(temporal_mesh.n_elements-1) + [True])
    # temporal_mesh.plot_mesh()

    for iteration_dwr in range(1, MAX_DWR_ITERATIONS+1):
        print(f"\nDWR ITERATION {iteration_dwr}:")
        print("================\n")

        print("Solve primal problem:")
        primal_solutions = spatial_fe.solve_primal(temporal_mesh.mesh)

        print("Compute goal functional:")
        goal_functional = spatial_fe.compute_goal_functional(temporal_mesh.mesh, primal_solutions)
        #J_reference = 3.058290061946076e-05 # reference from uniform refinement with 262,144 temporal elements
        J_reference = 4.26326261e-03 # reference from uniform refinement with 131,072 temporal elements
        true_error = J_reference - goal_functional
        print(f"  n_k:           {temporal_mesh.n_elements}")
        print(f"  J(u_k):        {goal_functional:.8e}")
        print(f"  J(u) - J(u_k): {true_error:.8e}")

        print("Solve dual problem:")
        dual_solutions = spatial_fe.solve_dual(temporal_mesh.mesh, primal_solutions)

        print("Compute error estimator:")
        error_estimator = spatial_fe.compute_error_estimator(temporal_mesh.mesh, primal_solutions, dual_solutions)
        
        # get the start and end time values for each temporal element
        start_times = np.array([temporal_mesh.mesh[i][0] for i in range(temporal_mesh.n_elements)])
        end_times = np.array([temporal_mesh.mesh[i][1] for i in range(temporal_mesh.n_elements)])

        if PLOT_ESTIMATOR:
            # plot the error estimator in a bar chart
            plt.bar(0.5*(start_times+end_times), error_estimator, width=end_times-start_times, align='center', edgecolor='black')
            plt.title("Error estimator")
            plt.xlabel("Temporal element midpoint")
            plt.ylabel("Error estimate")
            plt.show()
        
        estimated_error = np.sum(error_estimator)
        effectivity_index = true_error / estimated_error
        convergence_table[temporal_mesh.n_elements] = {
            "J(u_k)": goal_functional, 
            "J(u) - J(u_k)": true_error, 
            "η_k": estimated_error,
            "effectivity index": effectivity_index
        }
        print(f"  η_k: {estimated_error}")
        print(f"  effectivity index: {effectivity_index:.8e}")

        if refinement_type == "uniform":
            # uniform refinement in time
            temporal_mesh.refine()
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
                refine_flags = [False for i in range(temporal_mesh.n_elements)]
                sum_abs_error = 0.
                for i in range(temporal_mesh.n_elements):
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
                temporal_mesh.refine(refine_flags=refine_flags)
            else:
                print(f"Temporal adaptivity finished! (estimated error = {np.abs(np.sum(error_estimator))} < {ERROR_TOL})")
                break

    # if using adaptive refinement
    if refinement_type == "adaptive":
        temporal_mesh.plot_mesh()

    # print convergence table as tabulate
    for tablefmt in ["simple", "latex"]:
        table = tabulate([[row[0], *row[1].values()] for row in convergence_table.items()], headers=["#DoFs", "$J(u_k)$", "$J(u) - J(u_k)$", "$\eta_k$", "$I_{eff}$"], tablefmt=tablefmt)
        print(table)
    
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