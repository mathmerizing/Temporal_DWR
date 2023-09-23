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
        
# primal right hand side function
class PrimalRHSExpression(UserExpression):
    _t = 0.

    def set_time(self, t):
        self._t = t

    def eval_cell(self, value, x, _):
        if ((x[0] - 0.5 - 0.25 * np.cos(2. * np.pi * self._t))**2 + (x[1] - 0.5 - 0.25 * np.sin(2. * np.pi * self._t))**2 < 0.125**2):
            value[0] = np.sin(4. * np.pi * self._t)
        else:
            value[0] = 0.

    def value_shape(self):
        return ()  # scalar function
    
# indicator function for upper half of domain
class IndicatorExpression(UserExpression):
    def eval_cell(self, value, x, _):
        if (x[1] > 0.5):
            value[0] = 1.
        else:
            value[0] = 0.

    def value_shape(self):
        return ()  # scalar function

class SpatialFE:
    def __init__(self):
        self.mesh = UnitSquareMesh(50, 50)
        self.V = FunctionSpace(self.mesh, 'P', 1) # linear FE in space
        self.bc = DirichletBC(self.V, Constant(0.), lambda _, on_boundary: on_boundary) # homogeneous Dirichlet BC everywhere
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.mass_form = self.u * self.v * dx
        self.laplace_form = inner(grad(self.u), grad(self.v)) * dx

        self.rhs = PrimalRHSExpression() # right hand side for primal problem
        self.indicator = IndicatorExpression() # indicator function for upper half of domain

    def solve_primal(self, temporal_mesh):
        solutions = []

        # initial condition
        u_0 = Constant(0.)
        # u_n: solution from last time step
        u_n = interpolate(u_0, self.V)
        # solution on current time step
        u = Function(self.V)

        # store initial condition as numpy array
        solutions.append(np.array(u_n.vector()))

        # for each temporal element:
        #    solve forward in time with backward Euler
        for i, temporal_element in enumerate(tqdm(temporal_mesh)):
            # print(f"Solve primal on I_{i} = ({temporal_element[0]}, {temporal_element[1]})")

            Δt = temporal_element[1] - temporal_element[0]
            L = u_n*self.v*dx
            for w_q, t_q in zip([Δt], [temporal_element[1]]):
                self.rhs.set_time(t_q)
                L += w_q * self.rhs * self.v * dx
            solve(self.mass_form + Δt*self.laplace_form == L, u, self.bc)

            # store solution as numpy array
            solutions.append(np.array(u.vector()))

            # c = plot(u)
            # plt.colorbar(c)
            # plt.show()

            u_n.assign(u)
        
        return solutions
    
    def solve_dual(self, temporal_mesh, primal_solutions):
        # NOTE: primal_solutions is only used for nonlinear PDEs or nonlinear goal functionals

        solutions = []

        # initial condition
        z_0 = Constant(0.)
        # z_n: solution from next time step
        z_n = interpolate(z_0, self.V)
        # solution on current time step
        z = Function(self.V)

        # store initial condition as numpy array
        solutions.append(np.array(z_n.vector()))

        # for each temporal element:
        #    solve backward in time with backward Euler
        for i, temporal_element in tqdm(list(enumerate(temporal_mesh))[::-1]):
            # print(f"Solve dual on I_{i} = ({temporal_element[0]}, {temporal_element[1]})")
            
            Δt = temporal_element[1] - temporal_element[0]
            solve(self.mass_form + Δt*self.laplace_form == z_n*self.v*dx + Δt*self.indicator*self.v*dx, z, self.bc)

            # store solution as numpy array
            solutions.append(np.array(z.vector()))

            # c = plot(z)
            # plt.colorbar(c)
            # plt.show()

            z_n.assign(z)
        
        return solutions[::-1] # sort solutions from t = t0 to t = T
    
    def compute_goal_functional(self, temporal_mesh, primal_solutions):
        value = 0.

        u = Function(self.V)
        for temporal_element, solution in tqdm(list(zip(temporal_mesh, primal_solutions[1:]))):
            u.vector()[:] = solution
            Δt = temporal_element[1] - temporal_element[0]
            value += Δt*assemble(u * self.indicator * dx)

        return value
    
    def compute_error_estimator(self, temporal_mesh, primal_solutions, dual_solutions):
        values = np.zeros(len(temporal_mesh))

        u = Function(self.V)   # u^{n+1}
        u_n = Function(self.V) # u^n
        z = Function(self.V)   # z^{n+1}
        z_n = Function(self.V) # z^n
        for i, temporal_element in enumerate(tqdm(temporal_mesh)):
            u.vector()[:] = primal_solutions[i+1]
            u_n.vector()[:] = primal_solutions[i]
            # z.vector()[:] = dual_solutions[i+1]
            # z_n.vector()[:] = dual_solutions[i]

            z.vector()[:] = dual_solutions[i]
            if i > 0:
                z_n.vector()[:] = dual_solutions[i-1]
            else:
                z_n.vector()[:] = dual_solutions[i]

            _z = dual_solutions[i]
            _z_n = None
            if i > 0:
                _z_n = dual_solutions[i-1]
            else:
                _z_n = dual_solutions[i]
            
            # I_k z_k: for dG(0) this is a linear interpolation between z_k(t_{m-1}) and z_k(t_m) on the temporal element I_m = (t_{m-1}, t_m)
            def z_fine(t):
                assert t >= temporal_element[0] and t <= temporal_element[1]
                z_fine = Function(self.V)
                z_fine.vector()[:] = _z_n + (_z - _z_n) * (t - temporal_element[0]) / (temporal_element[1] - temporal_element[0])
                return z_fine

            # z_k: for dG(0) this is a constant function on the temporal element I_m = (t_{m-1}, t_m)
            def z_coarse(t):
                assert t >= temporal_element[0] and t <= temporal_element[1]
                z_coarse = Function(self.V)
                z_coarse.vector()[:] = _z
                return z_coarse

            Δt = temporal_element[1] - temporal_element[0]
            
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
            # The jump terms in the primal residual are zero because (I_k z_k - z_k)^+_{m-1} = 0 on I_m.
            # The time derivative of u_k is zero because u_k is a dG(0) solution.
            # Hence, only laplace term and the right hand side remain in the primal residual.
            # Using the trapezoidal rule for the temporal integral, we get:

            # TODO: adapt theory here!!!
            #   ρ(u_k)(I_k z_k - z_k) = (Δt / 2) * ( (f^m, z_k^m - z_k^{m-1}) - (∇_x u_k^m, z_k^m - z_k^{m-1}) ) 

            # jump term: (u_k^{m} - u_k^{m-1}) * (I_k z_k - z_k)^+_{m-1}
            values[i] -= assemble((u-u_n) * (z_fine(temporal_element[0]) - z_coarse(temporal_element[0])) * dx)

            # laplace term: (∇_x u_k^m, I_k z_k - z_k)           [trapezoidal rule for temporal integral]
            for w_q, t_q in zip([Δt / 2., Δt / 2.], [temporal_element[0], temporal_element[1]]): 
                values[i] -= w_q * assemble(inner(grad(u), grad(z_fine(t_q) - z_coarse(t_q))) * dx)

            #  values[i] += - assemble((u - u_n) * (z - z_n) * dx) - (Δt / 2.) * assemble(inner(grad(u), grad(z - z_n)) * dx)

            # right hand side term: (f^m, I_k z_k - z_k)           [Simpson's rule for temporal integral]
            # Simpson: for w_q, t_q in zip([Δt / 6., 4. * Δt / 6., Δt / 6.], [temporal_element[0], (temporal_element[0] + temporal_element[1]) / 2., temporal_element[1]]):
            for w_q, t_q in zip([Δt / 2., Δt / 2.], [temporal_element[0], temporal_element[1]]):
                self.rhs.set_time(t_q)
                values[i] += w_q * assemble(self.rhs * (z_fine(t_q) - z_coarse(t_q)) * dx)

            # self.rhs.set_time(temporal_element[0])
            # values[i] += (Δt / 2.) * assemble(self.rhs * (z - z_n) * dx)

            # self.rhs.set_time(temporal_element[1])
            # values[i] += (Δt / 2.) * assemble(self.rhs * z * dx)
            # self.rhs.set_time(temporal_element[0])
            # values[i] -= (Δt / 2.) * assemble(self.rhs * z_n * dx)

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
    MAX_DWR_ITERATIONS = 3#5
    PLOT_ESTIMATOR = False
    temporal_mesh = TemporalMesh(
        t0 = 0.0, # start time 
        T = 2.0, # end time
        Δt = 0.05 #0.125 # initial uniform time step size
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
        J_reference = 3.058290061946076e-05 # reference from uniform refinement with 262,144 temporal elements
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
        print("   TODO: debug estimator")
        # TODO: debug error estimator

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
    
    # print convergence table using pandas and scientific notation
    import pandas as pd
    print("\nConvergence table:")
    print("==================\n")
    pd.set_option('display.float_format', lambda x: f"{x:.3e}")
    print(pd.DataFrame.from_dict(convergence_table, orient='index'))


        
        