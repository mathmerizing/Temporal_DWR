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
            self.rhs.set_time(temporal_element[1])
            solve(self.mass_form + Δt*self.laplace_form == u_n*self.v*dx + Δt*self.rhs*self.v*dx, u, self.bc)

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
        
        return reversed(solutions) # sort solutions from t = t0 to t = T
    
    def compute_goal_functional(self, temporal_mesh, primal_solutions):
        value = 0.

        u = Function(self.V)
        for temporal_element, solution in zip(temporal_mesh, primal_solutions[1:]):
            u.vector()[:] = solution
            Δt = temporal_element[1] - temporal_element[0]
            value += Δt*assemble(u * self.indicator * dx)

        return value

if __name__ == "__main__":
    # hyperparameters
    ERROR_TOL = 1e-4 # stopping criterion for DWR loop
    MAX_DWR_ITERATIONS = 5
    temporal_mesh = TemporalMesh(
        t0 = 0.0, # start time 
        T = 2.0, # end time
        Δt = 0.125 # initial uniform time step size
    )
    spatial_fe = SpatialFE()
    
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
        J_reference = 2.9125264677148095e-05
        print(f"n_k: {temporal_mesh.n_elements}, J(u_k): {goal_functional}, J(u) - J(u_k): {J_reference - goal_functional}")

        # uniform refinement
        temporal_mesh.refine()

        # print("Solve dual problem:")
        # dual_solutions = spatial_fe.solve_dual(temporal_mesh.mesh, primal_solutions)

        # print("Compute error estimator:")
        # print("  TODO...")

        # error_estimator = 1. # TODO

        # if error_estimator > ERROR_TOL:
        #     print("Mark temporal elements for refinement:")
        #     print("  TODO...")

        #     print("Refine temporal mesh:")
        #     print("  TODO...")
        # else:
        #     print(f"Temporal adaptivity finished! (estimated error = {error_estimator} < {ERROR_TOL})")
        #     break
        # quit()
        
        
        