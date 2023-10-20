# Temporal Dual-Weighted Residual (DWR) Method
Dual-weighted residual error estimation for temporal error estimation applied to tensor-product space-time finite elements

## Dual-Weighted Residual (DWR) Method
The dual-weighted residual method consists of two problems: the primal and the dual problem.

- Primal problem (original ODE/PDE): Find weak solution U such that

$$
A(U)(\Phi) = F(\Phi) \qquad \forall \Phi
$$

- Dual problem (measures adjoint sensitivity): Find weak solution Z such that

$$
A^{'}_U(U)(\Psi, Z) = J^{'}_U(U)(\Psi) \qquad \forall \Psi
$$

Here, J(U) denotes some quantity of interest (QoI) of our solution, e.g. it could be the value at the end time in the case of an ordinary differential equation (ODE): $J(U) := U(T)$.
To make our lives easier we consider here linear differential equations and linear goal functionals $J$ such that the dual problem simplifies to

$$
A(\Psi)(Z) = J(\Psi) \qquad \forall \Psi.
$$

To estimate the error caused by the temporal discretization, we have the error estimator

$$
J(U) - J(U_k) \approx - A(U_k)(Z - Z_k) + F(Z - Z_k),
$$

which means that we just need to insert the primal and dual solutions into the residuum to be able to estimate the error on all temporal elements. For error localization, we can just assemble the residual on individual elements and then refine the elements with the largest error. 
We need to use some approximation for $Z - Z_k$, since the analytical solution is unknown.
In this repisotory, we demonstrate this method by solving the primal problem with a dG(0) discretization in time, i.e. we use piecewise constant finite elements for the time discretization. Note that for our numerical problems this leads to a backward Euler time discretization. 
For the dual problem, we then use either an equal order discretization by also using dG(0) in time or a higher order discretization by using dG(1) in time, i.e. piecewise linear finite elements for the time discretization.
For more information on dG(r) time discretizations for ODEs and PDEs, we refer to our [exercises for the course on Space-time finite element methods](https://github.com/mathmerizing/SpaceTimeFEM_2023-2024).
The dual weights $Z - Z_k$ are then either approximated by patchwise higher order interpolation for the dG(0) solution (here: linear interpolating solution at neighboring time points), i.e.


$$
Z - Z_k \approx I_{2k}^{dG(0)} Z_k^{dG(0)} - Z_k^{dG(0)},
$$

or by low order interpolation for the dG(1) solution (here: evaluate at left end point of temporal element), i.e.

$$
Z - Z_k \approx Z_k^{dG(1)} - I_k^{dG(0)}Z_k^{dG(1)}.
$$

## Application: Ordinary Differential Equation (ODE)

### Problem description
Find $u: [0,1] \rightarrow \mathbb{R}$ such that

$$
\partial_t u(t) - u(t) = 0 \qquad \forall t \in (0,1),
$$


$$
u(0) = 1.
$$

The analytical solution for the primal problem is $u(t) = \exp(t)$. As a quantity of interest, we choose the end time value

$$
J(u) := u(1) = e
$$

Then the dual problem is: Find $z: [0,1] \rightarrow \mathbb{R}$ such that

$$
 -\partial_t z(t) - z(t) = 0 \qquad \forall t \in (0,1),
$$

$$
z(1) = 1,
$$

which runs backward in time.

### Discrete formulations
The dG(0) formulation of the primal problem reads

$$
u_k^m = \frac{1}{1 - k_m}u_k^{m-1},
$$

where $k_m := t_m - t_{m-1}$ denotes the time step size.

The dG(0) formulation of the dual problem reads


$$
z_k^{m-1} = \frac{1}{1 - k_m}z_k^{m}
$$

and the dG(1) formulation of the dual problem is given by

$$
 \left[ \frac{1}{2}\begin{pmatrix}
        1 & -1 \\ 
        1 & 1
    \end{pmatrix} - 
    \frac{k}{6}\begin{pmatrix}
        2 & 1 \\
        1 & 2
    \end{pmatrix} \right] \begin{pmatrix}
        z_k^m(t_{m-1}) \\ 
        z_k^m(t_{m})
    \end{pmatrix} = \begin{pmatrix}
        0 \\ 
        z_k^{m+1}(t_{m}) 
    \end{pmatrix}.
$$

## Application: Partial Differential Equation (PDE)

### Problem description

Find $u: \Omega \times I \rightarrow \mathbb{R}$ such that

$$
\partial_t u - \Delta_x u = f \qquad \text{in } \Omega \times I,
$$

$$
u = 0 \qquad \text{on } \partial \Omega \times I,
$$

$$
u = 0 \qquad \text{on } \Omega \times \{ 0 \},
$$

with spatial domain $\Omega = (0,1)^2$, temporal domain $I = (0,2)$ and right hand side

$$
 f(x, t) = \begin{cases}
        1 & \text{for } (x,t) \in \left(\frac{1}{2},1\right)^2 \times \left(0,\frac{1}{2}\right) \cup \left(0,\frac{1}{2}\right)^2 \times \left(1,\frac{3}{2}\right), \\
        0 & \text{else}.
    \end{cases}
$$

As a quantity of interest, we choose the time integral over the spatial integral over the upper half of the domain, i.e.

$$
    J(u) = \int_I \int_{(0,1) \times (\frac{1}{2},1)}u\ \mathrm{d}x\ \mathrm{d}t.
$$

