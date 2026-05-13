The general structure and documentation should be sufficient, but questions can be directed via email at aarnavpanda11@gmail.com. The overview of the code is of four parts: parameter input, lattice class, slave-rotor solver class, plotting options. Updates will continue to be made, especially with regards to cleaning redundancies, upgrading to the new FHS method [1], and making further quality-of-life improvements. The end goal is that the codebase can be easily adapted to any tight binding system, but the nature of solving mean-field equations means that over-packaging into pre-prepared functions is unlikely to be useful, as is done in the currently available TopologicalNumber.jl package for Julia[2]. This method will not work well for situations where a Green's function needs to be used (interacting systems), so it is more useful to have the direct program as opposed to a library or package. 

[1] K. Shiozaki, A discrete formulation for three-dimensional winding number, en, arXiv:2403.05291
[cond-mat], Mar. 2026.
[2] K. Adachi and M. Kanega, “TopologicalNumbers.jl: A Julia package for topological number
computation”, Journal of Open Source Software 10, 6944 (2025).

We also evaluate the 1D SSH chain and the 2D honeycomb lattice. We then extend this to see the topological and trivial gaps that can be generated. 
Using the FHS algorithm I defined link variables to rapidly calculate the berry curvature of a given system. My partner simulates an infinite graphene lattice and how its band structure changes as we alter parameters.
We reproduce, as shown in Haldane's original paper, the values for M which set the transition boundary between phases.


