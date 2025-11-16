We evaluate the 1D SSH chain and the 2D honeycomb lattice. We then extend this to see the topological and trivial gaps that can be generated. 
Using the FHS algorithm I defined link variables to rapidly calculate the berry curvature of a given system. My partner simulates an infinite graphene lattice and how its band structure changes as we alter parameters.
We reproduce, as shown in Haldane's original paper, the values for M which set the transition boundary between phases.

The .py program is the most up to date version of the code.

Key coding takeaways:
- np.linalg library is very efficient for eignevalues/vectors
- the BZ may sometimes be more efficiently considered for a different area
  - My "brute force" method of calculating Berry Curvature works better if I consider the BZ as being defined by the Gamma points and not the K+- points
- 3D plots offer a lot of clarity.
