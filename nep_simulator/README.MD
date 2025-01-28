This code performs Molecular Dynamics simulations of anisotropic particles
using pre-trained NeuroEvolutionPotentials.

Packages required:
Pytorch
Cupy
Numpy
Scipy
Gsd

Computationally demanding parts are implemented with Cupy and custom CUDA kernels
of cupy.


Python scripts that use NeuroEvolution Potential adapted to rigid bodies.
Currently only Nose-Hoover thermostat is implemented.

How to add a new particle geometry:
