import numpy as np
from Sampler import *
import matplotlib.pyplot as plt
import argparse
import os

"""
4 data types:

****************************
Raw (not invariant can't be used as input)
- 14 dimensional
- pos1, quat1, pos2, quat2
pos = (x,y,z), center of mass coordinates
quat = (w,x,y,z), orientation quaternion

*****************************
NotSymmetrized
- 6 dimensional
- euler1, euler2, euler3, r, azimuthal, polar,
euler angles follow ZXZ convention
last 3 dimensions are the spherical coordinates of relative position

This data type is invariant to global rotations and translations but not
invariant to the symmetries of 3D object
it can be used as input for shapes with no symmetry

******************************
Symmetrized
- 6 dimensional
- x1, x2, x3, alpha, beta, gamma

First 3 numbers are the relative position
Last 3 angles are the relative orientation (they are not equivalent to euler angles)
This data type is invariant to global rotations and translations AND
invariant to the symmetries of cubic objects
it can be used as input for shapes with octahedral symmetry

****************************
12pts
- 12x3 = 36 dimensional
- (p1, p2, p3, ..., p12)
Consists of 12 points which are placed to the centers of the faces of the cubes
Not invariant to anything. It can be directly used as input for deep MLPs such as
SchNet, DimeNet, etc.
Or it can be used to get invariant descriptors with SOAP, MTP, etc.


=====================================================================
=====================================================================

The data is first sampled as NotSymmetrized using LatinHypercubeSampling
Than it is converted to other types for accurate comparisons.
Finally it is saved to the disk as .npy format.
5 files generated per seed and sample size. 4 files for 4 different input types
and 1 output file for corresponding energies. 

If sample size is N
Raw (N, 6)
NotSymmetrized (N, 6)
Symmetrized (N, 6)
12pts (N,12,3)
y (N,)

=====================================================================
=====================================================================
Packages required:

CuPy : Here I only showcase small sample sizes but for larger sample sizes >> 10k
It is beneficial to use GPU in energy calculations. So this script won't run
without a gpu and cupy.

Gsd : The structure of the cube is stored as a gsd file.

"""


if not os.path.exists('datadir'):
    os.makedirs('datadir')

sample_size = 500
sample_seed = 42
data_id_string = '500_s42'

SampleNotSymmetrized(sample_size,sample_seed,data_id_string,50)
ConvertNotSymmetrizedToRaw(data_id_string)
ConvertRawToSymmetrized(data_id_string,50)
ConvertNotSymmetrizedTo12pts(data_id_string)







print('Done')
