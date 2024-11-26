import numpy as np
from OptimizeHyper import *
from FFNN import *

"""
- GPR training might take longer for training sizes of 1k and larger, keep that
in mind when setting n_calls

- Neural Net method is not tuned as it was done previously. Here it only serves
as a benchmark for the other methods
"""

n_calls = 2
data_sizes = ['100','500','1k']

optim_linear = []
optim_kernel = []
optim_gpr = []
optim_FFNN = []

for i,ds in enumerate(data_sizes):

    training_files_12pts = [
        '../../08/x_12pts_%s_s42.npy'%(ds),
        '../../08/y_%s_s42.npy'%(ds)
    ]

    test_files_12pts = [
        '../../08/x_12pts_5k_s42.npy',
        '../../08/y_5k_s42.npy'
    ]

    training_files_red = [
        '../../08/x_red_%s_s42.npy'%(ds),
        '../../08/y_%s_s42.npy'%(ds)
    ]

    test_files_red = [
        '../../08/x_red_5k_s42.npy',
        '../../08/y_5k_s42.npy'
    ]

    res_linear = OptimizeLinear(training_files_12pts,test_files_12pts,n_calls)
    optim_linear.append(res_linear.fun)
    res_kernel = OptimizeKernel(training_files_12pts,test_files_12pts,n_calls)
    optim_kernel.append(res_kernel.fun)
    res_gpr = OptimizeGPR(training_files_12pts,test_files_12pts,n_calls)
    optim_gpr.append(res_gpr.fun)

    res_NN = FeedForwardNeuralNet(training_files_red,test_files_red)
    optim_FFNN.append(res_NN)

plt.figure(1)
plt.scatter(data_sizes,optim_linear,label='LinearReg.')
plt.scatter(data_sizes,optim_kernel,label='KernelReg.')
plt.scatter(data_sizes,optim_gpr,label='GPR.')

plt.scatter(data_sizes,optim_FFNN,label='NeuralNetwork')
plt.xlabel('Training Data Size')
plt.ylabel('Mean Abs. Error')
plt.ylim(0.0,None)
plt.legend()
plt.show()



print('Done')
