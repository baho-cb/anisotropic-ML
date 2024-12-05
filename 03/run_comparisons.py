import numpy as np
from FFNN import *
from GeometricModel import *
# from Dimenetpp import *
import matplotlib.pyplot as plt

"""
Run trainings and plot/save errors for 3 models
- Schnet
- DimeNet++
- FFNN_Symmetrized

Improvements:

Fix the filepaths
Save the results with the hyperparameters as metadata
Include more models
Print some info on the progress of the training
Maybe save the best performing models 

"""

n_epochs = 5 # # of training epochs
data_sizes = ['10k','20k']
batch_size = 128
lr = 0.0005 # learning rate
gpu_id = 1 # -1 for cpu
error_type_index = 2 # mae, maape, training mse

for i,ds in enumerate(data_sizes):

    training_files_12pts = [
        '../../08/x_12pts_%s_s42.npy'%(ds),
        '../../08/y_%s_s42.npy'%(ds)
        # '../../08/x_12pts_%s_s23.npy'%(ds),
        # '../../08/y_%s_s23.npy'%(ds)
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
    #
    schnet = SchnetUtils(training_files_12pts,test_files_12pts,batch_size,lr,gpu_id)
    schnet.init_model([16,32,2,15,10.0,11]) ## hyperparameters specific to the model
    res_schnet = schnet.train(n_epochs)

    dimenetpp = DimeNetUtils(training_files_12pts,test_files_12pts,batch_size,lr,gpu_id)
    dimenetpp.init_model([16,2,4,10.0,11]) ## hyperparameters specific to the model
    res_dimenet = dimenetpp.train(n_epochs)

    ffnn_sym = FFNN(training_files_red,test_files_red,batch_size,lr,gpu_id)
    ffnn_sym.init_model([100,3]) ## hyperparameters specific to the model
    res_ffnn_sym = ffnn_sym.train(n_epochs)

    res_schnet = np.array(res_schnet)
    res_dimenet = np.array(res_dimenet)
    res_ffnn_sym = np.array(res_ffnn_sym)


    plt.figure(1)
    plt.title('Training Data Size %s '%ds)
    plt.plot(res_schnet[:,0],res_schnet[:,error_type_index+1],label='SchNet')
    plt.plot(res_dimenet[:,0],res_dimenet[:,error_type_index+1],label='DimeNetPP')
    plt.plot(res_ffnn_sym[:,0],res_ffnn_sym[:,error_type_index+1],label='FFNN_Symmetrized')
    plt.xlabel('Epochs')

    error_types = [
    'Mean Abs. Test Error',
    'Mean Abs. Arctan Test Error',
    'MSE Training',
    ]
    plt.ylabel(error_types[error_type_index])
    plt.legend()
    plt.show()



print('Done')
