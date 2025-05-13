import argparse, os
import itertools
import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from Processors import raw_to_pts, pts_to_nep, load_data
from FFNN import FFNN


def run_experiment(hparams):
    # Log hyperparameters for this run.
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    mlflow.log_params(hparams)
    model_id = hparams['manuel_id']
    
    # Load data.
    device_id = 1
    X_train, y_train, X_test, y_test = load_data(debug_mode=False)
    
    # Process features.
    print('Converting raw data to point data ...')
    X_train = raw_to_pts(X_train, hparams)
    X_test  = raw_to_pts(X_test, hparams)


    print('Converting point data to nep data ...')
    X_train, nep_cutoff  = pts_to_nep(X_train, hparams,device_id, None)
    X_test,_   = pts_to_nep(X_test, hparams,device_id, nep_cutoff)
    
    # Train the model.
    
    batch_size = 1024 
    learning_rate = 0.0001 
    width = 150 
    depth = 3
    n_epochs = 501 

    neural_net = FFNN(X_train, y_train, X_test, y_test)
    neural_net.set_device(device_id)
    neural_net.set_model_id(model_id)

    neural_net.set_model_params(width, depth, batch_size, learning_rate)
    model, final_test_loss = neural_net.train(n_epochs)
    
    # Log the test loss and model.
    # mlflow.log_metric("test_loss", final_test_loss)
    # mlflow.pytorch.log_model(model, artifact_path="model")
    # print(f"Finished run with hparams: {hparams} and test_loss: {final_test_loss}")

def main():
 

    all_hyperparams = []
    experiment0 = {'pts_placement':'along_axis','npts':6,'nep_hyper':[10,10,10,100]}
    experiment1 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'none'}
    experiment2 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'none','pts_types':[0,0,0,1,1,1]}
    experiment3 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'none','pts_types':[0,1,2,3,4,5]}
    experiment4 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'gyration'}
    experiment5 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'inverse_gyration'}
    experiment6 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'inverse_gyration','pts_types':[0,0,0,1,1,1]}
    experiment7 = {'pts_placement':'xyz','nep_hyper':[10,10,10,100], 'scaling':'inverse_gyration','pts_types':[0,1,2,3,4,5]}
    experiment8 = {'pts_placement':'cubic','nep_hyper':[10,10,10,100], 'scaling':'none','pts_types':[0,0,0,1,1,1,0,0,0,1,1,1]}
    experiment9 = {'pts_placement':'cubic','nep_hyper':[10,10,10,100], 'scaling':'none','pts_types':[0,0,0,1,1,1,2,2,2,3,3,3]}
    
    all_hyperparams.append(experiment0)
    all_hyperparams.append(experiment1)
    all_hyperparams.append(experiment2)
    all_hyperparams.append(experiment3)
    all_hyperparams.append(experiment4)
    all_hyperparams.append(experiment5)
    all_hyperparams.append(experiment6)
    all_hyperparams.append(experiment7)
    all_hyperparams.append(experiment8)
    all_hyperparams.append(experiment9)

    mlflow.set_experiment("TwistedCylinderExperiment1")
    
    print(all_hyperparams)
    with mlflow.start_run(run_name="hyperparam_search") as parent_run:
        iid = 100
        for hparams in all_hyperparams:
            hparams['manuel_id'] = iid
            iid+=1
      
            with mlflow.start_run(run_name=f"run_nep_{iid}", nested=True):
                run_experiment(hparams)

if __name__ == "__main__":
    main()
