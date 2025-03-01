import argparse
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
    mlflow.log_params(hparams)
    
    # Load data.
    X_train, y_train, X_test, y_test = load_data()
    
    # Process features.
    print('Converting raw data to point data ...')
    X_train = raw_to_pts(X_train, hparams["factor"], hparams["pts_placement_type"])
    X_test  = raw_to_pts(X_test, hparams["factor"], hparams["pts_placement_type"])


    print('Converting point data to nep data ...')
    X_train, nep_cutoff  = pts_to_nep(X_train, hparams["nep_hyper"], None)
    X_test,_   = pts_to_nep(X_test, hparams["nep_hyper"], nep_cutoff)
    
    # Train the model.
    device_id = 1
    batch_size = 1024 
    learning_rate = 0.0001 
    width = 150 
    depth = 3 
    n_epochs = 500 

    neural_net = FFNN(X_train, y_train, X_test, y_test)
    neural_net.set_device(device_id)
    neural_net.set_model_params(width, depth, batch_size, learning_rate)
    model, final_test_loss = neural_net.train(n_epochs)
    
    # Log the test loss and model.
    mlflow.log_metric("test_loss", final_test_loss)
    mlflow.pytorch.log_model(model, artifact_path="model")
    print(f"Finished run with hparams: {hparams} and test_loss: {final_test_loss}")

def main():
    factors = [0.8,1.0,1.25,1.5]
    pts_placement_types = ['edges', 'faces', 'vertices']
    nep_hypers = [[10,10,10,100],[10,8,4,104]]
    
    mlflow.set_experiment("TetraHedraExperiment")
    
    with mlflow.start_run(run_name="hyperparam_search") as parent_run:
        for factor, pts_placement_type, nep_hyper in itertools.product(factors, pts_placement_types, nep_hypers):
            hparams = {
                "factor": factor,
                "pts_placement_type": pts_placement_type,
                "nep_hyper": nep_hyper,
            }
            with mlflow.start_run(run_name=f"run_nep_{nep_hyper[3]}_factor_{factor}_ptstype_{pts_placement_type}", nested=True):
                run_experiment(hparams)

if __name__ == "__main__":
    main()
