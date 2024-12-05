import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import SchNet, DimeNetPlusPlus
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import random


class GeometricModel():
    """
    Train-test-optimize a model from pytorch geometric
    This is the parent class for models that work with 12pts input
    """
    def __init__(self, training_files,test_files,batch_size,learning_rate,device):
        self.training_files = training_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.lr = learning_rate
        self.set_device(device)
        self.results = [] ## [epoch mae_test_error maape_test_error]


    def set_device(self,device):
        self.device = torch.device('cpu')
        if(device>-0.5):
            dev_str = 'cuda:%d' %(device)
            self.device = torch.device(dev_str)
        self.process_data()
        self.preprocess_batches()
        self.process_output()


    def process_data(self):
        """
        Remove non-interesting interactions and normalize output
        """
        self.en_min = 0.0
        self.en_max = 15.0

        self.x_train = np.load(self.training_files[0])
        self.y_train = np.load(self.training_files[1])
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.x_train = self.x_train[(self.y_train<self.en_max-0.05)&(self.y_train>self.en_min+0.05)]
        self.y_train = self.y_train[(self.y_train<self.en_max-0.05)&(self.y_train>self.en_min+0.05)]

        self.y_train = (self.y_train - self.en_min) / (self.en_max - self.en_min)
        self.x_train = torch.from_numpy(self.x_train)
        self.y_train = torch.from_numpy(self.y_train)
        self.y_train = self.y_train.view(-1,1)

        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.N_sample = len(self.x_train)

    def process_output(self): # not exaclty the same

        self.x_test = np.load(self.test_files[0])
        self.y_test = np.load(self.test_files[1])


        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        self.x_test = self.x_test[(self.y_test<self.en_max-0.05)&(self.y_test>self.en_min+0.05)]
        self.y_test = self.y_test[(self.y_test<self.en_max-0.05)&(self.y_test>self.en_min+0.05)]

        self.x_test = torch.from_numpy(self.x_test)
        self.y_test = torch.from_numpy(self.y_test)
        self.y_test = self.y_test.view(-1,1)
        self.x_test = self.x_test.to(self.device)
        self.y_test = self.y_test.to(self.device)

    def preprocess_batches(self):
        """
        Handle batch indexing, atom indexing arrays etc.
        """
        self.N_atom_per_mol = 12
        batch_atom_indexing = torch.arange(self.batch_size,dtype=torch.int64)
        batch_atom_indexing = batch_atom_indexing.repeat_interleave(self.N_atom_per_mol)
        atomic_numbers_per_batch = torch.ones(len(batch_atom_indexing),dtype=torch.int64)
        self.batch_atom_indexing = batch_atom_indexing.to(self.device)
        self.atomic_numbers_per_batch = atomic_numbers_per_batch.to(self.device)


    def train_step(self):
        self.model.train()
        batch_indexes = torch.split(torch.randperm(self.N_sample).to(self.device), self.batch_size, dim=0)
        batch_indexes = batch_indexes[:-1] # last batch will fail since it won't have the same size
        self.training_error = 0.0
        for j in range(len(batch_indexes)):
            b_ind = batch_indexes[j]
            x_data_j = self.x_train[b_ind].view((-1,3))
            y_data_j = self.y_train[b_ind]
            self.optimizer.zero_grad()
            out = self.model(self.atomic_numbers_per_batch, x_data_j, self.batch_atom_indexing)
            loss = self.criterion(out, y_data_j)
            self.training_error += loss.cpu().detach().item()
            # train_loss.append(loss.cpu().detach().item())
            loss.backward()
            self.optimizer.step()
        self.training_error /= j
        # print(self.training_error)


    def validate(self):
        self.model.eval()
        valid_loss = 0
        batch_atom_indexing = torch.arange(self.x_test.size()[0],dtype=torch.int64,device=self.device)
        x_test = self.x_test.view((-1,3))
        atom_nums = torch.ones(x_test.size()[0],dtype=torch.int64,device=self.device)
        batch_atom_indexing = batch_atom_indexing.repeat_interleave(self.N_atom_per_mol)
        with torch.no_grad():
            output = self.model(atom_nums,x_test,batch_atom_indexing)
            output = output*(self.en_max - self.en_min) + self.en_min
            # error = maape(output, y_data)
            ae = torch.abs(output - self.y_test)
            mae = torch.mean(ae).cpu().item()
            maape = torch.mean(torch.arctan(ae/self.y_test)).cpu().item()
            # print("%.3f %d"%(mae,self.epoch))
            self.results.append([self.epoch,mae,maape,self.training_error])
        # print(self.results)

    def train(self, N_epochs):
        self.epoch = 0
        for epoch in range(1, N_epochs + 1):
            self.train_step()
            self.epoch += 1
            if(epoch%1==0):
                self.validate()
        return self.results


class SchnetUtils(GeometricModel):
    """
    Generic hyperparameters: lr, batch_size
    Specific hyperparameters: n_inter, n_hidden, n_filter, n_gauss
    Task specific: cutoff, max_num_neighbors (also common to most models)
    """

    def __init__(self, training_files,test_files,batch_size,learning_rate,device):
        super().__init__(training_files,test_files,batch_size,learning_rate,device)


    def init_model(self,hypers):
        self.model = SchNet(
            hidden_channels=hypers[0],
            num_filters=hypers[1],
            num_interactions=hypers[2],
            num_gaussians=hypers[3],
            cutoff=hypers[4], # 10
            max_num_neighbors=hypers[5] # 11
        )

        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

class DimeNetUtils(GeometricModel):
    """
    Generic hyperparameters: lr, batch_size
    Specific hyperparameters: (there are 11 - we only tune for 3 )
    n_inter_blocks (num_blocks)
    n_basis_geo (num_spherical & num_radial)
    n_hidden (hidden_channels)
    Task specific: cutoff, max_num_neighbors (also common to most models)
    """

    def __init__(self, training_files,test_files,batch_size,learning_rate,device):
        super().__init__(training_files,test_files,batch_size,learning_rate,device)


    def init_model(self,hypers):
        self.model = DimeNetPlusPlus(
            hidden_channels=hypers[0],
            out_channels=1,
            num_blocks=hypers[1],
            int_emb_size=16,
            basis_emb_size=4,
            out_emb_channels=16,
            num_spherical=hypers[2],
            num_radial=hypers[2],
            cutoff=hypers[3],
            max_num_neighbors=hypers[4],
            envelope_exponent=5,
        )


        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)























def dummy():
    pass
