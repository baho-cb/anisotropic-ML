import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from GeometricModel import *

class FFNN(GeometricModel):
    """
    Train-test-optimize a model from pytorch ffnn model
    Works with the symmetrized input input
    """
    def __init__(self, training_files,test_files,batch_size,learning_rate,device):
        super().__init__(training_files,test_files,batch_size,learning_rate,device)


    def set_device(self,device):
        self.device = torch.device('cpu')
        if(device>-0.5):
            dev_str = 'cuda:%d' %(device)
            self.device = torch.device(dev_str)
        self.process_data()
        self.process_output()
        self.normalize_input()

    def normalize_input(self):
        self.mins = torch.tensor([2.7,  0.0,  0.,  0.0,    -3.142,  0.0]).to(self.device)
        self.maxs = torch.tensor([5.9, 4.8, 4.2, 1.87,    3.142, 1.571]).to(self.device)
        self.N_features = self.x_train.size()[1]
        self.x_train = (self.x_train - self.mins.view(1, -1)) / (self.maxs.view(1, -1) - self.mins.view(1, -1))
        self.x_test = (self.x_test - self.mins.view(1, -1)) / (self.maxs.view(1, -1) - self.mins.view(1, -1))


    def train_step(self):
        self.model.train()
        batch_indexes = torch.split(torch.randperm(self.N_sample).to(self.device), self.batch_size, dim=0)
        self.training_error = 0.0
        for j in range(len(batch_indexes)):
            b_ind = batch_indexes[j]
            y_hat = self.model.forward(self.x_train[b_ind])
            loss = self.criterion(y_hat,self.y_train[b_ind])
            self.optimizer.zero_grad()
            self.training_error += loss.cpu().detach().item()
            loss.backward()
            self.optimizer.step()
        self.training_error /= j
        # print(self.training_error)  

    def validate(self):
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            output = self.model(self.x_test)
            output = output*(self.en_max - self.en_min) + self.en_min
            ae = torch.abs(output - self.y_test)
            mae = torch.mean(ae).cpu().item()
            maape = torch.mean(torch.arctan(ae/self.y_test)).cpu().item()
            # print("%.3f %d"%(mae,self.epoch))
            self.results.append([self.epoch,mae,maape,self.training_error])


    def init_model(self,hypers):
        self.model = TheNet(hypers[0],hypers[1],self.N_features)
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


class TheNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width, num_hidden,input_size):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)

        self.layer_last = nn.Linear(nn_width, 1)

    def forward(self, x):
        activation = nn.ReLU()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u






























def dummy():
    pass
