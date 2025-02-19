import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class FFNN():
    def __init__(self, x_train, y_train, x_test, y_test):

        self.en_min = -6.7
        self.en_max = 15.0
        self.validate_per = 10 

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.N_features = x_train.shape[1]
        self.results = [] ## [epoch mae_test_error maape_test_error]

    def set_model_params(self,width, depth, batch_size, lr):
        self.model = TheNet(width,depth,self.N_features)
        self.batch_size = batch_size
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.process_train_data()
        self.process_test_data()


    def set_device(self,device):
        self.device = torch.device('cpu')
        if(device>-0.5):
            dev_str = 'cuda:%d' %(device)
            self.device = torch.device(dev_str)
        

    def process_train_data(self):
        """
        Remove non-interesting interactions and normalize output
        """
        
        self.x_train = self.x_train.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)

        print("Initial data size: ", len(self.y_data))
        yield_mask = np.abs(self.y_data)<15.0
        self.x_data = self.x_data[yield_mask]
        self.y_data = self.y_data[yield_mask]
        print("Final training size after eliminating overlaps: ", len(self.y_data))

        self.y_data = torch.from_numpy(self.y_data)
        self.x_data = torch.from_numpy(self.x_data)

        shape = self.x_data.size()
        self.n_samples = shape[0]
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))
        self.input_size = shape[1]

        self.y_data[self.y_data>self.en_max] = self.en_max
        self.y_data = (self.y_data - self.en_min) / (self.en_max - self.en_min)

        self.y_data[self.y_data>1.0] = 1.0
        self.y_data[self.y_data<0.0] = 0.0

        self.x_data = self.x_data.to(self.device)
        self.y_data = self.y_data.to(self.device)


    def process_test_data(self): 

        self.x_test = self.x_test.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        yield_mask = np.abs(self.y_test)<15.0
        self.x_test = self.x_test[yield_mask]
        self.y_test = self.y_test[yield_mask]

        self.x_test = torch.from_numpy(self.x_test)
        
        
    
    def train(self, N_epochs):
        print("Start Training ...")
        self.epoch = 0
        for epoch in range(1, N_epochs + 1):
            self.train_step()
            self.epoch += 1
            if(epoch%self.validate_per==0):
                self.validate()
        return self.results

    def train_step(self):
        self.model.train()
        batch_indexes = torch.split(torch.randperm(self.N_sample).to(self.device), self.batch_size, dim=0)
        train_loss = []
        for j in range(len(batch_indexes)):
            b_ind = batch_indexes[j]
            y_hat = self.model.forward(self.x_train[b_ind])
            loss = self.criterion(y_hat,self.y_train[b_ind])
            self.optimizer.zero_grad()
            self.training_error += loss.cpu().detach().item()
            loss.backward()
            self.optimizer.step()
        self.training_error = np.mean(np.array(train_loss))

    def validate(self):
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            y_pred = self.model(self.x_test)
            y_pred = y_pred*(self.en_max - self.en_min) + self.en_min
            y_pred = y_pred.cpu().detach().numpy()

            abs_err = np.abs(y_pred.flatten() - self.y_test.flatten())
            mae = np.mean(abs_err)
            aape = np.arctan(abs_err/self.y_test.flatten())
            index = np.isnan(aape)
            aape[index] = 0.0
            maape = np.mean(aape)

            # print("%.3f %d"%(mae,self.epoch))
            self.results.append([self.epoch,mae,maape,self.training_error])



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