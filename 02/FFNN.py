import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

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


class Dataset():
    def __init__(self,filenames):
        self.en_min = 0.0
        self.en_max = 15.0

        self.x_data = np.load(filenames[0])
        self.y_data = np.load(filenames[1])
        self.x_data = self.x_data.astype(np.float32)
        self.y_data = self.y_data.astype(np.float32)

        self.y_data = torch.from_numpy(self.y_data)
        self.x_data = torch.from_numpy(self.x_data)

        shape = self.x_data.size()
        self.n_samples = shape[0]
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))
        self.input_size = shape[1]

        self.normalize_input()



    def normalize_input(self):
        """
        Max-min normalization
        """

        mins = [2.7,  0.0,  0.,  0.0,    -3.142,  0.0]
        maxs = [5.9, 4.8, 4.2, 1.87,    3.142, 1.571]
        N_features = self.x_data.size()[1]

        for i in range(N_features):
            data = torch.clone(self.x_data[:,i])
            data_normal = (data - mins[i]) / (maxs[i] - mins[i])
            self.x_data[:,i] = data_normal

    def normalize_output(self):

        self.y_data[self.y_data>self.en_max] = self.en_max
        self.y_data = (self.y_data - self.en_min) / (self.en_max - self.en_min)

        self.y_data[self.y_data>1.0] = 1.0
        self.y_data[self.y_data<0.0] = 0.0

    def get_y_min_max(self):
        return self.en_min,self.en_max

    def get_data(self):
        return self.x_data,self.y_data

    def get_input_size(self):
        return self.input_size

    def get_n_sample(self):
        shape = self.x_data.size()
        self.n_samples = shape[0]
        return self.n_samples


def validate(model,test_files,device):
    test_dataset = Dataset(test_files)
    x_test,y_test = test_dataset.get_data()
    model.eval()
    y_pred = model(x_test)
    y_min,y_max = test_dataset.get_y_min_max()
    y_pred = y_pred*(y_max-y_min) + y_min
    mae = torch.mean(torch.abs(y_pred.view(-1)-y_test.view(-1)))
    return mae


def FeedForwardNeuralNet(training_files,test_files,device='cpu'):
    """
    This method only serves as a benchmark for other methods so here we're not
    attemting to tune the hyperparameters
    """
    #### no parameter tuning
    N_epochs = 200
    lr = 0.0005
    width = 60
    depth = 6
    device = torch.device(device)

    train_dataset = Dataset(training_files)
    train_dataset.normalize_output()

    N_sample = train_dataset.get_n_sample()
    input_size = train_dataset.get_input_size()

    batch_size = N_sample//10

    model = TheNet(width,depth,input_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_data,y_data = train_dataset.get_data()
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    del(train_dataset)


    for i in range(N_epochs):
        model.train()
        train_loss = []
        batch_indexes = torch.split(torch.randperm(N_sample).to(device), batch_size, dim=0)
        for j in range(len(batch_indexes)):
            b_ind = batch_indexes[j]
            y_hat = model.forward(x_data[b_ind])
            loss = criterion(y_hat,y_data[b_ind])
            train_loss.append(loss.cpu().detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(np.mean(np.array(train_loss)))

    mae = validate(model,test_files,device)

    return mae.detach().cpu().numpy()

































def dummy():
    pass
