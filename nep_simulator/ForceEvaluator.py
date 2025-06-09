import torch
import torch.nn as nn
import cupy as cp
import numpy as np 
from typing import Sequence, Tuple, Callable
import matplotlib.pyplot as plt 

class TheEnergyNet(nn.Module):
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

def relu(z: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """Returns (activation, derivative_mask)"""
    mask = z > 0
    return z * mask, mask.astype(z.dtype)


class ForceEvaluator():
    """
    passes the nep descriptors through the pretrained neural net 
    to evaluate energies and forces-torques by taking the numerical gradient
    """
    def setDevice(self,gpu_id):
        self.gpu_id = gpu_id
        self.torch_device = torch.device('cuda:%d' %self.gpu_id if torch.cuda.is_available() else 'cpu')

    def setNN_dimensions(self,w,d):
        self.width = w 
        self.depth = d

    def set_dx_dteta(self,dx,dteta):
        self.dx = dx
        self.dteta = dteta

    def setNparticles(self,Nparticles):
        self.Nparticles = Nparticles    

    def setNd(self,Nd):
        self.Nd = Nd

    def setSync(self,is_sync):
        self.is_sync = is_sync    

    def read_model(self,model_path):
        energy_model =  TheEnergyNet(self.width,self.depth,self.Nd)
        energy_model.load_state_dict(torch.load(model_path,map_location='cpu'))
        gpu_str = 'cuda:%d' %(self.gpu_id)
        energy_model.to(gpu_str)
        energy_model.eval()
        self.energy_net = energy_model


    def set_energy_range(self,en_min,en_max):
        self.en_min = en_min
        self.en_max = en_max

    def move_data_to_device(self):
        pass

    def evaluate_interactions(self,g_all_cupy,pp,N_pair):
        self.evaluate_gradients(g_all_cupy,N_pair)
        self.evaluate_forces_torques(pp)
        return self.forces,self.torks    

    def evaluate_gradients(self,g_all_cupy,N_pair):
        """
        Forward the generated descriptors through the trained neural-net
        """
        g_all = torch.from_dlpack(g_all_cupy)

        with torch.no_grad():
            e_all = self.energy_net(g_all)


        e0 = e_all[:N_pair]
        edx = e_all[N_pair:N_pair*2]
        edy = e_all[N_pair*2:N_pair*3]
        edz = e_all[N_pair*3:N_pair*4]

        edtetax = e_all[N_pair*4:N_pair*5]
        edtetay = e_all[N_pair*5:N_pair*6]
        edtetaz = e_all[N_pair*6:N_pair*7]

        edtetax2 = e_all[N_pair*7:N_pair*8]
        edtetay2 = e_all[N_pair*8:N_pair*9]
        edtetaz2 = e_all[N_pair*9:N_pair*10]

        self.gradient_dx = torch.zeros((N_pair,3),device=self.torch_device)
        self.gradient_dx[:,0] = -(edx-e0).squeeze(1) / self.dx
        self.gradient_dx[:,1] = -(edy-e0).squeeze(1) / self.dx
        self.gradient_dx[:,2] = -(edz-e0).squeeze(1) / self.dx

        self.gradient_dtetax = torch.zeros((N_pair,3),device=self.torch_device)
        self.gradient_dtetax[:,0] = -(edtetax-e0).squeeze(1) / self.dteta
        self.gradient_dtetax[:,1] = -(edtetay-e0).squeeze(1) / self.dteta
        self.gradient_dtetax[:,2] = -(edtetaz-e0).squeeze(1) / self.dteta

        self.gradient_dtetax2 = torch.zeros((N_pair,3),device=self.torch_device)
        self.gradient_dtetax2[:,0] = -(edtetax2-e0).squeeze(1) / self.dteta
        self.gradient_dtetax2[:,1] = -(edtetay2-e0).squeeze(1) / self.dteta
        self.gradient_dtetax2[:,2] = -(edtetaz2-e0).squeeze(1) / self.dteta
        self.e0 = e0



    def evaluate_forces_torques(self,pp):

        en_range = self.en_max - self.en_min

        forces_inter = self.gradient_dx*en_range
        torks_inter1 = self.gradient_dtetax*en_range
        torks_inter2 = self.gradient_dtetax2*en_range


        forces_inter[forces_inter>100.0] = 100.0
        torks_inter1[torks_inter1>100.0] = 100.0
        torks_inter2[torks_inter2>100.0] = 100.0

        forces_inter[forces_inter<-100.0] = -100.0
        torks_inter1[torks_inter1<-100.0] = -100.0
        torks_inter2[torks_inter2<-100.0] = -100.0

        self.t_true = cp.from_dlpack(torks_inter1[:,0]) 
        self.f_true = cp.from_dlpack(forces_inter[:,0]) 


        forces_net = torch.zeros((self.Nparticles,3),device=self.torch_device)
        torks_net = torch.zeros((self.Nparticles,3),device=self.torch_device)

        forces_net.index_add_(0, pp[:,0], forces_inter)
        forces_net.index_add_(0, pp[:,1], -forces_inter)

        torks_net.index_add_(0, pp[:,0], torks_inter1)
        torks_net.index_add_(0, pp[:,1], torks_inter2)

        self.torks = cp.from_dlpack(torks_net)

        self.forces = cp.from_dlpack(forces_net)


    ######## FUNCTIONS NEEDED FOR ANALYTICAL DERICATIVE ######## 

    def evaluate_interactions_analytical(self,g_all_cupy,dqdtetax,pp,N_pair,dqdx):
        self.evaluate_gradients_analytical(g_all_cupy)

        # print(self.dudq.shape)
        # print(dqdtetax.shape)
        # print(dqdx.shape)
        # exit()

        self.tork_x_analytical = cp.sum(self.dudq * dqdtetax, axis=1)
        self.force_x_analytical = cp.sum(self.dudq * dqdx, axis=1)
        self.net_interactions(pp)
        return  self.torks_analytical 

    def net_interactions(self,pp):

        print(self.t_true.shape)
        en_range = self.en_max - self.en_min

        self.tork_x_analytical = self.tork_x_analytical * -en_range
        self.force_x_analytical = self.force_x_analytical * -en_range
        diff = cp.abs(self.tork_x_analytical - self.t_true)
        diff2 = cp.abs(self.force_x_analytical - self.f_true)

        xx = np.linspace(cp.min(self.t_true).get(),cp.max(self.t_true).get(),100)
        print(cp.max(diff))
        print(cp.max(diff2))
        plt.figure(1)
        plt.plot(xx,xx,'k--')
        plt.scatter(cp.asnumpy(self.tork_x_analytical),cp.asnumpy(self.t_true))
        plt.scatter(cp.asnumpy(self.force_x_analytical),cp.asnumpy(self.f_true))
        plt.show()
        exit()


        exit()


        tork_x_analytical = torch.from_dlpack(self.tork_x_analytical)
        torks_net = torch.zeros((self.Nparticles),device=self.torch_device)
        torks_net.index_add_(0, pp[:,0], tork_x_analytical)
        torks_net.index_add_(0, pp[:,1], tork_x_analytical)
        self.torks_analytical = cp.from_dlpack(torks_net)


    def read_model_weights(self):

        self.weights = []
        self.biases  = []

        # 2a) First layer:
        W0 = self.energy_net.layer_first.weight.detach().cpu().numpy()  # shape: (nn_width, input_size)
        b0 = self.energy_net.layer_first.bias.detach().cpu().numpy()    # shape: (nn_width,)
        W0 = cp.asarray(W0)
        b0 = cp.asarray(b0)

        self.weights.append(W0)
        self.biases.append(b0)

        # 2b) Hidden layers (if any):
        for hidden_layer in self.energy_net.layer_hidden:
            W_h = hidden_layer.weight.detach().cpu().numpy()  # shape: (nn_width, nn_width)
            b_h = hidden_layer.bias.detach().cpu().numpy()    # shape: (nn_width,)
            # print(W_h.shape, b_h.shape)
            W_h = cp.asarray(W_h)
            b_h = cp.asarray(b_h)
            self.weights.append(W_h)
            self.biases.append(b_h)


        # 2c) Last layer:
        W_last = self.energy_net.layer_last.weight.detach().cpu().numpy()  # shape: (1, nn_width)
        b_last = self.energy_net.layer_last.bias.detach().cpu().numpy()    # shape: (1,)
        W_last = cp.asarray(W_last)
        b_last = cp.asarray(b_last)
        self.weights.append(W_last)
        self.biases.append(b_last)

    def evaluate_gradients_analytical(self,x):

        # ------------ F O R W A R D -------------------------------------------
        zs   = []        # pre-activations   (for derivative masks)
        masks = []       # ReLU derivative masks
        a = x

        for W, b in zip(self.weights[:-1], self.biases[:-1]):     # all hidden layers

            z = a @ W.T + b            # (B, n_k)
            a, m = relu(z)       # activation + derivative mask
            zs.append(z)
            masks.append(m)

        # Last (output) layer – linear
        y = (a @ self.weights[-1].T + self.biases[-1]).squeeze(-1)   # (B,)

        # ------------ B A C K W A R D  (∂f/∂x) -------------------------------
        # Start with gradient of output w.r.t. last hidden activation:
        # y = a_{L-1} · W_L^T + b_L     → dy/da = W_L
        g = cp.broadcast_to(self.weights[-1], (x.shape[0], self.weights[-1].shape[1]))  # (B, n_{L-1})

        # Propagate through hidden layers in reverse
        for W, m in zip(reversed(self.weights[:-1]), reversed(masks)):   # ↩︎ hidden
            # g = (g @ W) * m 
            g = (g * m) @ W         # (B, n_{k-1})

        self.dudq = g         # (B, in_dim)













def dummy():
    pass