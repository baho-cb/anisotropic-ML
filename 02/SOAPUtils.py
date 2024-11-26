import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,LinearRegression,Lasso,BayesianRidge
from dscribe.descriptors import ACSF,SOAP
from ase import Atoms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

np.set_printoptions(suppress=True)

class CentralSOAP():
    """
    Class to generate SOAP descriptors for pair configurations in pts12 format
    Performs LinearR, kernelR and GPR and tests
    """

    def __init__(self,pos,hypers):
        self.pos = pos

        self.r_cut = hypers[0]
        self.n_max = hypers[1]
        self.l_max = hypers[2]
        self.sigma = hypers[3]
        self.poly_order = hypers[4]

        self.soap_data_train = self.generate_soap_dscribe(pos)


    def generate_soap_dscribe(self,pos):
        """
        use dscribe package to generate the SOAP descriptors
        https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html

        positions must be supplied as groups of atoms with ase package
        """
        N_data = pos.shape[0]

        soap = SOAP(
            species=["H"],
            periodic=False,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
        )

        centraal = np.average(pos[:,:],axis=1)
        _pos = np.zeros((N_data,13,3),dtype=np.float32)
        _pos[:,1:,:] = pos
        _pos[:,0,:] = centraal
        pos = _pos

        configs = []
        symbols = ['H'] * 13
        for idx, positions in enumerate(pos):
            configs.append(Atoms(symbols=symbols, positions=positions))


        centers = [[0]]*N_data
        soap_data = soap.create(configs,centers=centers)
        soap_data = soap_data.astype(np.float32)
        soap_data = soap_data[:, 0, :]

        if(self.poly_order==1):
            pass
        elif(self.poly_order==2):
            soap_data = np.hstack((soap_data,soap_data**2))
        elif(self.poly_order==3):
            soap_data = np.hstack((soap_data,soap_data**2,soap_data**3))
        else:
            print('Polynomial augmentation is implemented up to 3rd order')
            exit()

        return soap_data

    def calculate_kernel_matrix(self):
        kernel_matrix = np.dot(self.soap_data_train, self.soap_data_train.T)
        self.norms_train = np.linalg.norm(self.soap_data_train, axis=1)
        norms_outer = np.outer(self.norms_train, self.norms_train)
        self.kernel_matrix = kernel_matrix / norms_outer


    def normalize_x_data(self):

        x_min = np.min(self.soap_data_train,axis=0)
        x_max = np.max(self.soap_data_train,axis=0)
        self.soap_data_train = (self.soap_data_train - x_min) / (x_max - x_min)
        self.x_min = x_min
        self.x_max = x_max

    ######## TRAINING METHODS ############

    def KernelRidgeRegression(self,target,regul):
        self.calculate_kernel_matrix()
        N = self.kernel_matrix.shape[0]
        I = np.eye(N)
        K_reg = self.kernel_matrix + regul * I
        self.krr_coefs = np.linalg.solve(K_reg, target)

    def LinearRegression(self,target,regul):
        self.normalize_x_data()

        self.ridge_reg =  Ridge(alpha=regul)
        # self.ridge_reg =  LinearRegression(fit_intercept=False)
        self.ridge_reg.fit(self.soap_data_train, target)

    def GaussianProcessRegression(self,target):
        self.normalize_x_data()
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2,1e2))

        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=42,
            copy_X_train=False, n_restarts_optimizer=10,
            n_targets=1)

        self.gpr.fit(self.soap_data_train, target)


    ######## TESTING METHODS ############
    """
    Mean Absolute Error is used for all methods
    """

    def infer_GPR(self,x_test,y_test):
        soap_test = self.generate_soap_dscribe(x_test)
        soap_test = (soap_test - self.x_min) / (self.x_max - self.x_min)
        y_pred, sigma_pool = self.gpr.predict(soap_test, return_std=True)
        return np.mean(np.abs(y_pred-y_test))

    def infer_LR(self,x_test,y_test):
        soap_test = self.generate_soap_dscribe(x_test)
        soap_test = (soap_test - self.x_min) / (self.x_max - self.x_min)
        y_pred = self.ridge_reg.predict(soap_test)
        return np.mean(np.abs(y_pred-y_test))

    def infer_KRR(self,x_test,y_test):
        soap_test = self.generate_soap_dscribe(x_test)
        infer_dot = soap_test @ self.soap_data_train.T
        norms_test = np.linalg.norm(soap_test, axis=1)
        K_test = infer_dot / np.outer(norms_test, self.norms_train)
        y_pred = K_test @ self.krr_coefs

        return np.mean(np.abs(y_pred-y_test))


    def normalize_target(self,t):
        t[t>self.en_max] = self.en_max
        t = (t - self.en_min) / (en_max - self.en_min)

        t[t>1.0] = 1.0
        t[t<0.0] = 0.0
        return t

    def denormalize_target(self,t):
        t = t*(self.en_max - self.en_min) + self.en_min
        return t





def dummy():
    pass
