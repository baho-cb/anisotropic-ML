import numpy as np
import sys,os
from scipy import stats
import time
import argparse
import re
import gc
import matplotlib.pyplot as plt
from scipy.special import chebyt
from scipy.special import eval_legendre
np.set_printoptions(precision=3, threshold=None, suppress=True)
import cupy as cp

def evaluate_chebyshev(x, N):
    """
    Evaluate Chebyshev polynomials of the first kind up to degree N at points x.

    Parameters:
    - x (cupy.ndarray): Array of points where polynomials are evaluated. Should be in the range [-1, 1].
    - N (int): Maximum degree of Chebyshev polynomials to evaluate.

    Returns:
    - T (cupy.ndarray): Array of shape (N+1, len(x)) where T[n, i] = T_n(x[i]).
    """
    x = x.flatten()
    # Initialize an array to hold the polynomial values
    T = np.zeros((N + 1, x.size), dtype=x.dtype)

    # T_0(x) = 1
    T[0, :] = 1.0

    if N >= 1:
        # T_1(x) = x
        T[1, :] = x

    # Use the recursive relation to compute higher-order polynomials
    for n in range(1, N):
        T[n + 1, :] = 2 * x * T[n, :] - T[n - 1, :]

    return T

def evaluate_chebyshev_gpu(x, N):
    """
    Evaluate Chebyshev polynomials of the first kind up to degree N at points x.

    Parameters:
    - x (cupy.ndarray): Array of points where polynomials are evaluated. Should be in the range [-1, 1].
    - N (int): Maximum degree of Chebyshev polynomials to evaluate.

    Returns:
    - T (cupy.ndarray): Array of shape (N+1, len(x)) where T[n, i] = T_n(x[i]).
    """
    x = x.flatten()
    # Initialize an array to hold the polynomial values
    T = cp.zeros((N + 1, x.size), dtype=x.dtype)

    # T_0(x) = 1
    T[0, :] = 1.0

    if N >= 1:
        # T_1(x) = x
        T[1, :] = x

    # Use the recursive relation to compute higher-order polynomials
    for n in range(1, N):
        T[n + 1, :] = 2 * x * T[n, :] - T[n - 1, :]

    return T


def evaluate_legendre(x, N):
    """
    Evaluate Legendre polynomials up to degree N at points x.

    Parameters:
    - x (cupy.ndarray): Array of points where polynomials are evaluated. Should be in the range [-1, 1].
    - N (int): Maximum degree of Legendre polynomials to evaluate.

    Returns:
    - P (cupy.ndarray): Array of shape (N+1, len(x)) where P[n, i] = P_n(x[i]).
    """
    # Initialize an array to hold the polynomial values
    x = x.flatten()
    P = np.zeros((N + 1, x.size), dtype=x.dtype)

    # P_0(x) = 1
    P[0, :] = 1.0

    if N >= 1:
        # P_1(x) = x
        P[1, :] = x

    # Use the recursive relation to compute higher-order polynomials
    for n in range(1, N):
        P[n + 1, :] = ((2 * n + 1) * x * P[n, :] - n * P[n - 1, :]) / (n + 1)

    return P

def evaluate_legendre_gpu(x, N):
    """
    Evaluate Legendre polynomials up to degree N at points x.

    Parameters:
    - x (cupy.ndarray): Array of points where polynomials are evaluated. Should be in the range [-1, 1].
    - N (int): Maximum degree of Legendre polynomials to evaluate.

    Returns:
    - P (cupy.ndarray): Array of shape (N+1, len(x)) where P[n, i] = P_n(x[i]).
    """
    # Initialize an array to hold the polynomial values
    x = x.flatten()
    P = cp.zeros((N + 1, x.size), dtype=x.dtype)

    # P_0(x) = 1
    P[0, :] = 1.0

    if N >= 1:
        # P_1(x) = x
        P[1, :] = x

    # Use the recursive relation to compute higher-order polynomials
    for n in range(1, N):
        P[n + 1, :] = ((2 * n + 1) * x * P[n, :] - n * P[n - 1, :]) / (n + 1)

    return P


class NEP():
    """
    type 0 : center is CofM
    type 1 : center is CofM of cube 1 (i.e. origin)
    type 2 : center is CofM of cube 2
    """

    def __init__(self,pos,type):
        Np = len(pos)
        self.pos = pos
        self.r_ij = pos
        if(type==0):
            centraal = np.average(self.pos,axis=1)
            self.r_ij = self.pos - centraal[:,np.newaxis,:]
        if(type==2):
            centraal = np.average(self.pos[:,self.npts:],axis=1)
            self.r_ij = self.pos - centraal[:,np.newaxis,:]

    def set_npts(self,n):
        self.npts = n

    def set_hypers(self,hypers):
        self.nrad = hypers[0]
        self.nang = hypers[1]
        self.lmax = hypers[2]
        self.cutoff = hypers[3]

    def calculate_radial_fast_gpu(self):
        Np = len(self.pos)
        self.pos_gpu = cp.asarray(self.pos)
        self.r_ij_gpu = cp.asarray(self.r_ij)
        rij = cp.linalg.norm(self.r_ij_gpu,axis=-1)
        rij_rc = rij/self.cutoff
        fc = 0.5*(1. + cp.cos(np.pi*cp.copy(rij_rc)))
        x = 2.0*(rij_rc - 1.)**2 - 1.

        dat = rij.get()
        dat = dat.flatten()
        fc_flat = fc.get()
        fc_flat = fc_flat.flatten()


        plt.figure(1)
        # plt.hist(dat,bins=100)
        plt.scatter(dat,fc_flat)
        plt.show()
        exit()



        chebo_test = evaluate_chebyshev_gpu(x, self.nrad)
        chebo_test = chebo_test.reshape((self.nrad+1,Np,self.npts*2))
        chebo_test = (chebo_test+1)/2.
        chebo_test *= fc[cp.newaxis]
        self.chebo = cp.copy(chebo_test)
        chebo_test = cp.sum(chebo_test,axis=-1)
        chebo_test = chebo_test.T
        chebo_test = chebo_test.get()
        self.g_rad_fast_gpu = chebo_test.astype(np.float32)


    def calculate_angular_fast_gpu(self):
        chebo_test = self.chebo[:self.nang+1]

        Np = len(self.pos)
        r_ij = cp.asarray(self.r_ij)

        R_ij = np.tile(r_ij,(1,self.npts*2,1))
        R_ik = np.repeat(r_ij,self.npts*2,axis=1)

        norm_rij = cp.linalg.norm(R_ij,axis=-1)
        norm_rik = cp.linalg.norm(R_ik,axis=-1)

        cosine = cp.sum(R_ij*R_ik,axis=2)/(norm_rij*norm_rik)

        reso = (chebo_test[:, :, :, cp.newaxis] * chebo_test[:, :, cp.newaxis, :])
        reso = reso.reshape(chebo_test.shape[0], chebo_test.shape[1], -1)

        nnnn = (self.npts*2)**2
        lego2 = evaluate_legendre_gpu(cosine,self.lmax)
        lego2 = lego2.reshape((-1,Np,nnnn))
        lego2 = lego2[1:]

        product = cp.einsum('ibk,jbk->ijb', reso, lego2)
        product = product.reshape((self.nang+1)*self.lmax,Np)
        self.g_ang_fast_gpu = product.T
        self.g_ang_fast_gpu = self.g_ang_fast_gpu.get()
        self.g_ang_fast_gpu = self.g_ang_fast_gpu.astype(np.float32)


    def calculate_radial(self):
        Np = len(self.pos)
        rij = np.linalg.norm(self.r_ij,axis=-1)
        rij_rc = rij/self.cutoff
        fc = 0.5*(1. + np.cos(np.pi*np.copy(rij_rc)))
        x = 2.0*(rij_rc - 1.)**2 - 1.

        self.g_rad = np.zeros((Np,self.nrad+1))
        for i in range(self.nrad+1):
            chebo = chebyt(i)(x)
            chebo = (chebo+1)/2.
            chebo = chebo*fc
            chebo = np.sum(chebo,axis=-1)
            self.g_rad[:,i] = np.copy(chebo)

        # x = np.linspace(0,1,1000)
        # x1 = 2.0*(x - 1.)**2 - 1.
        # for i in range(4):
        #     chebo = (chebo+1)/2.
        #     chebo = chebo*fc
        #     plt.figure(1)
        #     plt.plot(x,chebo,label=i)
        # plt.show()
        # exit()

    def calculate_angular(self):
        Np = len(self.pos)
        r_ij = self.r_ij

        R_ij = np.tile(r_ij,(1,self.npts*2,1))
        R_ik = np.repeat(r_ij,self.npts*2,axis=1)

        norm_rij = np.linalg.norm(R_ij,axis=-1)
        norm_rik = np.linalg.norm(R_ik,axis=-1)
        cosine = np.sum(R_ij*R_ik,axis=2)/(norm_rij*norm_rik)
        r_ij_rc = norm_rij/self.cutoff
        r_ik_rc = norm_rik/self.cutoff

        x_ij = 2.*(r_ij_rc-1.0)**2 - 1.
        x_ik = 2.*(r_ik_rc-1.0)**2 - 1.
        fc_rij = 0.5*(1.+np.cos(np.pi*np.copy(r_ij_rc)))
        fc_rik = 0.5*(1.+np.cos(np.pi*np.copy(r_ik_rc)))

    

        Ndim = (self.nang+1)*self.lmax
        self.g_ang = np.zeros((Np,Ndim))
        i_col = 0
        for n in range(self.nang+1):
            g_ij = chebyt(n)(x_ij)
            g_ij = (g_ij+1)/2.
            g_ij = g_ij*fc_rij

            g_ik = chebyt(n)(x_ik)
            g_ik = (g_ik+1)/2.
            g_ik = g_ik*fc_rik

            if(n==0):
                print(g_ik.shape)
                print(g_ik*g_ij)

            for l in range(1,self.lmax+1):
                lego = eval_legendre(l, cosine)
                res = g_ij*g_ik*lego
                res = np.sum(res,axis=-1)
                self.g_ang[:,i_col] = res
                i_col += 1

    def get_g(self):
        g = np.hstack((self.g_rad,self.g_ang))
        return g

    def get_g_fast(self):
        self.device = cp.cuda.Device(1)
        self.device.use()
        self.calculate_radial_fast_gpu()
        self.calculate_angular_fast_gpu()
        g = np.hstack((self.g_rad_fast_gpu,self.g_ang_fast_gpu))
        return g
