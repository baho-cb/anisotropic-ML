from DescriptorGenerator import DescriptorGenerator
import cupy as cp
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from cudakernels.NepDescriptorKernels import angular_cuda_kernel, radial_cuda_kernel
import cudakernels.PosToPtsKernels as ptp
from cudakernels.SingleKernel import single_kernel
from cudakernels.SingleKernelDebug import single_kernel_debug
import sys
import cudakernels.DerivativeKernels as cuda_dk
import MathUtils as mu

"""
possible simplifications: 
for dpdteta_xyz instead of 3 seperate arrays you just need 6 columns of data 2 per dx
Similarly instead of storing dp1dtetax and dp2dtetax separately you can keep d12dtetax 
"""

np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)

def _dcosdteta(p, r, dp, dr):
    Np = p.shape[0]
    P_i  = p[:, :, None, :]    # shape → (100, 12, 1, 3)
    P_j  = p[:, None, :, :]    # shape → (100, 1, 12, 3)
    dP_i = dp[:, :, None, :]     # shape → (100, 12, 1, 3)
    dP_j = dp[:, None, :, :]     # shape → (100, 1, 12, 3)

    R_i  = r[:, :, None]       # shape → (100, 12, 1)
    R_j  = r[:, None, :]       # shape → (100, 1, 12)
    dR_i = dr[:, :, None]        # shape → (100, 12, 1)
    dR_j = dr[:, None, :]        # shape → (100, 1, 12)

    dot_p = cp.sum(P_i * P_j, axis=-1)      # shape → (100, 12, 12)
    dot_dp_p = cp.sum(dP_i * P_j, axis=-1)   # shape → (100, 12, 12)
    dot_p_dp = cp.sum(P_i * dP_j, axis=-1)   # shape → (100, 12, 12)
    num1 = dot_dp_p + dot_p_dp              # shape → (100, 12, 12)

    den = R_i * R_j                         # r_i * r_j, shape → (100, 12, 12)
    den2 = den * den                        # (r_i * r_j)^2, shape → (100, 12, 12)
    num2 = dot_p * (dR_i * R_j + R_i * dR_j)  # shape → (100, 12, 12)

    dcosine_dteta = (num1 * den - num2) / den2       # shape → (100, 12, 12)
    dcosine_dteta = dcosine_dteta.reshape(Np,-1)    
    return dcosine_dteta

def _dqdteta(drdp, dpdteta, dgdr, one_or_two):

    inner_x = cp.sum(drdp * dpdteta, axis=2)  # (100, 6)
    if(one_or_two == 1):
        dgraddtetax = dgdr[:,:,:6] * inner_x[cp.newaxis, :, :]  # → (3, 100)
    elif(one_or_two == 2):
        dgraddtetax = dgdr[:,:,6:] * inner_x[cp.newaxis, :, :]
    else:
        raise ValueError("one_or_two must be 1 or 2")    
    return cp.sum(dgraddtetax,  axis=2)

def _dgdteta(drdp, dpdteta, dgdr, one_or_two):

    inner_x = cp.sum(drdp * dpdteta, axis=2)  # (100, 6)
    if(one_or_two == 1):
        dgraddtetax = dgdr[:,:,:6] * inner_x[cp.newaxis, :, :]  # → (3, 100)
    elif(one_or_two == 2):
        dgraddtetax = dgdr[:,:,6:] * inner_x[cp.newaxis, :, :]
    else:
        raise ValueError("one_or_two must be 1 or 2")    
    return dgraddtetax



class DescriptorGeneratorAnalytical(DescriptorGenerator):
    def generate_nep_descriptors_derivatives(self,central_pos,orientations,Nlist):
        print('generating descriptors and derivatives')
        self.calculate_pts(central_pos,orientations,Nlist)
        
        self.calculate_derivatives()
        self.calculate_dx_derivatives()
        self.merge_derivatives()

        return self.dqalldteta_list, self.pp, self.N_pair

    def calculate_pts(self,central_pos,orientations,Nlist):
        translate = central_pos[Nlist[:,1]]-central_pos[Nlist[:,0]]
        translate = cp.where(translate > 0.5 * self.Lx, translate- self.Lx, translate)
        translate = cp.where(translate <- 0.5 * self.Lx, self.Lx + translate, translate)
        dist = cp.linalg.norm(translate,axis=1)
        mask = cp.where(dist < self.cutoff)[0]


        self.pairs = Nlist[mask]
        pair0 = self.pairs[:,0]
        pair1 = self.pairs[:,1]
        self.pp = torch.from_dlpack(self.pairs)

        translate = translate[mask]
        N_pair = len(self.pairs)
        self.N_pair = N_pair

        QUAT1 = orientations[pair0]
        QUAT2 = orientations[pair1]

        self.quat1 = QUAT1
        self.quat2 = QUAT2

        # blocks = (self.N_pair,)
        # threads_per_block = (32,)

        # self._P = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        # self._p1dtetax = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        # self._p1dtetay = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        # self._p1dtetaz = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        # self._p2dtetax = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        # self._p2dtetay = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        # self._p2dtetaz = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)

        # cuda_dk.dpts_kernel(
        #     blocks,
        #     threads_per_block,
        #     (
        #         QUAT1,
        #         QUAT2,
        #         translate,
        #         self.pts_rep,
        #         self._P,
        #         self._p1dtetax,
        #         self._p1dtetay,
        #         self._p1dtetaz,
        #         self._p2dtetax,
        #         self._p2dtetay,
        #         self._p2dtetaz,
        #         cp.int32(self.N_pair),
        #         cp.int32(self.Nd)
        #     )
        # )


        # print(self._p1dtetax[0])
        # print(self._p1dtetay[0])
        # print(self._p1dtetaz[0])
        # exit()

        self.pts_pair = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        blocks = (self.N_pair,)
        threads_per_block = (self.Nd*3,)
    
        ptp.get_pts_pairs_kernels1(
            blocks,
            threads_per_block,
            (
                QUAT1,
                QUAT2,
                translate,
                self.pts_rep,
                self.pts_pair,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

    def calculate_derivatives(self):
        self.calculate_dpdteta()
        self.calculate_dcosdteta()    
        self.calculate_cosine()
        self.calculate_T()
        self.calculate_dgdr()
        self.calculate_dqang_dteta()
        # self.dqdtetax = cp.concatenate((self.dqraddtetax,self.dqangdtetax),axis=0)
        # self.dqdtetax = self.dqdtetax.T    

    def calculate_dpdteta(self):
        self.P = self.pts_pair - np.average(self.pts_pair,axis=1)[:,np.newaxis,:]
        self.r = cp.linalg.norm(self.P,axis=-1)
        d = self.quat1[:,1:] @ self.pts_rep.T # (N_pair,Nd)
        k = self.quat1[:,0]**2 - cp.sum(self.quat1[:,1:]**2,axis=1) # (N_pair,)
        k = k[:,None]
        self.p1dtetax = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self.p1dtetay = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self.p1dtetaz = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)

        self.p1dtetax[:,:,1] = (
            - self.pts_rep[:,2] * k
            + 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,2,None] - self.pts_rep[:,1] * self.quat1[:,1,None] )
            - 2.0 * self.quat1[:,3,None] * d
        )  # shape (N_pair,6)
        self.p1dtetax[:,:,2] = (
            + self.pts_rep[:,1] * k
            + 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,3,None] - self.pts_rep[:,2] * self.quat1[:,1,None] )
            + 2.0 * self.quat1[:,2,None] * d
        )  # shape (N_pair,6)


        self.p1dtetay[:,:,0] = (
            + self.pts_rep[:,2] * k
            - 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,2,None] - self.pts_rep[:,1] * self.quat1[:,1,None] )
            + 2.0 * self.quat1[:,3,None] * d
        )  # shape (N_pair,6)

        self.p1dtetay[:,:,2] = (
            - self.pts_rep[:,0] * k
            + 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,1] * self.quat1[:,3,None] - self.pts_rep[:,2] * self.quat1[:,2,None] ) 
            - 2.0 * self.quat1[:,1,None] * d
        )  # shape (N_pair,6) ## TO DO this formula ??? 


        self.p1dtetaz[:,:,0] = (
            - self.pts_rep[:,1] * k
            - 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,3,None] - self.pts_rep[:,2] * self.quat1[:,1,None] )
            - 2.0 * self.quat1[:,2,None] * d
        )  # shape (N_pair,6)

        self.p1dtetaz[:,:,1] = (
            + self.pts_rep[:,0] * k
            - 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,1] * self.quat1[:,3,None] - self.pts_rep[:,2] * self.quat1[:,2,None] ) 
            + 2.0 * self.quat1[:,1,None] * d
        )  # shape (N_pair,6)  

        ################## P2/DTETAX ####################
        d = self.quat2[:,1:] @ self.pts_rep.T # (N_pair,Nd)
        k = self.quat2[:,0]**2 - cp.sum(self.quat2[:,1:]**2,axis=1) # (N_pair,)
        k = k[:,None]

        self.p2dtetax = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self.p2dtetay = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self.p2dtetaz = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)


        self.p2dtetax[:,:,1] = (
            - self.pts_rep[:,2] * k
            + 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,0] * self.quat2[:,2,None] - self.pts_rep[:,1] * self.quat2[:,1,None] )
            - 2.0 * self.quat2[:,3,None] * d
        )  # shape (N_pair,6)
        self.p2dtetax[:,:,2] = (
            + self.pts_rep[:,1] * k
            + 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,0] * self.quat2[:,3,None] - self.pts_rep[:,2] * self.quat2[:,1,None] )
            + 2.0 * self.quat2[:,2,None] * d
        )  # shape (N_pair,6)


        self.p2dtetay[:,:,0] = (
            + self.pts_rep[:,2] * k
            - 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,0] * self.quat2[:,2,None] - self.pts_rep[:,1] * self.quat2[:,1,None] )
            + 2.0 * self.quat2[:,3,None] * d
        )  # shape (N_pair,6)

        self.p2dtetay[:,:,2] = (
            - self.pts_rep[:,0] * k
            + 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,1] * self.quat2[:,3,None] - self.pts_rep[:,2] * self.quat2[:,2,None] ) 
            - 2.0 * self.quat2[:,1,None] * d
        )  # shape (N_pair,6) ## TO DO this formula ??? 


        self.p2dtetaz[:,:,0] = (
            - self.pts_rep[:,1] * k
            - 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,0] * self.quat2[:,3,None] - self.pts_rep[:,2] * self.quat2[:,1,None] )
            - 2.0 * self.quat2[:,2,None] * d
        )  # shape (N_pair,6)

        self.p2dtetaz[:,:,1] = (
            + self.pts_rep[:,0] * k
            - 2.0 * self.quat2[:,0,None] * (self.pts_rep[:,1] * self.quat2[:,3,None] - self.pts_rep[:,2] * self.quat2[:,2,None] ) 
            + 2.0 * self.quat2[:,1,None] * d
        )  # shape (N_pair,6)  

        # print(self.P[0])
        # print(self.p2dtetax[0])
        # print(self.p2dtetay[0])
        # print(self.p2dtetaz[0])
        # exit()

        # print(mu.maxerr(self.P, self._P))
        # print(mu.maxerr(self.p1dtetax, self._p1dtetax))
        # print(mu.maxerr(self.p1dtetay, self._p1dtetay))
        # print(mu.maxerr(self.p1dtetaz, self._p1dtetaz))
        # # print((self.p2dtetax[0]))
        # # print((self._p2dtetax[0]))


        # print(mu.maxerr(self.p2dtetax, self._p2dtetax))
        # print(mu.maxerr(self.p2dtetay, self._p2dtetay))
        # print(mu.maxerr(self.p2dtetaz, self._p2dtetaz))

        # exit()




    def calculate_dcosdteta(self):
        self.dcosinedteta_list = []

        dp12dtetax = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp12dtetax[:,0:self.Nd//2,:] = self.p1dtetax
        self.dp12dtetax = dp12dtetax
        self.dr12dtetax = cp.sum(self.P * dp12dtetax, axis=2)/self.r

        dp12dtetay = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp12dtetay[:,0:self.Nd//2,:] = self.p1dtetay
        self.dp12dtetay = dp12dtetay
        self.dr12dtetay = cp.sum(self.P * dp12dtetay, axis=2)/self.r

        dp12dtetaz = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp12dtetaz[:,0:self.Nd//2,:] = self.p1dtetaz
        self.dp12dtetaz = dp12dtetaz
        self.dr12dtetaz = cp.sum(self.P * dp12dtetaz, axis=2)/self.r
        
        
        dp21dtetax = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp21dtetax[:,self.Nd//2:,:] = self.p2dtetax
        self.dp21dtetax = dp21dtetax
        self.dr21dtetax = cp.sum(self.P * dp21dtetax, axis=2)/self.r

        dp21dtetay = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp21dtetay[:,self.Nd//2:,:] = self.p2dtetay
        self.dp21dtetay = dp21dtetay
        self.dr21dtetay = cp.sum(self.P * dp21dtetay, axis=2)/self.r

        dp21dtetaz = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dp21dtetaz[:,self.Nd//2:,:] = self.p2dtetaz
        self.dp21dtetaz = dp21dtetaz
        self.dr21dtetaz = cp.sum(self.P * dp21dtetaz, axis=2)/self.r
        
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetax, self.dr12dtetax))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetay, self.dr12dtetay))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetaz, self.dr12dtetaz))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetax, self.dr21dtetax))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetay, self.dr21dtetay))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetaz, self.dr21dtetaz))

        # print(len(self.dcosinedteta_list))
        # print(self.dcosinedteta_list[0].shape)
        # exit()
        # blocks = (self.N_pair,)
        # threads_per_block = (self.Nd*self.Nd,)


        # cuda_dk.cosine_kernel(
        #     blocks,
        #     threads_per_block,
        #     (
        #         self._P,
        #         self._p1dtetax,
        #         self._p1dtetay,
        #         self._p1dtetaz,
        #         self._p2dtetax,
        #         self._p2dtetay,
        #         self._p2dtetaz,
        #         cp.int32(self.N_pair),
        #         cp.int32(self.Nd)
        #     )
        # )

        

    def calculate_cosine(self):
        self.P = self.pts_pair - np.average(self.pts_pair,axis=1)[:,np.newaxis,:]

        self.r = cp.linalg.norm(self.P,axis=-1)
        R_ij = cp.tile(self.P,(1,6*2,1))
        R_ik = cp.repeat(self.P,6*2,axis=1)

        norm_rij = cp.linalg.norm(R_ij,axis=-1)
        norm_rik = cp.linalg.norm(R_ik,axis=-1)

        self.cosine = cp.sum(R_ij*R_ik,axis=2)/(norm_rij*norm_rik)

        self.dlegdcos = cp.zeros((self.lmax, self.cosine.shape[0], self.cosine.shape[1]), dtype=cp.float32)
        self.dlegdcos[0] = 1
        self.dlegdcos[1] = 3*self.cosine
        self.dlegdcos[2] = (15*self.cosine**2 - 3) / 2.0

        self.leg = cp.zeros((self.lmax, self.cosine.shape[0], self.cosine.shape[1]), dtype=cp.float32)
        self.leg[0] = self.cosine
        self.leg[1] = (3*self.cosine**2 - 1)*0.5
        self.leg[2] = (5*self.cosine**3 - 3*self.cosine)/2.0

        for i in range(3, self.lmax):
            self.leg[i] = (
                (2 * i + 1) * self.cosine * self.leg[i - 1]
                - i * self.leg[i - 2]
                ) / (i + 1)

        for i in range(3, self.lmax):
            n = i + 1 
            self.dlegdcos[i] = n*self.leg[i-1] + self.cosine*self.dlegdcos[i-1]


        self.dlegdteta_list = []

        for i in range(6):
            self.dlegdteta_list.append(self.dlegdcos * self.dcosinedteta_list[i][cp.newaxis, :, :])


        # self.dpl_dtetax = self.dpldcosine * self.dcosine_dtetax[cp.newaxis, :, :]    


    def calculate_T(self):
        self.z = 2.0*(self.r/self.cutoff_nep - 1.0)**2 - 1.0    
      

        self.T = cp.zeros((self.nrad + 1,
                        self.z.shape[0],
                        self.z.shape[1]),
                        dtype=cp.float32)

        self.T[0, :, :] = 1.0             # T₀(z) = 1
        if self.nrad >= 1:
            self.T[1, :, :] = self.z      # T₁(z) = z

        for n in range(2, self.nrad + 1):
            self.T[n, :, :] = 2.0 * self.z * self.T[n - 1, :, :] \
                            - self.T[n - 2, :, :]    


        self.dTdz = cp.zeros((self.nrad + 1,
                            self.z.shape[0],
                            self.z.shape[1]),
                            dtype=cp.float32)

        # Base cases:
        #   T₀(z) = 1      ⇒  dT₀/dz = 0
        #   T₁(z) =  z     ⇒  dT₁/dz = 1
        self.dTdz[0, :, :] = 0.0
        if self.nrad >= 1:
            self.dTdz[1, :, :] = 1.0

        # Now use the recurrence for n = 2..n_rad:
        #   dTₙ/dz = 2·Tₙ₋₁ + 2·z·(dTₙ₋₁/dz) − (dTₙ₋₂/dz)
        for n in range(2, self.nrad + 1):
            # 2 · z · (dTₙ₋₁ / dz)
            term_z_dT_prev = 2.0 * self.z * self.dTdz[n - 1, :, :]

            # 2 · Tₙ₋₁
            term_T_prev = 2.0 * self.T[n - 1, :, :]

            # subtract dTₙ₋₂/dz
            term_dT_prev2 = self.dTdz[n - 2, :, :]

            self.dTdz[n, :, :] = term_T_prev + term_z_dT_prev - term_dT_prev2

        # self.g_rad = cp.zeros((self.nrad+1, self.z.shape[0], self.z.shape[1]), dtype=cp.float32)
        self.f_rcut = 0.5*(1.0 + cp.cos(np.pi*(self.r/self.cutoff_nep)))
        self.g_rad = ((self.T + 1.0) / 2.0 ) * self.f_rcut      

    def calculate_dgdr(self):
        r_rc = self.r/ self.cutoff_nep
        factor1 = (4.0 / self.cutoff_nep) * (r_rc - 1.0) * (1.0 + cp.cos(np.pi * r_rc))   # shape (6, N)
        factor2 = -cp.sin(np.pi * r_rc) * (np.pi / self.cutoff_nep)                      # shape (6, N)
        factor1 = factor1[cp.newaxis, :, :]  # → (1, 6, N)
        factor2 = factor2[cp.newaxis, :, :]  # → (1, 6, N)
                        
        self.dgdr = 0.25 * (
            self.dTdz * factor1
            + (self.T + 1.0) * factor2
        )    

        self.drdp = self.P/self.r[:, :, np.newaxis] 
        self.dr1dp1 = self.drdp[:,:6].copy()
        self.dr2dp2 = self.drdp[:,6:].copy()

        self.dqdteta_list = []
        self.dqdteta_list.append(_dqdteta(self.dr1dp1, self.p1dtetax, self.dgdr, 1))
        self.dqdteta_list.append(_dqdteta(self.dr1dp1, self.p1dtetay, self.dgdr, 1))
        self.dqdteta_list.append(_dqdteta(self.dr1dp1, self.p1dtetaz, self.dgdr, 1))

        self.dqdteta_list.append(_dqdteta(self.dr2dp2, self.p2dtetax, self.dgdr, 2))
        self.dqdteta_list.append(_dqdteta(self.dr2dp2, self.p2dtetay, self.dgdr, 2))
        self.dqdteta_list.append(_dqdteta(self.dr2dp2, self.p2dtetaz, self.dgdr, 2))

        self.dgdteta_list = []
        self.dgdteta_list.append(_dgdteta(self.dr1dp1, self.p1dtetax, self.dgdr, 1))
        self.dgdteta_list.append(_dgdteta(self.dr1dp1, self.p1dtetay, self.dgdr, 1))
        self.dgdteta_list.append(_dgdteta(self.dr1dp1, self.p1dtetaz, self.dgdr, 1))

        self.dgdteta_list.append(_dgdteta(self.dr2dp2, self.p2dtetax, self.dgdr, 2))
        self.dgdteta_list.append(_dgdteta(self.dr2dp2, self.p2dtetay, self.dgdr, 2))
        self.dgdteta_list.append(_dgdteta(self.dr2dp2, self.p2dtetaz, self.dgdr, 2))

    def calculate_dqang_dteta(self):

        gij = cp.tile(self.g_rad,(1,1,12))
        gik = cp.repeat(self.g_rad,12,axis=-1)
        n_desc_ang = (self.nang+1) * self.lmax
        self.dqangdteta_list = []

        for i in range(6):
            dgdteta = self.dgdteta_list[i]
            dgdteta12 = cp.zeros((dgdteta.shape[0],dgdteta.shape[1],dgdteta.shape[2]*2))
            if(i <3):
                dgdteta12[:, :, :dgdteta.shape[2]] = dgdteta
            else:
                dgdteta12[:, :, dgdteta.shape[2]:] = dgdteta

            dgdtetaij = cp.tile(dgdteta12,(1,1,12))
            dgdtetaik = cp.repeat(dgdteta12,12,axis=-1)


            dg_ang_dteta = cp.zeros((n_desc_ang,self.N_pair,144), dtype=cp.float32)
            for n in range(self.nang+1):
                for l in range(self.lmax):
                    term1 = dgdtetaij[n]*gik[n]*self.leg[l]
                    term2 = dgdtetaik[n]*gij[n]*self.leg[l]
                    term3 = gij[n]*gik[n]*self.dlegdteta_list[i][l]
                    dg_ang_dteta[n*self.lmax + l] = term1 + term2 + term3

            self.dqangdteta_list.append(cp.sum(dg_ang_dteta, axis=-1))


    def merge_derivatives(self):
        self.dqalldteta_list = []

        for i in range(6):
            dat = cp.concatenate((self.dqdteta_list[i], self.dqangdteta_list[i]), axis=0)
            dat = dat.T
            self.dqalldteta_list.append(dat)




################ FOR FORCE ################# 

    def calculate_dx_derivatives(self):
        self.calculate_drdx()
        self.calculate_dcosine_dx()
        self.calculate_dqang_dx()


    def calculate_drdx(self):

        # dpxdx = +-1/2, dpydy = +- 1/2     
        self.drdx = (self.P[:,:,0]/ self.r)*(+0.5)
        self.drdx[:,6:] *= -1.0
        self.dgdx = self.dgdr* self.drdx[np.newaxis, :, :]  # shape (3, N_pair, 3)
        self.dqdx = cp.sum(self.dgdx, axis=2)
        self.dqdx = self.dqdx.T
        self.dpdx = cp.zeros((self.N_pair, 6*2, 3), dtype=cp.float32)
        self.dpdx[:, :6, 0] = +0.5
        self.dpdx[:, 6:, 0] = -0.5

    def calculate_dcosine_dx(self):
        P_i  = self.P[:, :, None, :]    # shape → (100, 12, 1, 3)
        P_j  = self.P[:, None, :, :]    # shape → (100, 1, 12, 3)
        dP_i = self.dpdx[:, :, None, :]     # shape → (100, 12, 1, 3)
        dP_j = self.dpdx[:, None, :, :]     # shape → (100, 1, 12, 3)

        R_i  = self.r[:, :, None]       # shape → (100, 12, 1)
        R_j  = self.r[:, None, :]       # shape → (100, 1, 12)
        dR_i = self.drdx[:, :, None]        # shape → (100, 12, 1)
        dR_j = self.drdx[:, None, :]        # shape → (100, 1, 12)

        dot_p = cp.sum(P_i * P_j, axis=-1)      # shape → (100, 12, 12)
        dot_dp_p = cp.sum(dP_i * P_j, axis=-1)   # shape → (100, 12, 12)
        dot_p_dp = cp.sum(P_i * dP_j, axis=-1)   # shape → (100, 12, 12)
        num1 = dot_dp_p + dot_p_dp              # shape → (100, 12, 12)

        den = R_i * R_j                         # r_i * r_j, shape → (100, 12, 12)
        den2 = den * den                        # (r_i * r_j)^2, shape → (100, 12, 12)
        num2 = dot_p * (dR_i * R_j + R_i * dR_j)  # shape → (100, 12, 12)

        self.dcosine_dx = (num1 * den - num2) / den2       # shape → (100, 12, 12)
        # self.dcosine_dx[:, np.arange(12), np.arange(12)] = 0
        # print(self.dcosine_dx[0])
        # exit()
        self.dcosine_dx = self.dcosine_dx.reshape(self.N_pair,-1)
        
 

        self.dpl_dx = self.dlegdcos * self.dcosine_dx[cp.newaxis, :, :]
       
    def calculate_dqang_dx(self):

        # dgdx = cp.zeros((self.dgdx.shape[0],self.dgdx.shape[1],self.dgdx.shape[2]))
        # dgdx[:, :, :self.dgdx.shape[2]] = self.dgdx
        dgdx = self.dgdx    
        # print(dgdx)
        # exit()
        
        gij = cp.tile(self.g_rad,(1,1,12))
        gik = cp.repeat(self.g_rad,12,axis=-1)

        dgdxij = cp.tile(dgdx,(1,1,12))
        dgdxik = cp.repeat(dgdx,12,axis=-1)

        n_desc_ang = (self.nang+1) * self.lmax

        dg_ang_dx = cp.zeros((n_desc_ang,self.N_pair,144), dtype=cp.float32)
        g_ang = cp.zeros((n_desc_ang,self.N_pair,144), dtype=cp.float32)
        for n in range(self.nang+1):
            for l in range(self.lmax):
                term1 = dgdxij[n]*gik[n]*self.leg[l]
                term2 = dgdxik[n]*gij[n]*self.leg[l]
                term3 = gij[n]*gik[n]*self.dpl_dx[l]
                dg_ang_dx[n*self.lmax + l] = term1 + term2 + term3
        
        
        self.dqangdx = cp.sum(dg_ang_dx, axis=-1) 
        self.dqangdx = self.dqangdx.T
        self.dqdx = cp.concatenate((self.dqdx,self.dqangdx),axis=1)




    def dummy(self):
        pass 