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
np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)


class DescriptorGeneratorAnalytical(DescriptorGenerator):
    def generate_nep_descriptors_derivatives(self,central_pos,orientations,Nlist):
        print('generating descriptors and derivatives')
        self.calculate_pts(central_pos,orientations,Nlist)
        self.calculate_dpts_dtetax(central_pos,orientations,Nlist)
        self.calculate_derivatives()

        return self.dqdtetax, self.pp, self.N_pair

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

        # debug_raw = cp.zeros((N_pair,14),dtype=cp.float32)
        # debug_raw[:,3:7] = QUAT1
        # debug_raw[:,7:10] = translate
        # debug_raw[:,10:14] = QUAT2
        # ddd = np.load('debug_raw.npy')
        # diff = np.abs(ddd - cp.asnumpy(debug_raw))
        # print('max diff in debug_raw:',np.max(diff))
        # exit(   )

        # debug_raw = cp.asnumpy(debug_raw)
        # np.save('debug_raw.npy',debug_raw)
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

        self.quat1 = QUAT1

    def calculate_dpts_dtetax(self,central_pos,orientations,Nlist):
        print('calculating dpts_dtetax')
        d = self.quat1[:,1:] @ self.pts_rep.T # (N_pair,Nd)
        k = self.quat1[:,0]**2 - cp.sum(self.quat1[:,1:]**2,axis=1) # (N_pair,)
        k = k[:,None]
        # d = d[:,None]
        self.dpts_rep_dtetax = cp.zeros((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)

        self.dpts_rep_dtetax[:,:,1] = (
            - self.pts_rep[:,2] * k
            + 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,2,None] - self.pts_rep[:,1] * self.quat1[:,1,None] )
            - 2.0 * self.quat1[:,3,None] * d
        )  # shape (N_pair,6)
        self.dpts_rep_dtetax[:,:,2] = (
            + self.pts_rep[:,1] * k
            + 2.0 * self.quat1[:,0,None] * (self.pts_rep[:,0] * self.quat1[:,3,None] - self.pts_rep[:,2] * self.quat1[:,1,None] )
            + 2.0 * self.quat1[:,2,None] * d
        )  # shape (N_pair,6)

        ## TO DO : add dtetay and dtetaz 


    def calculate_derivatives(self):
        self.calculate_cosine()
        self.calculate_drdteta()    
        self.calculate_T()
        self.calculate_dgdr()
        self.calculate_dqang_dteta()
        self.dqdtetax = cp.concatenate((self.dqraddtetax,self.dqangdtetax),axis=0)
        self.dqdtetax = self.dqdtetax.T


    def calculate_cosine(self):
        self.P = self.pts_pair - np.average(self.pts_pair,axis=1)[:,np.newaxis,:]

        self.r = cp.linalg.norm(self.P,axis=-1)
        R_ij = cp.tile(self.P,(1,6*2,1))
        R_ik = cp.repeat(self.P,6*2,axis=1)

        norm_rij = cp.linalg.norm(R_ij,axis=-1)
        norm_rik = cp.linalg.norm(R_ik,axis=-1)

        self.cosine = cp.sum(R_ij*R_ik,axis=2)/(norm_rij*norm_rik)

        self.dpldcosine = cp.zeros((self.lmax, self.cosine.shape[0], self.cosine.shape[1]), dtype=cp.float32)
        self.dpldcosine[0] = 1
        self.dpldcosine[1] = 3*self.cosine
        self.dpldcosine[2] = (15*self.cosine**2 - 3) / 2.0

        self.dpl = cp.zeros((self.lmax, self.cosine.shape[0], self.cosine.shape[1]), dtype=cp.float32)
        self.dpl[0] = self.cosine
        self.dpl[1] = (3*self.cosine**2 - 1)*0.5
        self.dpl[2] = (5*self.cosine**3 - 3*self.cosine)/2.0

        for i in range(3, self.lmax):
            self.dpl[i] = (
                (2 * i + 1) * self.cosine * self.dpl[i - 1]
                - i * self.dpl[i - 2]
                ) / (i + 1)

        for i in range(3, self.lmax):
            n = i + 1 
            self.dpldcosine[i] = n*self.dpl[i-1] + self.cosine*self.dpldcosine[i-1]


    def calculate_drdteta(self):
        dpts_rep_dtetax_double = cp.zeros((self.N_pair, self.Nd, 3), dtype=cp.float32)
        dpts_rep_dtetax_double[:,0:self.Nd//2,:] = self.dpts_rep_dtetax
        self.dpts_rep_dtetax_double = dpts_rep_dtetax_double
        self.drdtetax = cp.sum(self.P * dpts_rep_dtetax_double, axis=2)/self.r

        P_i  = self.P[:, :, None, :]    # shape → (100, 12, 1, 3)
        P_j  = self.P[:, None, :, :]    # shape → (100, 1, 12, 3)
        dP_i = dpts_rep_dtetax_double[:, :, None, :]     # shape → (100, 12, 1, 3)
        dP_j = dpts_rep_dtetax_double[:, None, :, :]     # shape → (100, 1, 12, 3)

        R_i  = self.r[:, :, None]       # shape → (100, 12, 1)
        R_j  = self.r[:, None, :]       # shape → (100, 1, 12)
        dR_i = self.drdtetax[:, :, None]        # shape → (100, 12, 1)
        dR_j = self.drdtetax[:, None, :]        # shape → (100, 1, 12)

        dot_p = cp.sum(P_i * P_j, axis=-1)      # shape → (100, 12, 12)
        dot_dp_p = cp.sum(dP_i * P_j, axis=-1)   # shape → (100, 12, 12)
        dot_p_dp = cp.sum(P_i * dP_j, axis=-1)   # shape → (100, 12, 12)
        num1 = dot_dp_p + dot_p_dp              # shape → (100, 12, 12)

        den = R_i * R_j                         # r_i * r_j, shape → (100, 12, 12)
        den2 = den * den                        # (r_i * r_j)^2, shape → (100, 12, 12)
        num2 = dot_p * (dR_i * R_j + R_i * dR_j)  # shape → (100, 12, 12)

        self.dcosine_dtetax = (num1 * den - num2) / den2       # shape → (100, 12, 12)
        self.dcosine_dtetax = self.dcosine_dtetax.reshape(self.N_pair,-1)

        self.dpl_dtetax = self.dpldcosine * self.dcosine_dtetax[cp.newaxis, :, :]



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

        self.g_rad = cp.zeros((self.nrad+1, self.z.shape[0], self.z.shape[1]), dtype=cp.float32)
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
        self.drdp[:,6:] = 0.0

        inner_x = cp.sum(self.drdp * self.dpts_rep_dtetax_double, axis=2)  # (100, 6)


        self.dgdr[:,:,6:] = 0.0
        
        self.dgraddtetax = self.dgdr * inner_x[cp.newaxis, :, :]  # → (3, 100)

        self.dqraddtetax = cp.sum(self.dgdr * inner_x[cp.newaxis, :, :], axis=2)  # → (3, 100)

    def calculate_dqang_dteta(self):
        # dgdtetax = cp.zeros((self.dgraddtetax.shape[0],self.dgraddtetax.shape[1],self.dgraddtetax.shape[2]*2))

        # print(self.dgraddtetax.shape)
        # print(self.dgraddtetax[0,19])
        # exit()

        dgdtetax = cp.zeros((self.dgraddtetax.shape[0],self.dgraddtetax.shape[1],self.dgraddtetax.shape[2]))
        dgdtetax[:, :, :self.dgraddtetax.shape[2]] = self.dgraddtetax
        
        gij = cp.tile(self.g_rad,(1,1,12))
        gik = cp.repeat(self.g_rad,12,axis=-1)

        dgdtetaxij = cp.tile(dgdtetax,(1,1,12))
        dgdtetaxik = cp.repeat(dgdtetax,12,axis=-1)

        n_desc_ang = (self.nang+1) * self.lmax

        dg_ang_dtetax = cp.zeros((n_desc_ang,self.N_pair,144), dtype=cp.float32)
        g_ang = cp.zeros((n_desc_ang,self.N_pair,144), dtype=cp.float32)
        for n in range(self.nang+1):
            for l in range(self.lmax):
                term1 = dgdtetaxij[n]*gik[n]*self.dpl[l]
                term2 = dgdtetaxik[n]*gij[n]*self.dpl[l]
                term3 = gij[n]*gik[n]*self.dpl_dtetax[l]
                # g_ang[n*3 + l] = gij[n]*gik[n]*self.dpl[l]
                dg_ang_dtetax[n*self.lmax + l] = term1 + term2 + term3

        # cp.save('g_ang.npy',g_ang)
        # exit()
        
        
        self.dqangdtetax = cp.sum(dg_ang_dtetax, axis=-1) 


    def dummy(self):
        pass 