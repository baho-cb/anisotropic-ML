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
import cudakernels.DerivativeKernels2 as cuda_dk2
import cudakernels.SumKernels as cuda_sum
import MathUtils as mu

"""
possible simplifications: 
for dpdteta_xyz instead of 3 seperate arrays you just need 6 columns of data 2 per dx
Similarly instead of storing dp1dtetax and dp2dtetax separately you can keep d12dtetax 
"""

np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)


class DescriptorGeneratorAnalytical(DescriptorGenerator):
    def generate_nep_descriptors_derivatives(self,central_pos,orientations,Nlist):
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t0 = time.time()

        self.calc_pts_dpts(central_pos,orientations,Nlist)

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t1 = time.time()

        self.calculate_dcosdteta()
       
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t2 = time.time()

        self.kernel1()

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t3 = time.time()

        self.kernel2()

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t4 = time.time()

        self.kernel3()


        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t5 = time.time()
        
        self.kernel4a()

        # stream_1 = cp.cuda.Stream()
        # stream_2 = cp.cuda.Stream()
        stream_1 = None
        stream_2 = None

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t6 = time.time()
        self.kernel4b(stream_1)

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t7 = time.time()
        self.kernel5(stream_2)

        # stream_1.synchronize()
        # stream_2.synchronize()    

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t8 = time.time()
        # self.kernel6()
        self._kernel6()

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t9 = time.time()

        self._kernel7()

        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t10 = time.time()

        self.t_pts += t1-t0
        self.t_dcos += t2-t1
        self.t_k1 += t3-t2
        self.t_k2 += t4-t3
        self.t_k3 += t5-t4
        self.t_k4a += t6-t5
        self.t_k4b += t7-t6
        self.t_k5 += t8-t7
        self.t_k6 += t9-t8
        self.t_k7 += t10-t9


        return self._dqdteta, self._dqdxyz, self._q, self.pp, self.N_pair


    def set_timers(self):
        self.t_pts = 0 
        self.t_dcos = 0 
        self.t_k1 = 0
        self.t_k2 = 0
        self.t_k3 = 0
        self.t_k4a = 0
        self.t_k4b = 0
        self.t_k5 = 0
        self.t_k6 = 0
        self.t_k7 = 0

    def calc_pts_dpts(self,central_pos,orientations,Nlist):
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

        blocks = (self.N_pair,)
        threads_per_block = (32,)

        self._P = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        self._p1dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p2dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._r1 = cp.empty((self.N_pair,self.Nd//2),dtype=cp.float32) # (N_pair,6,3)
        self._r2 = cp.empty((self.N_pair,self.Nd//2),dtype=cp.float32) # (N_pair,6,3)
        self._dr1dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._dr2dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)

        cuda_dk.dpts_kernel(
            blocks,
            threads_per_block,
            (
                QUAT1,
                QUAT2,
                translate,
                self.pts_rep,
                self._P,
                self._p1dteta,
                self._p2dteta,
                self._r1,
                self._r2,
                self._dr1dteta,
                self._dr2dteta,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )



    def calculate_dcosdteta(self):

        blocks = (self.N_pair,)
        threads_per_block = (self.Nd*self.Nd,)

        self._r12 = cp.concatenate((self._r1, self._r2),axis=1)
        self._p12dteta = cp.concatenate((self._p1dteta, self._p2dteta),axis=1)
        self._dr12dteta = cp.concatenate((self._dr1dteta, self._dr2dteta),axis=1)

        self._cosine = cp.empty((self.N_pair, self.Nd, self.Nd), dtype=cp.float32)
        self._dcosdteta = cp.zeros((self.N_pair, self.Nd, self.Nd, 6), dtype=cp.float32)
        self._dcosdxyz = cp.zeros((self.N_pair, self.Nd, self.Nd, 3), dtype=cp.float32)
        self._leg = cp.empty((self.N_pair, self.lmax, self.Nd, self.Nd), dtype=cp.float32)
        self._dlegdcos = cp.empty((self.N_pair, self.lmax, self.Nd, self.Nd), dtype=cp.float32)
    
        cuda_dk.cosine_kernel(
            blocks,
            threads_per_block,
            (
                self._P,
                self._r12,
                self._p12dteta,
                self._dr12dteta,
                self._cosine,
                self._dcosdteta,
                self._dcosdxyz,
                self._leg,
                self._dlegdcos,
                cp.int32(self.lmax),
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )


    def kernel1(self):
        self._dgdr = cp.empty((self.N_pair,self.nrad + 1,self.Nd),dtype=cp.float32) # (N_pair,6,3)
        self._drdp = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32) # (N_pair,6,3)
        self._grad = cp.empty((self.N_pair,self.nrad + 1,self.Nd),dtype=cp.float32) # (N_pair,6,3)

        blocks = (self.N_pair,)
        threads_per_block = (self.Nd,)
        n_chebysev = self.nrad + 1
        
        cuda_dk2.grad_kernel(
            blocks,
            threads_per_block,
            (
                self._P,
                self._r12,    
                self._dgdr,
                self._drdp,
                self._grad,
                self.nep_cutoff,
                n_chebysev,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

    def kernel2(self):

        self._dgdteta = cp.empty((self.N_pair,self.nrad + 1,self.Nd, 6),dtype=cp.float32) # (N_pair,6,3)
        self._dgdxyz = cp.empty((self.N_pair,self.nrad + 1,self.Nd, 3),dtype=cp.float32) # (N_pair,6,3)
        n_chebysev = self.nrad + 1
        blocks = (self.N_pair*n_chebysev,)
        threads_per_block = (self.Nd,)
        n_chebysev = self.nrad + 1
        
        cuda_dk2.dgdteta_kernel(
            blocks,
            threads_per_block,
            (
                self._dgdr,
                self._drdp,    
                self._p12dteta,
                self._dgdteta,
                self._dgdxyz,
                n_chebysev,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

    def kernel3(self):    

        self._dlegdcos = self._dlegdcos.reshape(self.N_pair, self.lmax, self.Nd * self.Nd) 
        self._dcosdteta = self._dcosdteta.reshape(self.N_pair, self.Nd * self.Nd, 6)  #
        self._dcosdxyz = self._dcosdxyz.reshape(self.N_pair,144,3)

        self._dlegdteta = self._dlegdcos[:,:,:,cp.newaxis] * self._dcosdteta[:,cp.newaxis,:,:]  # shape (N_pair, lmax, Nd, Nd)
        self._dlegdxyz = self._dlegdcos[:,:,:,cp.newaxis] * self._dcosdxyz[:,cp.newaxis,:,:]  # shape (N_pair, lmax, Nd, Nd)

    def kernel4a(self):             


        self.out = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144, 6), dtype=cp.float32)
        self.out_xyz = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144, 3), dtype=cp.float32)
        self._gang = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144), dtype=cp.float32)

        self._gradforang = self._grad[:,:self.nang+1]
        self._gradforang = cp.ascontiguousarray(self._gradforang)
        self._dgdtetaforang = self._dgdteta[:,:self.nang+1]
        self._dgdtetaforang = cp.ascontiguousarray(self._dgdtetaforang)
        self._dgdxyzforang = self._dgdxyz[:,:self.nang+1]
        self._dgdxyzforang = cp.ascontiguousarray(self._dgdxyzforang)

        self.cp_N_pair = cp.int32(self.N_pair)
        self.cp_nangp1 = cp.int32(self.nang + 1)
        self.cp_lmax = cp.int32(self.lmax)
        self.cp_Nd = cp.int32(self.Nd)
        self.cp_nradp1 = cp.int32(self.nrad + 1)


    def kernel4b(self,stream):
        if stream == None:
            blocks = (self.N_pair*(self.nang+1)*self.lmax,)
            threads_per_block = (self.Nd*self.Nd*6,)

            cuda_dk2.dqang_dteta_kernel(
            blocks, 
            threads_per_block,
            (self._gradforang, 
            self._leg, 
            self._dlegdteta, 
            self._dgdtetaforang,
            self.out, 
            self.cp_N_pair,
            self.cp_nangp1, 
            self.cp_lmax,
            self.cp_Nd
            ))
        else:
            with stream:     
                blocks = (self.N_pair*(self.nang+1)*self.lmax,)
                threads_per_block = (self.Nd*self.Nd*6,)

                cuda_dk2.dqang_dteta_kernel(
                blocks, 
                threads_per_block,
                (self._gradforang, 
                self._leg, 
                self._dlegdteta, 
                self._dgdtetaforang,
                self.out, 
                self.cp_N_pair,
                self.cp_nangp1, 
                self.cp_lmax,
                self.cp_Nd
                ))

    def kernel5(self,stream):  
        if(stream == None):  
            blocks = (self.N_pair*(self.nang+1)*self.lmax,)
            threads_per_block = (self.Nd*self.Nd*3,)
        
            cuda_dk2.dqang_dxyz_kernel(
            blocks, 
            threads_per_block,
            (self._gradforang, 
            self._leg, 
            self._dlegdxyz, 
            self._dgdxyzforang,
            self.out_xyz, 
            self._gang,
            self.cp_N_pair,
            self.cp_nangp1, 
            self.cp_lmax,
            self.cp_Nd
            ))
        else:    
            with stream:
                blocks = (self.N_pair*(self.nang+1)*self.lmax,)
                threads_per_block = (self.Nd*self.Nd*3,)
            
                cuda_dk2.dqang_dxyz_kernel(
                blocks, 
                threads_per_block,
                (self._gradforang, 
                self._leg, 
                self._dlegdxyz, 
                self._dgdxyzforang,
                self.out_xyz, 
                self._gang,
                self.cp_N_pair,
                self.cp_nangp1, 
                self.cp_lmax,
                self.cp_Nd
                ))

    def kernel6(self):    
        self._q_rad = cp.sum(self._grad,axis=-1)
        self._q_ang = cp.sum(self._gang,axis=-1)
        
        self._dqraddteta = cp.sum(self._dgdteta,axis=-2)
        self._dqraddxyz = cp.sum(self._dgdxyz,axis=-2)

        # self._dqangdteta = cp.sum(self.out,axis=-2)
        # self._dqangdxyz = cp.sum(self.out_xyz,axis=-2)

        tmp = self.out.transpose(0,1,3,2)   # shape (6000,36,6,144)
        tmp2 = self.out_xyz.transpose(0,1,3,2)   # shape (6000,36,6,144)
        self._dqangdteta = cp.sum(tmp, axis=-1)
        self._dqangdxyz = cp.sum(tmp2, axis=-1)

        self._dqdteta = cp.concatenate((self._dqraddteta,self._dqangdteta),axis=1)
        self._dqdxyz = cp.concatenate((self._dqraddxyz,self._dqangdxyz),axis=1)
        self._q = cp.concatenate((self._q_rad,self._q_ang),axis=1)



    def _kernel6(self):

        nradp1 = self.nrad + 1
        J2 = self.lmax*(self.nang + 1 ) 
        J1 = nradp1
        sizes = [
                self.N_pair*nradp1,
                self.N_pair*J2,
                self.N_pair*J1*3,
                self.N_pair*J1*6,
                self.N_pair*J2*3,
                self.N_pair*J2*6,
            ]
        
        total_out = sum(sizes)

        threads_per_block = 256
        blocks = (total_out + threads_per_block - 1) // threads_per_block

        self._q_rad = cp.empty((self.N_pair, J1), dtype=cp.float32)
        self._q_ang = cp.empty((self.N_pair, J2), dtype=cp.float32)
        self._dqraddteta = cp.empty((self.N_pair, J1, 6), dtype=cp.float32)
        self._dqraddxyz = cp.empty((self.N_pair, J1, 3), dtype=cp.float32)
        self._dqangdteta = cp.empty((self.N_pair, J2, 6), dtype=cp.float32)
        self._dqangdxyz = cp.empty((self.N_pair, J2, 3), dtype=cp.float32)



        cuda_sum.sum_kernel(    
                (blocks,),
                (threads_per_block,),
            (
                self._grad,
                self._q_rad,
                self._gang,
                self._q_ang,
                self._dgdteta,
                self._dqraddteta,
                self._dgdxyz,
                self._dqraddxyz,
                self.out,
                self._dqangdteta,
                self.out_xyz,
                self._dqangdxyz,
                self.cp_N_pair,
                self.cp_lmax,
                self.cp_nradp1, 
                self.cp_nangp1, 
                self.cp_Nd
            )
            )
        
    def _kernel7(self):
        
        self._dqdteta = cp.concatenate((self._dqraddteta,self._dqangdteta),axis=1)
        self._dqdxyz = cp.concatenate((self._dqraddxyz,self._dqangdxyz),axis=1)
        self._q = cp.concatenate((self._q_rad,self._q_ang),axis=1)






#################### DEBUG - CHECK MATH ETC ##############3
    def generate_nep_descriptors_derivatives_debug(self,central_pos,orientations,Nlist):
        """
        Not to be used in production 
        """
        self.calculate_pts_debug(central_pos,orientations,Nlist)
        self.calculate_derivatives()
        self.calculate_dx_derivatives()
        self.merge_derivatives()
        self.kernel_test()
        return self._dqdteta, self._dqdxyz, self._q, self.pp, self.N_pair




    def kernel_test(self):
        for i in range(6):
            q_true = self.dqalldteta_list[i]
            err = mu.maxerr(q_true,self._dqdteta[:,:,i])
            if(err > 0.01):
                print('e123')
                exit()

        for i in range(3):
            q_true = self.dqdxyz[i]
            err = mu.maxerr(q_true,self._dqdxyz[:,:,i])
            if(err > 0.01):
                print('e123')
                exit()


    def calculate_pts_debug(self,central_pos,orientations,Nlist):
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

        blocks = (self.N_pair,)
        threads_per_block = (32,)

        self._P = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        self._p1dtetax = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p1dtetay = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p1dtetaz = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p2dtetax = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p2dtetay = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p2dtetaz = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p1dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._p2dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._r1 = cp.empty((self.N_pair,self.Nd//2),dtype=cp.float32) # (N_pair,6,3)
        self._r2 = cp.empty((self.N_pair,self.Nd//2),dtype=cp.float32) # (N_pair,6,3)
        self._dr1dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)
        self._dr2dteta = cp.empty((self.N_pair,self.Nd//2,3),dtype=cp.float32) # (N_pair,6,3)

        cuda_dk.dpts_kernel(
            blocks,
            threads_per_block,
            (
                QUAT1,
                QUAT2,
                translate,
                self.pts_rep,
                self._P,
                self._p1dtetax,
                self._p1dtetay,
                self._p1dtetaz,
                self._p2dtetax,
                self._p2dtetay,
                self._p2dtetaz,
                self._p1dteta,
                self._p2dteta,
                self._r1,
                self._r2,
                self._dr1dteta,
                self._dr2dteta,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )



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

    def calculate_derivatives_debug(self):
        self.calculate_dpdteta()
        self.calculate_dcosdteta_debug()    
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
        # print(mu.maxerr(self.P, self._P))
        # print(mu.maxerr(self.p1dtetax, self._p1dtetax))
        # print(mu.maxerr(self.p1dtetay, self._p1dtetay))
        # print(mu.maxerr(self.p1dtetaz, self._p1dtetaz))
        # print(mu.maxerr(self.p2dtetax, self._p2dtetax))
        # print(mu.maxerr(self.p2dtetay, self._p2dtetay))
        # print(mu.maxerr(self.p2dtetaz, self._p2dtetaz))

        # exit()




    def calculate_dcosdteta_debug(self):
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

        # print(self.dr12dtetax[0])
        # print(self.dr12dtetay[0])
        # print(self.dr12dtetaz[0])
        # print(self.dr12dtetax[0])
        # print(self.dr12dtetay[0])
        # print(self.dr12dtetaz[0])
        # print(self._dr1dteta[0])

        
        
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
        

        # print(mu.maxerr(self.P, self._P))
        # print(mu.maxerr(self.p1dtetax, self._p1dtetax))
        # print(mu.maxerr(self.p1dtetay, self._p1dtetay))
        # print(mu.maxerr(self.p1dtetaz, self._p1dtetaz))
        # print(mu.maxerr(self.p2dtetax, self._p2dtetax))
        # print(mu.maxerr(self.p2dtetay, self._p2dtetay))
        # print(mu.maxerr(self.p2dtetaz, self._p2dtetaz))




        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetax, self.dr12dtetax))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetay, self.dr12dtetay))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp12dtetaz, self.dr12dtetaz))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetax, self.dr21dtetax))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetay, self.dr21dtetay))
        self.dcosinedteta_list.append(_dcosdteta(self.P, self.r, self.dp21dtetaz, self.dr21dtetaz))

        # print(len(self.dcosinedteta_list))
        # print(self.dcosinedteta_list[0].shape)
        # exit()
        blocks = (self.N_pair,)
        threads_per_block = (self.Nd*self.Nd,)

        self._r12 = cp.concatenate((self._r1, self._r2),axis=1)
        self._p12dteta = cp.concatenate((self._p1dteta, self._p2dteta),axis=1)
        self._dr12dteta = cp.concatenate((self._dr1dteta, self._dr2dteta),axis=1)

        self._cosine = cp.empty((self.N_pair, self.Nd, self.Nd), dtype=cp.float32)
        self._dcosdteta = cp.zeros((self.N_pair, self.Nd, self.Nd, 6), dtype=cp.float32)
        self._dcosdxyz = cp.zeros((self.N_pair, self.Nd, self.Nd, 3), dtype=cp.float32)
        self._leg = cp.empty((self.N_pair, self.lmax, self.Nd, self.Nd), dtype=cp.float32)
        self._dlegdcos = cp.empty((self.N_pair, self.lmax, self.Nd, self.Nd), dtype=cp.float32)

        # print(mu.maxerr(self.r, self._r12))
        # print(mu.maxerr(self.P, self._P))
        # print(mu.maxerr(self.dr12dtetax[:,:6], self._dr12dteta[:,:6,0]))
        # print(mu.maxerr(self.dr12dtetay[:,:6], self._dr12dteta[:,:6,1]))
        # print(mu.maxerr(self.dr12dtetaz[:,:6], self._dr12dteta[:,:6,2]))
        # print(mu.maxerr(self.dr21dtetax[:,6:], self._dr12dteta[:,6:,0]))
        # print(mu.maxerr(self.dr21dtetay[:,6:], self._dr12dteta[:,6:,1]))
        # print(mu.maxerr(self.dr21dtetaz[:,6:], self._dr12dteta[:,6:,2]))

        # print(mu.maxerr(cp.zeros_like(self.dp12dtetax[:,:6,0]), self.dp12dtetax[:,:6,0]))
        # print(mu.maxerr(-self._p12dteta[:,:6,2], self.dp12dtetax[:,:6,1]))
        # print(mu.maxerr(self._p12dteta[:,:6,1], self.dp12dtetax[:,:6,2]))

        # print(mu.maxerr(self._p12dteta[:,:6,2], self.dp12dtetay[:,:6,0]))
        # print(mu.maxerr(cp.zeros_like(self.dp12dtetay[:,:6,1]), self.dp12dtetay[:,:6,1]))
        # print(mu.maxerr(-self._p12dteta[:,:6,0], self.dp12dtetay[:,:6,2]))

        # print(mu.maxerr(-self._p12dteta[:,:6,1], self.dp12dtetaz[:,:6,0]))
        # print(mu.maxerr(self._p12dteta[:,:6,0], self.dp12dtetaz[:,:6,1]))
        # print(mu.maxerr(cp.zeros_like(self.dp12dtetaz[:,:6,2]), self.dp12dtetaz[:,:6,2]))

        # print(mu.maxerr(cp.zeros_like(self.dp12dtetax[:,:6,0]), self.dp21dtetax[:,6:,0]))
        # print(mu.maxerr(-self._p12dteta[:,6:,2], self.dp21dtetax[:,6:,1]))
        # print(mu.maxerr(self._p12dteta[:,6:,1], self.dp21dtetax[:,6:,2]))

        # print(mu.maxerr(self._p12dteta[:,6:,2], self.dp21dtetay[:,6:,0]))
        # print(mu.maxerr(cp.zeros_like(self.dp12dtetay[:,:6,1]), self.dp21dtetay[:,6:,1]))
        # print(mu.maxerr(-self._p12dteta[:,6:,0], self.dp21dtetay[:,6:,2]))

        # print(mu.maxerr(-self._p12dteta[:,6:,1], self.dp21dtetaz[:,6:,0]))
        # print(mu.maxerr(self._p12dteta[:,6:,0], self.dp21dtetaz[:,6:,1]))
        # print(mu.maxerr(cp.zeros_like(self.dp21dtetaz[:,:6,2]), self.dp21dtetaz[:,6:,2]))

    
        cuda_dk.cosine_kernel(
            blocks,
            threads_per_block,
            (
                self._P,
                self._r12,
                self._p12dteta,
                self._dr12dteta,
                self._cosine,
                self._dcosdteta,
                self._dcosdxyz,
                self._leg,
                self._dlegdcos,
                cp.int32(self.lmax),
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

        # print(mu.maxerr(self.dcosinedteta_list[0], self._dcosdteta[:, :, :, 0].reshape(-1,144)))
        # print(mu.maxerr(self.dcosinedteta_list[1], self._dcosdteta[:, :, :, 1].reshape(-1,144)))
        # print(mu.maxerr(self.dcosinedteta_list[2], self._dcosdteta[:, :, :, 2].reshape(-1,144)))
        # print(mu.maxerr(self.dcosinedteta_list[3], self._dcosdteta[:, :, :, 3].reshape(-1,144)))
        # print(mu.maxerr(self.dcosinedteta_list[4], self._dcosdteta[:, :, :, 4].reshape(-1,144)))
        # print(mu.maxerr(self.dcosinedteta_list[5], self._dcosdteta[:, :, :, 5].reshape(-1,144)))


        

    def calculate_cosine(self):
        self.P = self.pts_pair - np.average(self.pts_pair,axis=1)[:,np.newaxis,:]

        self.r = cp.linalg.norm(self.P,axis=-1)
        R_ij = cp.tile(self.P,(1,6*2,1))
        R_ik = cp.repeat(self.P,6*2,axis=1)

        norm_rij = cp.linalg.norm(R_ij,axis=-1)
        norm_rik = cp.linalg.norm(R_ik,axis=-1)

        self.cosine = cp.sum(R_ij*R_ik,axis=2)/(norm_rij*norm_rik)

        self.dlegdcos = cp.zeros((self.cosine.shape[0], self.lmax,self.cosine.shape[1]), dtype=cp.float32)
        self.dlegdcos[:,0] = 1
        self.dlegdcos[:,1] = 3*self.cosine
        self.dlegdcos[:,2] = (15*self.cosine**2 - 3) / 2.0

        self.leg = cp.zeros((self.cosine.shape[0], self.lmax, self.cosine.shape[1]), dtype=cp.float32)
        self.leg[:,0] = self.cosine
        self.leg[:,1] = (3*self.cosine**2 - 1)*0.5
        self.leg[:,2] = (5*self.cosine**3 - 3*self.cosine)/2.0

        for i in range(3, self.lmax):
            self.leg[:,i] = (
                (2 * i + 1) * self.cosine * self.leg[:,i - 1]
                - i * self.leg[:,i - 2]
                ) / (i + 1)

        # print(mu.maxerr(self.leg[:,0], self._leg[:,0].reshape(-1,144)))
        # print(mu.maxerr(self.leg[:,1], self._leg[:,1].reshape(-1,144)))
        # print(mu.maxerr(self.leg[:,1], self._leg[:,1].reshape(-1,144)))
        # print(mu.maxerr(self.leg[:,3], self._leg[:,3].reshape(-1,144)))


        for i in range(3, self.lmax):
            n = i + 1 
            self.dlegdcos[:,i] = n*self.leg[:,i-1] + self.cosine*self.dlegdcos[:,i-1]

        # print(mu.maxerr(self.dlegdcos[:,0], self._dlegdcos[:,0].reshape(-1,144)))
        # print(mu.maxerr(self.dlegdcos[:,1], self._dlegdcos[:,1].reshape(-1,144)))
        # print(mu.maxerr(self.dlegdcos[:,2], self._dlegdcos[:,2].reshape(-1,144)))
        # print(mu.maxerr(self.dlegdcos[:,3], self._dlegdcos[:,3].reshape(-1,144)))
        # exit()

        self.dlegdteta_list = []

        for i in range(6):
            self.dlegdteta_list.append(self.dlegdcos * self.dcosinedteta_list[i][:, cp.newaxis, :])


        blocks = (self.N_pair,)
        threads_per_block = (self.Nd*self.Nd*6,)

        
        # cuda_dk.legendre_kernel(
        #     blocks,
        #     threads_per_block,
        #     (
        #         self._cosine,
        #         self._dcosdteta,    
                
        #         self._dlegdcos,
        #         self._dlegdteta,
        #         cp.int32(self.lmax),
        #         cp.int32(self.N_pair),
        #         cp.int32(self.Nd)
        #     )
        # )



    def calculate_T(self):
        
        self.z = 2.0*(self.r/self.cutoff_nep - 1.0)**2 - 1.0    
        self.T = cp.zeros((self.z.shape[0],
                        self.nrad + 1,
                        self.z.shape[1]),
                        dtype=cp.float32)

        self.T[:, 0, :] = 1.0             # T₀(z) = 1
        if self.nrad >= 1:
            self.T[:, 1, :] = self.z      # T₁(z) = z

        for n in range(2, self.nrad + 1):
            self.T[:, n, :] = 2.0 * self.z * self.T[:, n - 1, :] \
                            - self.T[:, n - 2, :]    


        self.dTdz = cp.zeros((self.z.shape[0],
                              self.nrad + 1,
                            self.z.shape[1]),
                            dtype=cp.float32)

        # Base cases:
        #   T₀(z) = 1      ⇒  dT₀/dz = 0
        #   T₁(z) =  z     ⇒  dT₁/dz = 1
        self.dTdz[:, 0, :] = 0.0
        if self.nrad >= 1:
            self.dTdz[:, 1, :] = 1.0

        # Now use the recurrence for n = 2..n_rad:
        #   dTₙ/dz = 2·Tₙ₋₁ + 2·z·(dTₙ₋₁/dz) − (dTₙ₋₂/dz)
        for n in range(2, self.nrad + 1):
            # 2 · z · (dTₙ₋₁ / dz)
            term_z_dT_prev = 2.0 * self.z * self.dTdz[:, n - 1, :]

            # 2 · Tₙ₋₁
            term_T_prev = 2.0 * self.T[:, n - 1, :]

            # subtract dTₙ₋₂/dz
            term_dT_prev2 = self.dTdz[:, n - 2, :]

            self.dTdz[:, n, :] = term_T_prev + term_z_dT_prev - term_dT_prev2

        # self.g_rad = cp.zeros((self.nrad+1, self.z.shape[0], self.z.shape[1]), dtype=cp.float32)
        self.f_rcut = 0.5*(1.0 + cp.cos(np.pi*(self.r/self.cutoff_nep)))
        self.g_rad = ((self.T + 1.0) / 2.0 ) * self.f_rcut[:,cp.newaxis,:]


        self._dgdr = cp.empty((self.N_pair,self.nrad + 1,self.Nd),dtype=cp.float32) # (N_pair,6,3)
        self._drdp = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32) # (N_pair,6,3)
        self._grad = cp.empty((self.N_pair,self.nrad + 1,self.Nd),dtype=cp.float32) # (N_pair,6,3)

        blocks = (self.N_pair,)
        threads_per_block = (self.Nd,)
        n_chebysev = self.nrad + 1
        
        cuda_dk2.grad_kernel(
            blocks,
            threads_per_block,
            (
                self._P,
                self._r12,    
                self._dgdr,
                self._drdp,
                self._grad,
                self.nep_cutoff,
                n_chebysev,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )




    def calculate_dgdr(self):
        r_rc = self.r/ self.cutoff_nep
        factor1 = (4.0 / self.cutoff_nep) * (r_rc - 1.0) * (1.0 + cp.cos(np.pi * r_rc))   # shape (6, N)
        factor2 = -cp.sin(np.pi * r_rc) * (np.pi / self.cutoff_nep)                      # shape (6, N)
        factor1 = factor1[:, cp.newaxis, :]  # → (1, 6, N)
        factor2 = factor2[:, cp.newaxis, :]  # → (1, 6, N)
                        
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


        self._dgdteta = cp.empty((self.N_pair,self.nrad + 1,self.Nd, 6),dtype=cp.float32) # (N_pair,6,3)
        self._dgdxyz = cp.empty((self.N_pair,self.nrad + 1,self.Nd, 3),dtype=cp.float32) # (N_pair,6,3)
        n_chebysev = self.nrad + 1
        blocks = (self.N_pair*n_chebysev,)
        threads_per_block = (self.Nd,)
        n_chebysev = self.nrad + 1
        
        cuda_dk2.dgdteta_kernel(
            blocks,
            threads_per_block,
            (
                self._dgdr,
                self._drdp,    
                self._p12dteta,
                self._dgdteta,
                self._dgdxyz,
                n_chebysev,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

        # print(mu.maxerr(self.dgdteta_list[0], self._dgdteta[:, :, :6, 0]))
        # print(mu.maxerr(self.dgdteta_list[1], self._dgdteta[:, :, :6, 1]))
        # print(mu.maxerr(self.dgdteta_list[2], self._dgdteta[:, :, :6, 2]))
        # print(mu.maxerr(self.dgdteta_list[3], self._dgdteta[:, :, 6:, 3]))
        # print(mu.maxerr(self.dgdteta_list[4], self._dgdteta[:, :, 6:, 4]))
        # print(mu.maxerr(self.dgdteta_list[5], self._dgdteta[:, :, 6:, 5]))
        # print('done')
        # exit()
        self._dlegdcos = self._dlegdcos.reshape(self.N_pair, self.lmax, self.Nd * self.Nd) 
        self._dcosdteta = self._dcosdteta.reshape(self.N_pair, self.Nd * self.Nd, 6)  #
        self._dcosdxyz = self._dcosdxyz.reshape(self.N_pair,144,3)

        self._dlegdteta = self._dlegdcos[:,:,:,cp.newaxis] * self._dcosdteta[:,cp.newaxis,:,:]  # shape (N_pair, lmax, Nd, Nd)
        self._dlegdxyz = self._dlegdcos[:,:,:,cp.newaxis] * self._dcosdxyz[:,cp.newaxis,:,:]  # shape (N_pair, lmax, Nd, Nd)
        # print(self._dlegdteta.shape)
        # exit()
        # print(mu.maxerr(self.dlegdteta_list[0], self._dlegdteta[:, :, :,0]))
        # print(mu.maxerr(self.dlegdteta_list[1], self._dlegdteta[:, :, :,1]))
        # print(mu.maxerr(self.dlegdteta_list[2], self._dlegdteta[:, :, :,2]))
        # print(mu.maxerr(self.dlegdteta_list[3], self._dlegdteta[:, :, :,3]))
        # print(mu.maxerr(self.dlegdteta_list[4], self._dlegdteta[:, :, :,4]))
        # print(mu.maxerr(self.dlegdteta_list[5], self._dlegdteta[:, :, :,5]))
        # exit()




    def calculate_dqang_dteta(self):
        gij = cp.tile(self.g_rad,(1,1,12))
        gik = cp.repeat(self.g_rad,12,axis=-1)
        n_desc_ang = (self.nang+1) * self.lmax
        self.dqangdteta_list = []
        self.dgangdteta_list = []

        for i in range(6):
            dgdteta = self.dgdteta_list[i]
            dgdteta12 = cp.zeros((dgdteta.shape[0],dgdteta.shape[1],dgdteta.shape[2]*2))
            if(i <3):
                dgdteta12[:, :, :dgdteta.shape[2]] = dgdteta
            else:
                dgdteta12[:, :, dgdteta.shape[2]:] = dgdteta

            dgdtetaij = cp.tile(dgdteta12,(1,1,12))
            dgdtetaik = cp.repeat(dgdteta12,12,axis=-1)

            # print('dgdtetaij shape:', dgdtetaij.shape)
            # print('dgdtetaik shape:', dgdtetaik.shape)
            # print('gij shape:', gij.shape)
            # print('gik shape:', gik.shape)
            # print('self.leg shape:', self.leg.shape)
            # exit()
            dg_ang_dteta = cp.zeros((self.N_pair,n_desc_ang,144), dtype=cp.float32)
            for n in range(self.nang+1):
                for l in range(self.lmax):
                    term1 = dgdtetaij[:,n]*gik[:,n]*self.leg[:,l]
                    term2 = dgdtetaik[:,n]*gij[:,n]*self.leg[:,l]
                    term3 = gij[:,n]*gik[:,n]*self.dlegdteta_list[i][:,l]
                    dg_ang_dteta[:,n*self.lmax + l] = term1 + term2 + term3
                    # dg_ang_dteta[:,n*self.lmax + l] = dgdtetaik[:,n]

            self.dqangdteta_list.append(cp.sum(dg_ang_dteta, axis=-1))
            self.dgangdteta_list.append(dg_ang_dteta)

        
        out = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144, 6), dtype=cp.float32)
        self.out_xyz = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144, 3), dtype=cp.float32)
        self.g_ang = cp.empty((self.N_pair, (self.nang+1) * self.lmax, 144), dtype=cp.float32)

        self._gradforang = self._grad[:,:self.nang+1]
        self._gradforang = cp.ascontiguousarray(self._gradforang)
        self._dgdtetaforang = self._dgdteta[:,:self.nang+1]
        self._dgdtetaforang = cp.ascontiguousarray(self._dgdtetaforang)
        self._dgdxyzforang = self._dgdxyz[:,:self.nang+1]
        self._dgdxyzforang = cp.ascontiguousarray(self._dgdxyzforang)
        blocks = (self.N_pair*(self.nang+1)*self.lmax,)
        threads_per_block = (self.Nd*self.Nd*6,)

        cuda_dk2.dqang_dteta_kernel(
            blocks, 
            threads_per_block,
            (self._gradforang, 
            self._leg, 
            self._dlegdteta, 
            self._dgdtetaforang,
            out, 
            cp.int32(self.N_pair),
            cp.int32(self.nang + 1), 
            cp.int32(self.lmax),
            cp.int32(self.Nd)
            ))
        
        blocks = (self.N_pair*(self.nang+1)*self.lmax,)
        threads_per_block = (self.Nd*self.Nd*3,)
        
        cuda_dk2.dqang_dxyz_kernel(
            blocks, 
            threads_per_block,
            (self._gradforang, 
            self._leg, 
            self._dlegdxyz, 
            self._dgdxyzforang,
            self.out_xyz, 
            self.g_ang,
            cp.int32(self.N_pair),
            cp.int32(self.nang + 1), 
            cp.int32(self.lmax),
            cp.int32(self.Nd)
            ))
        
        self._q_rad = cp.sum(self.g_rad,axis=-1)
        self._q_ang = cp.sum(self.g_ang,axis=-1)
        
        self._dqraddteta = cp.sum(self._dgdteta,axis=-2)
        self._dqraddxyz = cp.sum(self._dgdxyz,axis=-2)

        self._dqangdteta = cp.sum(out,axis=-2)
        self._dqangdxyz = cp.sum(self.out_xyz,axis=-2)

        self._dqdteta = cp.concatenate((self._dqraddteta,self._dqangdteta),axis=1)
        self._dqdxyz = cp.concatenate((self._dqraddxyz,self._dqangdxyz),axis=1)
        self._q = cp.concatenate((self._q_rad,self._q_ang),axis=1)



    def merge_derivatives(self):
        self.dqalldteta_list = []

        for i in range(6):
            dat = cp.concatenate((self.dqdteta_list[i], self.dqangdteta_list[i]), axis=1)
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
        self.drdy = (self.P[:,:,1]/ self.r)*(+0.5)
        self.drdy[:,6:] *= -1.0
        self.drdz = (self.P[:,:,2]/ self.r)*(+0.5)
        self.drdz[:,6:] *= -1.0


        self.dgdx = self.dgdr* self.drdx[:, np.newaxis, :]  # shape (3, N_pair, 3)
        self.dqdx = cp.sum(self.dgdx, axis=2)
        self.dqdx = self.dqdx.T

        self.dgdy = self.dgdr* self.drdy[:, np.newaxis, :]  # shape (3, N_pair, 3)
        self.dqdy = cp.sum(self.dgdy, axis=2)
        self.dqdy = self.dqdy.T

        self.dgdz = self.dgdr* self.drdz[:, np.newaxis, :]  # shape (3, N_pair, 3)
        self.dqdz = cp.sum(self.dgdz, axis=2)
        self.dqdz = self.dqdz.T

        # print(mu.maxerr(self._dgdxyz[:,:,:,0],self.dgdx))
        # print(mu.maxerr(self._dgdxyz[:,:,:,1],self.dgdy))
        # print(mu.maxerr(self._dgdxyz[:,:,:,2],self.dgdz))
        # exit()
   
        self.dpdx = cp.zeros((self.N_pair, 6*2, 3), dtype=cp.float32)
        self.dpdy = cp.zeros((self.N_pair, 6*2, 3), dtype=cp.float32)
        self.dpdz = cp.zeros((self.N_pair, 6*2, 3), dtype=cp.float32)
        self.dpdx[:, :6, 0] = +0.5
        self.dpdx[:, 6:, 0] = -0.5
        self.dpdy[:, :6, 1] = +0.5
        self.dpdy[:, 6:, 1] = -0.5
        self.dpdz[:, :6, 2] = +0.5
        self.dpdz[:, 6:, 2] = -0.5



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
        self.dcosine_dx = self.dcosine_dx.reshape(self.N_pair,-1)
        self.dcosine_dx = _dcosdteta(self.P, self.r, self.dpdx, self.drdx)
        self.dcosine_dy = _dcosdteta(self.P, self.r, self.dpdy, self.drdy)
        self.dcosine_dz = _dcosdteta(self.P, self.r, self.dpdz, self.drdz)

        # print(mu.maxerr(self._dcosdxyz[:,:,0],self.dcosine_dx))
        # print(mu.maxerr(self._dcosdxyz[:,:,1],self.dcosine_dy))
        # print(mu.maxerr(self._dcosdxyz[:,:,2],self.dcosine_dz))
        # exit()

        self.dpl_dx = self.dlegdcos * self.dcosine_dx[:, cp.newaxis, :]
        self.dpl_dy = self.dlegdcos * self.dcosine_dy[:, cp.newaxis, :]
        self.dpl_dz = self.dlegdcos * self.dcosine_dz[:, cp.newaxis, :]
       
    def calculate_dqang_dx(self):

        dgdx = self.dgdx    
        dgdy = self.dgdy    
        dgdz = self.dgdz    
        
        gij = cp.tile(self.g_rad,(1,1,12))
        gik = cp.repeat(self.g_rad,12,axis=-1)

        dgdxij = cp.tile(dgdx,(1,1,12))
        dgdxik = cp.repeat(dgdx,12,axis=-1)
        dgdyij = cp.tile(dgdy,(1,1,12))
        dgdyik = cp.repeat(dgdy,12,axis=-1)
        dgdzij = cp.tile(dgdz,(1,1,12))
        dgdzik = cp.repeat(dgdz,12,axis=-1)

        n_desc_ang = (self.nang+1) * self.lmax

        dg_ang_dx = cp.zeros((self.N_pair,n_desc_ang,144), dtype=cp.float32)
        dg_ang_dy = cp.zeros((self.N_pair,n_desc_ang,144), dtype=cp.float32)
        dg_ang_dz = cp.zeros((self.N_pair,n_desc_ang,144), dtype=cp.float32)

        for n in range(self.nang+1):
            for l in range(self.lmax):
                term1dx = dgdxij[:,n]*gik[:,n]*self.leg[:,l]
                term2dx = dgdxik[:,n]*gij[:,n]*self.leg[:,l]
                term3dx = gij[:,n]*gik[:,n]*self.dpl_dx[:,l]
                dg_ang_dx[:,n*self.lmax + l] = term1dx + term2dx + term3dx

                term1dy = dgdyij[:,n]*gik[:,n]*self.leg[:,l]
                term2dy = dgdyik[:,n]*gij[:,n]*self.leg[:,l]
                term3dy = gij[:,n]*gik[:,n]*self.dpl_dy[:,l]
                dg_ang_dy[:,n*self.lmax + l] = term1dy + term2dy + term3dy

                term1dz = dgdzij[:,n]*gik[:,n]*self.leg[:,l]
                term2dz = dgdzik[:,n]*gij[:,n]*self.leg[:,l]
                term3dz = gij[:,n]*gik[:,n]*self.dpl_dz[:,l]
                dg_ang_dz[:,n*self.lmax + l] = term1dz + term2dz + term3dz
        
        self.dqangdx = cp.sum(dg_ang_dx, axis=-1) 
        self.dqangdx = self.dqangdx.T
        self.dqdx = cp.concatenate((self.dqdx,self.dqangdx),axis=0)

        self.dqangdy = cp.sum(dg_ang_dy, axis=-1) 
        self.dqangdy = self.dqangdy.T
        self.dqdy = cp.concatenate((self.dqdy,self.dqangdy),axis=0)

        self.dqangdz = cp.sum(dg_ang_dz, axis=-1) 
        self.dqangdz = self.dqangdz.T
        self.dqdz = cp.concatenate((self.dqdz,self.dqangdz),axis=0)

        # print(mu.maxerr(self.out_xyz[:,:,:,0],dg_ang_dx))
        # print(mu.maxerr(self.out_xyz[:,:,:,1],dg_ang_dy))
        # print(mu.maxerr(self.out_xyz[:,:,:,2],dg_ang_dz))
        # exit()

        self.dqdxyz = []
        self.dqdxyz.append(self.dqdx.T)
        self.dqdxyz.append(self.dqdy.T)
        self.dqdxyz.append(self.dqdz.T)


    def dummy(self):
        pass 


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

    # print(num1[0])
    # print(num2[0])
    # exit()

    dcosine_dteta = (num1 * den - num2) / den2       # shape → (100, 12, 12)
    dcosine_dteta = dcosine_dteta.reshape(Np,-1)    
    return dcosine_dteta

def _dqdteta(drdp, dpdteta, dgdr, one_or_two):

    inner_x = cp.sum(drdp * dpdteta, axis=2)  # (100, 6)
    if(one_or_two == 1):
        dgraddtetax = dgdr[:,:,:6] * inner_x[:, cp.newaxis, :]  # → (3, 100)
    elif(one_or_two == 2):
        dgraddtetax = dgdr[:,:,6:] * inner_x[:, cp.newaxis, :]
    else:
        raise ValueError("one_or_two must be 1 or 2")    
    return cp.sum(dgraddtetax,  axis=2)

def _dgdteta(drdp, dpdteta, dgdr, one_or_two):

    # print("dgdr shape:", dgdr.shape)
    # print("drdp shape:", drdp.shape)
    # print("dpdteta shape:", dpdteta.shape)
    # exit()

    inner_x = cp.sum(drdp * dpdteta, axis=2)  # (100, 6)
    if(one_or_two == 1):
        dgraddtetax = dgdr[:,:,:6] * inner_x[:, cp.newaxis, :]  # → (3, 100)
    elif(one_or_two == 2):
        dgraddtetax = dgdr[:,:,6:] * inner_x[:, cp.newaxis, :]
    else:
        raise ValueError("one_or_two must be 1 or 2")    
    return dgraddtetax
