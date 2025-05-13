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

class DescriptorGenerator():
    """
    This class generates the NEP descriptors from positions 
    at each times step of the simulation.      
    """

    def set_timers(self):
        self.t_pos_to_pts = 0
        self.t_dx = 0 
        self.t_pts_to_nep = 0

        self.t_nep1 = 0
        self.t_nep2 = 0
        self.t_nep3 = 0
        self.t_nep4 = 0
        self.t_nep5 = 0
        self.is_sync = 0


    def setDevice(self,gpu_id):
        self.set_timers()
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)

    def setHyperParameters(self,hypers):
        self.nrad = int(hypers[0])
        self.nang = int(hypers[1])
        self.lmax = int(hypers[2])
        self.cutoff_nep = hypers[3]
        self.N_descriptors = (self.nrad + 1) + (self.nang + 1)*self.lmax
        self.buffer_size = 0 

    def setSync(self,is_sync):
        self.is_sync = is_sync

    def getN_descriptor(self):
        return self.N_descriptors    

    def setBoxSize(self,Lx):
        self.Lx = Lx

    def set_dx_dteta(self,dx,dteta):
        self.dx_gpu = cp.asarray(dx,dtype=cp.float32)
        self.dteta_gpu = cp.asarray(dteta,dtype=cp.float32)    

    def setShape(self,shape_str):
        """
        vertices with the shape symmetry group
        should be consistent with the training
        """
        if(shape_str == 'cube_v2'):
            self.pts_rep = cp.array([
            [1.0,0.0,0.0],
            [-1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,-1.0,0.0],
            [0.0,0.0,1.0],
            [0.0,0.0,-1.0],
            ],dtype=cp.float32)

            self.pts_rep *= 1.7
            self.Npts = 6
            self.Nd = self.Npts*2
            self.en_min = -5.1
            self.en_max = 15.0
            self.cutoff = 5.80
            self.nep_cutoff = 4.5

        elif(shape_str == 'th_v1'):
            self.pts_rep = cp.array([
            [-0.45,-0.45,-0.45],
            [-0.45,0.45,0.45],
            [0.45,-0.45,0.45],
            [0.45,0.45,-0.45]
            ],dtype=cp.float32)

            self.pts_rep *= 1.0
            self.Npts = 4
            self.Nd = self.Npts*2

            self.en_min = -6.7
            self.en_max = 15.0
            self.cutoff = 5.75
            self.nep_cutoff = 4.5


        elif(shape_str == 'pbp_v3'):
            self.pts_rep = cp.array([
            [-1.46946,  2.02254,  0.     ],
            [-2.37764, -0.77254,  0.     ],
            [-0.     , -2.5    ,  0.     ],
            [ 2.37764, -0.77254,  0.     ],
            [ 1.46946,  2.02254,  0.     ],
            [ 0.     ,  0.     ,  3.5    ],
            [ 0.     ,  0.      ,-3.5    ]
            ],dtype=cp.float32)

            self.pts_rep *= 1.0
            self.Npts = 7
            self.Nd = self.Npts*2

            self.en_min = -2.5
            self.en_max = 15.0
            self.cutoff = 8.1
            self.nep_cutoff = 7.5



        else:
            print('shape not found ')
            exit()

        self.nep_cutoff = cp.asarray(self.nep_cutoff,dtype=cp.float32)

    def get_en_min_max(self):
        return self.en_min, self.en_max

    def move_data_to_device(self):
        self.pts_rep = cp.asarray(self.pts_rep,dtype=cp.float32)
        self.dx_gpu = cp.asarray(self.dx_gpu,dtype=cp.float32)
        self.dteta_gpu = cp.asarray(self.dteta_gpu,dtype=cp.float32)        
        

    def generate_nep_descriptors(self,central_pos,orientations,Nlist):
        """
        Generate NEP descriptors from the positions and orientations of the particles
        """
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        self.pos_to_pts(central_pos,orientations,Nlist)
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        self.apply_dx_dteta()
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t2 = time.time()
        self.pts_to_nep_descriptors()
        # self.pts_to_nep_descriptors2()
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()
        t3 = time.time()

        self.t_pos_to_pts += t1-t0
        self.t_dx += t2-t1
        self.t_pts_to_nep += t3-t2

        return self.g_all_cupy, self.pp, self.N_pair

    def pos_to_pts(self,central_pos,orientations,Nlist):
        """
        convert particle center of mass position and orientations to 
        point representation of the particles and populate the possibly interacting pairs
        """
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

        self.pts_pair_rot2 = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
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

        ptp.get_pts_pairs_kernels2(
            blocks,
            threads_per_block,
            (
                QUAT2,
                QUAT1,
                translate,
                self.pts_rep,
                self.pts_pair_rot2,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

    def apply_dx_dteta(self):
        """
        to calculate forces and torques apply dx translation and dteta rotation to 
        the point representation of the pairs 
        """
    
        blocks = (self.N_pair*7,)
        threads_per_block = (self.Nd*3,)
        self.all_pts1 = cp.empty((self.N_pair*7,self.Nd,3),dtype=cp.float32)
        self.all_pts2 = cp.empty((self.N_pair*3,self.Nd,3),dtype=cp.float32)

        # print(self.all_pts1.shape)
        ptp.apply_dx_dteta_kernel(
            blocks,
            threads_per_block,
            (
                self.pts_pair,
                self.all_pts1,
                self.dx_gpu,
                self.dteta_gpu,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

        ptp.apply_dteta_kernel2(
            blocks,
            threads_per_block,
            (
                self.pts_pair_rot2,
                self.all_pts2,
                self.dteta_gpu,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

    def pts_to_nep_descriptors2(self):
        pos_gpu = cp.vstack((self.all_pts1,self.all_pts2 ))
        n_chebysev = self.nrad + 1
        n_pairs = pos_gpu.shape[0]


        self.g_rad = cp.zeros((n_pairs, n_chebysev), dtype=cp.float32)
        self.g_ang = cp.zeros((n_pairs, (self.nang+1)*self.lmax), dtype=cp.float32)

        blocks = (5000,)
        threads_per_block = (32,)


        single_kernel(
            blocks,
            threads_per_block,
            (
                pos_gpu,
                self.g_rad,
                self.g_ang,
                self.nep_cutoff,
                n_pairs,
                n_chebysev,
                cp.int32(self.nang),
                cp.int32(self.lmax),
                cp.int32(self.Nd) 
            )
        )

        self.g_all_cupy = cp.hstack((self.g_rad,self.g_ang))









    def pts_to_nep_descriptors(self):

        

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t0 = time.time()    
        pos_gpu = cp.vstack((self.all_pts1,self.all_pts2 ))
        n_chebysev = self.nrad + 1
        n_pairs = pos_gpu.shape[0]


        if (n_pairs > self.buffer_size):
            print('resizing buffers')
            print(n_pairs)
            self.buffer_size = n_pairs + 2000 
            self.r_ij = cp.empty((self.buffer_size, self.Nd, 3), dtype=cp.float32)
            self.r_ij_norm = cp.empty((self.buffer_size, self.Nd), dtype=cp.float32)
            self.chebo = cp.empty((self.buffer_size, n_chebysev, self.Nd), dtype=cp.float32)
            self.g_rad = cp.empty((self.buffer_size, n_chebysev), dtype=cp.float32)
            
            # self.chebo_angular = cp.empty((self.buffer_size, nang+1, Nd*Nd), dtype=cp.float32)
            # self.lego = cp.empty((self.buffer_size, lmax+1, Nd*Nd), dtype=cp.float32)
            # self.lego_trans = cp.empty((self.buffer_size, Nd*Nd, lmax), dtype=cp.float32)
            # self.temp_matmul = cp.empty((self.buffer_size, nang+1, lmax), dtype=cp.float32)
            # total_cols = (n_chebysev) + (nang + 1) * lmax
            # self.g_all_cupy = cp.empty((self.buffer_size, total_cols), dtype=cp.float32)
        
        # r_ij = cp.empty_like(pos_gpu)
        # r_ij_norm = cp.empty((n_pairs,self.Nd),cp.float32)
        # chebo = cp.empty((n_pairs,n_chebysev,self.Nd),cp.float32)
        # g_rad = cp.empty((n_pairs,n_chebysev),cp.float32)
        self.g_rad2 = cp.zeros((self.buffer_size, n_chebysev), dtype=cp.float32)
        self.g_ang2 = cp.zeros((n_pairs, (self.nang+1)*self.lmax), dtype=cp.float32)
        

        
        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t1 = time.time()
        blocks = (n_pairs,)
        threads_per_block = (12,)

        radial_cuda_kernel(
            blocks,
            threads_per_block,
            (
                pos_gpu,
                self.r_ij,
                self.r_ij_norm,
                self.chebo,
                self.g_rad,
                self.nep_cutoff,
                n_pairs,
                n_chebysev,
                cp.int32(self.Nd) 
            )
        )

        # g_rad = cp.sum(chebo,axis=-1)

        # cp.sum(self.chebo,axis=-1,out=self.g_rad)

        # diff = cp.abs(self.g_rad[:n_pairs] - self.g_rad2[:n_pairs])
       
        # print(cp.max(diff))    

        # # print(self.g_rad[:10])
        # # print(self.g_rad2[:10])
        # exit()


        chebo_sliced = self.chebo[:,:self.nang+1]

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t2 = time.time()
        
        

        # chebo_angular = cp.einsum('...i,...j->...ij', chebo_sliced, chebo_sliced).reshape(n_pairs, self.nang+1, -1)
        chebo_angular = cp.einsum('...i,...j->...ij', chebo_sliced, chebo_sliced).reshape(self.buffer_size, self.nang+1, -1)

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t3 = time.time()
        
        blocks = (n_pairs,)
        threads_per_block = (self.Nd*self.Nd,)
        # lego = cp.empty((n_pairs,self.lmax + 1,self.Nd*self.Nd),cp.float32)
        lego = cp.empty((self.buffer_size,self.lmax + 1,self.Nd*self.Nd),cp.float32)

        angular_cuda_kernel(
            blocks,
            threads_per_block,
            (
                self.r_ij,
                self.r_ij_norm,
                lego,
                n_pairs,
                self.Nd,
                self.lmax
            )
        )

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t4 = time.time()

        lego_trans = lego[:, 1:, :].transpose(0, 2, 1)



        g_ang = cp.matmul(chebo_angular, lego_trans)
        # g_ang = g_ang.reshape(n_pairs, -1)
        g_ang = g_ang.reshape(self.buffer_size, -1)

        # plt.figure(1)
        # plt.hist(g_ang.get().flatten(),bins=100)
        # plt.hist(lego_trans.get().flatten(),bins=100)
        # plt.hist(self.g_ang2[:n_pairs].get().flatten(),bins=100)
        # plt.show()


        # print(g_ang[1])
        # print(self.g_ang2[1])
        # print(g_ang3)
        # exit()

        # diff = cp.abs(g_ang[:n_pairs] - self.g_ang2[:n_pairs])
       
        # print(cp.max(diff))    
        # exit()


        # self.g_all_cupy = cp.hstack((g_rad[:n_pairs],g_ang[:n_pairs]))
        self.g_all_cupy = cp.hstack((self.g_rad[:n_pairs],g_ang[:n_pairs]))

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t5 = time.time()

        self.t_nep1 += t1-t0
        self.t_nep2 += t2-t1
        self.t_nep3 += t3-t2
        self.t_nep4 += t4-t3
        self.t_nep5 += t5-t4





















def dummy():
    pass    