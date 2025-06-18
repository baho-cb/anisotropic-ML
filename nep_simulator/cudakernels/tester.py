import cupy as cp
import numpy as np 
from cupyx.profiler import benchmark
import DangDtetaKernel as ccc
import DerivativeKernels2 as cuda_dk2
import time 
import sys
np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)

Np = 5975 
nangp1 = 9 
Nd = 12 
lmax = 4 


np.random.seed(12)
# identical random inputs
# g   = np.random.rand(Np, nangp1, Nd).astype(np.float32)
# leg = np.random.rand(Np, lmax, Nd*Nd).astype(np.float32)
# dL  = np.random.rand(Np, lmax, Nd*Nd, 6).astype(np.float32)
# dg  = np.random.rand(Np, nangp1, Nd, 6).astype(np.float32)
g = np.load('../grad.npy')
leg = np.load('../leg.npy')
dg = np.load('../dg.npy')
dleg = np.load('../dleg.npy')
out_true = np.load('../out.npy')

g = cp.asarray(g,dtype=cp.float32)
leg = cp.asarray(leg,dtype=cp.float32)
dl = cp.asarray(dleg,dtype=cp.float32)
dg = cp.asarray(dg,dtype=cp.float32)
out_true = cp.asarray(out_true,dtype=cp.float32)


dg1 = dg[:,:,:,0]
dg2 = dg[:,:,:,1]
dg3 = dg[:,:,:,2]
dg4 = dg[:,:,:,3]
dg5 = dg[:,:,:,4]
dg6 = dg[:,:,:,5]

dl1 = dl[:,:,:,0]
dl2 = dl[:,:,:,1]
dl3 = dl[:,:,:,2]
dl4 = dl[:,:,:,3]
dl5 = dl[:,:,:,4]
dl6 = dl[:,:,:,5]

# print(dl1[0,0].reshape(12,12))
# print(dg1[0,:])
# exit()

out_ref = cp.empty((Np, nangp1 * lmax, 144, 6), dtype=cp.float32)
out_small1 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small2 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small3 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small4 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small5 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small6 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)

blocks = (Np*nangp1*lmax,)
threads_per_block = (Nd*Nd*6,)

cuda_dk2.dqang_dteta_kernel(
            blocks, 
            threads_per_block,
            (g, 
            leg, 
            dl, 
            dg,
            out_ref, 
            cp.int32(Np),
            cp.int32(nangp1),
            cp.int32(lmax),
            cp.int32(Nd)
            ))



# blocks = (Np*(nangp1)*lmax*6,)
# threads_per_block = (Nd*Nd,)

# blocks = (Np*(nangp1)*lmax,)
# blocks = (1400000,)
# threads_per_block = (12,12,)

# blocks = (6000,36,)
blocks = (216000,)
threads_per_block = (12,12,)

# blocks = (800000,)
# threads_per_block = (256,)

# blocks = (400000,)
# threads_per_block = (512,)


cuda_dk2.dqang_dteta_kernel2(
            blocks, 
            threads_per_block,
            (g, 
                leg,
                dl1, 
                dl2, 
                dl3, 
                dl4, 
                dl5, 
                dl6, 
                dg1,
                dg2,
                dg3,
                dg4,
                dg5,
                dg6,
                out_small1, 
                out_small2, 
                out_small3, 
                out_small4, 
                out_small5, 
                out_small6, 
                cp.int32(Np),
                cp.int32(nangp1),
                cp.int32(lmax),
                cp.int32(Nd),
                ))

# diff = cp.abs(out_small1 - out_true[:,:,:,0])
# diff = cp.abs(out_ref - out_true)
print(out_small1[0,0].reshape(12,12))
print(dg1[0,0])
print(out_ref[0,0,:,0].reshape(12,12))

# print(cp.max(diff))
exit()

# print(out_small4[0,0].reshape(12,12))
# exit()
t0 = time.time()

for i in range(2000):


    cuda_dk2.dqang_dteta_kernel2(
                blocks, 
                threads_per_block,
                (g, 
                leg,
                dl1, 
                dl2, 
                dl3, 
                dl4, 
                dl5, 
                dl6, 
                dg1,
                dg2,
                dg3,
                dg4,
                dg5,
                dg6,
                out_small1, 
                out_small2, 
                out_small3, 
                out_small4, 
                out_small5, 
                out_small6, 
                cp.int32(Np),
                cp.int32(nangp1),
                cp.int32(lmax),
                cp.int32(Nd),
                ))


cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(t1-t0)

print()


print(cp.max(out_small2))
print(cp.min(out_small2))
