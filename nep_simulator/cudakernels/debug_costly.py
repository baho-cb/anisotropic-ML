import cupy as cp
import numpy as np 
from cupyx.profiler import benchmark
import DangDtetaKernel as ccc
import DerivativeKernels2 as cuda_dk2
import time 
import sys
np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)

def maxerr(arr1,arr2):
    if(arr1.shape != arr2.shape):
        print("Shape of arr1:", arr1.shape)
        print("Shape of arr2:", arr2.shape)
        raise ValueError("Arrays must have the same shape for maxerr calculation.")
    
    return cp.max(cp.abs(arr1 - arr2))

Np = 5975 
nangp1 = 9 
Nd = 12 
lmax = 4 

# for blx in range(216000):
#     ig = blx//lmax
#     il = blx%lmax + (blx//36)*lmax 
#     print(ig,il)
#     if(blx==360):
#         exit()




#     for thy in range(12):
#         for thx in range(12):
#             ind_gij = (blx//lmax)*Nd + thx
#             # ind_leg = ()
#             multi_gij = np.unravel_index(ind_gij, (Np,9,12))
#             # print(multi_gij)
#             print(blx//lmax)

# exit()
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

dgxyz = np.load('../dgxyz.npy')
dlegxyz = np.load('../dlegxyz.npy')
gang_true = np.load('../gang.npy')
outxyz_true = np.load('../outxyz.npy')



g = cp.asarray(g,dtype=cp.float32)
leg = cp.asarray(leg,dtype=cp.float32)
dl = cp.asarray(dleg,dtype=cp.float32)
dg = cp.asarray(dg,dtype=cp.float32)
out_true = cp.asarray(out_true,dtype=cp.float32)

dgxyz = cp.asarray(dgxyz,dtype=cp.float32)
dlegxyz = cp.asarray(dlegxyz,dtype=cp.float32)
gang_true = cp.asarray(gang_true,dtype=cp.float32)
outxyz_true = cp.asarray(outxyz_true,dtype=cp.float32)


dgxyz1 = dgxyz[:,:,:,0]
dgxyz2 = dgxyz[:,:,:,1]
dgxyz3 = dgxyz[:,:,:,2]

dlegxyz1 = dlegxyz[:,:,:,0]
dlegxyz2 = dlegxyz[:,:,:,1]
dlegxyz3 = dlegxyz[:,:,:,2]

dgxyz1 = cp.ascontiguousarray(dgxyz1, dtype=cp.float32)
dgxyz2 = cp.ascontiguousarray(dgxyz2, dtype=cp.float32)
dgxyz3 = cp.ascontiguousarray(dgxyz3, dtype=cp.float32)

dlegxyz1 = cp.ascontiguousarray(dlegxyz1, dtype=cp.float32)
dlegxyz2 = cp.ascontiguousarray(dlegxyz2, dtype=cp.float32)
dlegxyz3 = cp.ascontiguousarray(dlegxyz3, dtype=cp.float32)



dg1 = dg[:,:,:,0]
dg2 = dg[:,:,:,1]
dg3 = dg[:,:,:,2]
dg4 = dg[:,:,:,3]
dg5 = dg[:,:,:,4]
dg6 = dg[:,:,:,5]

dg1 = cp.ascontiguousarray(dg1, dtype=cp.float32)
dg2 = cp.ascontiguousarray(dg2, dtype=cp.float32)
dg3 = cp.ascontiguousarray(dg3, dtype=cp.float32)
dg4 = cp.ascontiguousarray(dg4, dtype=cp.float32)
dg5 = cp.ascontiguousarray(dg5, dtype=cp.float32)
dg6 = cp.ascontiguousarray(dg6, dtype=cp.float32)


dl1 = dl[:,:,:,0]
dl2 = dl[:,:,:,1]
dl3 = dl[:,:,:,2]
dl4 = dl[:,:,:,3]
dl5 = dl[:,:,:,4]
dl6 = dl[:,:,:,5]

dl1 = cp.ascontiguousarray(dl1, dtype=cp.float32)
dl2 = cp.ascontiguousarray(dl2, dtype=cp.float32)
dl3 = cp.ascontiguousarray(dl3, dtype=cp.float32)
dl4 = cp.ascontiguousarray(dl4, dtype=cp.float32)
dl5 = cp.ascontiguousarray(dl5, dtype=cp.float32)
dl6 = cp.ascontiguousarray(dl6, dtype=cp.float32)


out_ref = cp.empty((Np, nangp1 * lmax, 144, 6), dtype=cp.float32)
out_small1 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small2 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small3 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small4 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small5 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small6 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small7 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small8 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small9 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)
out_small10 = cp.empty((Np, nangp1 * lmax, 144), dtype=cp.float32)

blocks = (Np*nangp1*lmax,)
threads_per_block = (Nd*Nd*6,)

# cuda_dk2.dqang_dteta_kernel(
#             blocks, 
#             threads_per_block,
#             (g, 
#             leg, 
#             dl, 
#             dg,
#             out_ref, 
#             cp.int32(Np),
#             cp.int32(nangp1),
#             cp.int32(lmax),
#             cp.int32(Nd)
#             ))


blocks = (216000,)
threads_per_block = (12,12,)

cuda_dk2.dqang_dteta_kernel_faster(
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
                dlegxyz1,
                dlegxyz2,
                dlegxyz3,
                dg1,
                dg2,
                dg3,
                dg4,
                dg5,
                dg6,
                dgxyz1,
                dgxyz2,
                dgxyz3,
                out_small1, 
                out_small2, 
                out_small3, 
                out_small4, 
                out_small5, 
                out_small6, 
                out_small7, 
                out_small8, 
                out_small9, 
                out_small10, 
                cp.int32(Np),
                cp.int32(nangp1),
                cp.int32(lmax),
                cp.int32(Nd),
                ))

print(maxerr(out_small1, out_true[:,:,:,0]))
print(maxerr(out_small2, out_true[:,:,:,1]))
print(maxerr(out_small3, out_true[:,:,:,2]))
print(maxerr(out_small4, out_true[:,:,:,3]))
print(maxerr(out_small5, out_true[:,:,:,4]))
print(maxerr(out_small6, out_true[:,:,:,5]))

print(maxerr(out_small7, outxyz_true[:,:,:,0]))
print(maxerr(out_small8, outxyz_true[:,:,:,1]))
print(maxerr(out_small9, outxyz_true[:,:,:,2]))

print(maxerr(out_small10, gang_true))
exit()


t0 = time.time()

for i in range(2000):


    cuda_dk2.dqang_dteta_kernel_faster(
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
                dlegxyz1,
                dlegxyz2,
                dlegxyz3,
                dg1,
                dg2,
                dg3,
                dg4,
                dg5,
                dg6,
                dgxyz1,
                dgxyz2,
                dgxyz3,
                out_small1, 
                out_small2, 
                out_small3, 
                out_small4, 
                out_small5, 
                out_small6, 
                out_small7, 
                out_small8, 
                out_small9, 
                out_small10, 
                cp.int32(Np),
                cp.int32(nangp1),
                cp.int32(lmax),
                cp.int32(Nd),
                ))


cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(t1-t0)


