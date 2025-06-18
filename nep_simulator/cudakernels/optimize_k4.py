import numpy as np 
import cupy as cp 
import time 

def func(_dlegdteta, _dlegdxyz, _dgdteta, _dgdxyz, _grad):

    nang = 8
    lmax = 4 
    N_pair = 6000 
    Nd = 12
    nrad = 11 

    _gradforang = _grad[:,:nang+1]
    _gradforang = cp.ascontiguousarray(_gradforang)
    _dgdtetaforang = _dgdteta[:,:nang+1]
    _dgdtetaforang = cp.ascontiguousarray(_dgdtetaforang)
    _dgdxyzforang = _dgdxyz[:,:nang+1]
    _dgdxyzforang = cp.ascontiguousarray(_dgdxyzforang)

    # dl1 = _dlegdteta[:,:,:,0]
    # dl2 = _dlegdteta[:,:,:,1]
    # dl3 = _dlegdteta[:,:,:,2]
    # dl4 = _dlegdteta[:,:,:,3]
    # dl5 = _dlegdteta[:,:,:,4]
    # dl6 = _dlegdteta[:,:,:,5]

    # dl7 = _dlegdxyz[:,:,:,0]
    # dl8 = _dlegdxyz[:,:,:,1]
    # dl9 = _dlegdxyz[:,:,:,2]

    # dg1 = _dgdtetaforang[:,:,:,0]
    # dg2 = _dgdtetaforang[:,:,:,1]
    # dg3 = _dgdtetaforang[:,:,:,2]
    # dg4 = _dgdtetaforang[:,:,:,3]
    # dg5 = _dgdtetaforang[:,:,:,4]
    # dg6 = _dgdtetaforang[:,:,:,5]

    # dg7 = _dgdxyzforang[:,:,:,0]
    # dg8 = _dgdxyzforang[:,:,:,1]
    # dg9 = _dgdxyzforang[:,:,:,2]

    dl1 = _dlegdteta[0,:,:,:]
    dl2 = _dlegdteta[1,:,:,:]
    dl3 = _dlegdteta[2,:,:,:]
    dl4 = _dlegdteta[3,:,:,:]
    dl5 = _dlegdteta[4,:,:,:]
    dl6 = _dlegdteta[5,:,:,:]

    dl7 = _dlegdxyz[0,:,:,:]
    dl8 = _dlegdxyz[1,:,:,:]
    dl9 = _dlegdxyz[2,:,:,:]

    dg1 = _dgdtetaforang[0,:,:,:]
    dg2 = _dgdtetaforang[1,:,:,:]
    dg3 = _dgdtetaforang[2,:,:,:]
    dg4 = _dgdtetaforang[3,:,:,:]
    dg5 = _dgdtetaforang[4,:,:,:]
    dg6 = _dgdtetaforang[5,:,:,:]

    dg7 = _dgdxyzforang[6,:,:,:]
    dg8 = _dgdxyzforang[7,:,:,:]
    dg9 = _dgdxyzforang[8,:,:,:]

    Nupper = 78
    _gang = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)

    out1 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out2 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out3 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out4 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out5 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out6 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out7 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out8 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)
    out9 = cp.empty((N_pair, (nang+1) * lmax, Nupper), dtype=cp.float32)

    dl1 = cp.ascontiguousarray(dl1, dtype=cp.float32)
    dl2 = cp.ascontiguousarray(dl2, dtype=cp.float32)
    dl3 = cp.ascontiguousarray(dl3, dtype=cp.float32)
    dl4 = cp.ascontiguousarray(dl4, dtype=cp.float32)
    dl5 = cp.ascontiguousarray(dl5, dtype=cp.float32)
    dl6 = cp.ascontiguousarray(dl6, dtype=cp.float32)
    dl7 = cp.ascontiguousarray(dl7, dtype=cp.float32)
    dl8 = cp.ascontiguousarray(dl8, dtype=cp.float32)
    dl9 = cp.ascontiguousarray(dl9, dtype=cp.float32)

    dg1 = cp.ascontiguousarray(dg1, dtype=cp.float32)
    dg2 = cp.ascontiguousarray(dg2, dtype=cp.float32)
    dg3 = cp.ascontiguousarray(dg3, dtype=cp.float32)
    dg4 = cp.ascontiguousarray(dg4, dtype=cp.float32)
    dg5 = cp.ascontiguousarray(dg5, dtype=cp.float32)
    dg6 = cp.ascontiguousarray(dg6, dtype=cp.float32)
    dg7 = cp.ascontiguousarray(dg7, dtype=cp.float32)
    dg8 = cp.ascontiguousarray(dg8, dtype=cp.float32)
    dg9 = cp.ascontiguousarray(dg9, dtype=cp.float32)

    cp_N_pair = cp.int32(N_pair)
    cp_nangp1 = cp.int32(nang + 1)
    nangp1 = nang + 1
    cp_lmax = cp.int32(lmax)
    cp_Nd = cp.int32(Nd)
    cp_nradp1 = cp.int32(nrad + 1)


Np = 6000 
lmax = 4
Nd = 12 



np.random.seed(12)

dlegdteta = np.random.rand(Np,lmax,Nd*Nd,6)
dlegdteta = cp.asarray(dlegdteta, dtype=cp.float32)

dlegdxyz = np.random.rand(Np,lmax,Nd*Nd,3)
dlegdxyz = cp.asarray(dlegdxyz, dtype=cp.float32)

dgdteta = np.random.rand(Np,11,Nd,6)
dgdteta = cp.asarray(dgdteta, dtype=cp.float32)

dgdxyz = np.random.rand(Np,11,Nd,3)
dgdxyz = cp.asarray(dgdxyz, dtype=cp.float32)

grad = np.random.rand(Np,11,12)
grad = cp.asarray(grad, dtype=cp.float32)


for i in range(2):
    func(dlegdteta, dlegdxyz, dgdteta, dgdxyz, grad)

cp.cuda.Stream.null.synchronize()
t0 = time.time()

for i in range(2000):
    func(dlegdteta, dlegdxyz, dgdteta, dgdxyz, grad)

cp.cuda.Stream.null.synchronize()
t1 = time.time()

print(t1-t0)





























print('done')