import cupy as cp 
import numpy as np
import time
import SumKernels as sk 

Nupper = 78 
Nd = 12 
Np = 6000
nrad = 11 
nang = 36

ins = []
outs = []
out_trues = []
np.random.seed(12)
for i in range(10):
    v = np.random.rand(Np,nang,Nupper)
    v = cp.asarray(v,dtype=cp.float32)
    ins.append(v)
    outs.append(cp.empty((Np,nang),dtype=cp.float32))
    out_trues.append(cp.sum(v,axis=-1))

blocks = ((Np*nang*10//256) + 100,)
threads = (256,)


# sk.sum_kernel3(
#         blocks,
#         threads,
#         (ins[0], outs[0],
#         ins[1], outs[1],
#         ins[2], outs[2],
#         ins[3], outs[3],
#         ins[4], outs[4],
#         ins[5], outs[5],
#         ins[6], outs[6],
#         ins[7], outs[7],
#         ins[8], outs[8],
#         ins[9], outs[9],
#           Np, nang, Nupper)
#     )

# for i in range(10):
#     diff = cp.abs(out_trues[i] - outs[i])
#     print(cp.max(diff))

# exit()
for i in range(2):
    sk.sum_kernel3(
        blocks,
        threads,
        (ins[0], outs[0],
        ins[1], outs[1],
        ins[2], outs[2],
        ins[3], outs[3],
        ins[4], outs[4],
        ins[5], outs[5],
        ins[6], outs[6],
        ins[7], outs[7],
        ins[8], outs[8],
        ins[9], outs[9],
          Np, nang, Nupper)
    )


    # sums = []
    # for i in range(10):
    #     sums.append(cp.sum(ins[i],axis=-1))
    # dgangdteta = cp.stack((ins[0],ins[1],ins[2],ins[3],ins[4],ins[5]),axis=2)
    # dgangdxyz = cp.stack((ins[6],ins[7],ins[8]),axis=2)
    # dqangdteta = cp.sum(dgangdteta, axis=-1)
    # dqangdxyz = cp.sum(dgangdxyz, axis=-1)

cp.cuda.Stream.null.synchronize()
t0 = time.time()


for i in range(2000):
    # sums = []
    # for i in range(10):
    #     sums.append(cp.sum(ins[i],axis=-1))

    sk.sum_kernel3(
        blocks,
        threads,
        (ins[0], outs[0],
        ins[1], outs[1],
        ins[2], outs[2],
        ins[3], outs[3],
        ins[4], outs[4],
        ins[5], outs[5],
        ins[6], outs[6],
        ins[7], outs[7],
        ins[8], outs[8],
        ins[9], outs[9],
          Np, nang, Nupper)
    )



    # dgangdteta = cp.stack((ins[0],ins[1],ins[2],ins[3],ins[4],ins[5]),axis=2)
    # dgangdxyz = cp.stack((ins[6],ins[7],ins[8]),axis=2)
    # dqangdteta = cp.sum(dgangdteta, axis=-1)
    # dqangdxyz = cp.sum(dgangdxyz, axis=-1)



cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(t1-t0)

































print('done')