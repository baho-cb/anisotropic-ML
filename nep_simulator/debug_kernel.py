import numpy as np 
import cupy as cp 

Np = 420 
nangp1 = 9 
lmax = 4 
Nd = 12 

g_rad = np.random.rand(420,9,12)
g_rad = cp.asarray(g_rad,dtype=cp.float32)
gij = cp.tile(g_rad,(1,1,12))
out_list = []
out1 = cp.zeros((Np,lmax*nangp1,Nd*Nd,6),dtype=cp.float32)

for i in range(6):
    out_i = cp.zeros((Np,nangp1*lmax,Nd*Nd), dtype=cp.float32)
    for n in range(nangp1):
        for l in range(lmax):
            term3 = gij[:,n]
            out_i[:,n*lmax + l] = term3
    out_list.append(out_i)


for n in range(nangp1):
    for l in range(lmax):
        term3 = gij[:,n]
            
        out1[:,n*lmax + l,:,0] = term3
        out1[:,n*lmax + l,:,1] = term3
        out1[:,n*lmax + l,:,2] = term3
        out1[:,n*lmax + l,:,3] = term3
        out1[:,n*lmax + l,:,4] = term3
        out1[:,n*lmax + l,:,5] = term3



# print(out1[1,0,:,1])
# print(out_list[1][1,0,:])
# diff = cp.abs(out1[:,:,:,3] - out_list[3])
# print(cp.max(diff))
# exit()


kernel = cp.RawKernel(r'''
extern "C" __global__
void populate(    
    const float *__restrict__ g_rad, // [Np, nangp1, Nd]
    float       *__restrict__ out, // [Np, nangp1*lmax, Nd*Nd, 6]
    const int Np,                       
    const int nangp1,
    const int lmax,
    const int Nd   
    )
{
    int blx = blockIdx.x; // 0 to Np*nangp1*lmax
    int thx = threadIdx.x; // 0 to Nd*Nd*6

    int index_grad_i = (blx/lmax) * Nd + (thx/6)/(Nd);  
    int index_grad_j = (blx/lmax) * Nd + (thx/6)%Nd;  
    int index_target = blx*Nd*Nd*6 + thx;                    
                      
    out[index_target] = g_rad[index_grad_j];                             
                      
}     
                      
                      
''','populate')

out2 = cp.zeros((Np,lmax*nangp1,Nd*Nd,6),dtype=cp.float32)
blocks = (Np*(nangp1)*lmax,)
threads_per_block = (Nd*Nd*6,)

kernel(
    blocks, 
    threads_per_block,
    (g_rad, 
    out2, 
    cp.int32(Np),
    cp.int32(nangp1), 
    cp.int32(lmax),
    cp.int32(Nd)
    ))


diff = cp.abs(out2[:,:,:,3] - out_list[3])
print(cp.max(diff))
exit()




print(out1[0,0,:,0])
print(out2[0,0,:,0])

diff = cp.abs(out1-out2)
print(cp.max(diff))




































print('done ')