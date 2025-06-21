import cupy as cp

get_pairs_kernel = cp.RawKernel(r'''
extern "C" __global__
void get_pairs(    
    const float* __restrict__ pos,
    const int* __restrict__ nlist,
    float* __restrict__ trans,
    int* __restrict__ mask, 
    const float lx,
    const float cutoff,
    const int N_total    )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int tid = blx*blockDim.x + thx;

    if(tid >= N_total) {return;}

    int idx1, idx2;
    float dx, dy, dz, dx2, dy2, dz2;

    idx1 = nlist[tid*2 + 0];    
    idx2 = nlist[tid*2 + 1];

    dx = (pos[idx2*3 + 0] - pos[idx1*3 + 0]);
    dy = (pos[idx2*3 + 1] - pos[idx1*3 + 1]);
    dz = (pos[idx2*3 + 2] - pos[idx1*3 + 2]);   

    if(dx > (0.5*lx)) 
    {
        dx = dx - lx;
    } 
    else if (dx < (-0.5*lx))
    {
        dx = dx + lx;
    }

    if(dy > (0.5*lx)) 
    {
        dy = dy - lx;
    } 
    else if (dy < (-0.5*lx))
    {
        dy = dy + lx;
    }

    if(dz > (0.5*lx)) 
    {
        dz = dz - lx;
    } 
    else if (dz < (-0.5*lx))
    {
        dz = dz + lx;
    }

    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    trans[tid*3 + 0] = dx; 
    trans[tid*3 + 1] = dy; 
    trans[tid*3 + 2] = dz; 

    if(dist < cutoff)
    {
        mask[tid] = 1;
    }
    else
    {
        mask[tid] = 0;
    }


}                           

''', 'get_pairs')


