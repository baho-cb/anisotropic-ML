import cupy as cp 



dudq_dqdx_sum_kernel = cp.RawKernel(r'''
extern "C" __global__
void dudq_dqdx_sum_kernel(    
    const float* __restrict__ dudq, // [Np, n_desc]
    const float* __restrict__ dqdx1, // [Np, n_desc]
    const float* __restrict__ dqdx2, // [Np, n_desc]
    const float* __restrict__ dqdx3, // [Np, n_desc]
    const float* __restrict__ dqdx4, // [Np, n_desc]
    const float* __restrict__ dqdx5, // [Np, n_desc]
    const float* __restrict__ dqdx6, // [Np, n_desc]
    const float* __restrict__ dqdx7, // [Np, n_desc]
    const float* __restrict__ dqdx8, // [Np, n_desc]
    const float* __restrict__ dqdx9, // [Np, n_desc]
    float* __restrict__ tork1, // [Np, 3]
    float* __restrict__ tork2, // [Np, 3]   
    float* __restrict__ force, // [Np, 3]   
    const int Np,
    const int n_desc,
    const float en_range
    )

{
    const int blx = blockIdx.x;
    const int thx = threadIdx.x;
    const int tid = blx*blockDim.x + thx;  

    if (tid >= Np*9) return;

    if (tid < Np*1)
    {
        const int c_id = tid - Np*0;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx1[base + i];        
        }
        c_sum *= -en_range;
        tork1[c_id*3 + 0] = c_sum;
        return;
    }   

    if (tid < Np*2)
    {
        const int c_id = tid - Np*1;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx2[base + i];        
        }
        c_sum *= -en_range;
        tork1[c_id*3 + 1] = c_sum;
        return;
    }   

    if (tid < Np*3)
    {
        const int c_id = tid - Np*2;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx3[base + i];        
        }
        c_sum *= -en_range;
        tork1[c_id*3 + 2] = c_sum;
        return;
    }   

    if (tid < Np*4)
    {
        const int c_id = tid - Np*3;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx4[base + i];        
        }
        c_sum *= -en_range;
        tork2[c_id*3 + 0] = c_sum;
        return;
    }   

    if (tid < Np*5)
    {
        const int c_id = tid - Np*4;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx5[base + i];        
        }
        c_sum *= -en_range;
        tork2[c_id*3 + 1] = c_sum;
        return;
    }   

    if (tid < Np*6)
    {
        const int c_id = tid - Np*5;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx6[base + i];        
        }
        c_sum *= -en_range;
        tork2[c_id*3 + 2] = c_sum;
        return;
    }   

    if (tid < Np*7)
    {
        const int c_id = tid - Np*6;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx7[base + i];        
        }
        c_sum *= -en_range;
        force[c_id*3 + 0] = c_sum;
        return;
    }   

    if (tid < Np*8)
    {
        const int c_id = tid - Np*7;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx8[base + i];        
        }
        c_sum *= -en_range;
        force[c_id*3 + 1] = c_sum;
        return;
    }   

    if (tid < Np*9)
    {
        const int c_id = tid - Np*8;
        int base = c_id*n_desc; 
        float c_sum = 0.0f;
        for (int i=0 ; i < n_desc ; i++)
        {
            c_sum += dudq[base + i]*dqdx9[base + i];        
        }
        c_sum *= -en_range;
        force[c_id*3 + 2] = c_sum;
        return;
    }   


}


''', 'dudq_dqdx_sum_kernel')








index_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void index_add_kernel(    
    const float* __restrict__ tork1, // [Npair, 3]
    const float* __restrict__ tork2, // [Npair, 3]
    const float* __restrict__ force, // [Npair, 3]
    const int* __restrict__ indices, // [Npair, 2] 
    float* __restrict__ net_tork,    // [Nparticle, 3]
    float* __restrict__ net_force,   // [Nparticle, 3]   
    const int Npair,
    const int Nparticle
    )

{

    const int blx = blockIdx.x;
    const int thx = threadIdx.x;
    const int tid = blx*blockDim.x + thx;  

    // if(tid >= 3*Npair) return;

    if(tid < Npair)
    {
        int target_idx = indices[tid*2 + 0];
        float c_val = tork1[tid*3 + 0];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 0],c_val);
        }
        return;
    }

    if(tid < 2*Npair)
    {
        int c_tid = tid - Npair;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = tork2[c_tid*3 + 0];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 0],c_val);
        }
        return;
    }

    if(tid < 3*Npair)
    {
        int c_tid = tid - Npair*2;
        int target_idx = indices[c_tid*2 + 0];
        float c_val = tork1[c_tid*3 + 1];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 1],c_val);
        }
        return;
    }

    if(tid < 4*Npair)
    {
        int c_tid = tid - Npair*3;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = tork2[c_tid*3 + 1];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 1],c_val);
        }
        return;
    }


    if(tid < 5*Npair)
    {
        int c_tid = tid - Npair*4;
        int target_idx = indices[c_tid*2 + 0];
        float c_val = tork1[c_tid*3 + 2];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 2],c_val);
        }
        return;
    }

    if(tid < 6*Npair)
    {
        int c_tid = tid - Npair*5;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = tork2[c_tid*3 + 2];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_tork[target_idx*3 + 2],c_val);
        }
        return;
    }



    ////////// FORCES /////////////

    if(tid < 7*Npair)
    {
        int c_tid = tid - Npair*6;
        int target_idx = indices[c_tid*2 + 0];
        float c_val = force[c_tid*3 + 0];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 0],c_val);
        }
        return;
    }

    if(tid < 8*Npair)
    {
        int c_tid = tid - Npair*7;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = -force[c_tid*3 + 0];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 0],c_val);
        }
        return;
    }

    if(tid < 9*Npair)
    {
        int c_tid = tid - Npair*8;
        int target_idx = indices[c_tid*2 + 0];
        float c_val = force[c_tid*3 + 1];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 1],c_val);
        }
        return;
    }

    if(tid < 10*Npair)
    {
        int c_tid = tid - Npair*9;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = -force[c_tid*3 + 1];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 1],c_val);
        }
        return;
    }

    if(tid < 11*Npair)
    {
        int c_tid = tid - Npair*10;
        int target_idx = indices[c_tid*2 + 0];
        float c_val = force[c_tid*3 + 2];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 2],c_val);
        }
        return;
    }

    if(tid < 12*Npair)
    {
        int c_tid = tid - Npair*11;
        int target_idx = indices[c_tid*2 + 1];
        float c_val = -force[c_tid*3 + 2];
        if(target_idx < Nparticle)
        {
        atomicAdd(&net_force[target_idx*3 + 2],c_val);
        }
        return;
    }

































}


''', 'index_add_kernel')




        # if(c_val > 100.0f){c_val = 100.0f;}
        # if(c_val < -100.0f){c_val = -100.0f;}

        # int target_idx2 = indices[tid*2 + 1];
        # c_val = tork2[tid*3 + 0];
        # if(c_val > 100.0f){c_val = 100.0f;}
        # if(c_val < -100.0f){c_val = -100.0f;}
        # atomicAdd(&net_tork[target_idx2*3 + 0],c_val);
        # __syncthreads();
        # return;



