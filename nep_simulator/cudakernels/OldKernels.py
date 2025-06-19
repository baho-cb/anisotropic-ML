dqang_dxyz_kernel = cp.RawKernel(r'''

extern "C" __global__
void dqang_dteta_kernel(
    const float *__restrict__ g_rad, // [Np, nangp1, Nd]
    const float *__restrict__ leg, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dlegdxyz, // [Np, lmax, Nd*Nd, 3]
    const float *__restrict__ dgdxyz, // [Np, nangp1, Nd, 3]
    float       *__restrict__ out, // [Np, nangp1*lmax, Nd*Nd, 3]
    float       *__restrict__ g_ang, // [Np, nangp1*lmax, Nd*Nd]
    const int Np,                       
    const int nangp1,
    const int lmax,
    const int Nd)   
{
    int blx = blockIdx.x; // 0 to Np*nangp1*lmax
    int thx = threadIdx.x; // 0 to Nd*Nd*3
                                  
    int index_grad_ij = (blx/lmax) * Nd + (thx/3)%Nd;  
    int index_grad_ik = (blx/lmax) * Nd + (thx/3)/(Nd);  
                                 
    int index_g_ang = blx*Nd*Nd + thx/3;                             

    int g = blx/lmax; 
    int base = g * Nd * 3;                                                             
    
    int mu = thx % 3;
    int tmp = thx / 3;    
    int j = tmp % Nd; 
    int i = tmp / Nd;     
                                  
    int p = blx / (nangp1 * lmax);      // 0  Np-1
    int l = blx % lmax;                 // 0  lmax-1

    int index_leg       = (p * lmax + l) * Nd*Nd     + thx/3;
    int index_dlegdteta = (p * lmax + l) * Nd*Nd*3   + thx;
                                                                                                                        
                                                                                                            
    int index_dgdteta_ik = base + i * 3 + mu;                                                            
    int index_dgdteta_ij = base + j * 3 + mu;                                                            

    int index_target = blx*Nd*Nd*3 + thx;  
                                 
    float var_leg = leg[index_leg];                             
    float grad_ij = g_rad[index_grad_ij];                             
    float grad_ik = g_rad[index_grad_ik];                             

    float term1 = dgdxyz[index_dgdteta_ij] * grad_ik * var_leg;
    float term2 = dgdxyz[index_dgdteta_ik] * grad_ij * var_leg;                                                                                                                          
    float term3 = grad_ij*grad_ik* dlegdxyz[index_dlegdteta];                                                                                                                          
    
    out[index_target] = term1 + term2 + term3;                              
    g_ang[index_g_ang] = grad_ij*grad_ik*var_leg;                                                                               
                                  
}
''', 'dqang_dteta_kernel')


dqang_dteta_kernel_faster = cp.RawKernel(r'''

extern "C" __global__
void dqang_dteta_kernel(
    const float *__restrict__ g_rad, // [Np, nangp1, Nd]
    const float *__restrict__ leg, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl1, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl2, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl3, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl4, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl5, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl6, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl7, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl8, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dl9, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dgdteta1, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta2, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta3, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta4, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta5, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta6, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta7, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta8, // [Np, nangp1, Nd]
    const float *__restrict__ dgdteta9, // [Np, nangp1, Nd]
    float       *__restrict__ out1, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out2, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out3, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out4, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out5, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out6, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out7, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out8, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out9, // [Np, nangp1*lmax, Nd*Nd]
    float       *__restrict__ out10, // [Np, nangp1*lmax, Nd*Nd]
    const int Np,                       
    const int nangp1,
    const int lmax,
    const int Nd)   
{
    int blx = blockIdx.x; // 0 to Np*nangp1*lmax
    int thx = threadIdx.x; // 0 to Nd*Nd*6
    int thy = threadIdx.y;                               

    int index_target = blx*Nd*Nd + thy*Nd + thx;

    if(index_target >= Np*nangp1*lmax*Nd*Nd) return;
                                         
                                                                   
    int index_g_ik = (blx/lmax) * Nd + thx;
    int index_g_ij = (blx/lmax) * Nd + thy;
    int index_leg =  ((blx%lmax) + (blx/(lmax*nangp1))*lmax) * Nd * Nd + thy*Nd + thx;
                                   
    float g_ij = g_rad[index_g_ij];                                     
    float g_ik = g_rad[index_g_ik];                                    
    float leg_var = leg[index_leg];
                                         

    //out1[index_target] = g_rad[index_g_ij]*dgdteta1[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta1[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl1[index_leg];                                                                               
    //out2[index_target] = g_rad[index_g_ij]*dgdteta2[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta2[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl2[index_leg];                                                                              
    //out3[index_target] = g_rad[index_g_ij]*dgdteta3[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta3[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl3[index_leg];                                                                      
    //out4[index_target] = g_rad[index_g_ij]*dgdteta4[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta4[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl4[index_leg];                                                                            
    //out5[index_target] = g_rad[index_g_ij]*dgdteta5[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta5[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl5[index_leg];                                                                   
    //out6[index_target] = g_rad[index_g_ij]*dgdteta6[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta6[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl6[index_leg];                                                                        
    //out7[index_target] = g_rad[index_g_ij]*dgdteta7[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta7[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl7[index_leg];                                                                        
    //out8[index_target] = g_rad[index_g_ij]*dgdteta8[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta8[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl8[index_leg];                                                                        
    //out9[index_target] = g_rad[index_g_ij]*dgdteta9[index_g_ik]*leg[index_leg] + g_rad[index_g_ik]*dgdteta9[index_g_ij]*leg[index_leg] + g_rad[index_g_ij]*g_rad[index_g_ik]*dl9[index_leg];                                                                        
    //out10[index_target] = g_rad[index_g_ij]*g_rad[index_g_ik]*leg[index_leg];
                                         

    out1[index_target] = g_ij*dgdteta1[index_g_ik]*leg_var + g_ik*dgdteta1[index_g_ij]*leg_var + g_ij*g_ik*dl1[index_leg];                                                                               
    out2[index_target] = g_ij*dgdteta2[index_g_ik]*leg_var + g_ik*dgdteta2[index_g_ij]*leg_var + g_ij*g_ik*dl2[index_leg];                                                                              
    out3[index_target] = g_ij*dgdteta3[index_g_ik]*leg_var + g_ik*dgdteta3[index_g_ij]*leg_var + g_ij*g_ik*dl3[index_leg];                                                                      
    out4[index_target] = g_ij*dgdteta4[index_g_ik]*leg_var + g_ik*dgdteta4[index_g_ij]*leg_var + g_ij*g_ik*dl4[index_leg];                                                                            
    out5[index_target] = g_ij*dgdteta5[index_g_ik]*leg_var + g_ik*dgdteta5[index_g_ij]*leg_var + g_ij*g_ik*dl5[index_leg];                                                                   
    out6[index_target] = g_ij*dgdteta6[index_g_ik]*leg_var + g_ik*dgdteta6[index_g_ij]*leg_var + g_ij*g_ik*dl6[index_leg];                                                                        
    out7[index_target] = g_ij*dgdteta7[index_g_ik]*leg_var + g_ik*dgdteta7[index_g_ij]*leg_var + g_ij*g_ik*dl7[index_leg];                                                                        
    out8[index_target] = g_ij*dgdteta8[index_g_ik]*leg_var + g_ik*dgdteta8[index_g_ij]*leg_var + g_ij*g_ik*dl8[index_leg];                                                                        
    out9[index_target] = g_ij*dgdteta9[index_g_ik]*leg_var + g_ik*dgdteta9[index_g_ij]*leg_var + g_ij*g_ik*dl9[index_leg];                                                                        
    out10[index_target] = g_ij*g_ik*leg_var;
                   
}
''', 'dqang_dteta_kernel')

dqang_dteta_kernel = cp.RawKernel(r'''

extern "C" __global__
void dqang_dteta_kernel(
    const float *__restrict__ g_rad, // [Np, nangp1, Nd]
    const float *__restrict__ leg, // [Np, lmax, Nd*Nd]
    const float *__restrict__ dlegdteta, // [Np, lmax, Nd*Nd, 6]
    const float *__restrict__ dgdteta, // [Np, nangp1, Nd, 6]
    float       *__restrict__ out, // [Np, nangp1*lmax, Nd*Nd, 6]
    const int Np,                       
    const int nangp1,
    const int lmax,
    const int Nd)   
{
    int blx = blockIdx.x; // 0 to Np*nangp1*lmax
    int thx = threadIdx.x; // 0 to Nd*Nd*6
                                  
    int index_grad_ij = (blx/lmax) * Nd + (thx/6)%Nd;  
    int index_grad_ik = (blx/lmax) * Nd + (thx/6)/(Nd);  

    int g = blx/lmax; 
    int base = g * Nd * 6;                                                             
    
    int mu = thx % 6;
    int tmp = thx / 6;    
    int j = tmp % Nd; 
    int i = tmp / Nd;     
                                  
    int p = blx / (nangp1 * lmax);      // 0  Np-1
    int l = blx % lmax;                 // 0  lmax-1

    int index_leg       = (p * lmax + l) * Nd*Nd     + thx/6;
    int index_dlegdteta = (p * lmax + l) * Nd*Nd*6   + thx;
                                                                                                                        
                                                                                                            
    int index_dgdteta_ik = base + i * 6 + mu;                                                            
    int index_dgdteta_ij = base + j * 6 + mu;                                                            


    int index_target = blx*Nd*Nd*6 + thx;  
    float var_leg = leg[index_leg];                             
    float grad_ij = g_rad[index_grad_ij];                             
    float grad_ik = g_rad[index_grad_ik];                             


                                  
    //// true part /////
    float term1 = dgdteta[index_dgdteta_ij] * grad_ik * var_leg;
    float term2 = dgdteta[index_dgdteta_ik] * grad_ij * var_leg;                                                                                                                          
    float term3 = grad_ij*grad_ik* dlegdteta[index_dlegdteta];                                                                                                                          
    
    out[index_target] = term1 + term2 + term3;   
    //// true part /////

    //// debug part /////
    //out[index_target] = var_leg;
    //out[index_target] = term1;
                                                                                   
    //// debug part /////
                                  
}
''', 'dqang_dteta_kernel')



sum_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sum_axis2(
        const float* v1, float* out1,
        const float* v2, float* out2,
        const float* v3, float* out3,
        const float* v4, float* out4,
        const float* v5, float* out5,
        const float* v6, float* out6, 
        const int Np, 
        const int lmax,        
        const int nradp1, 
        const int nangp1,
        const int Nd                                                                
                          
    ) {{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        const int N = Np;
        const int J1 = nradp1, L1 = Nd;
        const int J2 = lmax*nangp1, L2 = Nd*Nd;
        const int K3 = 6, K4 = 3, K5 = 6, K6 = 3;

        const int out1_size = N * J1;
        const int out2_size = N * J2;
        const int out3_size = N * J1 * K3;
        const int out4_size = N * J1 * K4;
        const int out5_size = N * J2 * K5;
        const int out6_size = N * J2 * K6;
        const int total_out = out1_size + out2_size
                            + out3_size + out4_size
                            + out5_size + out6_size;

        if (tid >= total_out) return;

        // v1: [N,J1,L1]  out1: [N,J1]
        if (tid < out1_size) {{
            int idx = tid;
            int i = idx / J1;
            int j = idx % J1;
            float s = 0;
            int base = (i*J1 + j)*L1;
            for (int l = 0; l < L1; ++l) s += v1[base + l];
            out1[idx] = s;
            return;
        }}

        // v2: [N,J2,L2]  out2: [N,J2]
        if (tid < out1_size + out2_size) {{
            int idx = tid - out1_size;
            int i = idx / J2;
            int j = idx % J2;
            float s = 0;
            int base = (i*J2 + j)*L2;
            for (int l = 0; l < L2; ++l) s += v2[base + l];
            out2[idx] = s;
            return;
        }}

        // v3: [N,J1,L1,K3]  out3: [N,J1,K3]
        if (tid < out1_size + out2_size + out3_size) {{
            int idx = tid - out1_size - out2_size;
            int i = idx / (J1*K3);
            int rem = idx % (J1*K3);
            int j = rem / K3;
            int k = rem % K3;
            float s = 0;
            int base = ((i*J1 + j)*L1)*K3 + k;
            for (int l = 0; l < L1; ++l) s += v3[base + l*K3];
            out3[idx] = s;
            return;
        }}

        // v4: [N,J1,L1,K4]  out4: [N,J1,K4]
        if (tid < out1_size + out2_size + out3_size + out4_size) {{
            int idx = tid - out1_size - out2_size - out3_size;
            int i = idx / (J1*K4);
            int rem = idx % (J1*K4);
            int j = rem / K4;
            int k = rem % K4;
            float s = 0;
            int base = ((i*J1 + j)*L1)*K4 + k;
            for (int l = 0; l < L1; ++l) s += v4[base + l*K4];
            out4[idx] = s;
            return;
        }}

        // v5: [N,J2,L2,K5]  out5: [N,J2,K5]
        if (tid < out1_size + out2_size + out3_size + out4_size + out5_size) {{
            int idx = tid - out1_size - out2_size - out3_size - out4_size;
            int i = idx / (J2*K5);
            int rem = idx % (J2*K5);
            int j = rem / K5;
            int k = rem % K5;
            float s = 0;
            int base = ((i*J2 + j)*L2)*K5 + k;
            for (int l = 0; l < L2; ++l) s += v5[base + l*K5];
            out5[idx] = s;
            return;
        }}

        // v6: [N,J2,L2,K6]  out6: [N,J2,K6]
        int idx = tid - out1_size - out2_size
                          - out3_size - out4_size
                          - out5_size;
        int i = idx / (J2*K6);
        int rem = idx % (J2*K6);
        int j = rem / K6;
        int k = rem % K6;
        float s = 0;
        int base = ((i*J2 + j)*L2)*K6 + k;
        for (int l = 0; l < L2; ++l) s += v6[base + l*K6];
        out6[idx] = s;
    }}
                          ''', 'sum_axis2')


sum_kernel2 = cp.RawKernel(r'''
    extern "C" __global__
    void sum_axis2(
        const float* v1, float* out1,
        const float* v2, float* out2,
        const float* v3, float* out3,
        const float* v4, float* out4,
        const float* v5, float* out5,
        const float* v6, float* out6, 
        const float* v7, float* out7, 
        const float* v8, float* out8, 
        const float* v9, float* out9, 
        const float* v10, float* out10, 
        const int Np, 
        const int nang,
        const int Nupper                                                                
                          
    ) {{
        int blx = blockIdx.x;
        int bly = blockIdx.y;
        int thx = threadIdx.x;                                      
        int tid = blx * blockDim.x + thx;
        __shared__ float sum;

        if (thx == 0)
        {
        sum = 0.0f;
        }
    
        __syncthreads();

        if(bly==0)
        {
            atomicAdd(&sum, v1[tid]);
            __syncthreads();  
            out1[blx] = sum;
            return;             
        }         
        if(bly==1)
        {
            atomicAdd(&sum, v2[tid]);
            __syncthreads();  
            out2[blx] = sum;
            return;             
        }         
        if(bly==2)
        {
            atomicAdd(&sum, v3[tid]);
            __syncthreads();  
            out3[blx] = sum;
            return;             
        }         
        if(bly==3)
        {
            atomicAdd(&sum, v4[tid]);
            __syncthreads();  
            out4[blx] = sum;
            return;             
        }         
        if(bly==4)
        {
            atomicAdd(&sum, v5[tid]);
            __syncthreads();  
            out5[blx] = sum;
            return;             
        }         
        if(bly==5)
        {
            atomicAdd(&sum, v6[tid]);
            __syncthreads();  
            out6[blx] = sum;
            return;             
        }         
        if(bly==6)
        {
            atomicAdd(&sum, v7[tid]);
            __syncthreads();  
            out7[blx] = sum;
            return;             
        }         
        if(bly==7)
        {
            atomicAdd(&sum, v8[tid]);
            __syncthreads();  
            out8[blx] = sum;
            return;             
        }         
        if(bly==8)
        {
            atomicAdd(&sum, v9[tid]);
            __syncthreads();  
            out9[blx] = sum;
            return;             
        }         
        if(bly==9)
        {
            atomicAdd(&sum, v10[tid]);
            __syncthreads();  
            out10[blx] = sum;
            return;             
        }         
                                                                         
                                  

    }}
                          ''', 'sum_axis2')


sum_kernel3 = cp.RawKernel(r'''
    extern "C" __global__
    void sum_axis2(
        const float* v1, float* out1,
        const float* v2, float* out2,
        const float* v3, float* out3,
        const float* v4, float* out4,
        const float* v5, float* out5,
        const float* v6, float* out6, 
        const float* v7, float* out7, 
        const float* v8, float* out8, 
        const float* v9, float* out9, 
        const float* v10, float* out10, 
        const int Np, 
        const int nang,
        const int Nupper                                                                
                          
    ) {{
        int blx = blockIdx.x;
        int thx = threadIdx.x;                                      
        int tid = blx * blockDim.x + thx;

        int total_out = Np*nang*10; 
        if(tid >= total_out){return;}

        int per_array_out = Np*nang;

        // in [Np,nang,Nupper] -> out [Np,nang]
        if(tid < per_array_out*1)
        {  
            int idx = tid - per_array_out*0;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v1[base + l];
            out1[idx] = s;
            return;        
        }
        if(tid < per_array_out*2)
        {  
            int idx = tid - per_array_out*1;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v2[base + l];
            out2[idx] = s;
            return;        
        }
        if(tid < per_array_out*3)
        {  
            int idx = tid - per_array_out*2;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v3[base + l];
            out3[idx] = s;
            return;        
        }
        if(tid < per_array_out*4)
        {  
            int idx = tid - per_array_out*3;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v4[base + l];
            out4[idx] = s;
            return;        
        }
        if(tid < per_array_out*5)
        {  
            int idx = tid - per_array_out*4;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v5[base + l];
            out5[idx] = s;
            return;        
        }
        if(tid < per_array_out*6)
        {  
            int idx = tid - per_array_out*5;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v6[base + l];
            out6[idx] = s;
            return;        
        }
        if(tid < per_array_out*7)
        {  
            int idx = tid - per_array_out*6;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v7[base + l];
            out7[idx] = s;
            return;        
        }
        if(tid < per_array_out*8)
        {  
            int idx = tid - per_array_out*7;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v8[base + l];
            out8[idx] = s;
            return;        
        }
        if(tid < per_array_out*9)
        {  
            int idx = tid - per_array_out*8;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v9[base + l];
            out9[idx] = s;
            return;        
        }
        if(tid < per_array_out*10)
        {  
            int idx = tid - per_array_out*9;
            int i = idx / nang;
            int j = idx % nang;
            float s = 0;
            int base = (i*nang + j)*Nupper;
            for (int l = 0; l < Nupper; ++l) s += v10[base + l];
            out10[idx] = s;
            return;        
        }               
                                  

    }}
                          ''', 'sum_axis2')


dgdteta_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_dgdteta(    
    const float* __restrict__ dgdr, // [Np, n_cheb, Nd]
    const float* __restrict__ drdp, // [Np, Nd, 3]   
    const float* __restrict__ dpdteta, // [Np, Nd, 3]   
    float* __restrict__ dgdteta, // [Np, n_cheb, Nd, 6] 
    float* __restrict__ dgdxyz, // [Np, n_cheb, Nd, 6] 
    const int n_cheb,                       
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )                               

                               
{
    int blx = blockIdx.x; // 0 to Np*n_cheb  
    int thx = threadIdx.x; // 0 to Nd
    int ndh = Nd/2; // half of Nd              
                               
    if (blx >= (Np*n_cheb) || thx >= Nd) return;
                           
    int index_target = blx*Nd*6 + thx*6;  
    int index_dgdxyz = blx*Nd*3 + thx*3;  
                              
                              
    //int index_dpdteta = (blx%Np)*Nd*3 + thx*3; 
    int index_dpdteta = (blx/n_cheb)*Nd*3 + thx*3; 
    int index_dgdr = blx*Nd + thx;    
    float v_dgdr = dgdr[index_dgdr];                                                                           

    float dpdtetax = dpdteta[index_dpdteta + 0];
    float dpdtetay = dpdteta[index_dpdteta + 1];
    float dpdtetaz = dpdteta[index_dpdteta + 2];
                              
    float drdp_x = drdp[index_dpdteta + 0];
    float drdp_y = drdp[index_dpdteta + 1];
    float drdp_z = drdp[index_dpdteta + 2];

    float inner_dx1 = -dpdtetaz*drdp_y + dpdtetay*drdp_z;
    float inner_dy1 = dpdtetaz*drdp_x - dpdtetax*drdp_z;
    float inner_dz1 = -dpdtetay*drdp_x + dpdtetax*drdp_y;     

    float inner_dx2 = -dpdtetaz*drdp_y + dpdtetay*drdp_z;
    float inner_dy2 = dpdtetaz*drdp_x - dpdtetax*drdp_z;
    float inner_dz2 = -dpdtetay*drdp_x + dpdtetax*drdp_y;                                                     
                                                                                                        
    float dpdxyz = 0.5f; 
                              
    if(thx >= ndh)
    {
       inner_dx1 = 0.0f;
       inner_dy1 = 0.0f;
       inner_dz1 = 0.0f;  
       dpdxyz = -0.5f;                                            
    }                   

    if(thx < ndh)
    {                                                           
        inner_dx2 = 0.0f;
        inner_dy2 = 0.0f;
        inner_dz2 = 0.0f;
    }
                              
    dgdxyz[index_dgdxyz + 0] = v_dgdr * drdp_x * dpdxyz;                         
    dgdxyz[index_dgdxyz + 1] = v_dgdr * drdp_y * dpdxyz;                         
    dgdxyz[index_dgdxyz + 2] = v_dgdr * drdp_z * dpdxyz;                         
                              
    dgdteta[index_target + 0] = v_dgdr * inner_dx1;                          
    dgdteta[index_target + 1] = v_dgdr * inner_dy1;                          
    dgdteta[index_target + 2] = v_dgdr * inner_dz1;                          
    dgdteta[index_target + 3] = v_dgdr * inner_dx2;                          
    dgdteta[index_target + 4] = v_dgdr * inner_dy2;                          
    dgdteta[index_target + 5] = v_dgdr * inner_dz2;                          


                             }                           
''', 'calculate_dgdteta')


cosine_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_dcosdtheta(    
    const float* __restrict__ pts12, // [Np, Nd, 3]
    const float* __restrict__ r12, // [Np, Nd]   
    const float* __restrict__ dp12dteta, // [Np, Nd, 3]   
    const float* __restrict__ dr12dteta, // [Np, Nd, 3] 
    float* __restrict__ cosine, // [Np, Nd*Nd] 
    float* __restrict__ dcosdteta, // [Np, Nd*Nd, 6]
    float* __restrict__ dcosdxyz, // [Np, Nd*Nd, 3]
    float* __restrict__ leg, // [Np, lmax, Nd, Nd]                                                     
    float* __restrict__ dlegdcos, // [Np, lmax, Nd, Nd]                                                     
    const int lmax,
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    const int ndh = Nd >> 1; // Nd / 2 = 6
    if (blx >= Np || thx >= Nd*Nd) return;                         

    int i12 = thx / Nd; // integer division (gives the row index in the 12x12 dot matrix)
    int j12 = thx % Nd; 

    int indexi = blx * (Nd * 3) + i12 * 3;
    int indexj = blx * (Nd * 3) + j12 * 3;                                                  
    int index_cos = blx * Nd * Nd + thx; // Index for the cosine value in the output array
    int index_dcos = blx * Nd * Nd * 6 + thx * 6; // Index for the dcos value in the output array
    int index_dxyz = blx * Nd * Nd * 3 + thx * 3; 
    int index_ri = blx * Nd + i12; // Index for the norm of the first point
    int index_rj = blx * Nd + j12; // Index for the norm of the second point   
 
    float dotp = 0.0f;
    for (int k = 0; k < 3; ++k) {
        float a = pts12[indexi + k];
        float b = pts12[indexj + k];
        dotp += a * b;
    }
    float norm1 = r12[blx * Nd + i12];                         
    float norm2 = r12[blx * Nd + j12];
    cosine[index_cos] = dotp/(norm1*norm2); // Store the cosine value

    
    float dpi1x = 0.f;
    float dpi1y = 0.f;
    float dpi1z = 0.f;
    float dri1x = 0.f;                         
    float dri1y = 0.f;                         
    float dri1z = 0.f;   

    float dpi_trans = -0.5f;    
    float drix_trans = -(pts12[indexi + 0]/r12[index_ri])*0.5f;
    float driy_trans = -(pts12[indexi + 1]/r12[index_ri])*0.5f;
    float driz_trans = -(pts12[indexi + 2]/r12[index_ri])*0.5f;
                                                                                                 
    if( i12 < ndh)
    {
    dpi1x = dp12dteta[indexi + 0];
    dpi1y = dp12dteta[indexi + 1];
    dpi1z = dp12dteta[indexi + 2];
    dri1x = dr12dteta[indexi + 0];                         
    dri1y = dr12dteta[indexi + 1];                         
    dri1z = dr12dteta[indexi + 2];

    dpi_trans = 0.5f;            
    drix_trans = -drix_trans;                                                                
    driy_trans = -driy_trans;                                                                
    driz_trans = -driz_trans;                                                                
    }
 
    float dpj1x = 0.f;
    float dpj1y = 0.f;
    float dpj1z = 0.f;
    float drj1x = 0.f;                         
    float drj1y = 0.f;                         
    float drj1z = 0.f; 
                                                     
    float dpj_trans = -0.5f;
    float drjx_trans = -(pts12[indexj + 0]/r12[index_rj])*0.5f;
    float drjy_trans = -(pts12[indexj + 1]/r12[index_rj])*0.5f;
    float drjz_trans = -(pts12[indexj + 2]/r12[index_rj])*0.5f;
                                                                               
    if( j12 < ndh)
    {
    dpj1x = dp12dteta[indexj + 0];
    dpj1y = dp12dteta[indexj + 1];
    dpj1z = dp12dteta[indexj + 2];
    drj1x = dr12dteta[indexj + 0];                                                  
    drj1y = dr12dteta[indexj + 1];                                                  
    drj1z = dr12dteta[indexj + 2];

    dpj_trans = 0.5f;
    drjx_trans = -drjx_trans;                                                                
    drjy_trans = -drjy_trans;                                                                
    drjz_trans = -drjz_trans;                                                                
                                                                                                                                 
    }
                             
                     
    // for dx1 
    float dot_dpi_pj = -dpi1z*pts12[indexj + 1] + dpi1y*pts12[indexj + 2];
    float dot_pi_dpj = -dpj1z*pts12[indexi + 1] + dpj1y*pts12[indexi + 2];
    float den = norm1 * norm2;
    float num2 = r12[index_ri] * drj1x + r12[index_rj] * dri1x;  
    num2 *= dotp;
    float num1 = (dot_dpi_pj+dot_pi_dpj);                         

    dcosdteta[index_dcos + 0] = (num1*den - num2) / (den*den); // dx1
                             
    // for dy1 
    dot_dpi_pj = dpi1z*pts12[indexj + 0] - dpi1x*pts12[indexj + 2];
    dot_pi_dpj = dpj1z*pts12[indexi + 0] - dpj1x*pts12[indexi + 2];
    num2 = r12[index_ri] * drj1y + r12[index_rj] * dri1y;
    num2 *= dotp;
    num1 = (dot_dpi_pj+dot_pi_dpj);
    dcosdteta[index_dcos + 1] = (num1*den - num2) / (den*den); // dy1
                             
    // for dz1
    dot_dpi_pj = dpi1x*pts12[indexj + 1] - dpi1y*pts12[indexj + 0];
    dot_pi_dpj = dpj1x*pts12[indexi + 1] - dpj1y*pts12[indexi + 0];
    num2 = r12[index_ri] * drj1z + r12[index_rj] * dri1z;
    num2 *= dotp;
    num1 = (dot_dpi_pj+dot_pi_dpj);
    dcosdteta[index_dcos + 2] = (num1*den - num2) / (den*den); // dz1                                                  

    // for dx_trans
    float dot_dpi_pj_trans = dpi_trans * pts12[indexj + 0];
    float dot_pi_dpj_trans = pts12[indexi + 0] * dpj_trans;                                
    
    num2 = r12[index_ri] * drjx_trans + r12[index_rj] * drix_trans;
    num2 *= dotp;
    num1 = (dot_dpi_pj_trans+dot_pi_dpj_trans);
    dcosdxyz[index_dxyz + 0] = (num1*den - num2) / (den*den); // dx_trans
                             
    // for dy_trans
    dot_dpi_pj_trans = dpi_trans * pts12[indexj + 1];
    dot_pi_dpj_trans = pts12[indexi + 1] * dpj_trans;                                
    
    num2 = r12[index_ri] * drjy_trans + r12[index_rj] * driy_trans;
    num2 *= dotp;
    num1 = (dot_dpi_pj_trans+dot_pi_dpj_trans);
    dcosdxyz[index_dxyz + 1] = (num1*den - num2) / (den*den); // dy_trans
                             
    // for dz_trans
    dot_dpi_pj_trans = dpi_trans * pts12[indexj + 2];
    dot_pi_dpj_trans = pts12[indexi + 2] * dpj_trans;                                
    
    num2 = r12[index_ri] * drjz_trans + r12[index_rj] * driz_trans;
    num2 *= dotp;
    num1 = (dot_dpi_pj_trans+dot_pi_dpj_trans);
    dcosdxyz[index_dxyz + 2] = (num1*den - num2) / (den*den); // dz_trans
                             

    // for dx2
    float dpi2x = 0.f;
    float dpi2y = 0.f;
    float dpi2z = 0.f;
    float dri2x = 0.f;                         
    float dri2y = 0.f;                         
    float dri2z = 0.f;                         
    if( i12 >= ndh)
    {
    dpi2x = dp12dteta[indexi + 0];
    dpi2y = dp12dteta[indexi + 1];
    dpi2z = dp12dteta[indexi + 2];
    dri2x = dr12dteta[indexi + 0];                         
    dri2y = dr12dteta[indexi + 1];                         
    dri2z = dr12dteta[indexi + 2];                         
    }
 
    float dpj2x = 0.f;
    float dpj2y = 0.f;
    float dpj2z = 0.f;
    float drj2x = 0.f;                         
    float drj2y = 0.f;                         
    float drj2z = 0.f;                         
                                                      
    if( j12 >= ndh)
    {
    dpj2x = dp12dteta[indexj + 0];
    dpj2y = dp12dteta[indexj + 1];
    dpj2z = dp12dteta[indexj + 2];
    drj2x = dr12dteta[indexj + 0];                                                  
    drj2y = dr12dteta[indexj + 1];                                                  
    drj2z = dr12dteta[indexj + 2];                                                  
    }   

    dot_dpi_pj = -dpi2z*pts12[indexj + 1] + dpi2y*pts12[indexj + 2];
    dot_pi_dpj = -dpj2z*pts12[indexi + 1] + dpj2y*pts12[indexi + 2];
    num2 = r12[index_ri] * drj2x + r12[index_rj] * dri2x;
    num2 *= dotp;
    num1 = (dot_dpi_pj+dot_pi_dpj);
    dcosdteta[index_dcos + 3] = (num1*den - num2) / (den*den); // dx2                                                   
                                                      
    dot_dpi_pj = dpi2z*pts12[indexj + 0] - dpi2x*pts12[indexj + 2];
    dot_pi_dpj = dpj2z*pts12[indexi + 0] - dpj2x*pts12[indexi + 2];
    num2 = r12[index_ri] * drj2y + r12[index_rj] * dri2y;
    num2 *= dotp;
    num1 = (dot_dpi_pj+dot_pi_dpj);
    dcosdteta[index_dcos + 4] = (num1*den - num2) / (den*den); // dy2

    dot_dpi_pj = dpi2x*pts12[indexj + 1] - dpi2y*pts12[indexj + 0];
    dot_pi_dpj = dpj2x*pts12[indexi + 1] - dpj2y*pts12[indexi + 0];
    num2 = r12[index_ri] * drj2z + r12[index_rj] * dri2z;
    num2 *= dotp;
    num1 = (dot_dpi_pj+dot_pi_dpj);
    dcosdteta[index_dcos + 5] = (num1*den - num2) / (den*den); // dz2 

    float cos = dotp / (norm1 * norm2); // Calculate the cosine value                         
    leg[blx * Nd * Nd * lmax + thx] = cos; // Initialize legendre polynomial for l=0
    float lego_n_1 = 1.0f;
    float lego_n = cos;
    float lego_next;

    float dlegdcos_i_1 = 1.0f;                                                  
    float dlegdcos_i;                         
                             
    dlegdcos[blx * Nd * Nd * lmax + thx] = 1.0f; 
    // dlegdcos[blx * Nd * Nd * lmax + thx + Nd*Nd] = 3.0f*cos; 
                             
    // Calculate higher order legendre polynomials
    for (int i_lego = 1; i_lego < lmax  ; i_lego += 1)
    {
        lego_next = ((2.0f*i_lego + 1.0f)*cos*lego_n - lego_n_1*i_lego) / (i_lego+1);
        lego_n_1 = lego_n;
        lego_n = lego_next;
        leg[blx * Nd * Nd * lmax + thx + Nd*Nd*(i_lego)] = lego_next;

        dlegdcos_i = (i_lego + 1) * lego_n_1 + cos*dlegdcos_i_1;                                          
        dlegdcos[blx * Nd * Nd * lmax + thx + Nd*Nd*(i_lego)] = dlegdcos_i;
        dlegdcos_i_1 = dlegdcos_i;                     
                             
    }                                                                          

                             
 
}                           
''', 'calculate_dcosdtheta')

