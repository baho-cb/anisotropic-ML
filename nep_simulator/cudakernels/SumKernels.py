import cupy as cp 





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