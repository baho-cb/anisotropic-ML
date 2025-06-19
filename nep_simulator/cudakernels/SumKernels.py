import cupy as cp 



sum_kernel4 = cp.RawKernel(r'''
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
        const float* dgdteta, float* dqdteta, //[6,Np,11,12] to [6,Np,11]
        const float* dgdxyz, float* dqdxyz, //[3,Np,11,12] to [3,Np,11]
        const float* grad, float* qrad,  // [Np,11,12] to [Np,11]
        const int Np, 
        const int nang,
        const int nrad_desc,
        const int Nupper                                                                
                          
    ) {{
        int blx = blockIdx.x;
        int thx = threadIdx.x;                                      
        int tid = blx * blockDim.x + thx;
        int Nd = 12;

        int total_out = Np*nang*10; 
        //if(tid >= total_out){return;}

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

        // dgdteta dqdteta [6,Np,11,12] to [6,Np,11]
        int first_ten_out = per_array_out*10; 
        int per_dgraddteta = Np*nrad_desc;
        int nrad = nrad_desc;

        for (int tt=0 ; tt < 6 ; tt++)
        {
        
            if(tid < (first_ten_out + per_dgraddteta*(tt+1)))
        {  
            int idx = tid - (first_ten_out+ per_dgraddteta*(tt));
            int i = idx / nrad;
            int j = idx % nrad;
            float s = 0;
            int base = (i*nrad + j)*Nd + tt*Np*nrad*Nd;
            for (int l = 0; l < Nd; ++l) s += dgdteta[base + l];
            dqdteta[idx+Np*nrad*tt] = s;
            return;        
        }

        }

        int continue_from = per_array_out*10 + per_dgraddteta*6 ; 
        int per_dgraddxyz = Np*nrad_desc;

        for (int tt=0 ; tt < 3 ; tt++)
        {
        
            if(tid < (continue_from + per_dgraddxyz*(tt+1)))
        {  
            int idx = tid - (continue_from+ per_dgraddxyz*(tt));
            int i = idx / nrad;
            int j = idx % nrad;
            float s = 0;
            int base = (i*nrad + j)*Nd + tt*Np*nrad*Nd;
            for (int l = 0; l < Nd; ++l) s += dgdxyz[base + l];
            dqdxyz[idx+Np*nrad*tt] = s;
            return;        
        }
        }

        continue_from = per_array_out*10 + per_dgraddteta*9 ; 
        int per_qrad = Np*nrad_desc;

        if(tid < (continue_from + per_qrad))
        {  
            int idx = tid - continue_from;
            int i = idx / nrad;
            int j = idx % nrad;
            float s = 0;
            int base = (i*nrad + j)*Nd ;
            for (int l = 0; l < Nd; ++l) s += grad[base + l];
            qrad[idx] = s;
            return;        
        }



        

               



                                              
                                  

    }}
                          ''', 'sum_axis2')