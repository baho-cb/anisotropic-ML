import cupy as cp

grad_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_grad(    
    const float* __restrict__ pts12, // [Np, Nd, 3]
    const float* __restrict__ r12, // [Np, Nd]   
    float* __restrict__ dgdr, // [Np, n_cheb, Nd]   
    float* __restrict__ drdp, // [Np, Nd, 3] 
    float* __restrict__ grad, // [Np, n_cheb, Nd] 
    const float* nep_cutoff,
    const int n_cheb,                       
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )                               

                               
{
    int blx = blockIdx.x; // 0 to Np-1  
    int thx = threadIdx.x; // 0 to Nd-1
                  
                               
    if (blx >= Np || thx >= Nd) return;
                           
    float r = r12[blx*Nd + thx];                                                              
    drdp[blx*Nd*3 + thx*3 + 0] = pts12[blx*Nd*3 + thx*3 + 0]/r;
    drdp[blx*Nd*3 + thx*3 + 1] = pts12[blx*Nd*3 + thx*3 + 1]/r;
    drdp[blx*Nd*3 + thx*3 + 2] = pts12[blx*Nd*3 + thx*3 + 2]/r;                                                               

    float r_rc = r/nep_cutoff[0];                       
    float z = 2.0f*(r_rc - 1.0f)*(r_rc - 1.0f) - 1.0f;
                           
    float factor1 = (4.0f / nep_cutoff[0])*(r_rc - 1.0f)*(1.0f + cosf(3.141592653f*r_rc));
    float factor2 = -sinf(3.141592653f*r_rc) * (3.141592653f / nep_cutoff[0]);
    float f_rcut = 0.5f * (1.0f +cosf(3.141592653f*r_rc));      

    float T_n_1 = 1.0f;                                         
    float T_n = z;
    float T_next;                        
    grad[blx*n_cheb*Nd + thx] = (T_n_1 + 1.0f)*0.5f*f_rcut;                        
    grad[blx*n_cheb*Nd + thx + Nd] = (T_n + 1.0f)*0.5f*f_rcut;    

    float dTdz_n_2 = 0.0f;
    float dTdz_n_1 = 1.0f;
    float dTdz_n = 4.0f * z;                       
    float dTdz_next;
    dgdr[blx*n_cheb*Nd + thx] = 0.25f*(factor1 * dTdz_n_2 + factor2 * (T_n_1 + 1.0f));                                                                                         
    dgdr[blx*n_cheb*Nd + thx + Nd] = 0.25f*(factor1 * dTdz_n_1 + factor2 * (T_n + 1.0f));
                           

    for (int i_cheb = 1; i_cheb < (n_cheb-1); i_cheb += 1)
    {
        T_next = 2.0f*z*T_n - T_n_1;
        dTdz_next = 2.0f * T_next + 2.0f * z * dTdz_n - dTdz_n_1;             

        grad[blx*n_cheb*Nd + thx + (i_cheb+1)*Nd] = (T_next + 1.0f)*0.5f*f_rcut;                   
        dgdr[blx*n_cheb*Nd + thx + (i_cheb+1)*Nd] = 0.25f*(factor1 * dTdz_n + factor2 * (T_next + 1.0f));
                           
        T_n_1 = T_n;
        T_n = T_next;

        dTdz_n_2 = dTdz_n_1;
        dTdz_n_1 = dTdz_n;
        dTdz_n = dTdz_next;
                           
                           
                           
    }                                              



                             }                           
''', 'calculate_grad')


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

    float term1 = dgdteta[index_dgdteta_ij] * g_rad[index_grad_ik] * leg[index_leg];
    float term2 = dgdteta[index_dgdteta_ik] * g_rad[index_grad_ij] * leg[index_leg];                                                                                                                          
    float term3 = g_rad[index_grad_ij]*g_rad[index_grad_ik]* dlegdteta[index_dlegdteta];                                                                                                                          
    
    out[index_target] = term1 + term2 + term3;                              
                                                                                   
                                  
}
''', 'dqang_dteta_kernel')


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

    float term1 = dgdxyz[index_dgdteta_ij] * g_rad[index_grad_ik] * leg[index_leg];
    float term2 = dgdxyz[index_dgdteta_ik] * g_rad[index_grad_ij] * leg[index_leg];                                                                                                                          
    float term3 = g_rad[index_grad_ij]*g_rad[index_grad_ik]* dlegdxyz[index_dlegdteta];                                                                                                                          
    
    out[index_target] = term1 + term2 + term3;                              
    g_ang[index_g_ang] = g_rad[index_grad_ij]*g_rad[index_grad_ik]*leg[index_leg];                                                                               
                                  
}
''', 'dqang_dteta_kernel')










debug_kernel = cp.RawKernel(r'''

extern "C" __global__
void dqang_dteta_kernel(
    const float *__restrict__ g_rad, // [Np, nangp1, Nd]
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

    int index_target = blx*Nd*Nd*6 + thx;  
    
    //out[index_target] = g_rad[index_grad_ij]; // correct                              
    out[index_target] = g_rad[index_grad_ik];                              
                                  
                                              
}
''', 'dqang_dteta_kernel')