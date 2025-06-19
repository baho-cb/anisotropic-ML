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



dgdteta_kernel2 = cp.RawKernel(r'''
extern "C" __global__
void calculate_dgdteta(    
    const float* __restrict__ dgdr, // [Np, n_cheb, Nd]
    const float* __restrict__ drdp, // [Np, Nd, 3]   
    const float* __restrict__ dpdteta, // [Np, Nd, 3]   
    float* __restrict__ dgdteta, // [6, Np, n_cheb, Nd] 
    float* __restrict__ dgdxyz, // [3, Np, n_cheb, Nd] 
    const int n_cheb,                       
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )                               

                               
{
    int blx = blockIdx.x; // 0 to Np*n_cheb  
    int thx = threadIdx.x; // 0 to Nd
    int ndh = Nd/2; // half of Nd              
                               
    if (blx >= (Np*n_cheb) || thx >= Nd) return;
                           
                              
                              
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
    int index_target = blx*Nd*6 + thx*6;  
    int index_dgdxyz = blx*Nd*3 + thx*3;  
                              
    dgdxyz[blx*Nd + thx] = v_dgdr * drdp_x * dpdxyz;                         
    dgdxyz[blx*Nd + thx + Np*n_cheb*Nd] = v_dgdr * drdp_y * dpdxyz;                         
    dgdxyz[blx*Nd + thx + Np*n_cheb*Nd*2] = v_dgdr * drdp_z * dpdxyz;                         
                              
    dgdteta[blx*Nd + thx] = v_dgdr * inner_dx1;                          
    dgdteta[blx*Nd + thx + Np*n_cheb*Nd] = v_dgdr * inner_dy1;                          
    dgdteta[blx*Nd + thx + Np*n_cheb*Nd*2] = v_dgdr * inner_dz1;                          
    dgdteta[blx*Nd + thx + Np*n_cheb*Nd*3] = v_dgdr * inner_dx2;                          
    dgdteta[blx*Nd + thx + Np*n_cheb*Nd*4] = v_dgdr * inner_dy2;                          
    dgdteta[blx*Nd + thx + Np*n_cheb*Nd*5] = v_dgdr * inner_dz2;                          


                             }                           
''', 'calculate_dgdteta')







dqang_dteta_kernel_faster_upper = cp.RawKernel(r'''

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
    if(thx < thy) return;                                                           

    int out_size = (Nd/2)*(Nd + 1);
    //int index_target = blx*out_size + thy*Nd + thx;
    int index_target = blx*out_size + ((2*Nd - thy + 1)*thy)/2 + thx - thy;
                                                                                        
    //if(index_target >= Np*nangp1*lmax*Nd*Nd) return; 
    if(index_target >= Np*nangp1*lmax*out_size) return; 
                                                                                     
                                         
                                                                   
    int index_g_ik = (blx/lmax) * Nd + thx;
    int index_g_ij = (blx/lmax) * Nd + thy;
    int index_leg =  ((blx%lmax) + (blx/(lmax*nangp1))*lmax) * Nd * Nd + thy*Nd + thx;
                                   
    float g_ij = g_rad[index_g_ij];                                     
    float g_ik = g_rad[index_g_ik];                                    
    float leg_var = leg[index_leg];

    float multiplier = 2.0f;                                                                                                                 
    if(thx==thy)
    {
    multiplier = 1.0f;                                            
    }                                           

    out1[index_target] = (g_ij*dgdteta1[index_g_ik]*leg_var + g_ik*dgdteta1[index_g_ij]*leg_var + g_ij*g_ik*dl1[index_leg])*multiplier;                                                                               
    out2[index_target] = (g_ij*dgdteta2[index_g_ik]*leg_var + g_ik*dgdteta2[index_g_ij]*leg_var + g_ij*g_ik*dl2[index_leg])*multiplier;                                                                              
    out3[index_target] = (g_ij*dgdteta3[index_g_ik]*leg_var + g_ik*dgdteta3[index_g_ij]*leg_var + g_ij*g_ik*dl3[index_leg])*multiplier;                                                                      
    out4[index_target] = (g_ij*dgdteta4[index_g_ik]*leg_var + g_ik*dgdteta4[index_g_ij]*leg_var + g_ij*g_ik*dl4[index_leg])*multiplier;                                                                            
    out5[index_target] = (g_ij*dgdteta5[index_g_ik]*leg_var + g_ik*dgdteta5[index_g_ij]*leg_var + g_ij*g_ik*dl5[index_leg])*multiplier;                                                                   
    out6[index_target] = (g_ij*dgdteta6[index_g_ik]*leg_var + g_ik*dgdteta6[index_g_ij]*leg_var + g_ij*g_ik*dl6[index_leg])*multiplier;                                                                        
    out7[index_target] = (g_ij*dgdteta7[index_g_ik]*leg_var + g_ik*dgdteta7[index_g_ij]*leg_var + g_ij*g_ik*dl7[index_leg])*multiplier;                                                                        
    out8[index_target] = (g_ij*dgdteta8[index_g_ik]*leg_var + g_ik*dgdteta8[index_g_ij]*leg_var + g_ij*g_ik*dl8[index_leg])*multiplier;                                                                        
    out9[index_target] = (g_ij*dgdteta9[index_g_ik]*leg_var + g_ik*dgdteta9[index_g_ij]*leg_var + g_ij*g_ik*dl9[index_leg])*multiplier;                                                                        
    out10[index_target] = (g_ij*g_ik*leg_var)*multiplier;
                                               
                   
}
''', 'dqang_dteta_kernel')









