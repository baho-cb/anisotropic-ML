import cupy as cp

dpts_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_dpdteta(    
    const float* __restrict__ quat1,
    const float* __restrict__ quat2,
    const float* __restrict__ trans,
    const float* __restrict__ pts_rep, // 6 by 3
    float* __restrict__ pts12, // [Np, Nd, 3]
    float* __restrict__ dp1dteta, // [Np, Nd/2, 3]   
    float* __restrict__ dp2dteta, // [Np, Nd/2, 3]   
    float* __restrict__ r1, // [Np, Nd/2]   
    float* __restrict__ r2, // [Np, Nd/2]   
    float* __restrict__ dr1dteta, // [Np, Nd/2, 3]   
    float* __restrict__ dr2dteta, // [Np, Nd/2, 3]   
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )

{
    const int ndh = Nd >> 1; // Nd / 2 = 6
    const int bidx = blockIdx.x;
    const int tidx   = threadIdx.x;

    //if (bidx >= Np || tidx >= ndh) return;
    if (bidx >= Np || tidx >= 6) return;

    // Load quaternions
    float qw1 = quat1[bidx*4 + 0];
    float qx1 = quat1[bidx*4 + 1];
    float qy1 = quat1[bidx*4 + 2];
    float qz1 = quat1[bidx*4 + 3];

    float qw2 = quat2[bidx*4 + 0];
    float qx2 = quat2[bidx*4 + 1];
    float qy2 = quat2[bidx*4 + 2];
    float qz2 = quat2[bidx*4 + 3];

    // Load translation (optional)
    float tx = 0.f, ty = 0.f, tz = 0.f;
    if (trans) {
        tx = trans[bidx*3 + 0];
        ty = trans[bidx*3 + 1];
        tz = trans[bidx*3 + 2];
    }

    // Reference point
    float px = pts_rep[tidx*3 + 0];
    float py = pts_rep[tidx*3 + 1];
    float pz = pts_rep[tidx*3 + 2];

    float d1 = qx1*px + qy1*py + qz1*pz;
    float d2 = qx2*px + qy2*py + qz2*pz;
                           
    float a1 = qw1*qw1 - qx1*qx1 - qy1*qy1 - qz1*qz1;
    float a2 = qw2*qw2 - qx2*qx2 - qy2*qy2 - qz2*qz2;
    
    float cross_yz1 = qy1*pz - qz1*py;                                
    float cross_xz1 = qz1*px - qx1*pz;                                
    float cross_xy1 = qx1*py - qy1*px;                                
    
    float cross_yz2 = qy2*pz - qz2*py;                                
    float cross_xz2 = qz2*px - qx2*pz;                                
    float cross_xy2 = qx2*py - qy2*px;                                
                           
    size_t base1 = ((size_t)bidx * Nd + tidx) * 3;
    int point_index = bidx* ndh + tidx; // Index for the point in the 6-point representation
    float posx = a1*px + 2.f*(qw1*cross_yz1 + d1*qx1) - tx*0.5f;                 
    float posy = a1*py + 2.f*(qw1*cross_xz1 + d1*qy1) - ty*0.5f;
    float posz = a1*pz + 2.f*(qw1*cross_xy1 + d1*qz1) - tz*0.5f;                                                     
    pts12[base1 + 0] = posx;
    pts12[base1 + 1] = posy;
    pts12[base1 + 2] = posz;
    float norm1 = sqrt(posx*posx + posy*posy + posz*posz);                       
    r1[point_index] = norm1;                 
         
    size_t base2 = base1 + ndh * 3;
    float posx2 = a2*px + 2.f*(qw2*cross_yz2 + d2*qx2) + tx*0.5f;
    float posy2 = a2*py + 2.f*(qw2*cross_xz2 + d2*qy2) + ty*0.5f;
    float posz2 = a2*pz + 2.f*(qw2*cross_xy2 + d2*qz2) + tz*0.5f;
    pts12[base2 + 0] = posx2;
    pts12[base2 + 1] = posy2;
    pts12[base2 + 2] = posz2;
    float norm2 = sqrt(posx2*posx2 + posy2*posy2 + posz2*posz2);
    r2[point_index] = norm2;                       

    size_t base_d = ((size_t)bidx * ndh + tidx) * 3;

    float dp1x = a1*px + 2.f*(qw1*cross_yz1 + d1*qx1);
    float dp1y = a1*py + 2.f*(qw1*cross_xz1 + d1*qy1);
    float dp1z = a1*pz + 2.f*(qw1*cross_xy1 + d1*qz1);                                               
    dp1dteta[base_d + 0] = dp1x;                        
    dp1dteta[base_d + 1] = dp1y;                        
    dp1dteta[base_d + 2] = dp1z;                     
                              
    float dp2x = a2*px + 2.f*(qw2*cross_yz2 + d2*qx2);
    float dp2y = a2*py + 2.f*(qw2*cross_xz2 + d2*qy2);
    float dp2z = a2*pz + 2.f*(qw2*cross_xy2 + d2*qz2);                                               
    dp2dteta[base_d + 0] = dp2x;                        
    dp2dteta[base_d + 1] = dp2y;                        
    dp2dteta[base_d + 2] = dp2z;                        


    dr1dteta[base_d + 0] = (posy*-dp1z + posz*dp1y) / norm1;                        
    dr1dteta[base_d + 1] = (posx*dp1z + posz*-dp1x) / norm1;
    dr1dteta[base_d + 2] = (posx*-dp1y + posy*dp1x) / norm1;   
                           
                                               
    dr2dteta[base_d + 0] = (posy2*-dp2z + posz2*dp2y) / norm2;                        
    dr2dteta[base_d + 1] = (posx2*dp2z + posz2*-dp2x) / norm2;
    dr2dteta[base_d + 2] = (posx2*-dp2y + posy2*dp2x) / norm2;   


                        
}                           

''', 'calculate_dpdteta')

"""
maybe this kernel should have 6x more threads for more parallelization 
and shorter code 
legendre step easier to fuse that way 
"""
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

