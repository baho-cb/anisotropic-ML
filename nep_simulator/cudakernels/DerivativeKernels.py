import cupy as cp

dpts_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_dpdteta(    const float* __restrict__ quat1,
                       const float* __restrict__ quat2,
                       const float* __restrict__ trans,
                       const float* __restrict__ pts_rep, // 6 by 3
                       float* __restrict__ pts12, // [Np, Nd, 3]
                       float* __restrict__ dp1dx, // [Np, Nd/2, 3]   
                       float* __restrict__ dp1dy, // [Np, Nd/2, 3]   
                       float* __restrict__ dp1dz, // [Np, Nd/2, 3]   
                       float* __restrict__ dp2dx, // [Np, Nd/2, 3]   
                       float* __restrict__ dp2dy, // [Np, Nd/2, 3]   
                       float* __restrict__ dp2dz, // [Np, Nd/2, 3]   
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
    pts12[base1 + 0] = a1*px + 2.f*(qw1*cross_yz1 + d1*qx1) - tx*0.5f;
    pts12[base1 + 1] = a1*py + 2.f*(qw1*cross_xz1 + d1*qy1) - ty*0.5f;
    pts12[base1 + 2] = a1*pz + 2.0f*(qw1*cross_xy1 + d1*qz1) - tz*0.5f;
            
    size_t base2 = base1 + ndh * 3;
    pts12[base2 + 0] = a2*px + 2.f*(qw2*cross_yz2 + d2*qx2) + tx*0.5f;
    pts12[base2 + 1] = a2*py + 2.f*(qw2*cross_xz2 + d2*qy2) + ty*0.5f;
    pts12[base2 + 2] = a2*pz  + 2.f*(qw2*cross_xy2 + d2*qz2) + tz*0.5f;

    size_t base_d = ((size_t)bidx * ndh + tidx) * 3;

    dp1dx[base_d + 0] = 0.f;
    dp1dx[base_d + 1] = -pz*a1 - 2.f*qw1*cross_xy1 - 2.f*qz1*d1;
    dp1dx[base_d + 2] = +py*a1 + 2.f*qw1*cross_xz1 + 2.f*qy1*d1;

    dp1dy[base_d + 0] = +pz*a1 + 2.f*qw1*cross_xy1 + 2.f*qz1*d1;
    dp1dy[base_d + 1] = 0.f;
    dp1dy[base_d + 2] = -px*a1 - 2.f*qw1*cross_yz1 - 2.f*qz1*d1;

    dp1dz[base_d + 0] = -py*a1 - 2.f*qw1*cross_xz1 - 2.f*qy1*d1;
    dp1dz[base_d + 1] = +px*a1 + 2.f*qw1*cross_yz1 + 2.f*qx1*d1;
    dp1dz[base_d + 2] = 0.f;                      


    dp2dx[base_d + 0] = 0.f;
    dp2dx[base_d + 1] = -pz*a2 - 2.f*qw2*cross_xy2 - 2.f*qz2*d2;
    dp2dx[base_d + 2] = +py*a2 + 2.f*qw2*cross_xz2 + 2.f*qy2*d2;

    dp2dy[base_d + 0] = +pz*a2 + 2.f*qw2*cross_xy2 + 2.f*qz2*d2;
    dp2dy[base_d + 1] = 0.f;
    dp2dy[base_d + 2] = -px*a2 - 2.f*qw2*cross_yz2 - 2.f*qz2*d2;

    dp2dz[base_d + 0] = -py*a2 - 2.f*qw2*cross_xz2 - 2.f*qy2*d2;
    dp2dz[base_d + 1] = +px*a2 + 2.f*qw2*cross_yz2 + 2.f*qx2*d2;
    dp2dz[base_d + 2] = 0.f;

}                           

''', 'calculate_dpdteta')


cosine_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_dcosctheta(    
    const float* __restrict__ pts12, // [Np, Nd, 3]
    const float* __restrict__ dp1dx, // [Np, Nd/2, 3]   
    const float* __restrict__ dp1dy, // [Np, Nd/2, 3]   
    const float* __restrict__ dp1dz, // [Np, Nd/2, 3]   
    const float* __restrict__ dp2dx, // [Np, Nd/2, 3]   
    const float* __restrict__ dp2dy, // [Np, Nd/2, 3]   
    const float* __restrict__ dp2dz, // [Np, Nd/2, 3]   
    const int Np,
    const int Nd // 12 for cube, 8 for tetrahedron
    )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    if (blx >= N0 || thx >= Nd*Nd) return;                         

    int i12 = thx / Nd; // integer division (gives the row index in the 12x12 dot matrix)
    int j12 = thx % Nd; 



}                           
''', 'calculate_dcosctheta')
