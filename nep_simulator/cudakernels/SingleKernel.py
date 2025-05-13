import cupy as cp

single_kernel = cp.RawKernel(r'''
extern "C" __global__
void g_all(const float* __restrict__ pos,
                       float* __restrict__ g_rad, 
                       float* __restrict__ g_ang, 
                       const float* nep_cutoff,
                       const int N_pairs, 
                       const int N_chebysev, 
                       const int N_ang,
                       const int lmax,
                       const int Nd)
{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int idx = blx*blockDim.x + thx;

    // Bounds check (especially important in case we over-launch)
    if (idx >= N_pairs) return;

  
    float meanx = 0.0;
    float meany = 0.0;
    float meanz = 0.0;
    

    for(int i = 0; i < Nd; i++)
    {
        meanx += pos[idx*Nd*3 + i*3 + 0];
        meany += pos[idx*Nd*3 + i*3 + 1];
        meanz += pos[idx*Nd*3 + i*3 + 2];
    }

    meanx /= static_cast<float>(Nd);
    meany /= static_cast<float>(Nd);
    meanz /= static_cast<float>(Nd);

    float norm;
    float x_gpu;
    float fc_gpu;

    for(int i = 0; i < Nd; i++)
        {
            norm = (pos[idx*Nd*3 + i*3 + 0] - meanx)*(pos[idx*Nd*3 + i*3 + 0] - meanx) ;
            norm += (pos[idx*Nd*3 + i*3 + 1] - meany)*(pos[idx*Nd*3 + i*3 + 1] - meany);
            norm += (pos[idx*Nd*3 + i*3 + 2] - meanz)*(pos[idx*Nd*3 + i*3 + 2] - meanz);
            norm = sqrt(norm);
            norm = norm/nep_cutoff[0];
            x_gpu =  2.0*(norm - 1.0)*(norm - 1.0) - 1.0;
            fc_gpu = 0.5*(1.0 + cosf(3.14159265359*norm));

            float chebo_n_1 = 1.0f;
            float chebo_n = x_gpu;
            float chebo_next;

            g_rad[idx*N_chebysev] += ((1.0 + 1.0) / 2.0)*fc_gpu;
            for (int i_cheb = 1; i_cheb < N_chebysev; i_cheb += 1)
            {
                g_rad[idx*N_chebysev + i_cheb] += ((1.0 + chebo_n)/2.0)*fc_gpu;

                chebo_next = chebo_n*x_gpu*2.0 - chebo_n_1;
                chebo_n_1 = chebo_n;
                chebo_n = chebo_next;
            }  

            norm = 0.0;
            
        }

    int n_total_ang = (N_ang + 1)*lmax ;
    float x_ij, fc_ij, x_ik, fc_ik;

    for(int j = 0; j < Nd-1; j++)
    {
        float r_ij_x = pos[idx*Nd*3 + j*3 + 0] - meanx;
        float r_ij_y = pos[idx*Nd*3 + j*3 + 1] - meany;
        float r_ij_z = pos[idx*Nd*3 + j*3 + 2] - meanz;

        float r_ij_norm = sqrt(r_ij_x*r_ij_x + r_ij_y*r_ij_y + r_ij_z*r_ij_z);
        float r_ij_norm_cut = r_ij_norm/nep_cutoff[0];
        x_ij =  2.0*(r_ij_norm_cut - 1.0)*(r_ij_norm_cut - 1.0) - 1.0;
        fc_ij = 0.5*(1.0 + cosf(3.14159265359*r_ij_norm_cut));

        

        for (int k = j+1; k < Nd; k++)
        {
            float r_ik_x = pos[idx*Nd*3 + k*3 + 0] - meanx;
            float r_ik_y = pos[idx*Nd*3 + k*3 + 1] - meany;
            float r_ik_z = pos[idx*Nd*3 + k*3 + 2] - meanz;

            float r_ik_norm = sqrt(r_ik_x*r_ik_x + r_ik_y*r_ik_y + r_ik_z*r_ik_z);

            float dot = r_ij_x*r_ik_x + r_ij_y*r_ik_y + r_ij_z*r_ik_z;
            dot = dot/(r_ij_norm*r_ik_norm);

            float r_ik_norm_cut = r_ik_norm/nep_cutoff[0];
            x_ik =  2.0*(r_ik_norm_cut - 1.0)*(r_ik_norm_cut - 1.0) - 1.0;
            fc_ik = 0.5*(1.0 + cosf(3.14159265359*r_ik_norm_cut));

            float chebo_ij_n_1 = 1.0f;
            float chebo_ij_n = x_ij;
            float chebo_ij_next;

            float chebo_ik_n_1 = 1.0f;
            float chebo_ik_n = x_ik;
            float chebo_ik_next;
            
            for (int i_cheb = 0; i_cheb < N_ang+1; i_cheb += 1)
            {

                float leg_n_1 = 1.0f;
                float leg_n = dot;
                float leg_next;

                g_ang[idx*n_total_ang + i_cheb*lmax] += 2.0*leg_n*((1.0 + chebo_ij_n_1)/2.0)*fc_ij*((1.0 + chebo_ik_n_1)/2.0)*fc_ik;

                for (int i_leg = 1; i_leg < lmax; i_leg += 1)
                {

                    leg_next = ((2.0f*i_leg + 1.0f)*dot*leg_n - i_leg*leg_n_1)/(i_leg + 1.0f);
                    leg_n_1 = leg_n;
                    leg_n = leg_next;

                    g_ang[idx*n_total_ang + i_cheb*lmax + i_leg] += 2.0*leg_next*((1.0 + chebo_ij_n_1)/2.0)*fc_ij*((1.0 + chebo_ik_n_1)/2.0)*fc_ik;
            
                }  

                chebo_ij_next = chebo_ij_n*x_ij*2.0 - chebo_ij_n_1;
                chebo_ik_next = chebo_ik_n*x_ik*2.0 - chebo_ik_n_1;
                chebo_ij_n_1 = chebo_ij_n;
                chebo_ik_n_1 = chebo_ik_n;
                chebo_ij_n = chebo_ij_next;
                chebo_ik_n = chebo_ik_next;
            }
            




        }
    
    }





    

}
''', 'g_all')








    # // r_ij[blx*Nd*3 + thx*3 + 0] = x - avgx;
    # // r_ij[blx*Nd*3 + thx*3 + 1] = y - avgy;
    # // r_ij[blx*Nd*3 + thx*3 + 2] = z - avgz;

    # // float norm = sqrt((x - avgx)*(x - avgx) + (y - avgy)*(y - avgy) + (z - avgz)*(z - avgz));
    # // r_ij_norm[blx*Nd+thx] = norm;
    # // //norm = norm/4.5;
    # // norm = norm/nep_cutoff[0];
    # // float x_gpu =  2.0*(norm - 1.0)*(norm - 1.0) - 1.0;
    # // float fc_gpu = 0.5*(1.0 + cosf(3.14159265359*norm));

    # // float chebo_n_1 = 1.0f;
    # // float chebo_n = x_gpu;
    # // float chebo_next;

    # // chebo[idx1] = ((1.0 + 1.0) / 2.0)*fc_gpu;
    # // chebo[idx1 + Nd] = ((1.0 + x_gpu) / 2.0)*fc_gpu;

    # // #pragma unroll
    # // for (int i_cheb = 1; i_cheb < (N_chebysev-1); i_cheb += 1)
    # // {
    # //     chebo_next = chebo_n*x_gpu*2.0 - chebo_n_1;
    # //     chebo_n_1 = chebo_n;
    # //     chebo_n = chebo_next;
    # //     // chebo_carry[idx1 + Nd*(i_cheb+1)] = ((1.0 + chebo_next)/2.0)*fc_gpu;
    # //     chebo[idx1 + Nd*(i_cheb+1)] = ((1.0 + chebo_next)/2.0)*fc_gpu ;
    # // }
    # // }


    # // if(thx < N_chebysev)
    # //     {
    # //     g_rad[blx*N_chebysev + thx] = 0.0;
    # //     #pragma unroll
    # //         for (int ii = 0; ii < Nd; ii++)
    # //     {
    # //         g_rad[blx*N_chebysev + thx] += chebo[blx*N_chebysev*Nd + thx*Nd + ii];
    # //     }

    # // }

