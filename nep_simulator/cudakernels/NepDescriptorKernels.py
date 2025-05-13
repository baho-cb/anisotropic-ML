import cupy as cp

radial_cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void radial_grad(const float* __restrict__ pos,
                       float* __restrict__ r_ij, // (Np,12,3)
                       float* __restrict__ r_ij_norm, // (Np,12)
                       float* __restrict__ chebo, // (Np,11,12)
                       float* __restrict__ g_rad, // (Np,11,12)
                       const float* nep_cutoff,
               const int N_pairs, const int N_chebysev, const int Nd)
{
    int blx = blockIdx.x;
    int thx = threadIdx.x;

    // Bounds check (especially important in case we over-launch)
    if (blx >= N_pairs) return;

    __shared__ float sums[3];

    if(thx < Nd)
    {

    if (thx == 0)
        {
            sums[0] = 0.0f;
            sums[1] = 0.0f;
            sums[2] = 0.0f;
        }
    __syncthreads();

    int idx1 =  blx*Nd*N_chebysev + thx;
    int idx2 =  blx*Nd + thx;
    float x = pos[blx*Nd*3+ thx*3 + 0];
    float y = pos[blx*Nd*3+ thx*3 + 1];
    float z = pos[blx*Nd*3+ thx*3 + 2];

    atomicAdd(&sums[0], x);
    atomicAdd(&sums[1], y);
    atomicAdd(&sums[2], z);

    __syncthreads();
    float float_Nd = static_cast<float>(Nd);

    float avgx = sums[0] / float_Nd;
    float avgy = sums[1] / float_Nd;
    float avgz = sums[2] / float_Nd;
    r_ij[blx*Nd*3 + thx*3 + 0] = x - avgx;
    r_ij[blx*Nd*3 + thx*3 + 1] = y - avgy;
    r_ij[blx*Nd*3 + thx*3 + 2] = z - avgz;

    float norm = sqrt((x - avgx)*(x - avgx) + (y - avgy)*(y - avgy) + (z - avgz)*(z - avgz));
    r_ij_norm[blx*Nd+thx] = norm;
    //norm = norm/4.5;
    norm = norm/nep_cutoff[0];
    float x_gpu =  2.0*(norm - 1.0)*(norm - 1.0) - 1.0;
    float fc_gpu = 0.5*(1.0 + cosf(3.14159265359*norm));

    float chebo_n_1 = 1.0f;
    float chebo_n = x_gpu;
    float chebo_next;

    chebo[idx1] = ((1.0 + 1.0) / 2.0)*fc_gpu;
    chebo[idx1 + Nd] = ((1.0 + x_gpu) / 2.0)*fc_gpu;

    #pragma unroll
    for (int i_cheb = 1; i_cheb < (N_chebysev-1); i_cheb += 1)
    {
        chebo_next = chebo_n*x_gpu*2.0 - chebo_n_1;
        chebo_n_1 = chebo_n;
        chebo_n = chebo_next;
        // chebo_carry[idx1 + Nd*(i_cheb+1)] = ((1.0 + chebo_next)/2.0)*fc_gpu;
        chebo[idx1 + Nd*(i_cheb+1)] = ((1.0 + chebo_next)/2.0)*fc_gpu ;
    }
    }


    if(thx < N_chebysev)
        {
        g_rad[blx*N_chebysev + thx] = 0.0;
        #pragma unroll
            for (int ii = 0; ii < Nd; ii++)
        {
            g_rad[blx*N_chebysev + thx] += chebo[blx*N_chebysev*Nd + thx*Nd + ii];
        }

    }




}
''', 'radial_grad')


angular_cuda_kernel = cp.RawKernel(r'''
extern "C" __global__
void angular_g(        const float* __restrict__ r_ij, // (Np,12,3)
                       const float* __restrict__ r_ij_norm, // (Np,12)
                       float* __restrict__ lego, // (Np,4,144)
                       const int N0,
                       const int Nd,
                       const int lmax
                       )
{
    int blx = blockIdx.x;
    int thx = threadIdx.x;

    if (blx >= N0 || thx >= Nd*Nd) return;

    int i = thx / Nd; // integer division (gives the row index in the 12x12 dot matrix)
    int j = thx % Nd;

    int index1 = blx * (Nd * 3) + i * 3;
    int index2 = blx * (Nd * 3) + j * 3;

    float dot = 0.0f;

    for (int k = 0; k < 3; ++k) {
        float a = r_ij[index1 + k];
        float b = r_ij[index2 + k];
        dot += a * b;
    }
    dot /= (r_ij_norm[blx*Nd + i]*r_ij_norm[blx*Nd + j]);

    lego[blx * Nd * Nd * (lmax+1) + thx] = 1.0f;
    lego[blx * Nd * Nd * (lmax+1) + thx + Nd*Nd] = dot;

    float lego_n_1 = 1.0f;
    float lego_n = dot;
    float lego_next;

    #pragma unroll
    for (int i_lego = 1; i_lego < lmax; i_lego += 1)
    {
        lego_next = ((2.0f*i_lego + 1.0f)*dot*lego_n - lego_n_1*i_lego) / (i_lego+1);
        lego_n_1 = lego_n;
        lego_n = lego_next;
        lego[blx * Nd * Nd * (lmax+1) + thx + Nd*Nd*(i_lego+1)] = lego_next;
    }


}
''', 'angular_g')
