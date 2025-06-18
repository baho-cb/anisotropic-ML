import cupy as cp

outer_prod_kernel = cp.RawKernel(r'''
extern "C" __global__
void outer_prod(
    const float* __restrict__ dg1,  // [Np, nangp1, Nd]
    float*       __restrict__ out1, // [Np, nangp1*lmax, Nd*Nd]
    int Np,
    int nangp1,
    int lmax
) {
    const int Nd = 12;                        // inner-dimension
    int blk = blockIdx.x;                     // 0 .. Np*nangp1*lmax-1
    int tx  = threadIdx.x;                    // 0 .. Nd-1  (row)
    int ty  = threadIdx.y;                    // 0 .. Nd-1  (col)

    // decode (i, n, l) from blk:
    int i   = blk / (nangp1 * lmax);          // particle index
    int tmp = blk % (nangp1 * lmax);
    int n   = tmp / lmax;                     // angular index
    // int l = tmp % lmax;                    // repetition index (unused)

    // load dg1[i,n,tx] and dg1[i,n,ty]
    int base = (i * nangp1 + n) * Nd;
    float v1 = dg1[ base + tx ];
    float v2 = dg1[ base + ty ];

    // compute product
    float p = v1 * v2;

    // compute flat output index:
    // out1[(i, tmp, tx*Nd+ty)]  >  ((i*(nangp1*lmax)+tmp)*(Nd*Nd) + tx*Nd + ty)
    int out_flat = ((blk) * (Nd*Nd)) + (tx * Nd + ty);

    out1[out_flat] = p;
}
''', 'outer_prod')






k1 = cp.RawKernel(r'''
// ===============================================
//  dqang_dteta_kernel_fast  (grid: 1-D)
//  Launch exactly as you do today:
//
//  blocks  = (Np*(nangp1)*lmax,)
//  threads = (Nd, Nd, 1)         // (12,12,1) = 144 threads
//  shmem   = (Nd + Nd*6) * 4     // 336 bytes
// ===============================================
extern "C" __global__
void dqang_dteta_kernel_fast(
        const float* __restrict__ g_rad,     // [Np, nangp1, Nd]
        const float* __restrict__ leg,       // [Np, lmax, 144]
        const float* __restrict__ dlegdt,    // [Np, lmax, 144, 6]
        const float* __restrict__ dgdt,      // [Np, nangp1, Nd, 6]
        float*       __restrict__ out,       // [Np, nangp1*lmax, 144, 6]
        const int    Np,
        const int    nangp1,
        const int    lmax,
        const int    Nd)                     // ==12, but passed for safety
{
    // ---------- compile-time friendly ----------
    const int nMu = 6;

    // ---------- block & thread indices ----------
    const int blx = blockIdx.x;            // 0 ... N
    const int j   = threadIdx.x;           // 0 ... 11 (column)
    const int i   = threadIdx.y;           // 0 ... 11 (row)

    // ---------- decode (p, n, l) from blx ----------
    const int g   = blx / lmax;            // == n
    const int l   = blx % lmax;            // 0 ... lmax-1

    const int p   = g   / nangp1;          // 0 ... Np-1
    const int n   = g   - p * nangp1;      // == g % nangp1

    // ---------- shared memory (one Nd-vector plus Nd x 6 matrix) ----------
    extern __shared__ float sh[];
    float* sh_g  = sh;                     // Nd floats
    float* sh_dg = sh + Nd;                // Nd_6 floats

    // Only the first row of threads loads the tile
    if (threadIdx.y == 0) {
        // ---- g_rad column ----
        const int base_g = (g * Nd) + j;                     //   (p,n,j)
        sh_g[j] = g_rad[base_g];

        // ---- dgdt column (6 mus) ----
        const int base_dg = base_g * nMu;                    //   (p,n,j,0)
        #pragma unroll
        for (int mu = 0; mu < nMu; ++mu)
            sh_dg[j * nMu + mu] = dgdt[base_dg + mu];
    }
    __syncthreads();

    // ---------- handy aliases ----------
    const float gij = sh_g[j];               // g_rad(p,n,j)
    const float gik = sh_g[i];               // g_rad(p,n,i)

    const float* dg_row_j = &sh_dg[j * nMu]; // dgdt(p,n,j,_)
    const float* dg_row_i = &sh_dg[i * nMu]; // dgdt(p,n,i,_)

    // ---- leg & dlegdt indices (identical to reference) ----
    const int ij_flat   = i * Nd + j;                    // 0 ... 143
    const int legBase   = (p * lmax + l) * Nd * Nd + ij_flat;

    const float legVal  = leg[legBase];
    const int   dlegBase= legBase * nMu;                 // ..,0

    // ---------- main mu-loop ----------
    #pragma unroll
    for (int mu = 0; mu < nMu; ++mu)
    {
        const float tij     = dg_row_j[mu];              // dgdt(p,n,j,mu)
        const float tik     = dg_row_i[mu];              // dgdt(p,n,i,mu)
        const float dlegVal = dlegdt[dlegBase + mu];     // dlegdt(p,l,ij,mu)

        // term1 + term2 + term3
        const float res =
              (tij * gik + tik * gij) * legVal
            + (gij * gik) * dlegVal;

        // ---- final output index (same formula as original) ----
        const int outIdx =
              (blx * Nd * Nd + ij_flat) * nMu + mu;      // flattened

        out[outIdx] = res;
    }
}
                                  
                                  ''', 'dqang_dteta_kernel_fast')



gemini_kernel = cp.RawKernel(r'''
extern "C" __global__
void dqang_dteta_kernel(
    const float *__restrict__ g_rad, 
    const float *__restrict__ leg, 
    const float *__restrict__ dlegdteta, 
    const float *__restrict__ dgdteta,
    float       *__restrict__ out,
    const int Np,                       
    const int nangp1,
    const int lmax,
    const int Nd)   
{
    // --- Shared Memory Declaration ---
    // Cache the reused portions of g_rad and dgdteta.
    // Nd=12, nangp1=9
    __shared__ float s_grad[12];     // Caches g_rad[g, :]
    __shared__ float s_dgdteta[72];  // Caches dgdteta[g, :, :] (Nd*6 = 72)

    // --- Indexing ---
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    
    // g corresponds to the flattened (p, a) index
    int g = blx / lmax;

    // --- Step 1: Coalesced Load into Shared Memory ---
    // Use the first threads of the block to perform one coalesced read.
    // Load the needed slice of g_rad (12 floats)
    if (thx < Nd) {
        s_grad[thx] = g_rad[g * Nd + thx];
    }
    // Load the needed slice of dgdteta (72 floats)
    if (thx < Nd * 6) {
        s_dgdteta[thx] = dgdteta[g * Nd * 6 + thx];
    }
    
    // Synchronize to ensure all shared memory loads are complete
    // before any thread proceeds to computation.
    __syncthreads();

    // --- Step 2: Computation using Shared Memory ---
    
    // Decompose thread index into individual components (i, j, mu)
    int mu = thx % 6;
    int tmp = thx / 6;    // Corresponds to flattened (i, j) index
    int j = tmp % Nd; 
    int i = tmp / Nd;     
                                  
    // Decompose block index to get p and l
    int p = blx / (nangp1 * lmax);
    int l = blx % lmax;

    // Read from global memory (these accesses are already coalesced)
    int leg_base_idx = (p * lmax + l) * Nd * Nd;
    float leg_val = leg[leg_base_idx + tmp];
    float dlegdteta_val = dlegdteta[leg_base_idx * 6 + thx];
                                                                                                            
    // Read from FAST shared memory instead of SLOW global memory
    float grad_i = s_grad[i];
    float grad_j = s_grad[j];
    float dgdteta_ij = s_dgdteta[j * 6 + mu];
    float dgdteta_ik = s_dgdteta[i * 6 + mu];
    
    // The calculation logic remains the same
    float term1 = dgdteta_ij * grad_i * leg_val;
    float term2 = dgdteta_ik * grad_j * leg_val;                                                                                                                          
    float term3 = grad_j * grad_i * dlegdteta_val;                                                                                                                          
    
    // --- Step 3: Coalesced Write to Output ---
    int index_target = blx * Nd * Nd * 6 + thx;
    out[index_target] = term1 + term2 + term3;                              
}
''', 'dqang_dteta_kernel')