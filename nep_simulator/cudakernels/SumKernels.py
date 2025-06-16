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