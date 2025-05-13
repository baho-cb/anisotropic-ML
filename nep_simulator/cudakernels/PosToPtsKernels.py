import cupy as cp

get_pts_pairs_kernels1 = cp.RawKernel(r'''
extern "C" __global__
void get_pts_pairs(    const float* __restrict__ quat1,
                       const float* __restrict__ quat2,
                       const float* __restrict__ trans,
                       const float* __restrict__ pts_rep, // 6 by 3
                       float* __restrict__ pts12,
                       const int Np,
                       const int Nd // 12 for cube, 8 for tetrahedron
                       )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int idx = blx * blockDim.x + thx;
    //int pts_index = (thx%18)/3;
    int pts_index = (thx%((Nd/2)*3))/3;
    int xyz_index = thx%3;
    //if (blx >= Np || thx > 36) return;
    if (blx >= Np || thx > (Nd*3)) return;

    float qw,qx,qy,qz;
    float cross_prod;
    float transx = 0.0f;
    float transy = 0.0f;
    float transz = 0.0f;

    //if(thx<18)
    if(thx<((Nd/2)*3))
    {
        qw = quat1[blx*4];
        qx = quat1[blx*4+1];
        qy = quat1[blx*4+2];
        qz = quat1[blx*4+3];

    }
    else
    {
        qw = quat2[blx*4];
        qx = quat2[blx*4+1];
        qy = quat2[blx*4+2];
        qz = quat2[blx*4+3];

        transx = trans[blx*3];
        transy = trans[blx*3+1];
        transz = trans[blx*3+2];
    }

    float px = pts_rep[pts_index*3];
    float py = pts_rep[pts_index*3+1];
    float pz = pts_rep[pts_index*3+2];
    float a = qw*qw - (qx*qx + qy*qy + qz*qz);
    float dot = qx*px + qy*py + qz*pz;

    if(xyz_index==0)
    {
        cross_prod = qy*pz - qz*py;
        pts12[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx) + transx;
    }
    else if (xyz_index==1)
    {
        cross_prod = qz*px - qx*pz;
        pts12[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy) + transy;
    }
    else
    {
        cross_prod = qx*py - qy*px;
        pts12[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz) + transz;
    }






}
''', 'get_pts_pairs')

get_pts_pairs_kernels2 = cp.RawKernel(r'''
extern "C" __global__
void get_pts_pairs2(    const float* __restrict__ quat1,
                       const float* __restrict__ quat2,
                       const float* __restrict__ trans,
                       const float* __restrict__ pts_rep, // 6 by 3
                       float* __restrict__ pts12,
                       const int Np,
                       const int Nd
                       )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int idx = blx * blockDim.x + thx;
    int pts_index = (thx%((Nd/2)*3))/3;
    int xyz_index = thx%3;
    if (blx >= Np || thx > (Nd*3)) return;

    float qw,qx,qy,qz;
    float cross_prod;
    float transx = 0.0f;
    float transy = 0.0f;
    float transz = 0.0f;

    if(thx<((Nd/2)*3))
    {
        qw = quat1[blx*4];
        qx = quat1[blx*4+1];
        qy = quat1[blx*4+2];
        qz = quat1[blx*4+3];

    }
    else
    {
        qw = quat2[blx*4];
        qx = quat2[blx*4+1];
        qy = quat2[blx*4+2];
        qz = quat2[blx*4+3];
        transx = trans[blx*3];
        transy = trans[blx*3+1];
        transz = trans[blx*3+2];

    }

    float px = pts_rep[pts_index*3];
    float py = pts_rep[pts_index*3+1];
    float pz = pts_rep[pts_index*3+2];
    float a = qw*qw - (qx*qx + qy*qy + qz*qz);
    float dot = qx*px + qy*py + qz*pz;

    if(xyz_index==0)
    {
        cross_prod = qy*pz - qz*py;
        pts12[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx) - transx;
    }
    else if (xyz_index==1)
    {
        cross_prod = qz*px - qx*pz;
        pts12[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy) - transy;
    }
    else
    {
        cross_prod = qx*py - qy*px;
        pts12[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz) - transz;
    }


}
''', 'get_pts_pairs2')


apply_dx_dteta_kernel = cp.RawKernel(r'''
extern "C" __global__
void apply_dx_dteta(   const float* __restrict__ orig_pts,
                       float* __restrict__ all_pts,
                       const float* dx,
                       const float* dteta,
                       const int Np,
                       const int Nd
                       )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int idx = blx * blockDim.x + thx;
    int operation_index = blx/Np;
    // int orig_idx = (blx%Np)*36 + thx;
    int orig_idx = (blx%Np)*(Nd*3) + thx;
    int xyz_index = thx%3;
    int pts_index = orig_idx-xyz_index;

    if (blx >= Np*7 || thx > (Nd*3)) return;

    if(thx > ((Nd/2)*3)-1)
    {
        all_pts[idx] = orig_pts[orig_idx];
        return;
    }

    if(operation_index==0)
    {
        all_pts[idx] = orig_pts[orig_idx];
        return;
    }

    if(operation_index==1)
    {
        if(xyz_index==0)
        {
            all_pts[idx] = orig_pts[orig_idx] + dx[0];
            return;
        }
        else
        {
            all_pts[idx] = orig_pts[orig_idx];
            return;
        }
    }

    if(operation_index==2)
    {
        if(xyz_index==1)
        {
            all_pts[idx] = orig_pts[orig_idx] + dx[0];
        }
        else
        {
            all_pts[idx] = orig_pts[orig_idx];
        }
    }

    if(operation_index==3)
    {
        if(xyz_index==2)
        {
            all_pts[idx] = orig_pts[orig_idx] + dx[0];
        }
        else
        {
            all_pts[idx] = orig_pts[orig_idx];
        }
    }

    float qw = cosf(dteta[0]*0.5f);
    float qx = 0.0;
    float qy = 0.0;
    float qz = 0.0;

    float px,py,pz,cross_prod,a,dot;

    if(operation_index==4)
    {
        qx = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }


    if(operation_index==5)
    {
        qy = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }

    if(operation_index==6)
    {
        qz = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }

}
''', 'apply_dx_dteta')


apply_dteta_kernel2 = cp.RawKernel(r'''
extern "C" __global__
void apply_dteta2(   const float* __restrict__ orig_pts,
                       float* __restrict__ all_pts,
                       const float* dteta,
                       const int Np,
                       const int Nd
                       )

{
    int blx = blockIdx.x;
    int thx = threadIdx.x;
    int idx = blx * blockDim.x + thx;
    int operation_index = blx/Np;
    int orig_idx = (blx%Np)*(Nd*3) + thx;
    int xyz_index = thx%3;
    int pts_index = orig_idx-xyz_index;

    if (blx >= Np*3 || thx > (Nd*3)) return;

    if(thx > ((Nd/2)*3-1))
    {
        all_pts[idx] = orig_pts[orig_idx];
        return;
    }

    float qw = cosf(dteta[0]*0.5f);
    float qx = 0.0;
    float qy = 0.0;
    float qz = 0.0;

    float px,py,pz,cross_prod,a,dot;

    if(operation_index==0)
    {
        qx = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }


    if(operation_index==1)
    {
        qy = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }

    if(operation_index==2)
    {
        qz = sinf(dteta[0]*0.5f);
        px = orig_pts[pts_index];
        py = orig_pts[pts_index+1];
        pz = orig_pts[pts_index+2];
        a = qw*qw - (qx*qx + qy*qy + qz*qz);
        dot = qx*px + qy*py + qz*pz;

        if(xyz_index==0)
        {
            cross_prod = qy*pz - qz*py;
            all_pts[idx] = a*px + 2.0f*(qw*cross_prod + dot*qx);
            return;
        }
        else if (xyz_index==1)
        {
            cross_prod = qz*px - qx*pz;
            all_pts[idx] = a*py + 2.0f*(qw*cross_prod + dot*qy);
            return;
        }
        else
        {
            cross_prod = qx*py - qy*px;
            all_pts[idx] = a*pz + 2.0f*(qw*cross_prod + dot*qz);
            return;
        }
    }

}
''', 'apply_dteta2')
