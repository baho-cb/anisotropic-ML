import cupy as cp

step_one_rotation_kernel = cp.RawKernel(r'''
extern "C" __global__
void step_one_rotation(const float* __restrict__ dp, // (Np,4)
                       float* __restrict__ orient, // (Np,4)
                       float* __restrict__ angmom, // (Np,4)
                       const float* integ_var2,
                       const float* dt,
                       const float* moi,
                       const int Np
                       )
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Np) return;
    int thx4 = idx * 4;
    float timestep = dt[0];
    float exp_fac = expf((-timestep/2.0f)*integ_var2[0]);


    float angmom0 = angmom[thx4] + dp[thx4];
    float angmom1 = angmom[thx4+1] + dp[thx4+1];
    float angmom2 = angmom[thx4+2] + dp[thx4+2];
    float angmom3 = angmom[thx4+3] + dp[thx4+3];

    float ori0 = orient[thx4];
    float ori1 = orient[thx4+1];
    float ori2 = orient[thx4+2];
    float ori3 = orient[thx4+3];

    angmom0 *= exp_fac;
    angmom1 *= exp_fac;
    angmom2 *= exp_fac;
    angmom3 *= exp_fac;

    //angmom[thx4] = angmom0;
    //angmom[thx4+1] = angmom1;
    //angmom[thx4+2] = angmom2;
    //angmom[thx4+3] = angmom3;


    // permutation 1

    float anew_0 = -angmom3;
    float anew_1 = angmom2;
    float anew_2 = -angmom1;
    float anew_3 = angmom0;

    float onew0 = -ori3;
    float onew1 = ori2;
    float onew2 = -ori1;
    float onew3 = ori0;

    float phinew = ((1.0f/4.0f)/moi[2]) * ((angmom0*onew0) + (angmom1*onew1) + (angmom2*onew2) + (angmom3*onew3) );
    float cphinew = cosf(0.5f*timestep*phinew);
    float sphinew = sinf(0.5f*timestep*phinew);

    angmom0 = cphinew*angmom0 + sphinew*anew_0;
    angmom1 = cphinew*angmom1 + sphinew*anew_1;
    angmom2 = cphinew*angmom2 + sphinew*anew_2;
    angmom3 = cphinew*angmom3 + sphinew*anew_3;

    ori0 = cphinew*ori0 + sphinew*onew0;
    ori1 = cphinew*ori1 + sphinew*onew1;
    ori2 = cphinew*ori2 + sphinew*onew2;
    ori3 = cphinew*ori3 + sphinew*onew3;

    // permutation 2

    anew_0 = -angmom2;
    anew_1 = -angmom3;
    anew_2 = angmom0;
    anew_3 = angmom1;

    onew0 = -ori2;
    onew1 = -ori3;
    onew2 = ori0;
    onew3 = ori1;

    phinew = ((1.0f/4.0f)/moi[1]) * ((angmom0*onew0) + (angmom1*onew1) + (angmom2*onew2) + (angmom3*onew3) );
    cphinew = cosf(0.5f*timestep*phinew);
    sphinew = sinf(0.5f*timestep*phinew);

    angmom0 = cphinew*angmom0 + sphinew*anew_0;
    angmom1 = cphinew*angmom1 + sphinew*anew_1;
    angmom2 = cphinew*angmom2 + sphinew*anew_2;
    angmom3 = cphinew*angmom3 + sphinew*anew_3;

    ori0 = cphinew*ori0 + sphinew*onew0;
    ori1 = cphinew*ori1 + sphinew*onew1;
    ori2 = cphinew*ori2 + sphinew*onew2;
    ori3 = cphinew*ori3 + sphinew*onew3;

    // permutation 3

    anew_0 = -angmom1;
    anew_1 = angmom0;
    anew_2 = angmom3;
    anew_3 = -angmom2;

    onew0 = -ori1;
    onew1 = ori0;
    onew2 = ori3;
    onew3 = -ori2;

    phinew = ((1.0f/4.0f)/moi[0]) * ((angmom0*onew0) + (angmom1*onew1) + (angmom2*onew2) + (angmom3*onew3) );
    cphinew = cosf(timestep*phinew);
    sphinew = sinf(timestep*phinew);

    angmom0 = cphinew*angmom0 + sphinew*anew_0;
    angmom1 = cphinew*angmom1 + sphinew*anew_1;
    angmom2 = cphinew*angmom2 + sphinew*anew_2;
    angmom3 = cphinew*angmom3 + sphinew*anew_3;

    ori0 = cphinew*ori0 + sphinew*onew0;
    ori1 = cphinew*ori1 + sphinew*onew1;
    ori2 = cphinew*ori2 + sphinew*onew2;
    ori3 = cphinew*ori3 + sphinew*onew3;

    // permutation 2

    anew_0 = -angmom2;
    anew_1 = -angmom3;
    anew_2 = angmom0;
    anew_3 = angmom1;

    onew0 = -ori2;
    onew1 = -ori3;
    onew2 = ori0;
    onew3 = ori1;

    phinew = ((1.0f/4.0f)/moi[1]) * ((angmom0*onew0) + (angmom1*onew1) + (angmom2*onew2) + (angmom3*onew3) );
    cphinew = cosf(0.5f*timestep*phinew);
    sphinew = sinf(0.5f*timestep*phinew);

    angmom0 = cphinew*angmom0 + sphinew*anew_0;
    angmom1 = cphinew*angmom1 + sphinew*anew_1;
    angmom2 = cphinew*angmom2 + sphinew*anew_2;
    angmom3 = cphinew*angmom3 + sphinew*anew_3;

    ori0 = cphinew*ori0 + sphinew*onew0;
    ori1 = cphinew*ori1 + sphinew*onew1;
    ori2 = cphinew*ori2 + sphinew*onew2;
    ori3 = cphinew*ori3 + sphinew*onew3;

    // permutation 1

    anew_0 = -angmom3;
    anew_1 = angmom2;
    anew_2 = -angmom1;
    anew_3 = angmom0;

    onew0 = -ori3;
    onew1 = ori2;
    onew2 = -ori1;
    onew3 = ori0;

    phinew = ((1.0f/4.0f)/moi[2]) * ((angmom0*onew0) + (angmom1*onew1) + (angmom2*onew2) + (angmom3*onew3) );
    cphinew = cosf(0.5f*timestep*phinew);
    sphinew = sinf(0.5f*timestep*phinew);

    angmom[thx4] = cphinew*angmom0 + sphinew*anew_0;
    angmom[thx4+1] = cphinew*angmom1 + sphinew*anew_1;
    angmom[thx4+2] = cphinew*angmom2 + sphinew*anew_2;
    angmom[thx4+3] = cphinew*angmom3 + sphinew*anew_3;

    ori0 = cphinew*ori0 + sphinew*onew0;
    ori1 = cphinew*ori1 + sphinew*onew1;
    ori2 = cphinew*ori2 + sphinew*onew2;
    ori3 = cphinew*ori3 + sphinew*onew3;

    // renormalize orientation quaternion

    float norm = sqrt(ori0*ori0 + ori1*ori1 + ori2*ori2 + ori3*ori3);
    orient[thx4] = ori0/norm;
    orient[thx4+1] = ori1/norm;
    orient[thx4+2] = ori2/norm;
    orient[thx4+3] = ori3/norm;

}
''', 'step_one_rotation')


step_one_translation_kernel = cp.RawKernel(r'''
extern "C" __global__
void step_one_translation(
                       float* __restrict__ pos,
                       float* __restrict__ vel,
                       const float* __restrict__ accel,
                       const float* thermo_fac,
                       const float* dt,
                       const float* Lx,
                       const int Np
                       )

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Np*3) return;
    float box_size = Lx[0];

    float new_vel = (vel[idx] + accel[idx]*0.5f*dt[0])*thermo_fac[0];
    vel[idx] = new_vel;
    float new_pos = pos[idx] + new_vel*dt[0];
    new_pos = (new_pos > 0.5 * box_size) ? (new_pos - box_size) : new_pos;
    new_pos = (new_pos < -0.5 * box_size) ? (new_pos + box_size) : new_pos;
    pos[idx] = new_pos;

}
''', 'step_one_translation')


rotational_kinetic_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void rotational_kinetic_energy(
                       const float* __restrict__ orient,
                       const float* __restrict__ angmom,
                       const float* moi,
                       float* __restrict__ rot_en,
                       const int Np
                       )

{
    // process w,x,y,z (i.e. particle per thread)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Np) return;
    int thx4 = idx*4;
    int thx3 = idx*3;

    float conj0 = orient[thx4];
    float conj1 = -orient[thx4+1];
    float conj2 = -orient[thx4+2];
    float conj3 = -orient[thx4+3];

    float ang0 = angmom[thx4];
    float ang1 = angmom[thx4+1];
    float ang2 = angmom[thx4+2];
    float ang3 = angmom[thx4+3];

    // quaternion multiplication - we don't care about the scalar part
    // float scalar = -(ang1*conj1+ang2*conj2+ang3*conj3) + ang0*conj0;

    float sx = (conj0*ang1 + ang0*conj1 + (conj2*ang3 - conj3*ang2))*0.5;
    float sy = (conj0*ang2 + ang0*conj2 + (conj3*ang1 - conj1*ang3))*0.5;
    float sz = (conj0*ang3 + ang0*conj3 + (conj1*ang2 - conj2*ang1))*0.5;

    rot_en[idx] = ((sx*sx)/moi[0] + (sy*sy)/moi[1] + (sz*sz)/moi[2])*0.5;

}
''', 'rotational_kinetic_energy')
