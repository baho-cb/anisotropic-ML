import cupy as cp
step_two_rotation_kernel = cp.RawKernel(r'''
extern "C" __global__
void step_two_rotation(
                       const float* __restrict__ orient,
                       const float* __restrict__ torque,
                       float* __restrict__ angmom,
                       float* __restrict__ dp,
                       const float* integ_var2,
                       const float* dt,
                       const int Np
                       )

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Np) return;
    int thx4 = idx * 4;
    int thx3 = idx * 3;
    float timestep = dt[0];
    // rotate torks to body frame (self.orientations,self.torks)

    float conj0 = orient[thx4];
    float conj1 = -orient[thx4+1];
    float conj2 = -orient[thx4+2];
    float conj3 = -orient[thx4+3];

    float t1 = torque[thx3];
    float t2 = torque[thx3+1];
    float t3 = torque[thx3+2];

    float coef1 = conj0*conj0 - (conj1*conj1 + conj2*conj2 + conj3*conj3);

    float term2_x = (2.0f) * conj0 * (conj2*t3 - conj3*t2);
    float term2_y = (2.0f) * conj0 * (conj3*t1 - conj1*t3);
    float term2_z = (2.0f) * conj0 * (conj1*t2 - conj2*t1);

    float sub_sum = conj1*t1 + conj2*t2 + conj3*t3 ;

    float term3_x = (2.0f) * sub_sum * conj1;
    float term3_y = (2.0f) * sub_sum * conj2;
    float term3_z = (2.0f) * sub_sum * conj3;

    float ttx = t1*coef1 + term2_x + term3_x;
    float tty = t2*coef1 + term2_y + term3_y;
    float ttz = t3*coef1 + term2_z + term3_z;


    // calculate_dp get_dp_gpu(self.orientations,self.tt,self.dt)

    float dp0 = (conj1*ttx + conj2*tty + conj3*ttz)*timestep;
    float dp1 = ((-conj2*ttz) - (-conj3*tty) + conj0*ttx)*timestep;
    float dp2 = ((-conj3*ttx) - (-conj1*ttz) + conj0*tty)*timestep;
    float dp3 = ((-conj1*tty) - (-conj2*ttx) + conj0*ttz)*timestep;

    dp[thx4] = dp0;
    dp[thx4+1] = dp1;
    dp[thx4+2] = dp2;
    dp[thx4+3] = dp3;


    // update angular momentum
    float exp_fac = expf((-timestep/2.0)*integ_var2[0]);
    angmom[thx4] = angmom[thx4]*exp_fac + dp0;
    angmom[thx4+1] = angmom[thx4+1]*exp_fac + dp1;
    angmom[thx4+2] = angmom[thx4+2]*exp_fac + dp2;
    angmom[thx4+3] = angmom[thx4+3]*exp_fac + dp3;



}
''', 'step_two_rotation')


step_two_update_variables_kernel = cp.RawKernel(r'''
extern "C" __global__
void update_variables(
                       const float* N_dof,
                       const float* RN_dof,
                       const float* trans_kin_en,
                       const float* rot_kin_en,
                       const float* dt,
                       const float* tau,
                       const float* kT,

                       float* var0,
                       float* var1,
                       float* var2,
                       float* var3,
                       float* thermo_fac
                       )

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2) return;


    // translation
    if(idx==0)
    {
    float timestep = dt[0];
    float trans_temp = (2.0f/N_dof[0]) * trans_kin_en[0];
    float tau_ = tau[0];
    float kT_ = kT[0];

    float xi_prime = var0[0] + 0.5f*((timestep/tau_)/tau_)*((trans_temp/kT_) - 1.0f);
    float new_var0 = xi_prime + 0.5f*((timestep/tau_)/tau_)*((trans_temp/kT_) - 1.0f);
    var1[0] = var1[0] + xi_prime*timestep;

    thermo_fac[0] = expf(-0.5f*new_var0*timestep);
    var0[0] = new_var0;

    }

    // rotation
    if(idx==1)
    {
    float timestep = dt[0];
    float tau_ = tau[0];
    float kT_ = kT[0];
    float rot_temp = (2.0f*rot_kin_en[0])/RN_dof[0];

    float xi_prime_rot = var2[0] + 0.5f*((timestep/tau_)/tau_)*((rot_temp  /kT_) - 1.0f);
    var2[0] = xi_prime_rot +  0.5f*((timestep/tau_)/tau_)*((rot_temp  /kT_) - 1.0f);
    var3[0] += xi_prime_rot*timestep;
    }



}
''', 'update_variables')
