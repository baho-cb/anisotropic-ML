import cupy as cp
import numpy as np
import time
import cudakernels.IntegrateStepOneKernels as ik1
import cudakernels.IntegrateStepTwoKernels as ik2
import DescriptorGenerator
from Sim import Sim

class NVT(Sim):


    def integrate_step_one(self,):
        self.translate1()
        self.rotate1()
        self.advance_thermostat1()

    def integrate_step_two(self):

        Nthreads = 10
        blocks = (1,)
        threads_per_block = (Nthreads,)
        ### this part is implemented at the end of step one in hoomd 
        ik2.step_two_update_variables_kernel(
            blocks,
            threads_per_block,
            (
                self.N_dof,
                self.RN_dof,
                self.trans_kin_en,
                self.rot_kin_en,
                self.dt,
                self.tau,
                self.kT,
                self.integrator_vars0,
                self.integrator_vars1,
                self.integrator_vars2,
                self.integrator_vars3,
                self.m_exp_thermo_fac
            )
        )

        self.accel = self.forces/self.mass.reshape(-1,1)
        self.velocities = self.velocities*self.m_exp_thermo_fac
        self.velocities = self.velocities + (0.5)*(self.dt)*(self.accel) 
        
        Nthreads = 512
        blocks = (10,)
        threads_per_block = (Nthreads,)

        ik2.step_two_rotation_kernel(
            blocks,
            threads_per_block,
            (
                self.orientations,
                self.torks,
                self.angmom,
                self.dp,
                self.integrator_vars2,
                self.dt,
                cp.int32(self.Nparticles)
            )
        )

        self.timestep += 1

    
    def translate1(self):

        Nthreads = 512
        blocks = (10,)
        threads_per_block = (Nthreads,)

        ik1.step_one_translation_kernel(
            blocks,
            threads_per_block,
            (
                self.central_pos,
                self.velocities,
                self.accel,
                self.m_exp_thermo_fac,
                self.dt,
                self.Lx,
                cp.int32(self.Nparticles)
            )
        )

    def rotate1(self):    

        self.dp = self.dp.astype(cp.float32)
        Nthreads = 512
        blocks = (4,)
        threads_per_block = (Nthreads,)

        ik1.step_one_rotation_kernel(
            blocks,
            threads_per_block,
            (
                self.dp,
                self.orientations,
                self.angmom,
                self.integrator_vars2,
                self.dt,
                self.moi_gpu,
                cp.int32(self.Nparticles)
            )
        )

    def advance_thermostat1(self):
        Nthreads = 512
        blocks = (4,)
        threads_per_block = (Nthreads,)

        trans_kin_en = (0.5)*(self.mass)*(cp.sum(self.velocities*self.velocities,axis=1))
        self.trans_kin_en = cp.sum(trans_kin_en)
 
        rotational_energy = cp.empty((self.Nparticles),dtype=cp.float32)
        ik1.rotational_kinetic_energy_kernel(
            blocks,
            threads_per_block,
            (
                self.orientations,
                self.angmom,
                self.moi_gpu,
                rotational_energy,
                cp.int32(self.Nparticles)
            )
        )

        self.rot_kin_en = cp.sum(rotational_energy)


        



















































def dummy():
    pass 
