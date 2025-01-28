import numpy as np
import torch
from EnergyModel import EnergyModel
import gsd.hoomd
from sklearn import tree
from scipy.spatial import cKDTree as KDTree
from HelperFunctions import *
from GsdHandler import GsdHandler
from CudaKernels import angular_cuda_kernel, radial_cuda_kernel
from IntegrateStepOneKernels import step_one_rotation_kernel, step_one_translation_kernel, rotational_kinetic_energy_kernel
from IntegrateStepTwoKernels import step_two_rotation_kernel, step_two_update_variables_kernel
from GenPtsKernels import get_pts_pairs_kernels1, get_pts_pairs_kernels2, apply_dx_dteta_kernel, apply_dteta_kernel2
import time
import matplotlib.pyplot as plt
import sys
import cupy as cp


np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)
torch.set_printoptions(precision=3,threshold=sys.maxsize)


class Sim():

    def __init__(self):
        print("A new simulation instance created.")

    def setBox(self,L):
        self.Lx = L


    def placeParticlesFromGsd(self,gsd_file):
        """
        Reads the gsd file that contains
        """
        gsd_handler = GsdHandler(gsd_file)
        self.central_pos = gsd_handler.get_central_pos()
        self.Nparticles = len(self.central_pos)
        self.N_dof = (self.Nparticles - 1)*3
        self.RN_dof = (self.Nparticles)*3
        self.velocities = gsd_handler.get_COM_velocities()
        self.accel = np.zeros_like(self.velocities)
        self.mass = gsd_handler.getMass()
        ### Angular Data
        self.orientations = gsd_handler.getOrientations()
        self.angmom = gsd_handler.getAngmom()
        self.moi = gsd_handler.getMoi()
        self.moi = self.moi[0]
        self.Lx = gsd_handler.getLx()

        self.moi_to_dump = gsd_handler.getMoi()
        self.charges = gsd_handler.getCharges()
        # print(self.moi_to_dump)
        # exit()


        print("Simulation will be initialized from %s"%(gsd_file))

    def move_to_gpu(self):
        self.mass = cp.asarray(self.mass, dtype=cp.float32)
        self.accel = cp.asarray(self.accel, dtype=cp.float32)
        self.velocities = cp.asarray(self.velocities, dtype=cp.float32)
        self.orientations = cp.asarray(self.orientations, dtype=cp.float32)
        self.angmom = cp.asarray(self.angmom, dtype=cp.float32)
        self.central_pos = cp.asarray(self.central_pos, dtype=cp.float32)
        trans_kin_en = (0.5)*(self.mass)*(cp.sum(self.velocities*self.velocities,axis=1))
        self.trans_kin_en = cp.sum(trans_kin_en)
        self.rot_kin_en = self.getRotKin_gpu()
        self.moi_gpu = cp.asarray(self.moi, dtype=cp.float32)
        self.dp = cp.empty_like(self.angmom)

        self.N_dof = cp.asarray(self.N_dof,dtype=cp.float32)
        self.RN_dof = cp.asarray(self.RN_dof,dtype=cp.float32)
        self.dt = cp.asarray(self.dt,dtype=cp.float32)
        self.tau = cp.asarray(self.tau,dtype=cp.float32)
        self.kT = cp.asarray(self.kT,dtype=cp.float32)

        self.Lx = cp.asarray(self.Lx,dtype=cp.float32)

        self.integrator_vars0 = cp.asarray(self.integrator_vars0,dtype=cp.float32)
        self.integrator_vars1 = cp.asarray(self.integrator_vars1,dtype=cp.float32)
        self.integrator_vars2 = cp.asarray(self.integrator_vars2,dtype=cp.float32)
        self.integrator_vars3 = cp.asarray(self.integrator_vars3,dtype=cp.float32)
        self.m_exp_thermo_fac = cp.asarray(self.m_exp_thermo_fac,dtype=cp.float32)


    def setkT(self,kT):
        self.kT = kT

    def setTau(self,tau):
        self.tau = tau

    def setdt(self,dt):
        self.dt = dt


    def setShape_gpu(self,shape_str):
        """
        vertices with the shape symmetry group
        should be consistent with the training
        """
        if(shape_str == 'cube_v2'):
            self.pts_rep = cp.array([
            [1.0,0.0,0.0],
            [-1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,-1.0,0.0],
            [0.0,0.0,1.0],
            [0.0,0.0,-1.0],
            ],dtype=cp.float32)

            self.pts_rep *= 1.7
            # self.Npts = len(self.pts_rep)
            self.Npts = 6
            self.Nd = self.Npts*2
            self.en_min = -5.1
            self.en_max = 15.0
            self.cutoff = 5.80

        elif(shape_str == 'th_v1'):
            self.pts_rep = cp.array([
            [-0.45,-0.45,-0.45],
            [-0.45,0.45,0.45],
            [0.45,-0.45,0.45],
            [0.45,0.45,-0.45]
            ],dtype=cp.float32)

            self.pts_rep *= 1.0
            # self.Npts = len(self.pts_rep)
            self.Npts = 4
            self.Nd = self.Npts*2

            self.en_min = -6.7
            self.en_max = 15.0
            self.cutoff = 5.75
        else:
            print('shape not found ')
            exit()

    def setHypersNep(self,hypers):
        self.nrad = int(hypers[0])
        self.nang = int(hypers[1])
        self.lmax = int(hypers[2])
        self.cutoff_nep = hypers[3]
        self.N_descriptors = (self.nrad + 1) + (self.nang + 1)*self.lmax

    def setForces(self,force_path,gpu_id):
        self.energy_model = EnergyModel(force_path,gpu_id,self.N_descriptors)
        self.dx = 0.00001
        self.dteta = 0.00001
        self.device = cp.cuda.Device(gpu_id)
        self.device.use()
        self.dx_gpu = cp.asarray(self.dx,dtype=cp.float32)
        self.dteta_gpu = cp.asarray(self.dteta,dtype=cp.float32)

        gpu_str = 'cuda:%d' %(gpu_id)
        self.torch_device = torch.device(gpu_str if torch.cuda.is_available() else 'cpu')



    def setNeighborList_gpu(self,freq):
        self.Nlist_cutoff = self.cutoff + 2.0
        self.Nlist_freq = freq
        pos_tree = self.central_pos + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        self.Nlist = tree.query_pairs(r=self.Nlist_cutoff,output_type='ndarray')
        self.Nlist = cp.asarray(self.Nlist)
        self.cutoff_gpu = cp.ones(self.Nlist.shape[0], dtype=cp.float32)*self.cutoff
        self.set_rotations_gpu()


    def refreshNList_gpu(self):

        pos_tree = self.central_pos + self.Lx*0.5
        pos_tree = pos_tree.get()
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx.get()+0.0001)
        self.Nlist = tree.query_pairs(r=self.Nlist_cutoff,output_type='ndarray')
        self.Nlist = cp.asarray(self.Nlist)
        self.cutoff_gpu = cp.ones(self.Nlist.shape[0], dtype=cp.float32)*self.cutoff
        self.set_rotations_gpu()

    def set_rotations_gpu(self):
        self.quat_x = cp.array([np.cos(self.dteta*0.5),np.sin(self.dteta*0.5),0.0,0.0], dtype=cp.float32)
        self.quat_y = cp.array([np.cos(self.dteta*0.5),0.0,np.sin(self.dteta*0.5),0.0], dtype=cp.float32)
        self.quat_z = cp.array([np.cos(self.dteta*0.5),0.0,0.0,np.sin(self.dteta*0.5)], dtype=cp.float32)


    def set_integrator(self,ints):
        self.integrator_vars = np.copy(ints)
        self.integrator_vars0 = ints[0]
        self.integrator_vars1 = ints[1]
        self.integrator_vars2 = ints[2]
        self.integrator_vars3 = ints[3]

    def set_m_exp_factor(self,val):
        self.m_exp_thermo_fac = val

    def set_accelerations(self,accel):

        if(self.is_cont==0):
            self.accel = np.zeros_like(self.velocities)
        elif(self.is_cont==1):
            if(accel=='no_accel'):
                print('you must supply an acceleration file if continuing')
                exit()
            accel = np.load(accel)
            self.accel = accel


    def setDumpFreq(self,df):
        self.dumpFreq = df

    def is_continue(self,is_cont):
        """
        if is cont = 1, this means we continue a simulation (probably for debugging)
        so it will replace the usual integrator variables with the ones that come in
        the initiator gsd file.
        """
        self.is_cont = is_cont
        if(is_cont==1):
            print("Integrator variables are set from gsd file - Continuing ")
            self.integrator_vars = np.copy(self.charges[:4])
            self.m_exp_thermo_fac = np.copy(self.charges[4])
            print(self.charges[:5])
        elif(is_cont==0):
            print("Integrator variables are set as default - Not continuing")
        else:
            print("Invalid --is_cont argument should be 0 or 1.")

    def setDump(self,filename):
        """
        Each dump file has 14 columns
        central_pos(3) - orientation(4) - velocities(3) - angular_moms(4)
        All of the above can be dumped as a gsd file
        """
        self.dumpFilename = filename

    def dumpAcceleration(self):
        """
        numpy save replaces/overwrites the existing data which is what we want
        """
        data = self.accel
        names = self.dumpFilename[:-4] + '_accel.npy'
        np.save(names,data)


    def dumpConfig(self,ts):
        pos_all = self.central_pos
        typeid_all = np.zeros(len(self.central_pos),dtype=int)
        velocities_all = self.velocities
        orientations_all = self.orientations
        angmoms_all = self.angmom

        charges = np.zeros_like(self.mass)
        # charges[:4] = self.integrator_vars
        # charges[4] = self.m_exp_thermo_fac
        snap = gsd.hoomd.Frame()
        # snap = gsd.hoomd.Snapshot()
        snap.configuration.step = ts
        snap.configuration.box = [self.Lx, self.Lx, self.Lx, 0, 0, 0]
        snap.particles.N = len(pos_all)
        snap.particles.position = pos_all
        snap.particles.types  = ['A','B']
        snap.particles.typeid  = typeid_all
        snap.particles.moment_inertia = self.moi_to_dump
        snap.particles.mass = self.mass
        # snap.particles.charge = charges

        snap.particles.orientation  = orientations_all
        snap.particles.angmom  = angmoms_all
        snap.particles.velocity  = velocities_all

        if(ts==0):
            with gsd.hoomd.open(name=self.dumpFilename, mode='w') as f:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='wb') as f:
                f.append(snap)
        else:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='rb+') as f:
            with gsd.hoomd.open(name=self.dumpFilename, mode='r+') as f:
                f.append(snap)

    def dumpConfig_gpu(self,ts):
        pos_all = self.central_pos.get()
        typeid_all = np.zeros(len(self.central_pos),dtype=int)
        velocities_all = self.velocities.get()
        orientations_all = self.orientations.get()
        angmoms_all = self.angmom.get()

        charges = np.zeros_like(self.mass)
        # charges[:4] = self.integrator_vars
        # charges[4] = self.m_exp_thermo_fac
        snap = gsd.hoomd.Frame()
        # snap = gsd.hoomd.Snapshot()
        snap.configuration.step = ts
        snap.configuration.box = [self.Lx.get(), self.Lx.get(), self.Lx.get(), 0, 0, 0]
        snap.particles.N = len(pos_all)
        snap.particles.position = pos_all
        snap.particles.types  = ['A','B']
        snap.particles.typeid  = typeid_all
        snap.particles.moment_inertia = self.moi_to_dump
        snap.particles.mass = self.mass.get()
        # snap.particles.charge = charges

        snap.particles.orientation  = orientations_all
        snap.particles.angmom  = angmoms_all
        snap.particles.velocity  = velocities_all

        if(ts==0):
            with gsd.hoomd.open(name=self.dumpFilename, mode='w') as f:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='wb') as f:
                f.append(snap)
        else:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='rb+') as f:
            with gsd.hoomd.open(name=self.dumpFilename, mode='r+') as f:
                f.append(snap)

    def print_performance_info(self):
        total_time_passed = time.time() - self.timer
        total_tps = self.timestep / total_time_passed
        current_time_passed = time.time() - self.epoch_timer
        epoch_tps = self.dumpFreq / current_time_passed

        print('TPS : %.2f   - last TPS : %.2f' %(total_tps,epoch_tps))
        self.epoch_timer = time.time()

    def run_gpu(self,N_steps):

        self.move_to_gpu()
        self.is_sync = 0
        c_dump = 0
        N_dump = N_steps//self.dumpFreq
        self.timestep = 0
        self.timer = time.time()
        self.epoch_timer = time.time()
        self.t_s1 = 0.0
        self.t_mid = 0.0
        self.t_s2 = 0.0
        self.get_pts = 0
        self.gen_nep = 0
        self.eval_force = 0
        self.t_1 = 0
        self.t_infer = 0

        self.t1_gen = 0
        self.t2_gen = 0
        self.t3_gen = 0
        self.t4_gen = 0
        self.t5_gen = 0

        self.t1_nep = 0
        self.t2_nep = 0
        self.t3_nep = 0
        self.t4_nep = 0
        self.t5_nep = 0

        # self.t_s1_1 = 0
        # self.t_s1_2 = 0
        # self.t_s1_3 = 0
        # self.t_s1_4 = 0
        # self.t_s1_5 = 0

        for i in range(N_steps+1):
            # print(i)
            if(i%self.Nlist_freq==0 and i>0):
                self.refreshNList_gpu()
            if(i%self.dumpFreq==0 and i>0):
                print("%d/%d"%(c_dump,N_dump))
                self.print_performance_info()
                self.dumpConfig_gpu(c_dump)
                c_dump += 1
            self.integrate_gpu()
            # if(i==500):
            # #     # print('NEP TIMES')
            # #     # print(self.t1_gen)
            # #     # print(self.t2_gen)
            # #     # print(self.t3_gen)
            # #     # print(self.t4_gen)
            # #     # print(self.t5_gen)
            # #
            # #     # print('NEP TIMES')
            # #     # print(self.t1_nep)
            # #     # print(self.t2_nep)
            # #     # print(self.t3_nep)
            # #     # print(self.t4_nep)
            # #     # print(self.t5_nep)
            #     print('----')
            #     print("Gen pts", self.get_pts)
            #     print("Gen neps", self.gen_nep)
            #     print("Eval ", self.eval_force)
            #     print("First integrate ", self.t_s1)
            #     print("Mid integrate ", self.t_mid)
            #     print("Second integrate ", self.t_s2)
            # #     # print(self.t_infer*0.001)
            # #
            #     exit()


    def integrate_gpu(self):
        """
        Nose-Hoover NVT, Two-step Integration
        Implemented from HOOMD Source Code
        """

        if(self.timestep==0):
            self.integrate_mid_step_gpu()
            self.integrate_step_two_gpu()

            # ftrue = np.load('forces37.npy')
            # fhere = self.forces.get()
            # ttrue = np.load('torks37.npy')
            # there = self.torks.get()

            # diff = np.abs(ftrue - fhere)
            # diff = np.abs(ttrue - there)
            # print(diff)
            # exit()


        else:
            if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
            t0 = time.time()
            self.integrate_step_one_gpu()

            if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()

            t1 = time.time()
            self.integrate_mid_step_gpu()
            if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()

            t2 = time.time()
            self.integrate_step_two_gpu()
            if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()

            t3 = time.time()

            self.t_s1 += t1-t0
            self.t_mid += t2-t1
            self.t_s2 += t3-t2


    def getRotKin_gpu(self):
        """
        From ComputeThermo.cc - ke_rot_total
        Scalar3 I = h_inertia.data[j];
        quat<Scalar> q(h_orientation.data[j]);
        quat<Scalar> p(h_angmom.data[j]);
        quat<Scalar> s(Scalar(0.5)*conj(q)*p);
        ke_rot_total /= Scalar(2.0);

        res = cp.zeros_like(a)

        s1 = a[:,0]*b[:,0]
        s2 = -cp.sum(a[:,1:]*b[:,1:],axis=1)
        scalar = s1+s2

        v1 = a[:,0].reshape(-1,1)*b[:,1:]
        v2 = b[:,0].reshape(-1,1)*a[:,1:]
        v3 = cp.cross(a[:,1:],b[:,1:])
        v = v1 + v2 + v3

        res[:,0] = scalar
        res[:,1:] = v
        """

        conj_q = cp.copy(self.orientations)
        conj_q = -conj_q
        conj_q[:,0] = -conj_q[:,0]
        s = quaternion_multiplication_gpu(conj_q,cp.copy(self.angmom))
        s = s*0.5
        rot_en = s[:,1]*s[:,1]/self.moi[0] + s[:,2]*s[:,2]/self.moi[1] + s[:,3]*s[:,3]/self.moi[2]
        rot_en = rot_en*0.5
        return cp.sum(rot_en)


    def initialize_integrator(self):
        """ see void TwoStepNVTMTK::randomizeVelocities """
        self.integrator_vars = np.zeros(4) # xi, eta, xi_rot, eta_rot
        sigmasq_t = 1.0/(self.N_dof*self.tau**2)
        s = np.random.normal(0.0, np.sqrt(sigmasq_t)) # TODO : I'm not sure about this
        self.integrator_vars[0] = s
        self.m_exp_thermo_fac = 1.0

    def integrate_step_one_gpu(self):

        """
        only works with cubic box (due to the way pbc is handled in kernel)
        Integrate Step 1 - Trans
        Nothing about the thermostat regular update - No need for for loop
        """
        Nthreads = 512
        blocks = (10,)
        threads_per_block = (Nthreads,)

        step_one_translation_kernel(
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

        # self.velocities = self.velocities+ (0.5)*self.accel*self.dt
        # self.velocities = self.m_exp_thermo_fac*self.velocities
        # self.central_pos = self.central_pos +  self.velocities*self.dt
        # self.central_pos = cp.where(self.central_pos > 0.5 * self.Lx, self.central_pos- self.Lx, self.central_pos)
        # self.central_pos = cp.where(self.central_pos <- 0.5 * self.Lx, self.Lx + self.central_pos, self.central_pos)


        """
        Integrate Step 1 - Rotation
        self.Nparticles
        """
        #exp_fac = np.exp((-self.dt/2.0)*self.integrator_vars2)
        self.dp = self.dp.astype(cp.float32)
        Nthreads = 512
        blocks = (4,)
        threads_per_block = (Nthreads,)

        step_one_rotation_kernel(
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

        ### high cost region ###
        # self.angmom += self.dp
        # self.angmom *= exp_fac
        # self.angmom,self.orientations = permutation1_gpu2(self.angmom,self.orientations,self.moi[2],self.dt)
        # self.angmom,self.orientations = permutation2_gpu2(self.angmom,self.orientations,self.moi[1],self.dt)
        # self.angmom,self.orientations = permutation3_gpu2(self.angmom,self.orientations,self.moi[0],self.dt)
        # self.angmom,self.orientations = permutation2_gpu2(self.angmom,self.orientations,self.moi[1],self.dt)
        # self.angmom,self.orientations = permutation1_gpu2(self.angmom,self.orientations,self.moi[2],self.dt)
        # self.orientations = renormalize_quat_gpu(self.orientations)
        ### high cost region ###

        """
        Advance Thermostat - Trans
        """

        trans_kin_en = (0.5)*(self.mass)*(cp.sum(self.velocities*self.velocities,axis=1))
        # trans_kin_en = cp.sum(trans_kin_en).get()

        self.trans_kin_en = cp.sum(trans_kin_en)
        # trans_temp = (2.0/self.N_dof)*trans_kin_en
        # xi_prime = self.integrator_vars[0] + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)
        # self.integrator_vars[0] = xi_prime + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)

        # self.integrator_vars[1] += xi_prime*self.dt
        # self.m_exp_thermo_fac = np.exp(-0.5*self.integrator_vars[0]*self.dt);


        """
        Advance Thermostat - Rot
        Scalar xi_prime_rot = xi_rot + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*
            (Scalar(2.0)*curr_ke_rot/ndof_rot/m_T->getValue(timestep) - Scalar(1.0));
        """
        # xi_rot = np.copy(self.integrator_vars[2])
        # eta_rot = np.copy(self.integrator_vars[3])

        rotational_energy = cp.empty((self.Nparticles),dtype=cp.float32)
        rotational_kinetic_energy_kernel(
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

        # xi_prime_rot = xi_rot + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*rot_kin_en)/self.RN_dof)/self.kT - 1.0)
        # xi_rot =  xi_prime_rot + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*rot_kin_en)/self.RN_dof)/self.kT - 1.0)
        #
        # eta_rot = eta_rot + xi_prime_rot*self.dt
        # self.integrator_vars[2] = xi_rot
        # self.integrator_vars[3] = eta_rot

    def integrate_step_two_gpu(self):

        Nthreads = 10
        blocks = (1,)
        threads_per_block = (Nthreads,)

        step_two_update_variables_kernel(
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


        # trans_temp = (2.0/self.N_dof)*self.trans_kin_en
        # xi_prime = self.integrator_vars0 + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)
        # self.integrator_vars0 = xi_prime + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)
        # self.integrator_vars1 += xi_prime*self.dt
        # self.m_exp_thermo_fac = cp.exp(-0.5*self.integrator_vars0*self.dt);
        #
        # xi_prime_rot = self.integrator_vars2 + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*self.rot_kin_en)/self.RN_dof)/self.kT - 1.0)
        # self.integrator_vars2 =  xi_prime_rot + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*self.rot_kin_en)/self.RN_dof)/self.kT - 1.0)
        #
        # self.integrator_vars3 = self.integrator_vars3 + xi_prime_rot*self.dt


        # print(self.integrator_vars0,self.integrator_vars1,self.integrator_vars2,self.integrator_vars3,self.m_exp_thermo_fac)


        """
        Integrate Step 2 - Trans
        4. velocity component is mass in HOOMD code

        """

        self.accel = self.forces/self.mass.reshape(-1,1)
        self.velocities = self.velocities*self.m_exp_thermo_fac
        self.velocities = self.velocities + (0.5)*(self.dt)*(self.accel) ### sikintili step bu
        # print(self.velocities[self.deb_i])

        """
        Integrate Step 2 - Rot

        Only advanced angular momentum, don't touch orientations

        Forces are fine at the box frame but the torks must be in the hoomd - shape
        frame which is defined by the orientation quaternion. To convert the box frame
        torks to just repeat the hoomd code rotate(conj(quat),tork)
        """
        Nthreads = 512
        blocks = (10,)
        threads_per_block = (Nthreads,)

        step_two_rotation_kernel(
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

        # self.tt = rotate_torks_to_body_frame_gpu(self.orientations,self.torks)
        # exp_fac = np.exp((-self.dt/2.0)*self.integrator_vars2)
        # self.angmom *= exp_fac
        # self.dp = get_dp_gpu(self.orientations,self.tt,self.dt)
        # self.angmom += self.dp


        self.timestep += 1


    def integrate_mid_step_gpu(self):
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()

        t0 = time.time()
        self.get_pts_represenation_gpu()
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()

        t1 = time.time()
        # self.generate_nep_descriptors2_gpu()
        self.generate_nep_cuda()
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()

        t2 = time.time()
        self.evaluate_force_torques_gpu()
        if(self.is_sync==1):
            cp.cuda.Stream.null.synchronize()

        t3 = time.time()
        self.get_pts += t1-t0
        self.gen_nep += t2-t1
        self.eval_force += t3-t2




    def get_pts_represenation_gpu(self):
        translate = self.central_pos[self.Nlist[:,1]]-self.central_pos[self.Nlist[:,0]]
        translate = cp.where(translate > 0.5 * self.Lx, translate- self.Lx, translate)
        translate = cp.where(translate <- 0.5 * self.Lx, self.Lx + translate, translate)
        dist = cp.linalg.norm(translate,axis=1)
        mask = cp.where(dist < self.cutoff_gpu)[0]


        self.pairs = self.Nlist[mask]
        pair0 = self.pairs[:,0]
        pair1 = self.pairs[:,1]
        self.pp = torch.from_dlpack(self.pairs)

        translate = translate[mask]
        N_pair = len(self.pairs)
        self.N_pair = N_pair

        QUAT1 = self.orientations[pair0]
        QUAT2 = self.orientations[pair1]

        # simraw = np.hstack((translate.get(),QUAT1.get(),QUAT2.get()))
        # unred = simraw_to_unred(simraw)
        # self.en_true = calc_en_unred(unred)
        self.pts_pair_rot2 = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        self.pts_pair = cp.empty((self.N_pair,self.Nd,3),dtype=cp.float32)
        blocks = (self.N_pair,)
        threads_per_block = (self.Nd*3,)

        get_pts_pairs_kernels1(
            blocks,
            threads_per_block,
            (
                QUAT1,
                QUAT2,
                translate,
                self.pts_rep,
                self.pts_pair,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

        get_pts_pairs_kernels2(
            blocks,
            threads_per_block,
            (
                QUAT2,
                QUAT1,
                translate,
                self.pts_rep,
                self.pts_pair_rot2,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )




        blocks = (self.N_pair*7,)
        threads_per_block = (self.Nd*3,)
        self.all_pts1 = cp.empty((self.N_pair*7,self.Nd,3),dtype=cp.float32)
        self.all_pts2 = cp.empty((self.N_pair*3,self.Nd,3),dtype=cp.float32)

        apply_dx_dteta_kernel(
            blocks,
            threads_per_block,
            (
                self.pts_pair,
                self.all_pts1,
                self.dx_gpu,
                self.dteta_gpu,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )

        apply_dteta_kernel2(
            blocks,
            threads_per_block,
            (
                self.pts_pair_rot2,
                self.all_pts2,
                self.dteta_gpu,
                cp.int32(self.N_pair),
                cp.int32(self.Nd)
            )
        )



        nn = self.N_pair
        # self.t1_gen += t1-t0
        # self.t2_gen += t2-t1
        # self.t3_gen += t3-t2
        # self.t4_gen += t4-t3
        # self.t5_gen += t5-t4
        #



    def generate_nep_cuda(self):

        pos_gpu = cp.vstack((self.all_pts1,self.all_pts2 ))
        pos_gpu = pos_gpu.astype(cp.float32)



        n_chebysev = self.nrad + 1
        n_pairs = pos_gpu.shape[0]

        r_ij = cp.empty_like(pos_gpu)
        r_ij_norm = cp.empty((n_pairs,self.Nd),cp.float32)
        chebo = cp.empty((n_pairs,n_chebysev,self.Nd),cp.float32)
        g_rad = cp.empty((n_pairs,n_chebysev),cp.float32)

        blocks = (n_pairs,)
        threads_per_block = (self.Nd,)


        radial_cuda_kernel(
            blocks,
            threads_per_block,
            (
                pos_gpu,
                r_ij,
                r_ij_norm,
                chebo,
                g_rad,
                n_pairs,
                n_chebysev,
                cp.int32(self.Nd) ## was 12 can be problematic
            )
        )

        g_rad = cp.sum(chebo,axis=-1)


        chebo = chebo[:,:self.nang+1]

        chebo_angular = cp.einsum('...i,...j->...ij', chebo, chebo).reshape(n_pairs, self.nang+1, -1)

        blocks = (n_pairs,)
        threads_per_block = (self.Nd*self.Nd,)
        lego = cp.empty((n_pairs,self.lmax + 1,self.Nd*self.Nd),cp.float32)
        angular_cuda_kernel(
            blocks,
            threads_per_block,
            (
                r_ij,
                r_ij_norm,
                lego,
                n_pairs,
                self.Nd,
                self.lmax
            )
        )

        lego_trans = lego[:, 1:, :].transpose(0, 2, 1)
        g_ang = cp.matmul(chebo_angular, lego_trans)
        g_ang = g_ang.reshape(n_pairs, -1)


        ###### ANGULAR ##########
        self.g_all_cupy = cp.hstack((g_rad,g_ang))


    def evaluate_gradients_gpu(self):
        """
        Forward the generated descriptors through the trained neural-net
        Can contatenate all inputs and pass together
        """
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        #
        # torch.cuda.synchronize()
        # start_event.record()
        #
        self.g_all = torch.from_dlpack(self.g_all_cupy)


        with torch.no_grad():
            e_all = self.energy_model.energy_net(self.g_all)

        # end_event.record()
        # torch.cuda.synchronize()
        # inference_time_ms = start_event.elapsed_time(end_event)
        # self.t_infer += inference_time_ms


        e0 = e_all[:self.N_pair]
        edx = e_all[self.N_pair:self.N_pair*2]
        edy = e_all[self.N_pair*2:self.N_pair*3]
        edz = e_all[self.N_pair*3:self.N_pair*4]

        edtetax = e_all[self.N_pair*4:self.N_pair*5]
        edtetay = e_all[self.N_pair*5:self.N_pair*6]
        edtetaz = e_all[self.N_pair*6:self.N_pair*7]

        edtetax2 = e_all[self.N_pair*7:self.N_pair*8]
        edtetay2 = e_all[self.N_pair*8:self.N_pair*9]
        edtetaz2 = e_all[self.N_pair*9:self.N_pair*10]

        self.gradient_dx = torch.zeros((self.N_pair,3),device=self.torch_device)
        self.gradient_dx[:,0] = -(edx-e0).squeeze(1) / self.dx
        self.gradient_dx[:,1] = -(edy-e0).squeeze(1) / self.dx
        self.gradient_dx[:,2] = -(edz-e0).squeeze(1) / self.dx

        self.gradient_dtetax = torch.zeros((self.N_pair,3),device=self.torch_device)
        self.gradient_dtetax[:,0] = -(edtetax-e0).squeeze(1) / self.dteta
        self.gradient_dtetax[:,1] = -(edtetay-e0).squeeze(1) / self.dteta
        self.gradient_dtetax[:,2] = -(edtetaz-e0).squeeze(1) / self.dteta

        self.gradient_dtetax2 = torch.zeros((self.N_pair,3),device=self.torch_device)
        self.gradient_dtetax2[:,0] = -(edtetax2-e0).squeeze(1) / self.dteta
        self.gradient_dtetax2[:,1] = -(edtetay2-e0).squeeze(1) / self.dteta
        self.gradient_dtetax2[:,2] = -(edtetaz2-e0).squeeze(1) / self.dteta
        self.e0 = e0
        ### dogrusu ikisi de eksi ###


    def evaluate_force_torques_gpu(self):

        self.evaluate_gradients_gpu()



        # self.e0 = self.e0*(en_max-en_min) + en_min

        en_range = self.en_max - self.en_min

        forces_inter = self.gradient_dx*en_range
        torks_inter1 = self.gradient_dtetax*en_range
        torks_inter2 = self.gradient_dtetax2*en_range


        forces_inter[forces_inter>100.0] = 100.0
        torks_inter1[torks_inter1>100.0] = 100.0
        torks_inter2[torks_inter2>100.0] = 100.0

        forces_inter[forces_inter<-100.0] = -100.0
        torks_inter1[torks_inter1<-100.0] = -100.0
        torks_inter2[torks_inter2<-100.0] = -100.0


        """
        Calculate Net Force-Tork On each Shape
        Forces are at the interaction frame, reorient them to be in the box frame
        """

        forces_net = torch.zeros((self.Nparticles,3),device=self.torch_device)
        torks_net = torch.zeros((self.Nparticles,3),device=self.torch_device)

        # #
        # # p0 = torch.from_numpy(self.pairs[:,0])
        # # p1 = torch.from_numpy(self.pairs[:,1])
        # # p0 = p0.to(device=self.torch_device)
        # # p1 = p1.to(device=self.torch_device)
        # p = torch.from_numpy(self.pairs)
        # p = p.to(device=self.torch_device)


        forces_net.index_add_(0, self.pp[:,0], forces_inter)
        forces_net.index_add_(0, self.pp[:,1], -forces_inter)

        torks_net.index_add_(0, self.pp[:,0], torks_inter1)
        torks_net.index_add_(0, self.pp[:,1], torks_inter2)


        # forces_net = torch.zeros((self.Nparticles,3),dtype=torch.float64)
        # torks_net = torch.zeros((self.Nparticles,3),dtype=torch.float64)
        #
        # forces_net.index_add_(0, torch.from_numpy(self.pairs[:,0]), torch.from_numpy(forces_inter))
        # forces_net.index_add_(0, torch.from_numpy(self.pairs[:,1]), torch.from_numpy(-forces_inter))
        #
        # torks_net.index_add_(0, torch.from_numpy(self.pairs[:,0]), torch.from_numpy(torks_inter1))
        # torks_net.index_add_(0, torch.from_numpy(self.pairs[:,1]), torch.from_numpy(torks_inter2))
        #

        # self.torks = torks_net.detach().numpy()
        # self.forces = forces_ net.detach().numpy()
        t0 = time.time()

        # self.torks = torks_net.cpu().detach().numpy()
        # self.forces = forces_net.cpu().detach().numpy()
        self.torks = cp.from_dlpack(torks_net)

        self.forces = cp.from_dlpack(forces_net)

        # ftrue = np.load('forces37.npy')
        # fhere = self.forces.get()
        # diff = np.abs(fhere-ftrue)
        # print(diff)
        # print(np.max(diff))
        # exit()


        t1 = time.time()
        self.t_1 += t1 - t0
