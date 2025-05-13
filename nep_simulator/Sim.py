import cupy as cp
import numpy as np 
from scipy.spatial import cKDTree as KDTree
import time 
import gsd.hoomd

from ForceEvaluator import ForceEvaluator
from DescriptorGenerator import DescriptorGenerator
from GsdHandler import GsdHandler
import MathUtils as mu

class Sim():
    
    def __init__(self):
        print("A new simulation instance created.")
        ## TODO : Add another class that contains the simulation data 
        ## like positions and orientations and move saving-reading gsd to that class

        ## TODO : Adjust block-thread sizes for different particle numbers 
        ## TODO : Initialize the Nose Hoover variables 

    def setDevice(self,gpu_id):
        self.gpu_id = gpu_id
        self.device = cp.cuda.Device(gpu_id)
        self.device.use()

    def initializeSystem(self,gsd_file):
        """
        Read the configuration gsd file 
        """
        print("Simulation will be initialized from %s"%(gsd_file))
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

        self.dx = 0.00001
        self.dteta = 0.00001

        
    def setEvaluator(self,model_path):
        """
        Set the evaluator for the forces and torques
        """
        
        self.evaluator = ForceEvaluator()
        self.evaluator.setDevice(self.gpu_id)
        self.evaluator.setNN_dimensions(150,3)
        self.evaluator.setNd(self.N_descriptors)
        self.evaluator.read_model(model_path)
        self.evaluator.set_dx_dteta(self.dx,self.dteta)
        self.evaluator.setNparticles(self.Nparticles)


    def setNepDescriptors(self,hypers):
        self.descriptor_generator = DescriptorGenerator()
        self.descriptor_generator.setDevice(self.device)
        self.descriptor_generator.setHyperParameters(hypers)    
        self.descriptor_generator.setBoxSize(self.Lx)
        self.N_descriptors = self.descriptor_generator.getN_descriptor()
        self.descriptor_generator.set_dx_dteta(self.dx,self.dteta)


    def setShape(self,shape_str):    
        self.descriptor_generator.setShape(shape_str)
        en_min, en_max = self.descriptor_generator.get_en_min_max()
        self.evaluator.set_energy_range(en_min, en_max)
        self.cutoff = self.descriptor_generator.cutoff

        self.evaluator.move_data_to_device()
        self.descriptor_generator.move_data_to_device()

    def setNeighborList(self):
        self.Nlist_cutoff = self.cutoff + 2.0
        self.Nlist_freq = 100
        pos_tree = self.central_pos + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        self.Nlist = tree.query_pairs(r=self.Nlist_cutoff,output_type='ndarray')
        self.Nlist = cp.asarray(self.Nlist)
        self.cutoff_gpu = cp.ones(self.Nlist.shape[0], dtype=cp.float32)*self.cutoff    

    def setGsdDump(self,dump_period,filename):
        self.gsdDumpPeriod = dump_period   
        self.gsdFilename = filename


    def setkT(self,kT):
        self.kT = kT
        self.integrator_vars0 = -0.0317711
        self.integrator_vars1 = 0.108609
        self.integrator_vars2 = 0.00814227
        self.integrator_vars3 = -0.014238
        self.m_exp_thermo_fac = 1.00011

        self.integrator_vars0 = cp.asarray(self.integrator_vars0,dtype=cp.float32)
        self.integrator_vars1 = cp.asarray(self.integrator_vars1,dtype=cp.float32)
        self.integrator_vars2 = cp.asarray(self.integrator_vars2,dtype=cp.float32)
        self.integrator_vars3 = cp.asarray(self.integrator_vars3,dtype=cp.float32)
        self.m_exp_thermo_fac = cp.asarray(self.m_exp_thermo_fac,dtype=cp.float32)

    def setdt(self,dt):
        self.dt = dt
        self.tau = 100.0*dt

    def dumpConfig(self,ts):
        pos_all = self.central_pos.get()
        typeid_all = np.zeros(len(self.central_pos),dtype=int)
        velocities_all = self.velocities.get()
        orientations_all = self.orientations.get()
        angmoms_all = self.angmom.get()


        snap = gsd.hoomd.Frame()
        snap.configuration.step = ts
        snap.configuration.box = [self.Lx.get(), self.Lx.get(), self.Lx.get(), 0, 0, 0]
        snap.particles.N = len(pos_all)
        snap.particles.position = pos_all
        snap.particles.types  = ['A','B']
        snap.particles.typeid  = typeid_all
        snap.particles.moment_inertia = self.moi_to_dump
        snap.particles.mass = self.mass.get()

        snap.particles.orientation  = orientations_all
        snap.particles.angmom  = angmoms_all
        snap.particles.velocity  = velocities_all
       
        if(ts==0):
            with gsd.hoomd.open(name=self.gsdFilename, mode='w') as f:
                f.append(snap)
        else:
            with gsd.hoomd.open(name=self.gsdFilename, mode='r+') as f:
                f.append(snap)        


    def move_data_to_device(self):
        self.central_pos = cp.asarray(self.central_pos,dtype=cp.float32)
        self.orientations = cp.asarray(self.orientations,dtype=cp.float32)
        self.velocities = cp.asarray(self.velocities,dtype=cp.float32)
        self.angmom = cp.asarray(self.angmom,dtype=cp.float32)
        self.mass = cp.asarray(self.mass,dtype=cp.float32)
        self.moi_gpu = cp.asarray(self.moi,dtype=cp.float32)
        self.dp = cp.empty_like(self.angmom)


        self.trans_kin_en = (0.5)*(self.mass)*(cp.sum(self.velocities*self.velocities,axis=1))
        self.rot_kin_en = self.getRotKin()

        self.N_dof = cp.asarray(self.N_dof,dtype=cp.float32)
        self.RN_dof = cp.asarray(self.RN_dof,dtype=cp.float32)
        self.dt = cp.asarray(self.dt,dtype=cp.float32)
        self.tau = cp.asarray(self.tau,dtype=cp.float32)
        self.kT = cp.asarray(self.kT,dtype=cp.float32)

        self.Lx = cp.asarray(self.Lx,dtype=cp.float32)

        self.dx = cp.asarray(self.dx,dtype=cp.float32)
        self.dteta = cp.asarray(self.dteta,dtype=cp.float32)

    def getRotKin(self):

        conj_q = cp.copy(self.orientations)
        conj_q = -conj_q
        conj_q[:,0] = -conj_q[:,0]
        s = mu.quaternion_multiplication(conj_q,cp.copy(self.angmom))
        s = s*0.5
        rot_en = s[:,1]*s[:,1]/self.moi[0] + s[:,2]*s[:,2]/self.moi[1] + s[:,3]*s[:,3]/self.moi[2]
        rot_en = rot_en*0.5
        return cp.sum(rot_en)

    def print_performance_info(self):
        if(self.timestep > 0):
            total_time_passed = time.time() - self.timer
            total_tps = self.timestep / total_time_passed
            current_time_passed = time.time() - self.epoch_timer
            epoch_tps = self.gsdDumpPeriod / current_time_passed

            print('TPS : %.2f   - last TPS : %.2f' %(total_tps,epoch_tps))
            self.epoch_timer = time.time()

    def refreshNList(self):
        pos_tree = self.central_pos + self.Lx*0.5
        pos_tree = pos_tree.get()
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx.get()+0.0001)
        self.Nlist = tree.query_pairs(r=self.Nlist_cutoff,output_type='ndarray')
        self.Nlist = cp.asarray(self.Nlist)
        self.cutoff_gpu = cp.ones(self.Nlist.shape[0], dtype=cp.float32)*self.cutoff    

    def set_timers(self):
        self.t_gen = 0
        self.t_eval = 0
        self.t_s1 = 0
        self.t_s2 = 0
        self.t_nlist = 0

    def run(self,Nsteps):
        self.is_sync = 1
        self.descriptor_generator.setSync(self.is_sync)
        self.evaluator.setSync(self.is_sync)
        self.set_timers()

        self.move_data_to_device()
        self.timestep = 0
        self.timer = time.time()
        self.epoch_timer = time.time()

        while(self.timestep < Nsteps):
           
            if(self.timestep%self.Nlist_freq==0 and self.timestep>0):
                if(self.is_sync==1):
                     cp.cuda.Stream.null.synchronize()
                t0 = time.time()     
                self.refreshNList()
                if(self.is_sync==1):
                     cp.cuda.Stream.null.synchronize()
                t1 = time.time()
                self.t_nlist += t1-t0

            if(self.timestep % self.gsdDumpPeriod == 0):
                self.print_performance_info()
                self.dumpConfig(self.timestep)

            self.step()

        print('Timers:')
        print('t_nlist : %.2f'%(self.t_nlist))
        print('t_gen : %.2f'%(self.t_gen))
        print('t_eval : %.2f'%(self.t_eval))
        print('t_s1 : %.2f'%(self.t_s1))
        print('t_s2 : %.2f'%(self.t_s2))    

        print('Descriptor Generator Timers:')
        print('t_pos_to_pts : %.2f'%(self.descriptor_generator.t_pos_to_pts))
        print('t_apply_dx_dteta : %.2f'%(self.descriptor_generator.t_dx))
        print('t_pts_to_nep : %.2f'%(self.descriptor_generator.t_pts_to_nep))

        print('NEP timers')
        print('t_nep1 : %.2f'%(self.descriptor_generator.t_nep1))
        print('t_nep2 : %.2f'%(self.descriptor_generator.t_nep2))
        print('t_nep3 : %.2f'%(self.descriptor_generator.t_nep3))
        print('t_nep4 : %.2f'%(self.descriptor_generator.t_nep4))
        print('t_nep5 : %.2f'%(self.descriptor_generator.t_nep5))

        print('Done')        

    def step(self):
        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t0 = time.time()        
        g_nep, pp, Npair = self.descriptor_generator.generate_nep_descriptors(self.central_pos,self.orientations,self.Nlist)

        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t1 = time.time()     
        self.forces, self.torks = self.evaluator.evaluate_interactions(g_nep,pp,Npair)
        
        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t2 = time.time()     
        self.integrate_step_two()
        
        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t3 = time.time()     
        self.integrate_step_one()
        
        if(self.is_sync==1):
                cp.cuda.Stream.null.synchronize()
        t4 = time.time()   

        self.t_gen += t1-t0
        self.t_eval += t2-t1
        self.t_s1 += t3-t2
        self.t_s2 += t4-t3



