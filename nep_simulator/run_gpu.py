import numpy as np
from EnergyModel import EnergyModel
from Sim import Sim
import argparse
import sys, os
from cupyx.profiler import benchmark



"""
April 2022

This script sets the parameters for Neural-Net Assisted rigid-body MD simulations
of cubes in NVT (Nose-Hoover thermostat).

Can be run with:
python3 run.py --init ./init/init.gsd --out output_gsd --ts 20000 --dump 2000 --kT 0.3 --dt 0.005 --cont 0

./init/ : Contains the initial state gsd_files that the simulation will start from
./models/ : Contains the neural-nets
./out/ : Simulation output is saved in this folder as a gsd file

"""

parser = argparse.ArgumentParser(description="Runs NN-assisted MD")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('--init', metavar="<dat>", type=str, dest="init_gsd",
required=True, help=".gsd files " )
non_opt.add_argument('--out', metavar="<dat>", type=str, dest="output",
required=True, help=".gsd files " )
non_opt.add_argument('--ts', metavar="<int>", type=int, dest="timesteps", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--dump', metavar="<int>", type=int, dest="dump_period", required=True, help="dump frequency ",default=1 )
non_opt.add_argument('--kT', metavar="<float>", type=float, dest="kT", required=True, help="temperature ",default=1 )
non_opt.add_argument('--dt', metavar="<float>", type=float, dest="dt", required=True, help="dt ",default=1 )
non_opt.add_argument('--cont', metavar="<int>", type=int, dest="is_cont", required=True, help="1 if continue, 0 if not(about thermostat variables) ",default=1 )
non_opt.add_argument('--hypers_nep', metavar="<float>", type=float, dest="hypers_nep", required=False, nargs='+')
non_opt.add_argument('--model_path', metavar="<dat>", type=str, dest="model_path", required=False, help="if continue you need to get accelerations as well ",default='no_accel')
non_opt.add_argument('--gpu_id', metavar="<int>", type=int, dest="gpu_id", required=False, help="if continue you need to get accelerations as well ",default='no_accel')
non_opt.add_argument('--shape', metavar="<str>", type=str, dest="shape", required=False, help="if continue you need to get accelerations as well ",default='no_accel')

args = parser.parse_args()
init_gsd = args.init_gsd
output = args.output
timesteps = args.timesteps
dump_period = args.dump_period
kT = args.kT
dt = args.dt
is_cont = args.is_cont
hypers_nep = args.hypers_nep
model_path = args.model_path
gpu_id = args.gpu_id
shape = args.shape

if not os.path.exists('./out/'):
    os.makedirs('./out/')

outname =  './out/' + output +'.gsd'

sim = Sim()
tau = 0.5
N_list_every = 100

sim.setHypersNep(hypers_nep)
sim.placeParticlesFromGsd(init_gsd)
sim.setForces(model_path,gpu_id)
sim.setShape_gpu(shape)
sim.setNeighborList_gpu(N_list_every)
sim.setDumpFreq(dump_period)
sim.setDump(outname)
sim.setkT(kT)
sim.setTau(tau)
sim.setdt(dt)
sim.setGradientMethod("forward")
sim.set_integrator(np.array([-0.0317711,0.108609,0.00814227,-0.014238]))
sim.set_m_exp_factor(1.00011)
sim.is_continue(is_cont)
# sim.set_accelerations(accel_file)

# print(benchmark(sim.run_gpu, (100,), n_repeat=1))


sim.run_gpu(timesteps)
print("Simulation completed.")
