from NVT import NVT
import argparse
import sys, os



"""

Can be run with:
python3 -W ignore sim_input.py --init ./init/initial_config.gsd --out output.gsd --ts 20000 --dump 1000 --kT 0.7 --dt 0.005 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1

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
non_opt.add_argument('--hypers_nep', metavar="<float>", type=float, dest="hypers_nep", required=True, nargs='+')
non_opt.add_argument('--model_path', metavar="<dat>", type=str, dest="model_path", required=True)
non_opt.add_argument('--gpu_id', metavar="<int>", type=int, dest="gpu_id", required=True)
non_opt.add_argument('--shape', metavar="<str>", type=str, dest="shape", required=True)

args = parser.parse_args()
init_gsd = args.init_gsd
output = args.output
timesteps = args.timesteps
dump_period = args.dump_period
kT = args.kT
dt = args.dt
hypers_nep = args.hypers_nep
model_path = args.model_path
gpu_id = args.gpu_id
shape = args.shape

if not os.path.exists('./out/'):
    os.makedirs('./out/')

gsd_filename =  './out/' + output +'.gsd'

sim = NVT()
sim.setDevice(gpu_id)
sim.initializeSystem(init_gsd)
sim.setNepDescriptors(hypers_nep)
sim.setEvaluator(model_path)
sim.setShape(shape)
sim.setNeighborList()
sim.setGsdDump(dump_period, gsd_filename)
sim.setkT(kT)
sim.setdt(dt)

sim.run(timesteps)
