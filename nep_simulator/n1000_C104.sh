#!/bin/sh

python3 -W ignore sim_input.py --init ./init/N1000_init.gsd --out analytical_N1000_dump100_kt05_dt005_C104_seed0 --ts 200000 --dump 100 --kT 0.5 --dt 0.005 --model_path ./models/model_nep_C104.pth --gpu_id 1 --hypers 10 8 4 4.5 --shape cube_v2
python3 -W ignore sim_input.py --init ./init/N1000_init.gsd --out analytical_N1000_dump100_kt06_dt005_C104_seed0 --ts 200000 --dump 100 --kT 0.6 --dt 0.005 --model_path ./models/model_nep_C104.pth --gpu_id 1 --hypers 10 8 4 4.5 --shape cube_v2
python3 -W ignore sim_input.py --init ./init/N1000_init.gsd --out analytical_N1000_dump100_kt07_dt005_C104_seed0 --ts 200000 --dump 100 --kT 0.7 --dt 0.005 --model_path ./models/model_nep_C104.pth --gpu_id 1 --hypers 10 8 4 4.5 --shape cube_v2