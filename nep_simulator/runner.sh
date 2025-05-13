#!/bin/sh
python3 -W ignore sim_input.py --init ./init/init_N200_seed10.gsd --out cube05.gsd --ts 100000 --dump 1000 --kT 0.5 --dt 0.005 --model_path ./models/model_nep_C104.pth --gpu_id 1 --hypers 10 8 4 4.5 --shape cube_v2 &> cube.log
wait
python3 -W ignore sim_input.py --init ./init/th_init_N200_seed20.gsd --out th05.gsd --ts 100000 --dump 1000 --kT 0.5 --dt 0.005 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1 &> th.log
wait
python3 -W ignore sim_input.py --init ./init/pbpv3_init_N200_seed60.gsd --out pbp025.gsd --ts 100000 --dump 1000 --kT 0.25 --dt 0.005 --model_path ./models/model_pbp_C104.pth --gpu_id 1 --hypers 10 8 4 7.5 --shape pbp_v3 &> pbp.log
wait
