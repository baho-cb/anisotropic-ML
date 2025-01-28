#!/bin/sh
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed20.gsd --out nep100_th_N200_dump100_kt05_dt005_ts2e5_seed20 --ts 200000 --dump 100 --kT 0.5 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed21.gsd --out nep100_th_N200_dump100_kt05_dt005_ts2e5_seed21 --ts 200000 --dump 100 --kT 0.5 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed22.gsd --out nep100_th_N200_dump100_kt05_dt005_ts2e5_seed22 --ts 200000 --dump 100 --kT 0.5 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1

python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed20.gsd --out nep100_th_N200_dump100_kt06_dt005_ts2e5_seed20 --ts 200000 --dump 100 --kT 0.6 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed21.gsd --out nep100_th_N200_dump100_kt06_dt005_ts2e5_seed21 --ts 200000 --dump 100 --kT 0.6 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed22.gsd --out nep100_th_N200_dump100_kt06_dt005_ts2e5_seed22 --ts 200000 --dump 100 --kT 0.6 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1

python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed20.gsd --out nep100_th_N200_dump100_kt07_dt005_ts2e5_seed20 --ts 200000 --dump 100 --kT 0.7 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed21.gsd --out nep100_th_N200_dump100_kt07_dt005_ts2e5_seed21 --ts 200000 --dump 100 --kT 0.7 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
python3 -W ignore run_gpu.py --init ./init/th_init_N200_seed22.gsd --out nep100_th_N200_dump100_kt07_dt005_ts2e5_seed22 --ts 200000 --dump 100 --kT 0.7 --dt 0.005 --cont 0 --model_path ./models/model_th_C100.pth --gpu_id 1 --hypers 10 10 10 4.5 --shape th_v1
