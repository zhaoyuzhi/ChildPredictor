#!/bin/bash
#MV2_USE_CUDA=0 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:0
#SBATCH --job-name=select
#SBATCH -o ./baby_select.txt
srun --mpi=pmi2 python -u main.py

