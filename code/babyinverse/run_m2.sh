#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:4
#SBATCH --job-name=c_g
#SBATCH -o ./logs/%j.txt
srun --mpi=pmi2 python -u main.py --config ./yaml/Inverse_ProGAN_GAN_start-with-code.yaml --mode train
