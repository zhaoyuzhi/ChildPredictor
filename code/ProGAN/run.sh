#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:8
#SBATCH --job-name=mixture1
#SBATCH -o ./logs/mixtureFace_GAN_MSGAN_ACGAN.txt
srun --mpi=pmi2 python -u train.py
