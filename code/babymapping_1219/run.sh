#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:4
#SBATCH --job-name=multigt
#SBATCH -o ./logs/%j.txt
srun --mpi=pmi2 sh ./yaml/script/Mapping_Xencoder_full_ProGAN_GAN_MSGAN_ACGAN_deepArch_multi-gt_v4.sh
