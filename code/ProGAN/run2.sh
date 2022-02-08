#!/bin/bash
#MV2_USE_CUDA=0 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:0
#SBATCH --job-name=datagenerate
#SBATCH -o ./data_generate_FFHQ.txt
srun --mpi=pmi2 python -u dataset_tool2.py create_from_images datasets/FFHQFace ../Datasets/FFHQ/FFHQ-128x128/

