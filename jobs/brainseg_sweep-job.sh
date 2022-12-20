#!/bin/bash

#SBATCH --output=/scratch_net/biwidl311/lhauptmann/segmentation_3D/sbatch_logs/TRAIN-%x.%A.%a.out
#SBATCH --gres=gpu:1
#SBATCH --job-name=brainseg_sweep
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-5%3

cd /scratch_net/biwidl311/lhauptmann/segmentation_3D

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate ArterySeg

srun wandb agent --count 1 lhauptmann/USZ_final/57wuc2cy

