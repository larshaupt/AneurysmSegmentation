#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32
#SBATCH --cpus-per-task=4

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate BrainSeg
nnUNet_plan_and_preprocess -t 544 --verify_dataset_integrity -tf 1 -tl 1
nnUNet_train 3d_fullres nnUNetTrainerV2 Task544_BrainArtery 5 --npz