#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --cpus-per-task=4

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate BrainSeg
python -u "$@"