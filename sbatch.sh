#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4
nvcc --version
source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate ArterySeg
cd tests
python -u "$@"