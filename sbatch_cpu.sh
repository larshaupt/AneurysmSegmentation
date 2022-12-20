#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate ArterySeg
python -u "$@"