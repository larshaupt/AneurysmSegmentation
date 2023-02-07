#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=2

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate ArterySeg
cd results
python -u "$@"