#!/bin/bash

NAME="Loss_Sweep_7"
SWEEP_ID="y3cn9n4s"
WANDBPROJECT="USZ_final"
NGPUS=1
JOB_FILE="$NAME-job.sh"
MAX_JOBS=7
NRUNS=5
MEM=32G
CPUS=4


cat << EOT > "jobs/$JOB_FILE"
#!/bin/bash


#SBATCH --gres=gpu:$NGPUS
#SBATCH --job-name=$NAME
#SBATCH --output=sbatch_logs/job-name-%x.%A.%a.out
#SBATCH --mem-per-cpu=$MEM
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-$(( $NRUNS-1 ))%$MAX_JOBS

cd /scratch_net/biwidl311/lhauptmann/segmentation_3D

source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate ArterySeg

srun wandb agent --count 1 lhauptmann/$WANDBPROJECT/$SWEEP_ID

EOT

sbatch "jobs/$JOB_FILE"