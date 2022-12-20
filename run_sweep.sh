#!/bin/bash

NAME="brainseg_sweep"
SWEEP_ID="57wuc2cy"
WANDBPROJECT="USZ_final"
NGPUS=1
JOB_FILE="$NAME-job.sh"
MAX_JOBS=3
NRUNS=6
MEM=32G
CPUS=4


cat << EOT > "jobs/$JOB_FILE"
#!/bin/bash

#SBATCH --output=/scratch_net/biwidl311/lhauptmann/segmentation_3D/sbatch_logs/TRAIN-%x.%A.%a.out
#SBATCH --gres=gpu:$NGPUS
#SBATCH --job-name=$NAME
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