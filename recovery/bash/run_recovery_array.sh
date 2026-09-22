#!/bin/bash -l
#SBATCH --job-name=ou_recov
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err

# Issue #1: OU LAN parameter recovery, one grid cell per array task (20 cells).
# `-l` on the shebang so `conda` is defined in the batch shell.
cd /users/igrahek/rnn_hssm
source ~/.conda/envs/pyHSSM_New_Nov24/bin/activate 2>/dev/null || true
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export PYTENSOR_FLAGS="optimizer=None,floatX=float32"
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export OMP_NUM_THREADS=1
echo "cell $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
$PY -u recovery/simulate_and_fit_ou.py --cell $SLURM_ARRAY_TASK_ID
