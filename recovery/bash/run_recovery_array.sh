#!/bin/bash -l
#SBATCH --job-name=ou_recov
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39
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
# tasks 0-19: no lapse mixture, 4.5 s deadline; tasks 20-39: no lapse, no deadline.
# (The first submission, job 6604819, ran tasks 0-19 with HSSM defaults = 5% lapse + deadline.)
CELL=$((SLURM_ARRAY_TASK_ID % 20)); BLOCK=$((SLURM_ARRAY_TASK_ID / 20))
FLAGS="--no-lapse"; [ "$BLOCK" -ge 1 ] && FLAGS="$FLAGS --no-deadline"
echo "task $SLURM_ARRAY_TASK_ID -> cell $CELL flags [$FLAGS] on $(hostname) at $(date)"
$PY -u recovery/simulate_and_fit_ou.py --cell $CELL $FLAGS
