#!/bin/bash -l
#SBATCH --job-name=ta_recov
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err

# Issue #3 (Track A) step 7: simulate from the pooled posterior means with g set to theory / to the
# fitted value, then refit with the same pipeline.  Task list = track_a/bash/tasks_recover.txt.
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export PYTENSOR_FLAGS="optimizer=None,floatX=float32"
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export OMP_NUM_THREADS=1
TASKFILE=${TASKFILE:-track_a/bash/tasks_recover.txt}
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKFILE")
[ -z "$ARGS" ] && { echo "no task"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID -> [$ARGS] on $(hostname) at $(date)"
$PY -u track_a/recover.py $ARGS
echo "done at $(date)"
