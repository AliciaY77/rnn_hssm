#!/bin/bash -l
#SBATCH --job-name=track_a
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err

# Issue #3 (Track A): one HSSM OU fit per array task.
# The task list is a text file of argument lines for track_a/fit_hssm_ou.py, one per array index
# (line 1 = task 0). Pass it with --export=TASKFILE=track_a/bash/tasks_<name>.txt and set
# --array to match its length.  `-l` on the shebang so conda is defined in the batch shell.
#
#   sbatch --array=0-2 --export=ALL,TASKFILE=track_a/bash/tasks_pooled.txt track_a/bash/run_track_a.sh
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export PYTENSOR_FLAGS="optimizer=None,floatX=float32"
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export OMP_NUM_THREADS=1

TASKFILE=${TASKFILE:?set TASKFILE=track_a/bash/tasks_<name>.txt}
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ARGS=$(sed -n "${LINE}p" "$TASKFILE")
if [ -z "$ARGS" ]; then echo "no task on line $LINE of $TASKFILE"; exit 1; fi
echo "task $SLURM_ARRAY_TASK_ID -> [$ARGS] on $(hostname) at $(date)"
$PY -u track_a/fit_hssm_ou.py $ARGS
echo "done at $(date)"
