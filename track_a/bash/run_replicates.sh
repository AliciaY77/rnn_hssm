#!/bin/bash -l
#SBATCH --job-name=ta_repl
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err
# Posterior-predictive replicates (omissions, median RT, accuracy) for the pooled Track A fits; one (bound, gain) per task.
#   sbatch --array=0-11 --export=ALL,TASKFILE=track_a/bash/tasks_replicates.txt track_a/bash/run_replicates.sh
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export OMP_NUM_THREADS=1
TASKFILE=${TASKFILE:?set TASKFILE}
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKFILE")
echo "task $SLURM_ARRAY_TASK_ID -> [$ARGS] on $(hostname) at $(date)"
$PY -u track_a/ppc_replicates.py $ARGS
echo "done at $(date)"
