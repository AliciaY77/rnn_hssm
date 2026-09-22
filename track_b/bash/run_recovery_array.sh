#!/bin/bash -l
#SBATCH --job-name=tb_rec2
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-22
#SBATCH --output=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.err
# Route-2 parameter recovery (issue #4 step 5): synthetic choices simulated on the real evidence.
# Task list: track_b/bash/tasks_recovery.txt, written by track_b/recover.py --stage make-r2.
cd /users/igrahek/rnn_hssm_track_b
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
read -r S G H NAME <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" track_b/bash/tasks_recovery.txt)"
echo "task $SLURM_ARRAY_TASK_ID: seed $S gain $G hit-mode $H dataset $NAME on $(hostname) $(date)"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
$PY -u track_b/fit_bounded.py --seed "$S" --gain "$G" --hit-mode "$H" --M 300 \
    --synthetic "output/track_b/recovery/${NAME}.npz" \
    --out output/track_b/bounded_recovery --tag "${NAME}_${H}"
echo "done $(date)"
