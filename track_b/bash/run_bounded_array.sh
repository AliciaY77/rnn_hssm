#!/bin/bash -l
#SBATCH --job-name=tb_bounded
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-79
#SBATCH --output=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.err
# Route 2 (issue #4): bounded OU MC likelihood, one (seed, gain, hit-mode) per task.
# Task list: track_b/bash/tasks_bounded.txt -- 0-79 = gains 0.8 and 1.2 (both variants),
# 80-119 = gain 1.0.  ~2200 likelihood evaluations per task.
cd /users/igrahek/rnn_hssm_track_b
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
read -r S G H <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" track_b/bash/tasks_bounded.txt)"
echo "task $SLURM_ARRAY_TASK_ID: seed $S gain $G hit-mode $H on $(hostname) $(date)"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
$PY -u track_b/fit_bounded.py --seed "$S" --gain "$G" --hit-mode "$H" --M 300
echo "done $(date)"
