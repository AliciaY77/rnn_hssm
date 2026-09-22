#!/bin/bash -l
#SBATCH --job-name=tb_pooled
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=05:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --output=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm_track_b/cluster/log/%x-%A_%a.err
# Pooled route-2 fits: 20 networks x 500 trials = 10 000 trials per gain, M = 100, 7-point profile.
# 6 tasks = gains {0.8, 1.2, 1.0} x hit-mode {none, cross}.
cd /users/igrahek/rnn_hssm_track_b
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
GAINS=(0.8 0.8 1.2 1.2 1.0 1.0); MODES=(none cross none cross none cross)
i=$SLURM_ARRAY_TASK_ID; G=${GAINS[$i]}; H=${MODES[$i]}
echo "task $i: pooled gain $G hit-mode $H on $(hostname) $(date)"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
$PY -u track_b/fit_bounded.py --pooled --gain "$G" --hit-mode "$H" --M 100 --sub-trials 500 --n-profile 7
echo "done $(date)"
