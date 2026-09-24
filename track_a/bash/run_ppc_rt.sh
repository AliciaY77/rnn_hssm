#!/bin/bash -l
#SBATCH --job-name=ta_ppcrt
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%j.err

# Issue #3 (Track A) step 5: RT/choice posterior predictive with the exact ssms `ornstein` simulator.
#   sbatch --export=ALL,PPC_ARGS="--all-pooled --k 10" track_a/bash/run_ppc_rt.sh
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export OMP_NUM_THREADS=1
echo "ppc_rt [$PPC_ARGS] on $(hostname) at $(date)"
$PY -u track_a/ppc_rt.py $PPC_ARGS
echo "done at $(date)"
