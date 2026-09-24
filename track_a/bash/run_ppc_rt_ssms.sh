#!/bin/bash -l
#SBATCH --job-name=ta_ppcssms
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%j.err

# Cross-check of the numpy Euler engine against ssm-simulators: ONE PROCESS PER FIT and only 2 draws,
# because ssms 0.8.3 corrupts the heap after ~38 calls with varying parameters (jobs 6614551 / 6614775).
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
export OMP_NUM_THREADS=1
for G in 0.8 1.0 1.2; do
  for ENG in numpy ssms; do
    echo "=== gain $G engine $ENG"
    $PY -u track_a/ppc_rt.py --tag g${G}_k10_b1.5_pooled --k 10 --engine $ENG \
        --n-draws 2 --n-per-coh 500 --out-suffix _xcheck_${ENG} 2>&1 | grep -E "quantile error|abs_coherence|^ +0\.|Traceback|Error"
    echo "   [exit ${PIPESTATUS[0]}]"
  done
done
echo "cross-check finished"
