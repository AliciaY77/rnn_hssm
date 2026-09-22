#!/bin/bash -l
#SBATCH --job-name=ta_diag2
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%j.err
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
for MT in 2 5 7.81 20 60 200; do
  for V in 0.0 -1.2624; do
    $PY -u track_a/diag_ssms2.py --v $V --max-t $MT 2>&1 | tr '\n' ' '
    echo "  [exit $?]"
  done
done
echo "sweep finished"
