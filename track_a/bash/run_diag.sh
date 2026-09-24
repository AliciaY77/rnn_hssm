#!/bin/bash -l
#SBATCH --job-name=ta_diag
#SBATCH --account=carney-mjfrank-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%j.err
cd /users/igrahek/rnn_hssm
/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python -u track_a/diag_ssms.py $DIAG_ARGS
