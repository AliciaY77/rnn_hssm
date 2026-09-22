#!/bin/bash -l
#SBATCH --job-name=ou_match
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-29
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err
# 30 tasks = gains {0.8,1.0,1.2} x bounds {1.5,1.25} x stretch {1,2,4,6,8}
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
GAINS=(0.8 1.0 1.2); BOUNDS=(1.5 1.25); KS=(1 2 4 6 8)
i=$SLURM_ARRAY_TASK_ID; K=${KS[$((i % 5))]}; B=${BOUNDS[$(((i / 5) % 2))]}; G=${GAINS[$((i / 10))]}
echo "task $i: gain $G bound $B stretch $K on $(hostname) $(date)"
$PY -u feasibility/ou_match.py --gain $G --bound $B --stretch $K
