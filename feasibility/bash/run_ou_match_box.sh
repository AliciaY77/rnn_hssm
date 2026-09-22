#!/bin/bash -l
#SBATCH --job-name=ou_box
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-11
#SBATCH --output=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.out
#SBATCH --error=/users/igrahek/rnn_hssm/cluster/log/%x-%A_%a.err
# Does the LAN box, not the OU family, force "leaky at every gain"?  12 tasks = 3 gains x 4 variants:
#   V1 native time, box wide open (v 10, g 10, a >= 0.05): the OU family itself, bound 1.5
#   V2 native time, wide open, bound 2.0 (samples more of the landscape)
#   V3 k=4, drift relaxed to 8, LAN g box kept
#   V4 k=4, LAN drift box kept, g relaxed to 3
cd /users/igrahek/rnn_hssm
PY=/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python
GAINS=(0.8 1.0 1.2); i=$SLURM_ARRAY_TASK_ID; G=${GAINS[$((i % 3))]}; V=$((i / 3))
case $V in
  0) ARGS="--bound 1.5 --stretch 1 --v-max 10 --g-max 10 --a-min 0.05 --t-min 0";;
  1) ARGS="--bound 2.0 --stretch 1 --v-max 10 --g-max 10 --a-min 0.05 --t-min 0";;
  2) ARGS="--bound 1.5 --stretch 4 --v-max 8 --g-max 1";;
  3) ARGS="--bound 1.5 --stretch 4 --v-max 2 --g-max 3";;
esac
echo "task $i: gain $G $ARGS on $(hostname) $(date)"
$PY -u feasibility/ou_match.py --gain $G $ARGS
