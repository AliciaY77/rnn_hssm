#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/users/xyuan48/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/xyuan48/rnn_hssm/cluster/log/%x-%j.err

cd /users/xyuan48/rnn_hssm

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate hssm

export PYTHONPATH=/users/xyuan48/rnn_hssm:$PYTHONPATH
echo "PYTHONPATH is: $PYTHONPATH"

python -u data/process/process_data.py
