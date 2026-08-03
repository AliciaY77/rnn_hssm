#!/bin/bash
#SBATCH --job-name=ddm_weibull
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=batch
#SBATCH --qos=carney-condo2
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/users/xyuan48/rnn_hssm/cluster/log/%x-%j.out
#SBATCH --error=/users/xyuan48/rnn_hssm/cluster/log/%x-%j.err

GAIN=${1:-1.0}

cd /users/xyuan48/rnn_hssm
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate hssm
export PYTENSOR_FLAGS="optimizer=None"
export PYTHONPATH=/users/xyuan48/rnn_hssm:$PYTHONPATH
echo "PYTHONPATH is: $PYTHONPATH"
echo "Fitting gain: $GAIN"

python -u data/process/process_data.py --gain $GAIN
python -u model/fit_ddm_weibull.py --gain $GAIN
