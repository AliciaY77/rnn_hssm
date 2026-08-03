#!/bin/bash
# Submit DDM fitting jobs for low/medium/high gain levels
for gain in 0.8 1.0 1.2; do
    sbatch bash/run_fit_ddm_fixed.sh $gain
    sbatch bash/run_fit_ddm_weibull.sh $gain
done
