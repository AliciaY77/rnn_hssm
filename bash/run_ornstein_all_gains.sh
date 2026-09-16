#!/bin/bash
for gain in 0.8 1.0 1.2; do
    sbatch bash/run_fit_ornstein.sh $gain
done
