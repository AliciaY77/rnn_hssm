# rnn_hssm

Fitting drift-diffusion models (DDM) to RNN behavioral data using HSSM.

## Repo structure
- `data/data_raw/` — raw RNN trial data (not tracked by git)
- `data/process/` — data processing scripts
- `data/processed/` — HSSM-ready processed data (not tracked by git)
- `model/` — HSSM fitting scripts
- `bash/` — SLURM job submission scripts
- `output/` — fitted model results (not tracked by git)
- `analysis/` — Jupyter notebooks for analysis
- `cluster/log/` — SLURM logs (not tracked by git)

## First pass
One network: nxx1, seed=42, gain=1.0
Two models: DDM fixed bound, DDM Weibull collapsing bound
Drift rate regressed on coherence: `v ~ 1 + coherence`
