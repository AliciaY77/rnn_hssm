# Track B plan (issue #4): likelihood-based evidence-conditioned OU fit per network, with intervals

Branch `track-b-evidence-conditioned-fit`. Local worktree `…/Studies/rnn_hssm_track_b` (data symlinked under `data/processed`);
Oscar worktree `/users/igrahek/rnn_hssm_track_b` (same branch; symlinked data). SLURM scripts must `cd` to the Oscar worktree
and write logs to its `cluster/log/`, not to `/users/igrahek/rnn_hssm`. Sign convention: **g > 0 leaky (recency)**.

Model, native time, dt = 1 ms, T = 750, σ = 1 fixed (state it): a_0 = a_bias; a_{t+1} = a_t + (v·e_t − g·a_t)dt + √dt·ξ_t;
sticky bound ±B; choice = 1 iff a_T > 0. Route 1 is B = ∞, where a_T is Gaussian: μ_i = a_bias·ρ^T + v·dt·Σ_t ρ^{T−1−t} e_{i,t},
s² = dt·Σ_t ρ^{2(T−1−t)}, ρ = 1 − g·dt, P(choice = 1) = Φ(μ_i/s).

Smoke numbers already obtained by the coordinator (seed 42, route 1 MLE, Nelder–Mead from g₀ ∈ {−5, 0, 5}, SE from a numerical
Hessian; `rel` cast to float32): gain 0.8 → g = +4.55 ± 0.24 /s, v = 56.2; gain 1.0 → g = +0.75 ± 0.18, v = 51.8;
gain 1.2 → g = −2.04 ± 0.20, v = 50.2. Total 4 s. Your step 1 must reproduce these within ±0.05 in g.

## Steps (each gated)

1. **`track_b/ou_lik.py`**: analytic negative log-likelihood (route 1), a vectorised simulator with sticky bound (route 2,
   common random numbers, M realisations per trial), and unit tests: (i) analytic P(choice) vs Monte Carlo with 20 000
   realisations on 20 random trials, max |Δ| < 0.01; (ii) seed 42 MLEs match the smoke numbers above. Run tests locally with
   `/opt/homebrew/anaconda3/bin/python`.
2. **Route 1, 60 fits (`track_b/fit_analytic.py`)**: MLE + Hessian SE for every (seed, gain), plus Bayesian NUTS with numpyro
   (priors g ~ N(0, 10), v ~ HalfNormal(100), a_bias ~ N(0, 1); 4 chains × 1000/1000; run on Oscar or in a local venv with
   `pip install numpyro`; `python -m venv --system-site-packages` off anaconda base works). Also pooled fits per gain (40 000
   trials). Output `output/track_b/analytic_per_network.csv` (g, v, a_bias, SE, 94 % HDI, R-hat), `analytic_pooled.csv`.
   Checks: R-hat ≤ 1.01; MLE and posterior mean agree within 1 SE. Expected: 20/20 g > 0 at 0.8 with intervals excluding 0;
   ≥ 15/20 g < 0 at 1.2 with most intervals excluding 0; gain 1.0 centred near +0.5 with mixed signs; pooled ≈ +4 / +0.5 / −2.
   Note a_bias is unidentified when g is strongly positive (its effect is a_bias·ρ^T ≈ 0.03·a_bias at g = +4.5); say so.
3. **Bound-hit export**: add `tcross2_g{gain}` (first index with |dv| ≥ 2.0, NaN if never; also `tcross15_g{gain}` at 1.5) to
   `kernel_fit/export_evidence.py`, run it with the `rnn_gain` env (`RNN_T=750 RNN_EPOCHS=100 /opt/homebrew/anaconda3/envs/rnn_gain/bin/python
   kernel_fit/export_evidence.py --n-trials 2000`, ≈ 15 min for 20 nets) into the worktree's data dir (it is a symlink to the
   shared folder: write to a temp dir first, verify `choice_g*` and `rel` are bit-identical to the existing npz, then replace,
   and `scp` to Oscar `/users/igrahek/rnn_hssm/data/processed/kernel/`). Report per gain the fraction of trials that cross 2.0
   before T (coordinator's quick estimate from dvT alone: ≥ 0.32 / 0.67 / 0.86 for |dvT| ≥ 2.5).
4. **Route 2 (`track_b/fit_bounded.py`)**: MC likelihood, M = 300, per-trial P(choice = 1) = mean σ((a_T)/τ) with τ = 0.1 (state
   τ), Nelder–Mead on (g, v, B, a_bias) started from the route-1 MLE with B ∈ {1.0, 3.0, 10} (three starts, to probe the
   leak/bound degeneracy); JAX `lax.scan` jitted on Oscar CPUs (≈ 1 s per evaluation is the target). **Bound-hit term**: per
   trial, the model's P(hit) = fraction of realisations with |a_t| ≥ B before T, and the network's hit = 1[tcross2 < T]; add the
   Bernoulli log-likelihood of hits to the choice log-likelihood (this is the judgment call; record it in RESULTS.md).
   Fit each (seed, gain) **with and without** the hit term. Intervals: profile likelihood on g (11-point grid around the MLE,
   re-optimising v, B, a_bias; the 1.92-unit drop defines the 95 % interval). Smoke first on seed 42 gain 1.2 (both variants),
   check the likelihood at the route-1 MLE with B = 50 equals the analytic value within 1 %; then Oscar array: 60 (seed, gain)
   × 2 variants = 120 tasks, gains 0.8 and 1.2 first, 4 CPUs, ≤ 2 h each. Expected: with the hit term, no fit lands on g ≳ +8
   with B < 2; without it, a few do (the `kernel_fit/` caveat: seeds 58 at 1.0 and 1.2, 59 at 1.2 had g ≈ +12–13 / +2.7 with low B).
5. **Recovery (`track_b/recover.py`)**: 5 networks (42, 46, 51, 58, 61) × 3 gains, choices simulated on the real evidence with
   (i) the fitted route-1 parameters, (ii) g = +4.3 / −1.6 / −6.2 with fitted v and a_bias; refit route 1. Also route 2 for
   (ii) at gains 0.8 and 1.2 with the fitted B, and one degenerate dataset (g = +12, B = 1.2, gain 1.2 evidence) fitted with and
   without the hit term. Check: sign of g recovered in every case, truth inside the interval in ≥ 90 %; the degenerate case is
   resolved only with the hit term.
6. **Checks/figures (`track_b/checks.py`)**: kernel reproduction per gain (pooled and per network; simulate choices from the
   fitted parameters on the real evidence with fresh noise; `kernel_fit/fit_ou_kernel.py::kernel` verbatim; figure like
   `output/kernel_fit/ou_kernel_fits.png`), psychometric per gain, per-network g with 94 % intervals against gain (with
   the landscape values +4.3 / −1.6 / −6.2 marked as approximate and the Fig 1F zero crossing f = 1.016 noted), g vs network
   kernel slope with error bars, sensitivity 1000 vs 2000 trials. Deadline-vs-bounded-choice sensitivity: the fixed-bound
   data come from a different rollout (`mscont_cal_pertrial.parquet`), so trials cannot be matched to the npz stimuli; skip
   and note this (coordinator's call).
7. **RESULTS.md + issue comment**: per gain the median g [IQR] across networks, the fraction of networks whose interval excludes
   0 on the predicted side, pooled estimates with intervals, route 1 vs route 2 (with/without hit term), recovery table.

Rules: smoke before any array; commit small, push often; `!output/track_b/` in `.gitignore`; no npz or large arrays in git;
numbers not adjectives; `rel` float16 → float32; deadline choice only; common random numbers. Report to the coordinator after
steps 2, 4 and 5.
