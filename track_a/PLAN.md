# Track A plan (issue #3): corrected HSSM OU fits on constant-bound data, 20 networks

Branch `track-a-hssm-ou-corrected`. Local worktree `…/Studies/rnn_hssm_track_a`; Oscar clone `/users/igrahek/rnn_hssm`
(already on this branch; data under `data/processed/{fixed_bound,kernel}`). Sign convention everywhere: **g > 0 leaky**.
Verified today with ssms 0.8.3: median first-passage time 0.59 / 0.78 / 1.10 s at g = −1 / 0 / +1 (v = 0, a = 1), and the
stretch rule (RT×k ⇔ v/√k, a√k, g/k) reproduces native quantiles within 4 % at k = 10 while the inverted v rule fails by 10×.

Data facts (checked): b = 1.5 CSVs have 38 640 / 39 881 / 39 988 trials at gains 0.8 / 1.0 / 1.2, min RT 6 / 6 / 5 ms,
median 145 / 99 / 76 ms, 11 signed coherences (−0.15 … +0.15 in steps of 0.03), 1869–2000 trials per seed, P(choice +) ≈ 0.49.
With k = 10 and c = 0.3 s the fixed t is 0.36 / 0.36 / 0.35 s and no RT falls below it.

Expected parameters at k = 10, from the native exact-simulator optimum in `output/feasibility/matches_table.csv`
(box v10_g10_a0.05, b = 1.5): gain 0.8 → g ≈ +0.59, a ≈ 1.24, v(0.15) ≈ 1.26; gain 1.0 → g ≈ +0.44, a ≈ 1.12, v(0.15) ≈ 1.27;
gain 1.2 → g ≈ +0.93, a ≈ 0.88, v(0.15) ≈ 1.32. Treat these as the expected sign pattern and rough magnitude, not truth.

## Steps (each gated; do not start the next before the check passes)

1. **Fit script `track_a/fit_hssm_ou.py`** (start from `recovery/simulate_and_fit_ou.py`, not from `model/`): args `--gain --k
   --n-sub --seed-filter --smoke --draws --tune --tag`. Loads the b = 1.5 CSV, `rt = k·rt + 0.3`, `response = response_choice`,
   `v ~ 1 + coherence_signed`, `t` passed as a float = 0.3 + k·min(rt_native), `p_outlier=None, lapse=None`, sampler
   `nuts_numpyro`, 4 chains, `target_accept 0.95`, `log_likelihood=False`. Writes `output/track_a/<tag>_summary.csv`
   (mean, sd, hdi 3/97, r_hat, ess_bulk, ess_tail per parameter), `<tag>_meta.json` (n, k, t, omission fraction, divergences,
   minutes, edge mass = posterior fraction within 0.02 of a LAN bound for a, g, v(0.15)), and the netcdf (gitignored).
   Check: `print(model)` shows t fixed and no lapse; a DataFrame row count equals the CSV's.
2. **Smoke on Oscar (login node or 1 short job)**: gain 1.0, `--n-sub 2000 --smoke` (1 chain, 20/20). Check: runs end to end,
   summary written, no NaN. Then **timing fit**: gain 1.0, 2000 trials, full sampling. Check: R-hat ≤ 1.01, divergences < 1 %,
   record minutes. Expected 30–45 min. Extrapolate to 20 000 trials; if the projected time exceeds 6 h choose the largest
   stratified subsample (equal per seed) that fits in 6 h, and record it in RESULTS.md.
3. **Main array (`track_a/bash/run_track_a.sh`, from `recovery/bash/run_recovery_array.sh`)**: tasks = 3 pooled fits at k = 10
   (n = 20 000 stratified by seed, or all trials if step 2 allows) + 60 per-network fits at k = 10 (all trials of that seed)
   + 3 pooled fits at k = 8 (sanity). 16–32 GB, 4 CPUs, 6 h. Check on landing: R-hat ≤ 1.01, ESS_bulk ≥ 400, divergences
   < 1 % for every fit; edge mass for g < 0.5 (if the g posterior piles on +1 at k = 10, rerun the pooled fits at k = 12 and say so).
   Expected: g > 0 at every gain in every converged fit; g largest at gain 1.2; 20/20 networks g > 0 at each gain.
4. **Hierarchical attempt** (one job per gain, 12 h budget): `g ~ 1 + (1|seed)`, `a ~ 1 + (1|seed)`, `v ~ 1 + coherence_signed
   + (1|seed)`, on the step-3 subsample. If it does not finish or converge in budget, report that and rely on the 60 per-network
   fits. Do not let this block steps 5–7.
5. **PPC on RT/choice (`track_a/ppc_rt.py`)**: for each pooled fit, 200 posterior draws × ≥ 20 000 trials per coherence with
   ssms `ornstein` (max_t ≥ 2 s stretched horizon = 7.5 s applied afterwards; simulate to max(7.5, 2) and drop non-crossers).
   Table of RT quantile errors (10/30/50/70/90, correct and error, native ms), accuracy and omission per coherence;
   one figure per gain (network histograms vs model). Expected: quantile errors ≲ 10 ms at gains 1.0 and 1.2, worse at 0.8
   (its errors are as fast as its corrects); omissions within 2 points.
6. **Kernel PPC — the decisive check (`track_a/ppc_kernel.py`)**: posterior mean and 20 draws per gain, converted to native units
   (g_nat = g·k, a_nat = a/√k, v_nat(coh) = v(coh)·√k, t_nat = (t − 0.3)/k). Drive the OU with the real evidence streams
   `data/processed/kernel/seed*.npz::rel` (all 20 networks, 40 000 trials per gain, float32), `kernel_fit/fit_ou_kernel.py::kernel`
   verbatim. Run two variants and report both: (a) **as fitted**: sticky bound at ±a_nat, start z, evidence enters as
   v1_nat·e_t with unit intrinsic diffusion, choice = sign(a_T) (document that evidence noise adds diffusion on top of the
   fitted dW; also run a version with intrinsic diffusion σ² = 1 − v1_nat²·dt so total per-step variance is 1); (b) **leak only**:
   same g_nat and v_nat, no bound (B = ∞), choice = sign(a_T). Compare 8 weights and slope to `output/kernel_fit/network_kernels.csv`
   (pooled slopes +0.039 / +0.002 / −0.026). Expected: (b) gives recency (slope > 0) at all three gains; (a) will very likely
   give an early-commitment primacy shape that is the same at all gains because a_nat ≈ 0.3 is hit within ~100 ms. Either way
   the network's ordering (recency → flat → primacy) is absent; if a variant reproduces the ordering, stop and report.
   One panel with network vs model kernels for the three gains (per variant).
7. **Recovery at k = 10 (`track_a/recover.py`)**: 6 datasets from the pooled posterior means with g replaced by theory
   (+0.43 / −0.16 / −0.62) and by the fitted values, n = 20 000, 7.5 s horizon applied (drop non-crossers, report the fraction),
   same fit pipeline. Check: sign of g recovered in all 6 and 94 % HDI covers truth in ≥ 4. Report as a table.
8. **Appendix**: one fit on the original Weibull data at gain 1.0 (`data/processed/hssm_ready_nxx1_s42_g1.0.parquet` → ship as
   CSV) with the same corrections. Expected: still g < 0 or on an edge, showing the collapse confound.
9. **RESULTS.md + collect.py + issue comment** with the headline table (gain × g mean [HDI] native and stretched × R-hat ×
   edge mass × kernel verdict), native-unit translation table, omission fractions, and the truncation caveat.

Rules: smoke before any array; commit small, push often; `!output/track_a/` in `.gitignore`, no netcdfs in git; numbers not
adjectives; never fit t, never force a deadline choice, lapse off. Report to the coordinator after steps 2, 3 and 6.
