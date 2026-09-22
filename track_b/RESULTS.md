# Track B (issue #4): likelihood-based evidence-conditioned OU fit, per network, with intervals

Branch `track-b-evidence-conditioned-fit`, base `ou-round2-base`. This turns the simulation-matching
proof of concept of `kernel_fit/` into a Brunton-style likelihood fit: the probability of each trial's
**deadline** choice given **that trial's own evidence stream**, maximised (and sampled) over the OU leak
g, the evidence scaling v, the sticky bound B and the start point a_bias.

**Sign convention, everywhere in this note: g > 0 = leaky = recency; g < 0 = unstable = primacy.**
The landscape "drift slope" of `RNN_Gain_Mod/lab/2026-09-21_gain-leaky-to-bistable` has the **opposite**
sign (its −4.3 / +1.6 / +6.2 per s at gains 0.8 / 1.0 / 1.2 correspond to g = +4.3 / −1.6 / −6.2 here).

## Model and data

    a_0   = a_bias
    a_t+1 = a_t + (v·e_t − g·a_t)·dt + sqrt(dt)·xi_t,      xi_t ~ N(0, 1)
    sticky bound: once |a| ≥ B the accumulator freezes for the rest of the trial
    choice = 1 ("+") iff a_T > 0

dt = 1 ms, T = 750 steps, **σ = 1 fixed** — σ and v are not jointly identifiable from choices alone, so
v and B are in noise units, as in `kernel_fit/`. e_t is the network's own relevant-stream evidence
(coherence + N(0,1) per ms) from `data/processed/kernel/seed{42..61}.npz`, cast float16 → float32.
Choice is the deadline choice `choice_g{gain}` = (dv[:, −1] < 0), 2000 trials per network, the same 2000
stimuli at all three gains, 20 networks.

**Route 1** is the B → ∞ limit, where a_T is Gaussian given the evidence:
μ_i = a_bias·ρ^T + v·dt·Σ_t ρ^(T−1−t)·e_it, s² = dt·Σ_t ρ^(2(T−1−t)), ρ = 1 − g·dt,
P(choice_i = 1) = Φ(μ_i/s). Three free parameters (g, v, a_bias); exact, no Monte Carlo.

**Route 2** adds the sticky bound and is fitted by Monte-Carlo likelihood with common random numbers:
M = 300 realisations per trial, P(choice = 1) = mean of logistic(a_T/τ) with **τ = 0.1**, the M×N×T noise
block fixed at construction so the objective is deterministic in the parameters. Four free parameters
(g, v, B, a_bias); Nelder-Mead on (g, v, log B, a_bias) from the route-1 MLE with B ∈ {1, 3, 10};
95 % intervals by profile likelihood on g (11 points, the other three re-optimised, 1.92-unit drop).

## Judgment calls (recorded as required by the issue)

1. **What the bound-hit term is.** The plan specified the Bernoulli log-likelihood of
   1[the network's |dv| reached 2.0 before T]. That indicator is nearly constant in these data
   (see the bound-hit table below), so it carries little information. I therefore implemented **two**
   bound-hit variants and ran both: `bern` (the plan's Bernoulli) and `cross` (the categorical
   log-likelihood of the *first-crossing-time bin*: 0–50, 50–100, 100–150, 150–250, 250–400, 400–750 ms,
   never — which contains the Bernoulli as its coarsest margin). The issue explicitly allows the
   crossing-time distribution as an extra likelihood term ("where feasible"). `cross` is used as the
   headline "with hit term" variant; `bern` is reported beside it.
2. **The identification the hit term assumes.** The model's bound B is in accumulator noise units and the
   network's dv is in logit units, so matching them requires a choice. Both variants identify the model's
   sticky bound B with the network's |dv| = 2.0 level. This is an assumption, not a measurement; it is the
   price of using the bound-hit observable at all. `tcross15` (the |dv| = 1.5 level) was exported as well
   so the assumption can be varied later.
3. **τ = 0.1** for the logistic smoothing of the MC choice probability, as planned: with B = 50 the MC
   log-likelihood then reproduces the analytic one to +0.18 % (test iii) and to +0.007 % in the JAX
   implementation at the route-1 MLE of seed 42 / gain 1.2 (273.37 vs 273.35).
4. **a_bias** is weakly identified wherever g is strongly positive (its effect on a_T is a_bias·ρ^T ≈
   0.03·a_bias at g = +4.5), so the Bayesian prior a_bias ~ N(0, 1) shrinks it noticeably at gain 0.8
   while leaving g essentially unchanged (see the MLE-vs-posterior comparison).
5. **Deadline vs bounded choice sensitivity was skipped** (coordinator's call, carried over from the plan):
   the fixed-bound data come from a different rollout (`mscont_cal_pertrial.parquet`), so its trials cannot
   be matched to the npz stimuli.
6. **Profile-likelihood grid**: ±max(1.0, 3·SE_route1) around the route-2 MLE. Where the 1.92 drop is not
   reached inside the grid, the interval is reported at the grid edge (flagged in the per-fit JSON).

## Unit tests (`python track_b/ou_lik.py --test`, all pass)

| test | result |
|---|---|
| (i) analytic P(choice) vs Monte Carlo, 20 000 realisations, 4 parameter sets × 20 random trials | max |ΔP| = 0.0049 (< 0.01) |
| (ii) seed-42 route-1 MLEs vs the coordinator's smoke numbers | g = +4.5524 / +0.7471 / −2.0414 vs +4.55 / +0.75 / −2.04; max |Δ| = 0.0029 |
| (iii) MC log-likelihood with B = 50 vs analytic, τ sweep | +0.05 / +0.18 / +0.28 / +0.30 % at τ = 0.2 / 0.1 / 0.05 / 0.02 |
| (iv) the recursion vs `kernel_fit/fit_ou_kernel.py::simulate` | identical choices on 200 trials |

## Data regeneration (bound-hit export)

`kernel_fit/export_evidence.py` now also writes `tcross2_g{gain}` and `tcross15_g{gain}` (the first ms at
which |dv| ≥ 2.0 / 1.5, NaN if never) and takes `--out-dir` / `--tab-out`. The 20 npz files were
regenerated into a temp directory with the `rnn_gain` env and checked key by key against the existing
files before replacing them: `rel`, `labels`, `goals`, `coh_signed`, `choice_g*` and `dvT_g*` are
**bit-identical** for all 20 networks (verification log in this note below), so Track A's inputs are
unchanged.

---

## 1. Route 1 (unbounded, exact likelihood): the transition, per network, with intervals

60 fits (20 networks × 3 gains, 2000 trials each) by maximum likelihood with numerical-Hessian SEs, and
the same 60 by NUTS (4 chains × 1000 warmup / 1000 draws, priors g ~ N(0, 10), v ~ HalfNormal(100),
a_bias ~ N(0, 1)); plus one pooled fit per gain over all 40 000 trials.

### Per network (`output/track_b/analytic_per_network.csv`)

| gain | median g [IQR] (MLE) | g > 0 | 95 % Wald excludes 0 on the predicted side | median g [IQR] (NUTS) | 94 % HDI excludes 0 |
|---|---|---|---|---|---|
| 0.8 | **+4.13** [+3.02, +4.63] | 20/20 | **20/20** (predicted g > 0) | +3.95 [+2.96, +4.50] | **20/20** |
| 1.0 | **+0.32** [−0.34, +0.79] | 12/20 | 15/20 (either side: 10 above, 5 below) | +0.33 [−0.34, +0.81] | 15/20 (10 above, 5 below) |
| 1.2 | **−2.63** [−3.56, −2.04] | 0/20 | **20/20** (predicted g < 0) | −2.63 [−3.56, −2.03] | **20/20** |

Range of the per-network MLE: +2.50 … +6.03 at gain 0.8, −0.90 … +1.48 at 1.0, −5.17 … −0.93 at 1.2.
Median v = 51.3 / 45.6 / 42.9 (noise units).

### Pooled per gain, 40 000 trials (`output/track_b/analytic_pooled.csv`)

| gain | g (MLE) ± SE | 95 % Wald | g (posterior mean) | 94 % HDI | R-hat | v | a_bias |
|---|---|---|---|---|---|---|---|
| 0.8 | **+3.946** ± 0.058 | [+3.831, +4.060] | +3.947 | [+3.834, +4.055] | 1.000 | 45.73 | −1.10 |
| 1.0 | **+0.229** ± 0.044 | [+0.143, +0.315] | +0.231 | [+0.151, +0.316] | 1.001 | 40.95 | −0.16 |
| 1.2 | **−2.730** ± 0.056 | [−2.840, −2.620] | −2.728 | [−2.827, −2.623] | 1.003 | 38.90 | −0.06 |

`kernel_fit/`'s simulation-matching pooled fit gave +3.98 / +0.24 / −2.50 — the likelihood fit reproduces
it to within 0.23 per s at every gain, with intervals.

### Sampler and MLE-vs-posterior checks

Max R-hat over all 63 Bayesian fits and all three parameters: **1.0036** (≤ 1.01); min ESS 1058.
|MLE − posterior mean| / SE ≤ 1 in **55/60** cells. The five exceptions are all at gain 0.8 (seeds 54,
46, 53, 60, 51; worst 2.89 SE for seed 54) and are entirely the a_bias prior: where g is strongly positive
a_bias only enters as a_bias·ρ^T ≈ 0.03·a_bias, the MLE runs away (seed 54: a_bias = −12.5 ± 2.9) and
N(0, 1) shrinks it (to −3.9), moving g from +6.03 to +5.23. Signs of g agree in 60/60 cells.

### Comparison with the landscape and with Fig 1F

| gain | landscape g (= −drift slope; approximate, 3 networks) | route-1 pooled g [95 %] |
|---|---|---|
| 0.8 | +4.3 | +3.946 [+3.831, +4.060] |
| 1.0 | −1.6 | +0.229 [+0.143, +0.315] |
| 1.2 | −6.2 | −2.730 [−2.840, −2.620] |

Same ordering and the same sign change; the fitted magnitudes are smaller at the extremes (the linear OU
under-reports the network's saturating repulsion at gain 1.2, as `kernel_fit/RESULTS.md` noted), and at
gain 1.0 the fit is slightly leaky where the 3-network landscape estimate is slightly unstable.
Zero crossing of the fitted g against gain (piecewise-linear interpolation): **pooled f₀ = 1.0155**
(posterior mean 1.0156), per network median **1.0227** [IQR 0.980, 1.055], mean 1.021 ± 0.011 (n = 20).
Fig 1F's kernel-slope crossing is f = 1.016; the landscape curvature crossing is 0.94 ± 0.07.

### Kernel and psychometric reproduction (`output/track_b/kernel_reproduction*.csv`, `kernel_reproduction.png`)

Choices simulated from the fitted parameters on the real evidence streams with fresh noise; kernels
computed with `kernel_fit/fit_ou_kernel.py::kernel` verbatim (8 bins, L1-normalised).

| gain | pooled kernel slope, network | model (pooled fit) | model (per-network fits) | accuracy network / model |
|---|---|---|---|---|
| 0.8 | +0.0392 | +0.0409 | +0.0392 | 0.866 / 0.865 |
| 1.0 | +0.0021 | +0.0024 | +0.0019 | 0.887 / 0.888 |
| 1.2 | −0.0261 | −0.0308 | −0.0306 | 0.869 / 0.869 |

Per network, model slope vs network slope: 0.896 × network + 0.0041 (r = 0.940) at gain 0.8,
1.061 × (r = 0.980) at 1.0, 1.160 × (r = 0.948) at 1.2 — i.e. the unbounded model slightly over-produces
primacy at gain 1.2, which is where the sticky bound matters (route 2).
Fitted g against the network's own kernel slope: **r = 0.996** over the 60 fits (g = 105.8 × slope − 0.061);
per gain r = 0.981 / 0.995 / 0.970.

### Recovery, route 1 (`output/track_b/recovery_route1.csv`, `recovery.png`)

5 networks (42, 46, 51, 58, 61) × 3 gains × 2 truths × 2 repetitions = 60 refits on the real evidence.

| truth | gain | mean true g | mean recovered g | bias | RMSE | mean SE | sign correct | 95 % covers truth |
|---|---|---|---|---|---|---|---|---|
| fitted | 0.8 | +3.681 | +3.738 | +0.057 | 0.299 | 0.249 | 10/10 | 9/10 |
| fitted | 1.0 | −0.025 | −0.074 | −0.050 | 0.173 | 0.190 | 10/10 | 9/10 |
| fitted | 1.2 | −2.784 | −2.840 | −0.056 | 0.207 | 0.234 | 10/10 | 10/10 |
| landscape | 0.8 | +4.300 | +4.360 | +0.060 | 0.223 | 0.275 | 10/10 | 10/10 |
| landscape | 1.0 | −1.600 | −1.610 | −0.010 | 0.141 | 0.200 | 10/10 | 10/10 |
| landscape | 1.2 | −6.200 | −6.214 | −0.014 | 0.322 | 0.336 | 10/10 | 10/10 |

60/60 signs correct, 56/60 intervals cover the truth (nominal 57), |bias| ≤ 0.06 per s everywhere.
(The gain-1.0 "sign correct" entries are weak evidence: the true values there are near zero.)

### Sensitivity to the number of trials (`output/track_b/sensitivity_ntrials.csv`)

1000 vs 2000 trials per network: median g +4.25 / +0.39 / −2.66 vs +4.13 / +0.32 / −2.63; per-network
r = 0.975 / 0.962 / 0.955; mean |Δg| = 0.19 / 0.21 / 0.29; SEs grow by ×1.43 (≈ √2); 60/60 signs
unchanged; still 20/20 intervals excluding 0 at gains 0.8 and 1.2 (11/20 at gain 1.0, vs 15/20).
