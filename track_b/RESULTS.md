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

---

## 2. Route 2 (sticky bound, Monte-Carlo likelihood): 186 fits

M = 300 realisations per trial, common random numbers, τ = 0.1, Nelder-Mead on (g, v, log B, a_bias) from
the best three of a coarse (B × a_bias) start grid, 95 % intervals by profile likelihood on g.
20 networks × {gains 0.8, 1.2} × {no hit term, `term`, `bern`, `cross`} + 20 × gain 1.0 without the hit
term + 6 pooled fits. Per fit ≈ 10–20 min on 8 Oscar cores, 0.57 s per likelihood evaluation
(2000 trials × 750 steps × 300 realisations, JAX `lax.scan`).

### Per network (`output/track_b/route2_per_network.csv`, `summary_table.md`)

| variant | gain | median g [IQR] | g > 0 | interval excludes 0 as predicted | median B | model vs network hit fraction | fits with g ≥ 8 |
|---|---|---|---|---|---|---|---|
| no hit term | 0.8 | **+4.13** [+3.01, +4.58] | 20/20 | **20/20** | 6.59 | 0.004 vs 0.858 | 0 |
| no hit term | 1.0 | **+0.52** [−0.20, +0.79] | 13/20 | 14/20 | 6.04 | 0.197 vs 0.955 | 0 |
| no hit term | 1.2 | **−2.59** [−3.77, −2.11] | 1/20 | **19/20** | 6.38 | 0.664 vs 0.979 | 0 |
| hit term, deadline commitment (`term`) | 0.8 | **+2.51** [+1.88, +3.06] | 20/20 | **20/20** | 1.44 | 0.448 vs 0.463 | 0 |
| hit term, deadline commitment (`term`) | 1.2 | **−2.53** [−3.23, −2.19] | 0/20 | **20/20** | 1.80 | 0.874 vs 0.880 | 0 |
| hit term, Bernoulli ever-crossed (`bern`) | 0.8 | **+5.90** [+4.56, +6.82] | 20/20 | **20/20** | 0.97 | 0.814 vs 0.858 | 3 |
| hit term, Bernoulli ever-crossed (`bern`) | 1.2 | **−3.51** [−4.32, −2.95] | 1/20 | **19/20** | 1.38 | 0.964 vs 0.979 | 0 |
| hit term, crossing-time (`cross`) | 0.8 | **+8.28** [+7.63, +8.97] | 20/20 | **20/20** | 0.94 | 0.846 vs 0.858 | 13 |
| hit term, crossing-time (`cross`) | 1.2 | **−1.19** [−2.43, +0.32] | 6/20 | 14/20 | 0.83 | 0.983 vs 0.979 | 0 |

Without the hit term, route 2 is route 1: g agrees per network with r = 0.998 / 0.926 / 0.952 and mean
difference +0.012 / +0.077 / +0.009 per s at gains 0.8 / 1.0 / 1.2. The bound is identified only from
below (median B ≈ 6.4, range 1.28–430); the choice data alone do not pin it.

### Pooled, 20 networks × 500 trials = 10 000 trials per gain, M = 100 (`route2_pooled.csv`)

| variant | gain 0.8 | gain 1.0 | gain 1.2 |
|---|---|---|---|
| no hit term | **+3.948** [+3.797, +4.106], B = 9.36 | **+0.497** [+0.228, +0.618], B = 7.18 | **−2.542** [−2.738, −2.418], B = 7.13 |
| crossing-time hit term | **+9.273** [+9.200, +9.497], B = 0.94 | **+5.025** [+4.684, +5.494], B = 0.82 | **−0.628** [−0.670, −0.221], B = 0.80 |

(Route-1 pooled on all 40 000 trials: +3.946 / +0.229 / −2.730.)

### The one degenerate per-network fit

Without the hit term, exactly **one** of the 60 fits lands on the leak-vs-bound degenerate solution:
seed 61, gain 1.2, g = **+1.055** [+0.639, +1.055] with B = 1.50 (nll 329.4 against 331.9 for the
unbounded route-1 solution at g = −0.93). Its profile interval hits the grid edge, so it is flagged.
No fit at any gain reaches the g ≈ +12 / low-B solution that `kernel_fit/` hit in 3 of 60
simulation-matching fits — the full likelihood is more informative than the kernel + psychometric summary
it was matched on. With the `term` hit term seed 61 becomes g = **−0.361** [−0.501, −0.185], i.e. the
degeneracy is resolved and the network joins the other 19.

### Why `bern` and `cross` inflate the leak (`output/track_b/bound_hit_absorption.csv`)

A sticky bound implies "ever hit" ⇔ |a_T| ≥ B. The network's |dv| = 2.0 crossing is **not** absorbing:
of the trials that cross before the deadline, **46.0 % / 20.9 % / 10.4 %** are back below 2.0 at T
(gains 0.8 / 1.0 / 1.2; per-network range 36–58 % at gain 0.8). Forcing a sticky bound to be hit at the
network's *transient* crossing rate therefore drives B down to ≈ 0.9 and the leak up: at gain 0.8 the
median g goes +4.13 → +5.90 (`bern`) → +8.28 (`cross`), i.e. the more of the crossing-time distribution
the bound is made to reproduce, the larger the inflation, and 13/20 `cross` fits land at g ≥ 8 with B < 1.
The cost is visible in the observable the fits were *not* fitted to — the psychophysical kernel:

| gain | network kernel slope (mean over 20) | no hit term | `term` | `cross` |
|---|---|---|---|---|
| 0.8 | +0.0386 | +0.0396 (r = 0.955) | +0.0208 (r = 0.675) | **−0.0097** (r = 0.082) |
| 1.2 | −0.0257 | −0.0303 (r = 0.950) | −0.0300 (r = 0.909) | −0.0539 (r = 0.343) |

The `cross` fits reproduce the choices and the crossing times and get the **sign of the kernel wrong at
gain 0.8** (the strong leak's recency is cancelled by early bound commitment). The `term` variant —
the same observable measured at the deadline, which is what a sticky bound actually implies — keeps the
kernel, identifies B (1.30–2.13 at gain 0.8, 1.43–4.75 at 1.2), matches the commitment fraction to
within 0.015, and gives **20/20 leaky at gain 0.8 and 20/20 unstable at gain 1.2**.

Pooled kernel reproduction, route 2 without the hit term: +0.0404 / +0.0030 / −0.0307 against the
networks' +0.0392 / +0.0021 / −0.0261; accuracy 0.865 / 0.888 / 0.869 against 0.866 / 0.887 / 0.869.

## 3. Recovery, route 2 (`output/track_b/recovery_route2.csv`)

Choices simulated on the real evidence streams with the landscape g (+4.3 / −6.2), the route-1 v and
a_bias and the fitted B, refitted with and without the hit term; 23 fits.

| | n | sign of g correct | truth inside the 95 % interval | bias |
|---|---|---|---|---|
| no hit term | 11 | 11/11 | 9/11 | +0.17 |
| crossing-time hit term | 11 | 11/11 | 8/11 | +0.08 |
| Bernoulli hit term (degenerate case only) | 1 | 1/1 | 1/1 | −0.87 |
| gain 0.8 only | 10 | 10/10 | 5/10 | +0.30 |
| gain 1.2 only | 13 | 13/13 | 13/13 | −0.02 |

**The degenerate case** (g = +12, B = 1.2 simulated on seed 42's gain-1.2 evidence; commitment fraction
0.403 against that network's real 0.9995):

| variant | recovered g [95 %] | B | model vs data hit fraction |
|---|---|---|---|
| no hit term | **+11.87** [+10.50, +12.92] | 1.50 | 0.363 vs 0.403 |
| Bernoulli hit term | **+11.14** [+10.46, +12.42] | 1.31 | 0.398 vs 0.403 |
| crossing-time hit term | **+11.88** [+10.60, +12.78] | 1.23 | 0.401 vs 0.403 |

All three recover it, including the choice-only fit: when the data really come from a low bound, the
2000-trial likelihood identifies it without any hit term. The hit term is therefore not needed to break
the degeneracy in this model — and when its data-side observable is the network's non-absorbing dv
crossing, it actively misleads. Coverage at gain 0.8 is the weak point (5/10): there the bound is barely
hit, B is weakly identified, and the fits are biased +0.2…+0.4 per s.

## 4. What Track B establishes

1. The leaky → near-perfect → unstable transition is recoverable from **behaviour alone** by a
   likelihood fit conditioned on the trial-by-trial evidence, **per network, with intervals**:
   20/20 networks leaky at gain 0.8 and 20/20 unstable at gain 1.2 with intervals excluding zero
   (route 1, both MLE and NUTS; route 2 without the hit term gives 20/20 and 19/20; route 2 with the
   deadline-commitment hit term gives 20/20 and 20/20).
2. The pooled estimates are +3.95 [+3.83, +4.06] / +0.23 [+0.14, +0.32] / −2.73 [−2.84, −2.62] per s,
   reproducing `kernel_fit/`'s +3.98 / +0.24 / −2.50 with uncertainty, and the zero crossing of g against
   gain lands at f = 1.0155 pooled (per-network median 1.0227) against Fig 1F's 1.016.
3. The fitted g tracks each network's own kernel slope with r = 0.996 across the 60 fits, and the fitted
   models reproduce the kernel bin by bin and the psychometric curve.
4. The sticky bound adds nothing at gain 0.8 (it is not hit) and little at 1.2; the leak-vs-bound
   degeneracy appears in exactly 1 of 60 choice-only fits and is resolved by the deadline-commitment
   observable.
5. Not in scope / not done: a two-timescale (fast + slow leak) model; a bounded-choice replication
   (the fixed-bound data come from a different rollout and cannot be matched trial by trial);
   gains 1.0 for the `bern` and `cross` variants (cancelled to save queue time).
