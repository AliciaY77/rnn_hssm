# Track A (issue #3) — corrected HSSM OU fits to the constant-bound RNN behaviour

Branch `track-a-hssm-ou-corrected` (from `ou-round2-base`). Code in `track_a/`, tables and figures in
`output/track_a/` (netcdfs stay on Oscar under the same path). Oscar jobs: 6613431 (timing), 6613593
(12 pooled), 6613594 (120 per-network), 6613838 (appendix), 6613948 (hierarchical smoke), 6614115
(RT-PPC smoke), 6614157 (hierarchical), 6614167 (recovery), 6614186 (k = 20 / 24 pooled).

**Sign convention, stated once and repeated in every table: ssm-simulators and HSSM implement
`dx = (v − g·x) dt + dW`, so `g > 0` is LEAKY (recency) and `g < 0` is UNSTABLE / attractive (primacy).**
The landscape's drift slope uses the opposite sign (negative = restoring = leaky), so the theoretical
prediction in *this* convention is g_native ≈ **+4.3 / −1.6 / −6.2 per s** at gains 0.8 / 1.0 / 1.2.
Boundaries are at −a and +a (a = half separation); z is the relative start point (0.5 = unbiased).

## 0. What was fitted

`data/processed/fixed_bound/hssm_ready_nxx1_fixed_b1.5_g{0.8,1.0,1.2}.csv`, 20 networks pooled:
38 640 / 39 881 / 39 988 trials, **omissions (dropped, never forced) 3.40 % / 0.30 % / 0.03 %**,
11 signed coherences (−0.15 … +0.15), P(choice "+") 0.488 / 0.490 / 0.491, accuracy 0.850 / 0.816 / 0.791.

| gain | min RT | q10 | median | q90 | q99 | max | q90/q50 | RT > 300 ms |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 6 ms | 46 | 145 | 419 | 685 | 750 | 2.89 | 20.4 % |
| 1.0 | 6 ms | 34 | 99 | 274 | 550 | 750 | 2.77 | 8.0 % |
| 1.2 | 5 ms | 28 | 76 | 205 | 413 | 744 | 2.70 | 3.3 % |

Model (`track_a/fit_hssm_ou.py`): HSSM `ornstein`, `loglik_kind="approx_differentiable"` (pretrained LAN),
sampler `nuts_numpyro` (hssm 0.2.4), 4 chains × 1000 tune / 1000 draws, `target_accept = 0.95`,
`idata_kwargs=dict(log_likelihood=False)`. `response = response_choice`, `v ~ 1 + coherence_signed`
(signed, raw units, so `v_coherence_signed` is per unit coherence). **No lapse mixture**
(`p_outlier=None, lapse=None`). **t is never fitted**: passed as a float.

Priors (explicit, not HSSM's defaults): Uniform over the LAN box —
`v_Intercept ~ U(−2, 2)`, `v_coherence_signed ~ U(−13.333, 13.333)` (= ±2/0.15),
`a ~ U(0.3, 3)`, `z ~ U(0.1, 0.9)`, `g ~ U(−1, 1)`.

**Deviation from the issue, recorded here.** The issue prescribes `t = 0.3 + k · min(rt_native)`, which
gives the fastest trial a decision time of exactly zero; every one of the first 132 array tasks refused to
fit on that guard. An RT recorded as 6 ms means the crossing fell in (5, 6] ms, so t is fixed one **native**
millisecond (the data's own time step) lower: `t = 0.3 + k · (min(rt_native) − 0.001)`, i.e.
**t = 0.35 / 0.35 / 0.34 s at k = 10** (t_native = 5 / 5 / 4 ms). No RT is at or below t in any fit.

## 1. Headline: every fit converges, and every fit puts g on the LAN's leaky ceiling

`output/track_a/headline_pooled.csv`, `native_units.csv`, `all_fits.csv`.

*(filled in below)*

## 2. Per-network fits (120 fits, job 6613594): 20/20 networks leaky at every gain

`output/track_a/per_network.csv`. All 120 converged (R-hat ≤ 1.01, ESS_bulk ≥ 400, 0 divergences);
median 1.5–2.2 min each.

| gain | k | networks with g > 0 | 94 % HDI excludes 0 (positive) | median g_native (IQR) | median a_native | median v_native(0.15) | median edge mass for g |
|---|---|---|---|---|---|---|---|
| 0.8 | 10 | 20/20 | 20/20 | **+9.81** (9.76–9.84) | 0.376 | 3.78 | 0.65 |
| 1.0 | 10 | 20/20 | 20/20 | **+9.66** (9.56–9.76) | 0.324 | 4.05 | 0.42 |
| 1.2 | 10 | 20/20 | 19/20 | **+9.44** (9.31–9.65) | 0.288 | 4.17 | 0.26 |
| 0.8 | 16 | 20/20 | 20/20 | **+15.78** (15.73–15.80) | 0.350 | 4.08 | 0.75 |
| 1.0 | 16 | 20/20 | 20/20 | **+15.47** (15.25–15.64) | 0.304 | 4.20 | 0.44 |
| 1.2 | 16 | 20/20 | 19/20 | **+14.98** (14.51–15.45) | 0.273 | 4.26 | 0.22 |

**0/120 fits put g below zero.** The theoretical prediction is g < 0 at gains 1.0 and 1.2.
The per-network ordering (0.8 > 1.0 > 1.2) does run in the direction theory predicts, but every value is
pinned against the leaky ceiling, none crosses zero, and the span (9.81 → 9.44 at k = 10, i.e. 0.37 per s)
is 3 % of the span theory predicts (+4.3 → −6.2, i.e. 10.5 per s). The ordering also tracks a_native
(0.376 → 0.288), i.e. the bound, not the leak.
