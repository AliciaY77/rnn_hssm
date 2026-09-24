# Track A (issue #3) — corrected HSSM OU fits to the constant-bound RNN behaviour

Branch `track-a-hssm-ou-corrected` (from `ou-round2-base`). Code in `track_a/`, tables and figures in
`output/track_a/` (netcdfs stay on Oscar under the same path). Oscar jobs: 6613431 (timing);
6613460 / 6613461 (first pooled and per-network arrays, cancelled — see §0); 6613593 (12 pooled),
6613594 (120 per-network), 6613838 (appendix), 6613884 / 6613948 (hierarchical smokes), 6614115 (RT-PPC
smoke), 6614157 (hierarchical), 6614167 (recovery), 6614186 (k = 20 / 24 pooled), 6614551 (RT PPC, died in
ssms), 6614751 / 6614775 / 6614808 (ssms crash diagnostics), 6614865 (engine cross-check).

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
The first two array submissions (jobs 6613460, 12 pooled tasks, and 6613461, 120 per-network tasks) all
exited on that guard within 20 s; 6613461 was cancelled after 25 of its tasks were still queued, and both
were resubmitted as 6613593 and 6613594 once t was fixed. No result in this note comes from those runs.

## 1. Headline: every fit converges, every fit says LEAKY, and g is not identified

`output/track_a/headline_pooled.csv`, `native_units.csv`, `all_fits.csv`. **All 18 pooled fits at
k = 8/10/12/16/20/24 converged: R-hat max 1.00, ESS_bulk min 1661, 0 divergences out of 4000 draws each,
23–34 min on 4 cores.** Headline at the plan's k = 10 (g > 0 = leaky):

| gain | n | t fixed | omissions | g [94 % HDI] (stretched) | g_native per s [94 % HDI] | a | a_native | v(0.15) | v_native(0.15) | z | R-hat | ESS_bulk | div | edge mass g | kernel verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 38 640 | 0.35 | 3.40 % | **0.998** [0.994, 1.000] | **+9.98** [+9.94, +10.00] | 1.201 | 0.380 | 1.209 | 3.82 | 0.495 | 1.00 | 2497 | 0/4000 | **1.00** | fails |
| 1.0 | 39 881 | 0.35 | 0.30 % | **0.996** [0.988, 1.000] | **+9.96** [+9.88, +10.00] | 1.023 | 0.324 | 1.292 | 4.09 | 0.495 | 1.00 | 2474 | 0/4000 | **0.99** | fails |
| 1.2 | 39 988 | 0.34 | 0.03 % | **0.921** [0.872, 0.968] | **+9.21** [+8.72, +9.68] | 0.920 | 0.291 | 1.298 | 4.10 | 0.496 | 1.00 | 1815 | 0/4000 | 0.01 | fails |

Theory, same sign convention, predicts g_native = **+4.3 / −1.6 / −6.2**. The fits give **+10.0 / +10.0 /
+9.2**: leaky at every gain, no zero crossing, and the 94 % HDIs exclude the predicted values by 6–16 per s.

**But g is on the box edge at two of the three gains, so those magnitudes are not estimates.** The `g`
posterior is pressed against the LAN's +1 ceiling: 100 % / 99 % of the draws lie within 0.02 of it at gains
0.8 and 1.0. The plan's remedy was to raise k, which lowers g by a factor k. It was run out to k = 24
(jobs 6613593 and 6614186, 18 pooled fits, all R-hat 1.00, ESS_bulk ≥ 1661, 0 divergences):

| k | g_native 0.8 (edge mass) | g_native 1.0 (edge) | g_native 1.2 (edge) | a_native (0.8/1.0/1.2) | v_native(0.15) (0.8/1.0/1.2) |
|---|---|---|---|---|---|
| 8 | +7.98 (**1.00**) | +7.97 (**1.00**) | +7.58 (0.11) | 0.392 / 0.332 / 0.296 | 3.74 / 4.03 / 4.07 |
| 10 | +9.98 (**1.00**) | +9.96 (**0.99**) | +9.21 (0.01) | 0.380 / 0.324 / 0.291 | 3.82 / 4.09 / 4.10 |
| 12 | +11.98 (**1.00**) | +11.94 (**0.98**) | +10.68 (0.00) | 0.370 / 0.316 / 0.286 | 3.91 / 4.14 / 4.13 |
| 16 | +15.98 (**1.00**) | +15.87 (**0.94**) | +12.93 (0.00) | 0.353 / 0.303 / 0.280 | 4.07 / 4.24 / 4.17 |
| 20 | +19.98 (**1.00**) | +19.88 (**0.97**) | +14.02 (0.00) | 0.340 / 0.292 / 0.278 | 4.19 / 4.34 / 4.19 |
| 24 | +24.00 (**1.00**) | +23.95 (**1.00**) | **+14.66** (0.00) | 0.330 / 0.283 / **0.277** | 4.29 / 4.43 / **4.21** |

Time rescaling is exact for the process (RT × k ⇔ g → g/k, a → a√k, v → v/√k), so a well-identified fit
must return the same native values at every k. The sweep separates two situations:

* **Gains 0.8 and 1.0: g is not identified.** The posterior re-pins on the ceiling at every stretch from
  k = 8 to k = 24 (edge mass 0.94–1.00), so g_native = k·1 is a boundary artefact, and a_native drifts with
  it (0.392 → 0.330 and 0.332 → 0.283, i.e. 16 %). All that can be read off is **g_native > 24 per s**.
  The likelihood is monotone in g up to whatever bound it is given.
* **Gain 1.2: g leaves the ceiling from k = 10 and converges.** g_native = +7.58, +9.21, +10.68, +12.93,
  +14.02, **+14.66 [+13.51, +15.82] at k = 24** — successive increments +1.63, +1.47, +2.25, +1.09, +0.64,
  i.e. saturating near +15 per s. Its a_native converges in step (0.296 → 0.278 → 0.277, the last three
  values within 1 %) and v_native(0.15) sits at 4.2. So the fit becomes stretch-consistent exactly when g
  comes off the edge, which confirms that the non-invariance at the other two gains is the ceiling and not
  an error in the rescaling.

Either way the sign is unambiguous and the same at every gain: **leaky, by +14.7 to more than +24 per s,
where theory predicts +4.3 / −1.6 / −6.2.** The one gain whose leak is identified (1.2) is the one theory
says should be the most *unstable*, and it comes out at +14.7 per s.

This is not an artefact of the LAN alone: the exact-simulator random search in `feasibility/RESULTS.md`
§1 reports the same thing — "every good boxed fit has g at or near +1 (leaky) at all gains".

Why the likelihood wants so much leak: the constant-bound RT distributions have a 6 ms minimum and a
750 ms maximum with q90/q50 ≈ 2.8 (table above). With t fixed at the minimum, the only way a one-boundary
OU produces both the very fast leading edge and the long flat tail is strong mean reversion, which creates
a quasi-stationary population that leaks across the bound slowly.

## 2. Per-network fits (120 fits, job 6613594): 20/20 networks leaky at every gain

`output/track_a/per_network.csv`. All 120 converged: R-hat ≤ 1.01 and ESS_bulk ≥ 400 in every fit,
0 divergences in 119 of them and 1/4000 in the last (gain 1.2, k = 16, seed 44); median 1.5–2.2 min each.

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

## 3. The decisive check: method for the posterior-predictive psychophysical kernel

`track_a/ppc_kernel.py`, run locally with `/opt/homebrew/anaconda3/bin/python`; outputs
`output/track_a/kernel_ppc*.csv` and `kernel_ppc*.png`. The fitted parameters are converted to native
time (`g_nat = g·k`, `a_nat = a/√k`, `x0_nat = a_nat(2z − 1)`, `v_nat(coh) = v(coh)·√k`,
`t_nat = (t − 0.3)/k`) and the SAME OU is then driven by the networks' own per-trial evidence streams
(`data/processed/kernel/seed*.npz::rel`, 20 networks × 2000 trials × 750 steps, per-step mean = signed
coherence, per-step sd = 1.000, autocorrelation < 0.003 at every lag out to 20 — i.e. white).
The kernel is `kernel_fit/fit_ou_kernel.py::kernel` verbatim (8 bins, logistic regression on z-scored bin
means, L1-normalised, slope = linear fit over bins).

**Evidence mapping (as the issue requires it be documented).** The fit's drift is linear in signed
coherence, `v_nat(coh) = v0_nat + v1_nat·coh` with `v0_nat = v_Intercept·√k` and
`v1_nat = v_coherence_signed·√k` (≈ 27.5 per unit coherence at k = 10). Because the stream has
`E[e_t] = coh` and `sd(e_t) = 1`, substituting `e_t` for its mean reproduces the fitted mean drift exactly:
`dx = (v0_nat + v1_nat·e_t − g_nat·x) dt + σ dW`. That substitution adds evidence-driven noise of per-step
variance `(v1_nat·dt)² = 0.756·dt` on top of the fitted diffusion, so both treatments are reported:
`σ = 1` (the fitted diffusion; total per-step variance 1.76·dt) and **matched**,
`σ² = 1 − v1_nat²·dt = 0.244` (total per-step variance exactly dt).
*Validation of the mapping*: with the matched σ the evidence-driven simulation reproduces the equivalent
constant-drift OU to within its own Monte-Carlo error (pooled accuracy 0.8085 vs 0.8098 at dt = 0.1 ms),
so the mapping introduces no artefact.

Variants, both run: **(a) as fitted** — sticky bound at ±a_nat, start x0_nat, choice = sign(x_T);
**(b) leak only** — same g_nat and v_nat, no bound (B = ∞), choice = sign(x_T).

## 4. Posterior predictive on RT and choice: the fits describe the marginals well

`track_a/ppc_rt.py`, 200 posterior draws × 100 trials per signed coherence = **20 000 simulated trials per
coherence** per gain, horizon 7.80 s stretched (= 0.3 + k·0.75) applied afterwards, non-crossers counted
as omissions. Tables `output/track_a/ppc_rt_g*_k10_b1.5_pooled.csv` and `ppc_rt_summary.csv`; one figure
per gain, `ppc_rt_g*_k10_b1.5_pooled.png`.

| gain | median abs. quantile error, correct | max | median, error trials | max | omissions obs / sim | max abs. accuracy error |
|---|---|---|---|---|---|---|
| 0.8 | **6.5 ms** | 36.8 ms | 12.3 ms | 48.4 ms | 3.4 % / 1.8 % | 0.021 |
| 1.0 | **1.6 ms** | 14.8 ms | 2.2 ms | 24.7 ms | 0.30 % / 0.20 % | 0.009 |
| 1.2 | **0.8 ms** | 12.4 ms | 1.6 ms | 11.1 ms | 0.03 % / 0.02 % | 0.014 |

Quantiles are 10/30/50/70/90 for correct and error trials at each |coherence|, in native ms. This matches
the plan's expectation: errors ≲ 10 ms at gains 1.0 and 1.2, worse at 0.8 (whose errors are as fast as its
corrects, which a leaky OU cannot reproduce), omissions within 2 points everywhere.
**The model reproduces what it was fitted to.**

**Simulation engine — deviation from the issue, and why.** The issue asks for the ssm-simulators
`ornstein` simulator. With ssms 0.8.3 that simulator aborts the process
(`double free or corruption (out)`, core dumped) when called repeatedly with *varying* parameters: job
6614551 died after 4 s, and a replay that prints every theta (`track_a/diag_ssms.py`) died at call 38 of
2200. Repeating a single theta is fine — 30 repeats each of six (n, max_t) configurations, including ones
where 4–53 % of trials do not terminate, all passed (job 6614808) — so it is cumulative across parameter
changes, the same family of bug as the documented max_t ≈ 0.75 s crash. The PPC therefore uses an
Euler–Maruyama integrator of the same process at the same delta_t = 1 ms (`--engine numpy`).
**Cross-check (job 6614865, 2 draws × 500 trials per coherence, one process per fit so ssms survives):**

| gain | median abs. quantile error, numpy | ssms | error trials, numpy | ssms |
|---|---|---|---|---|
| 0.8 | 7.6 ms | 7.8 ms | 13.5 ms | 18.5 ms |
| 1.0 | 3.9 ms | 2.9 ms | 6.8 ms | 3.8 ms |
| 1.2 | 1.8 ms | 1.6 ms | 3.0 ms | 5.4 ms |

The two engines agree within the Monte-Carlo noise of a 2-draw run.

## 5. The kernel PPC result: the transition is absent in both variants

`output/track_a/kernel_ppc.csv`, `kernel_ppc_draws.csv`, figure `kernel_ppc.png` (four panels, one per
variant, plus slope-vs-gain). Posterior mean and 20 thinned draws per gain; 40 000 real evidence streams
per gain. Network slopes from `output/kernel_fit/ou_kernel_fits.csv` (all 40 000 trials).

| gain | network slope | (a) as fitted, σ = 1 | (a) matched | (b) leak only, σ = 1 | (b) matched | g_native used | a_native used | frac. bounded, (a) |
|---|---|---|---|---|---|---|---|---|
| 0.8 | **+0.0392** | −0.0619 | −0.0408 | +0.0684 | +0.0675 | +9.98 | 0.380 | 0.999 |
| 1.0 | **+0.0021** | −0.0721 | −0.0566 | +0.0678 | +0.0671 | +9.96 | 0.323 | 1.000 |
| 1.2 | **−0.0261** | −0.0756 | −0.0643 | +0.0660 | +0.0655 | +9.20 | 0.291 | 1.000 |

(Draw-to-draw sd of the simulated slope is ≤ 0.002 in every cell, so these differences are not noise.)

**Verdict: neither variant reproduces the network's ordering, and neither reproduces any single gain.**

* **(b) leak only** gives *recency at all three gains* (+0.066 to +0.068) with essentially **no gain
  dependence**: the span across gain is 0.002, against the network's 0.065. That is the direct consequence
  of §1 — the fitted g_native is the same large positive number at every gain, so the leak-only model has
  nothing to vary.
* **(a) as fitted** gives *primacy at all three gains* (−0.041 to −0.064). The fitted bound is
  a_native ≈ 0.29–0.38, which the accumulator reaches within ~100 ms on 99.9–100 % of trials, so the
  model commits before most of the evidence arrives. Its weak ordering (−0.041 → −0.064) tracks
  **a_native (0.380 → 0.291), not g_native (+9.98 → +9.20)**: it is the bound falling with gain, not the
  leak changing sign.
* Accuracy under the evidence-driven simulation is 0.83 / 0.82 / 0.79 (variant a, matched) and
  0.80 / 0.82 / 0.82 (variant b) against the networks' 0.87 / 0.89 / 0.87, i.e. the fitted OU also loses
  4–8 points of accuracy once it has to integrate the real evidence stream rather than a constant drift.

For comparison, the evidence-conditioned fit of `kernel_fit/RESULTS.md`, which is *fitted to* the kernel,
reproduces it bin by bin with g = +3.98 / +0.24 / −2.50 per s and B = 2.6 / 2.1 / 4.9. The HSSM fit's
bound is 7–17× smaller and its leak 2.5–40× larger.

## 6. Recovery at k = 10 (job 6614167): the pipeline *can* see instability — the data never show it

`track_a/recover.py`, `output/track_a/recovery_table.csv`. Six datasets of 20 000 trials each, simulated
from the pooled posterior-mean parameters with g replaced by the theoretical value or by the fitted value,
same 750 ms native horizon (non-crossers dropped), then refitted with the identical pipeline (same fixed t,
same box-uniform priors, 4 × 1000/1000, target_accept 0.95). All six: R-hat 1.00, ESS_bulk ≥ 1525,
**0 divergences**, 15–21 min. g > 0 = leaky.

| g source | gain | true g (native) | recovered g [94 % HDI] (native) | sign | HDI covers truth | omissions | edge mass g |
|---|---|---|---|---|---|---|---|
| theory | 0.8 | +0.430 (+4.30) | **+0.430** [+0.356, +0.506] (+4.30) | ✓ | **yes** | 0.30 % | 0.00 |
| theory | 1.0 | −0.160 (−1.60) | **−0.015** [−0.107, +0.085] (−0.15) | ✓ | no | 0.01 % | 0.00 |
| theory | 1.2 | −0.620 (−6.20) | **−0.243** [−0.369, −0.121] (−2.43) | ✓ | no | 0.00 % | 0.00 |
| fitted | 0.8 | +0.979 (+9.79) | +0.651 [+0.581, +0.723] (+6.51) | ✓ | no | 1.52 % | 0.00 |
| fitted | 1.0 | +0.940 (+9.40) | +0.786 [+0.717, +0.853] (+7.86) | ✓ | no | 0.21 % | 0.00 |
| fitted | 1.2 | +0.908 (+9.08) | +0.763 [+0.692, +0.834] (+7.63) | ✓ | no | 0.02 % | 0.00 |

**Sign of g recovered 6/6. 94 % HDI covers the truth 1/6** — the plan's gate was ≥ 4/6, so that gate is
*not* met and the magnitudes must not be read as estimates. Two things matter for the interpretation:

1. **The pipeline is not biased toward "leaky".** When the generating g is negative the fit returns a
   negative g, and at gain 1.2 the 94 % HDI excludes zero ([−0.369, −0.121]). The same pipeline on the
   real data returns +0.92 to +1.00 with the HDI excluding zero from the other side. So "leaky at every
   gain" is a statement about the data, not about the machinery.
2. **The bias is a shrinkage toward zero, not toward the ceiling** (−0.62 → −0.24, +0.98 → +0.65). Even
   when the truth is g = +0.94, the refit returns +0.79 with edge mass 0.00 — it does *not* re-pin at +1.
   The real data therefore demand *more* leak than any parameter vector inside the box can generate,
   which is exactly the §1 diagnosis.

The one dataset whose HDI covers the truth is the only one whose true g is comfortably interior (+0.43).

## 7. Truncation caveat (required by the issue)

The data are horizon-truncated at 750 ms native (7.8 s stretched, = 0.3 + k·0.75 — the issue says 7.5 s,
which omits the 0.3 s offset) and HSSM has no censored likelihood. The recovery study of issue #1 showed
that fitting horizon-truncated data with an untruncated likelihood **biases g toward "unstable"**
(g < 0). The bias therefore runs *against* the outcome found here, which makes "leaky at every gain"
conservative. The omission fractions are small at two of the three gains (3.40 % / 0.30 % / 0.03 %),
and the gain with the largest truncation (0.8) is the one with the *most* leaky estimate — the opposite
of what the truncation bias would produce.

## 8. Hierarchical model (job 6614157): it converges, and it says the same thing

`track_a/fit_hssm_ou.py --hier`, one job per gain, n = 20 000 stratified by seed, k = 10, 64–87 min each.
Formulas `v ~ 1 + coherence_signed + (1|seed)`, `a ~ 1 + (1|seed)`, `g ~ 1 + (1|seed)`.

**Deviation, recorded here.** HSSM cannot put a regression on a doubly bounded parameter with an identity
link, so the hierarchical fits use `link_settings="log_logit"`, which places a **generalized logit** on
v, a and g. The sampled `*_Intercept` are therefore on the logit scale and every reported quantity is the
inverse link, `lo + (hi − lo)/(1 + exp(−η))`; the first smoke run, before this was noticed, printed
g_native = +7.56 per s and a_native = −0.297 (impossible). HSSM's own default Normal(0, 0.25) on every
coefficient also crushed the coherence slope (v(0.15) = 0.62 against 1.26 pooled), so the priors are
Normal(0, 1.5) on the three intercepts and Normal(0, 30) on the coherence slope, both on the logit scale.
Under this link v(coh) is not linear in coherence, so the hierarchical fits are reported but the PPCs of
record use the pooled fits.

| gain | population g [94 % HDI] | **g_native per s** [94 % HDI] | a_native | v(0.15) | sd(g) by seed (logit) | sd(a) | sd(v) | R-hat max | ESS_bulk min | div | edge mass g |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 0.991 [0.981, 0.999] | **+9.91** [+9.82, +9.99] | 0.381 | 1.181 | 0.255 | 0.077 | 0.179 | 1.01 | **222** | 0/4000 | 0.963 |
| 1.0 | 0.980 [0.959, 0.998] | **+9.80** [+9.59, +9.98] | 0.324 | 1.227 | 0.254 | 0.064 | 0.147 | 1.01 | 715 | 0/4000 | 0.562 |
| 1.2 | 0.902 [0.837, 0.960] | **+9.02** [+8.37, +9.60] | 0.292 | 1.220 | 0.241 | 0.063 | 0.134 | 1.01 | 855 | 0/4000 | 0.004 |

All three sampled: R-hat ≤ 1.01 and 0 divergences everywhere. **Gain 0.8 misses the ESS_bulk ≥ 400 gate**
(222), and the shortfall is on `v_Intercept` / the quantities derived from it (ESS 222–254); `g_Intercept`
itself has ESS 4560 and R-hat 1.00 at that gain, so the g conclusion is not affected. The population means
reproduce the pooled fits to within 0.7–2 % (g_native +9.91 / +9.80 / +9.02 against +9.98 / +9.96 / +9.21),
and g is again on the ceiling at gains 0.8 and 1.0 (edge mass 0.96 and 0.56).
The 60 per-network fits of §2 remain the primary evidence about between-network spread.

## 9. Appendix: the original Weibull-collapsing-bound data (job 6613838)

`track_a/make_weibull_appendix.py` reformats the stored Weibull outcome from the SIM-A per-trial cache
(`bounded_rt`, `choice`, `crossed`) into the same HSSM-ready columns; gain 1.0, seed 42, 2000 trials,
0 % omissions, min RT 35 ms, median 285 ms, max 588 ms. Same corrected pipeline (fixed t, no lapse,
signed-coherence drift, box-uniform priors). g > 0 = leaky.

| data | k | t fixed | g mean [94 % HDI] | g_native per s | a | a_native | v(0.15) | R-hat | ESS_bulk | div | edge mass g |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Weibull, gain 1.0, s42 | 4 | 0.436 | **−0.106** [−0.367, +0.160] | −0.43 | 1.206 | 0.603 | 1.782 | 1.00 | 1742 | 0/4000 | 0.00 |
| Weibull, gain 1.0, s42 | 10 | 0.640 | **−0.219** [−0.278, −0.154] | −2.19 | 2.193 | 0.694 | 1.173 | 1.00 | 1778 | 0/4000 | 0.00 |

The collapse confound reproduces: on the *same network at the same gain*, the Weibull-bound readout
gives **g < 0 (unstable)** with the 94 % HDI excluding zero at k = 10, while the constant-bound readout
gives g at the leaky ceiling. This is the sign that the three original issue-#1 fits reported, and it is
an artefact of the collapsing bound (which manufactures a long, flat RT tail), not a property of the
network. It also shows the corrected pipeline is not hard-wired to return g > 0.

## 10. Conclusions

1. **The corrected pipeline works.** 18 pooled fits, 120 per-network fits, 3 hierarchical fits, 6 recovery
   fits, 2 appendix fits: R-hat ≤ 1.01 and ESS_bulk ≥ 400 in all but one (the hierarchical gain-0.8 fit,
   ESS_bulk 222 on `v_Intercept`), and **1 divergence in 150 fits × 4000 = 600 000 draws** — that single one is
   in the gain-1.2, k = 16, seed-44 per-network fit (1/4000 = 0.025 %, far under the 1 % gate); every other
   fit has exactly 0. The three failures of the
   original issue-#1 fits (≈100 % divergences, t absorbing the 0.3 s offset, drift at the +2 edge) are all
   gone: t is fixed, the stretch puts a and v in the interior, and the lapse mixture is off.
2. **The fits describe RT and choice well.** Median absolute RT-quantile error 6.5 / 1.6 / 0.8 ms at gains
   0.8 / 1.0 / 1.2, accuracy within 0.021 everywhere, omissions within 1.6 points.
3. **They say "leaky" at every gain and show no transition.** g_native = +9.98 / +9.96 / +9.21 per s
   pooled at k = 10; 120/120 per-network fits give g > 0, with the 94 % HDI excluding zero in 118/120.
   Theory predicts +4.3 / −1.6 / −6.2. **This is the outcome the issue expected.**
4. **The magnitude is identified at only one gain.** g sits on the LAN's +1 ceiling at gains 0.8 and 1.0
   at every stretch from k = 8 to k = 24 (edge mass 0.94–1.00), so g_native = k·1 there is a boundary
   artefact and all that can be said is g_native > 24 per s. Gain 1.2 comes off the edge from k = 10 and
   converges to **g_native = +14.66 [+13.51, +15.82] per s at k = 24**, with a_native stable to 1 % over
   the last three stretches. Raising k — the plan's remedy — fixes gain 1.2 and not the other two.
5. **The pipeline could have seen the transition.** With g set to theory the same pipeline returns g < 0
   at gains 1.0 and 1.2, with the HDI excluding zero at 1.2 (sign recovered 6/6). The bias is shrinkage
   toward zero, and the truncation bias documented in #1 also pushes toward "unstable", so both known
   biases work *against* the leaky result actually obtained.
6. **The decisive check fails as predicted.** Driven by the networks' own evidence streams, the fitted OU
   gives recency at all three gains without a bound (slopes +0.068 / +0.068 / +0.066 against the networks'
   +0.039 / +0.002 / −0.026) and primacy at all three gains with the fitted bound (−0.041 / −0.057 /
   −0.064). The ordering the networks show is absent in both, and what little ordering variant (a) has
   comes from a_native, not from g.
7. **The hierarchical model converges and agrees**: population g_native = +9.91 / +9.80 / +9.02 per s at
   k = 10, R-hat ≤ 1.01, 0 divergences, within 2 % of the pooled fits (gain 0.8 misses the ESS gate at 222,
   on `v_Intercept`, not on g).
8. **The Weibull collapse is a confound, and it flips the sign.** The same network at the same gain gives
   g_native = −2.19 [−2.78, −1.54] on the collapsing-bound readout and g on the leaky ceiling on the
   constant-bound readout.

9. **Bottom line for the report: standard RT-and-choice fitting of a constant-bound OU to these networks
returns a converged, well-fitting model whose leak parameter is positive at every gain, is not identified
in magnitude, and reproduces neither the psychophysical kernel nor its ordering across gain. Track B's
   positive result is therefore not something Track A could have found.**

## 11. Bound sweep (2026-09-24, job 6668713; RT PPC jobs 6669698–6669700)

Question from the advisor discussion: at bound 1.5 the RTs are crossings by the fast readout fluctuations; at a higher
threshold the slow choice mode governs the crossing, so the network's dynamics should matter more. Same corrected pipeline
as §0 (k = 10, t fixed at 0.3 + k·(min RT − 1 ms) of each dataset, no lapse, v ~ 1 + signed coherence, a, z, g shared,
box-uniform priors, 4 × 1000/1000), pooled over 20 networks, constant bounds 2.0 / 2.5 / 2.94 on dv. **g > 0 = leaky.**
All 9 fits: R-hat 1.00, ESS_bulk ≥ 1525, 1 divergence in 36 000 draws (gain 0.8, bound 2.94). Bound-1.5 rows are from §1/§4.

| bound | gain | n fitted | g native [94 % HDI] (per s) | edge mass g | omissions network / model | median abs. RT-quantile error, correct (ms) | kernel slope, fitted OU with bound (network) |
|---|---|---|---|---|---|---|---|
| 1.5 | 0.8 | 38 640 | +10.0 [+9.9, +10.0] | 1.00 | 3.4 / 1.8 % | 6.5 | −0.041 (+0.039) |
| 1.5 | 1.0 | 39 881 | +10.0 [+9.9, +10.0] | 0.99 | 0.3 / 0.2 % | 1.6 | −0.057 (+0.002) |
| 1.5 | 1.2 | 39 988 | +9.2 [+8.7, +9.7] | 0.01 | 0.0 / 0.0 % | 0.8 | −0.064 (−0.026) |
| 2.0 | 0.8 | 34 639 | +10.0 [+9.9, +10.0] | 1.00 | 13.4 / 6.1 % | 20.9 | −0.024 (+0.039) |
| 2.0 | 1.0 | 39 321 | +9.9 [+9.8, +10.0] | 0.98 | 1.7 / 1.2 % | 3.7 | −0.045 (+0.002) |
| 2.0 | 1.2 | 39 914 | +4.9 [+4.3, +5.4] | 0.00 | 0.2 / 0.2 % | 2.4 | −0.057 (−0.026) |
| 2.5 | 0.8 | 27 722 | +6.7 [+6.2, +7.2] | 0.00 | 30.7 / 11.3 % | 33.7 | −0.011 (+0.039) |
| 2.5 | 1.0 | 35 978 | +5.8 [+5.3, +6.2] | 0.00 | 10.1 / 2.9 % | 10.1 | −0.035 (+0.002) |
| 2.5 | 1.2 | 37 772 | +4.7 [+4.2, +5.2] | 0.00 | 5.6 / 0.6 % | 3.5 | −0.052 (−0.026) |
| 2.94 | 0.8 | 13 240 | +9.9 [+9.6, +10.0] | 0.86 | 66.9 / 20.7 % | 37.8 | +0.006 (+0.039) |
| 2.94 | 1.0 | 27 855 | +2.3 [+1.9, +2.6] | 0.00 | 30.4 / 6.3 % | 24.1 | −0.025 (+0.002) |
| 2.94 | 1.2 | 34 276 | +0.8 [+0.5, +1.1] | 0.00 | 14.3 / 1.8 % | 4.3 | −0.043 (−0.026) |

Tables: `output/track_a/g*_k10_b{2.0,2.5,2.94}_pooled_{summary,meta,draws}`, `ppc_rt_summary_b*.csv`, `ppc_rt_*_b*.png`,
`kernel_ppc_b*.csv/png`; figure (rows = bound, correct up / error down) in `round2-report:output/report/fig_bound_sweep.png`.
Kernel slopes: variant (a) matched, posterior mean; leak-only variant (b) at 2.94: +0.067 / +0.027 / +0.011.

Reading, in order of confidence:

1. **g is positive in all 12 fits, every 94 % HDI excludes zero.** No threshold makes the RT-and-choice OU report instability.
2. **g falls as the threshold rises, and at 2.5 and 2.94 it is ordered by gain as the theory predicts** (2.94: +9.9 / +2.3 /
   +0.8). At 2.94 gain 1.2 is close to a perfect integrator (+0.8 [+0.5, +1.1]), not unstable (landscape: −6.2).
3. **Where the ordering appears, the fit fails for the leaky network.** Gain 1.2 fits at every bound (quantile error ≤ 4 ms).
   Gain 0.8 degrades steadily (6.5 → 20.9 → 33.7 → 37.8 ms) and the model predicts a fraction of the omissions (20.7 vs 66.9 %
   at 2.94): the network's RTs at high threshold have a long flat tail that a one-timescale linear OU cannot produce. At 2.94
   gain 0.8 is back on the ceiling.
4. **The kernel ordering appears too, shifted toward primacy.** With its bound the fitted model goes +0.006 / −0.025 / −0.043
   at 2.94 against the networks' +0.039 / +0.002 / −0.026: right ordering, span 0.049 vs 0.065, offset −0.02 to −0.03, so it
   still gets the sign wrong at gains 0.8 and 1.0.
5. **Truncation.** The non-crossers are dropped and HSSM has no censored likelihood; the model's own omission rate is far
   below the data's at high bounds. The recovery study (#1) showed this biases g toward unstable; g stays positive regardless.

So the advisor's reading holds in part: raising the threshold lets the slow dynamics show through in the *ordering* of g and of
the kernel across gain, but a linear one-timescale OU with a constant bound then misfits the leaky network's RTs and omissions
and still never reports instability. The model class, not the likelihood or the sampler, is what limits this route.

**Addendum (intervals for the round-3 figures).** `track_a/ppc_replicates.py` (job 6669949): 40 posterior draws per fit, one
replicate dataset of the data's size and coherence mix per draw → 95 % posterior-predictive intervals for omission rate,
median RT and accuracy (`output/track_a/ppc_replicates_g*_k10_b*.csv`). With 40 000 trials the intervals are narrow, so the
network lies outside them for some statistics even at 1.5 (gain 0.8 omissions 3.4 % vs [1.6, 1.9] %; gain 1.0 median RT 99 vs
[95, 97] ms); at 2.94 the misses are large (gain 0.8: median RT 321 vs [275, 283] ms, omissions 66.9 % vs [19.9, 21.3] %).
Kernel PPC with 40 draws (`kernel_ppc*_b{1.5,2.5,2.94}_d40.csv`): 95 % posterior interval of the model's kernel slope is
±0.001 or less; the network's own slope has a 95 % bootstrap interval of about ±0.0013 (round2-report
`output/report/network_kernel_boot.csv`), both far smaller than the model–network gaps of 0.02–0.08.
