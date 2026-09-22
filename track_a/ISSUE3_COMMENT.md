## Track A done — branch `track-a-hssm-ou-corrected`, full write-up in `track_a/RESULTS.md`

**Sign convention: HSSM/ssms implement `dx = (v − g·x)dt + dW`, so g > 0 = LEAKY (recency), g < 0 = UNSTABLE (primacy).** Theory in this convention predicts g_native = **+4.3 / −1.6 / −6.2 per s** at gains 0.8 / 1.0 / 1.2.

### Headline — pooled fits, b = 1.5, 20 networks, k = 10, t fixed, no lapse

| gain | n | omissions | g [94 % HDI] | **g_native /s** [94 % HDI] | a_native | v_native(0.15) | R-hat | ESS_bulk | div | edge mass g | kernel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 38 640 | 3.40 % | 0.998 [0.994, 1.000] | **+9.98** [+9.94, +10.00] | 0.380 | 3.82 | 1.00 | 2497 | 0/4000 | **1.00** | fails |
| 1.0 | 39 881 | 0.30 % | 0.996 [0.988, 1.000] | **+9.96** [+9.88, +10.00] | 0.324 | 4.09 | 1.00 | 2474 | 0/4000 | **0.99** | fails |
| 1.2 | 39 988 | 0.03 % | 0.921 [0.872, 0.968] | **+9.21** [+8.72, +9.68] | 0.291 | 4.10 | 1.00 | 1815 | 0/4000 | 0.01 | fails |

**The issue's expected outcome is confirmed: converged posteriors, g > 0 at every gain, no ordering that tracks the leaky → perfect → unstable prediction.** Every fit in the whole study converged: R-hat ≤ 1.01 everywhere, ESS_bulk ≥ 400 in all but one (the hierarchical gain-0.8 fit, 222 on `v_Intercept`), and **1 divergence in 150 fits × 4000 draws** (gain 1.2, k = 16, seed 44; 0 in every other fit).

### g is on the +1 ceiling at two of the three gains — raising k fixes only gain 1.2

The issue's remedy (raise k; g → g/k) was run to k = 24 (18 pooled fits, all R-hat 1.00, ESS_bulk ≥ 1661, 0 divergences):

| k | g_native 0.8 (edge mass) | 1.0 (edge) | 1.2 (edge) | a_native (0.8/1.0/1.2) | v_native(0.15) |
|---|---|---|---|---|---|
| 8 | +7.98 (**1.00**) | +7.97 (**1.00**) | +7.58 (0.11) | 0.392/0.332/0.296 | 3.74/4.03/4.07 |
| 10 | +9.98 (**1.00**) | +9.96 (**0.99**) | +9.21 (0.01) | 0.380/0.324/0.291 | 3.82/4.09/4.10 |
| 12 | +11.98 (**1.00**) | +11.94 (**0.98**) | +10.68 (0.00) | 0.370/0.316/0.286 | 3.91/4.14/4.13 |
| 16 | +15.98 (**1.00**) | +15.87 (**0.94**) | +12.93 (0.00) | 0.353/0.303/0.280 | 4.07/4.24/4.17 |
| 20 | +19.98 (**1.00**) | +19.88 (**0.97**) | +14.02 (0.00) | 0.340/0.292/0.278 | 4.19/4.34/4.19 |
| 24 | +24.00 (**1.00**) | +23.95 (**1.00**) | **+14.66** (0.00) | 0.330/0.283/**0.277** | 4.29/4.43/**4.21** |

Rescaling is exact for the process, so a well-identified fit must give the same native values at every k.

* **Gains 0.8 and 1.0: not identified.** The posterior re-pins on the ceiling at every stretch, so g_native = k·1 is a boundary artefact and a_native drifts 16 % with it. All that can be read off is **g_native > 24 per s**.
* **Gain 1.2: identified and converging.** g_native +7.58 → +9.21 → +10.68 → +12.93 → +14.02 → **+14.66 [+13.51, +15.82] at k = 24** (increments +1.63, +1.47, +2.25, +1.09, +0.64), with a_native stable to 1 % over the last three stretches. The fit becomes stretch-consistent exactly when g leaves the edge — which confirms the non-invariance elsewhere is the ceiling, not an error in the rescaling.

**The gain theory says should be the most *unstable* (1.2) is the one whose leak is identified, at +14.7 per s.** `feasibility/RESULTS.md` §1 found the same with the exact simulator ("every good boxed fit has g at or near +1").

### Per-network (120 fits at k = 10 and k = 16, all 20 networks)

| gain | k | g > 0 | HDI excludes 0 | median g_native (IQR) | median a_native |
|---|---|---|---|---|---|
| 0.8 | 10 | 20/20 | 20/20 | +9.81 (9.76–9.84) | 0.376 |
| 1.0 | 10 | 20/20 | 20/20 | +9.66 (9.56–9.76) | 0.324 |
| 1.2 | 10 | 20/20 | 19/20 | +9.44 (9.31–9.65) | 0.288 |
| 0.8 | 16 | 20/20 | 20/20 | +15.78 | 0.350 |
| 1.0 | 16 | 20/20 | 20/20 | +15.47 | 0.304 |
| 1.2 | 16 | 20/20 | 19/20 | +14.98 | 0.273 |

**0/120 fits put g below zero**, where theory wants g < 0 at gains 1.0 and 1.2.

### RT/choice PPC: the model reproduces what it was fitted to

200 draws × 100 trials per coherence (20 000 per coherence), 7.80 s stretched horizon:

| gain | median abs. RT-quantile error (correct) | max | error trials | omissions obs/sim | max abs. accuracy error |
|---|---|---|---|---|---|
| 0.8 | 6.5 ms | 36.8 | 12.3 ms | 3.4 % / 1.8 % | 0.021 |
| 1.0 | 1.6 ms | 14.8 | 2.2 ms | 0.30 % / 0.20 % | 0.009 |
| 1.2 | 0.8 ms | 12.4 | 1.6 ms | 0.03 % / 0.02 % | 0.014 |

### Kernel PPC — the decisive check — fails in both variants

Fitted parameters converted to native units and driven by the networks' own evidence streams (40 000 trials/gain, `kernel_fit/fit_ou_kernel.py::kernel` verbatim):

| gain | network | (a) as fitted (σ=1 / matched) | (b) leak only (σ=1 / matched) |
|---|---|---|---|
| 0.8 | **+0.0392** | −0.0619 / −0.0408 | +0.0684 / +0.0675 |
| 1.0 | **+0.0021** | −0.0721 / −0.0566 | +0.0678 / +0.0671 |
| 1.2 | **−0.0261** | −0.0756 / −0.0643 | +0.0660 / +0.0655 |

**(b)** gives recency at all three gains with a span across gain of 0.002 against the network's 0.065 — the fitted g is the same large positive number everywhere, so there is nothing to vary. **(a)** gives primacy at all three gains because a_native ≈ 0.29–0.38 is reached within ~100 ms on 99.9–100 % of trials; its weak ordering tracks a_native (0.380 → 0.291), not g_native (+9.98 → +9.20). **The network's recency → flat → primacy ordering is absent from both.** Figure: `output/track_a/kernel_ppc.png`.

### Recovery at k = 10: the pipeline *could* have seen the transition

6 datasets × 20 000 trials from the pooled posterior means, g replaced by theory / by the fitted value, same horizon, same pipeline. All 0 divergences.

| g source | gain | true g (native) | recovered [94 % HDI] (native) | sign | HDI covers |
|---|---|---|---|---|---|
| theory | 0.8 | +0.430 (+4.30) | +0.430 [+0.356, +0.506] (+4.30) | ✓ | **yes** |
| theory | 1.0 | −0.160 (−1.60) | −0.015 [−0.107, +0.085] (−0.15) | ✓ | no |
| theory | 1.2 | −0.620 (−6.20) | **−0.243 [−0.369, −0.121]** (−2.43) | ✓ | no |
| fitted | 0.8 | +0.979 | +0.651 [+0.581, +0.723] | ✓ | no |
| fitted | 1.0 | +0.940 | +0.786 [+0.717, +0.853] | ✓ | no |
| fitted | 1.2 | +0.908 | +0.763 [+0.692, +0.834] | ✓ | no |

**Sign recovered 6/6; 94 % HDI covers the truth 1/6** (the plan's gate was ≥ 4/6 — not met, so treat magnitudes as uninformative). When g < 0 is in the data the fit returns g < 0 with the HDI excluding zero at gain 1.2. The bias is shrinkage *toward zero*, not toward the ceiling, and the #1 truncation bias also pushes toward "unstable" — **both known biases run against the leaky result actually obtained**, which makes it conservative.

### Appendix: the Weibull collapse flips the sign

Same network (seed 42), same gain 1.0, original collapsing-bound readout, same corrected pipeline: g_native = **−2.19 [−2.78, −1.54]** at k = 10 (interior, 0 divergences) versus g on the leaky ceiling for the constant-bound readout. The sign the three original #1 fits reported is a property of the collapsing bound, not of the network — and it shows the pipeline is not hard-wired to return g > 0.

### Native-unit translation (`output/track_a/native_units.csv`)

`g_native = g·k`, `a_native = a/√k`, `v_native(coh) = v(coh)·√k`, `t_native = (t − 0.3)/k`. At k = 10:
t_native = 5 / 5 / 4 ms; a_native = 0.380 / 0.324 / 0.291; v_native(0.15) = 3.82 / 4.09 / 4.10;
v_native per unit signed coherence = 26.0 / 27.8 / 27.8; g_native = +9.98 / +9.96 / +9.21 per s.

### Two deviations from the issue, both forced and both documented in RESULTS.md

1. **t** is fixed at `0.3 + k·(min rt_native − 1 ms)`, not `0.3 + k·min rt_native`: the latter gives the fastest trial exactly zero decision time and killed all 132 tasks of the first array. t = 0.35 / 0.35 / 0.34 s at k = 10 (t_native 5 / 5 / 4 ms).
2. **k was swept to 24, not just 8/10/12** (the issue allows 12): needed to show the ceiling does not release at gains 0.8 and 1.0 and that gain 1.2 converges.
3. **The RT PPC uses an Euler–Maruyama engine, not ssm-simulators.** ssms 0.8.3 aborts the process (`double free or corruption`) after ~38 calls with varying parameters (jobs 6614551, 6614775); 30 repeats each of six fixed-parameter configurations all pass (job 6614808). A 2-draw ssms cross-check run one process per fit agrees with the numpy engine (median quantile error 7.8 vs 7.6 / 2.9 vs 3.9 / 1.6 vs 1.8 ms; job 6614865).

### Hierarchical model (`g ~ 1 + (1|seed)`, `a ~ 1 + (1|seed)`, `v ~ 1 + coherence_signed + (1|seed)`, n = 20 000, k = 10)

All three converged: R-hat ≤ 1.01, 0 divergences, 64–87 min. Population g_native = **+9.91 / +9.80 / +9.02 per s**, within 2 % of the pooled fits. Between-network sd of g on the logit scale 0.255 / 0.254 / 0.241. Gain 0.8 misses the ESS_bulk ≥ 400 gate (222) — the shortfall is on `v_Intercept`; `g_Intercept` there has ESS 4560 and R-hat 1.00. Note HSSM forces a **generalized logit** link on v, a and g for hierarchical regressions (`link_settings="log_logit"`), so all reported values are the inverse link; the PPCs of record use the pooled fits, where v is linear in coherence.

**Bottom line: standard RT-and-choice fitting of a constant-bound OU to these networks gives a converged, well-fitting model whose leak is positive at every gain, unidentified in magnitude, and reproduces neither the psychophysical kernel nor its ordering across gain — while the same pipeline does recover g < 0 when g < 0 is in the data. Track B's positive result is not something Track A could have found.**
