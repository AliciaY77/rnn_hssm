## Track B done: the transition is recoverable per network, with intervals

Branch `track-b-evidence-conditioned-fit`; full write-up in `track_b/RESULTS.md`, tables and figures in
`output/track_b/`. Sign convention: **g > 0 leaky (recency), g < 0 unstable (primacy)**; the landscape
drift slope is the negative of this g.

**Headline — per-network g (20 networks, 2000 deadline choices each, conditioned on each trial's own
evidence stream).**

| route / variant | gain | median g [IQR] | g > 0 | interval excludes 0 on the predicted side |
|---|---|---|---|---|
| route 1 MLE (95 % Wald) | 0.8 | **+4.13** [+3.02, +4.63] | 20/20 | **20/20** |
| route 1 MLE | 1.0 | +0.32 [−0.34, +0.79] | 12/20 | 15/20 (10 above 0, 5 below) |
| route 1 MLE | 1.2 | **−2.63** [−3.56, −2.03] | 0/20 | **20/20** |
| route 1 NUTS (94 % HDI) | 0.8 | +3.95 [+2.95, +4.50] | 20/20 | **20/20** |
| route 1 NUTS | 1.0 | +0.33 [−0.34, +0.81] | 12/20 | 15/20 |
| route 1 NUTS | 1.2 | −2.63 [−3.56, −2.03] | 0/20 | **20/20** |
| route 2, no hit term (95 % profile) | 0.8 | +4.13 [+3.01, +4.58] | 20/20 | **20/20** |
| route 2, no hit term | 1.0 | +0.52 [−0.20, +0.79] | 13/20 | 14/20 |
| route 2, no hit term | 1.2 | −2.59 [−3.77, −2.11] | 1/20 | **19/20** |
| route 2 + hit term, deadline commitment | 0.8 | +2.51 [+1.88, +3.06] | 20/20 | **20/20** |
| route 2 + hit term, deadline commitment | 1.2 | −2.53 [−3.23, −2.19] | 0/20 | **20/20** |
| route 2 + hit term, Bernoulli ever-crossed | 0.8 | +5.90 [+4.56, +6.82] | 20/20 | 20/20 |
| route 2 + hit term, Bernoulli ever-crossed | 1.2 | −3.51 [−4.32, −2.95] | 1/20 | 19/20 |
| route 2 + hit term, crossing-time | 0.8 | +8.28 [+7.63, +8.97] | 20/20 | 20/20 |
| route 2 + hit term, crossing-time | 1.2 | −1.19 [−2.43, +0.32] | 6/20 | 14/20 |

**Pooled per gain** (route 1, 40 000 trials): g = **+3.946** [+3.831, +4.060] / **+0.229** [+0.143, +0.315] /
**−2.730** [−2.840, −2.620] per s (MLE and NUTS agree to 0.002; max R-hat 1.0036, min ESS 1058).
Route 2 without the hit term, 10 000 trials: +3.948 [+3.797, +4.106] / +0.497 [+0.228, +0.618] /
−2.542 [−2.738, −2.418]. `kernel_fit/`'s simulation-matching fit gave +3.98 / +0.24 / −2.50.

Zero crossing of g against gain: pooled **f₀ = 1.0155**, per-network median 1.0227 [IQR 0.980, 1.055] —
against Fig 1F's kernel-slope crossing at f = 1.016. Landscape comparison (its sign flipped):
+4.3 / −1.6 / −6.2 approx. vs +3.95 / +0.23 / −2.73 fitted.

**Checks.** Kernels (computed with `fit_ou_kernel.py::kernel` verbatim) reproduced pooled: network
+0.0392 / +0.0021 / −0.0261 vs fitted model +0.0392 / +0.0019 / −0.0306 (route 1) and +0.0404 / +0.0030 /
−0.0307 (route 2); accuracies match to 0.001. Fitted g vs each network's own kernel slope: **r = 0.996**
over the 60 fits. Recovery (60 route-1 refits on the real evidence, truths = fitted g and the landscape
g): **60/60 signs correct**, 56/60 intervals cover, |bias| ≤ 0.06 per s; route 2, 23 refits: 23/23 signs,
18/23 cover. Halving the trials to 1000 leaves every sign unchanged and 20/20 intervals excluding 0 at
0.8 and 1.2.

**Bound-hit observable — a caveat worth reading.** The network's |dv| = 2.0 crossing is **not absorbing**:
46 % / 21 % / 10 % of the trials that cross are back below 2.0 at the deadline (gains 0.8 / 1.0 / 1.2).
A sticky bound can only produce |a_T| ≥ B, so matching the model's absorbing event to that transient
crossing (the `bern` and `cross` variants specified in the issue) forces B ≈ 0.9 and inflates the leak —
at gain 0.8 the median g goes +4.13 → +5.90 → +8.28, 13/20 `cross` fits land at g ≥ 8, and the fitted
model then gets the **kernel sign wrong** (mean model slope −0.0097 vs the network's +0.0386, r = 0.082).
Measuring the same observable at the deadline (1[|dv_T| ≥ 2.0], the `term` variant) keeps the kernel
(+0.0208, r = 0.675 at 0.8; −0.0300, r = 0.909 at 1.2), identifies B (1.3–2.1 / 1.4–4.7), matches the
commitment fraction to 0.015, and gives 20/20 and 20/20.

**Degeneracy.** Exactly 1 of 60 choice-only route-2 fits lands on the leak-vs-bound solution (seed 61,
gain 1.2: g = +1.06 with B = 1.50); none reaches the g ≈ +12 / low-B solution that 3 of 60 `kernel_fit/`
fits hit — the full likelihood is more informative than the kernel + psychometric summary. The
deadline-commitment hit term turns seed 61 into −0.36 [−0.50, −0.19]. In recovery, a genuinely degenerate
dataset (g = +12, B = 1.2) is recovered by **all** variants including the choice-only fit
(+11.87 [+10.50, +12.92], B = 1.50), so the hit term is not what breaks the degeneracy here.

Deliverables: `track_b/{ou_lik.py, fit_analytic.py, fit_bounded.py, recover.py, checks.py, bash/, RESULTS.md}`,
`output/track_b/` (per-network and pooled tables for every route and variant, recovery tables, bound-hit
fractions, sensitivity, 4 figures). `kernel_fit/export_evidence.py` now also exports `tcross2_g*` /
`tcross15_g*`; the regenerated npz were verified bit-identical on every pre-existing key before replacing
the shared data.
