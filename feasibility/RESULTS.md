# Can a constant-bound OU inside the pretrained LAN's box reproduce the networks' fixed-bound behaviour?

Follow-up to issues #1 and #2. Code: `feasibility/ou_match.py` (exact ssm-simulators `ornstein` simulator,
random search + Nelder–Mead on per-coherence RT quantiles, accuracy and omission rate), datasets from
`data/process/fixed_bound_from_traces.py` (nxx1, 20 seeds pooled, 3 gains). Oscar jobs 6606075/6606166
(LAN box, 5 time stretches), 6606270 (relaxed boxes), 6606439 (high bounds). Table: `output/feasibility/matches_table.csv`;
one figure per cell `output/feasibility/match_*.png`. Sign convention: g > 0 leaky, g < 0 unstable.
"Native" = the network's own time (1 step = 1 ms); parameters are in unit-diffusion OU units.

## 1. Inside the LAN box the fit is only acceptable after stretching time by 4–6, and then drift sits on its ceiling

| stretch k | typical loss | what binds |
|---|---|---|
| 1 | 7–15 | a at its floor (0.3), t at its floor |
| 2 | 2.6–4.7 | drift at ceiling (≈1.95) |
| 4 | 0.36–1.4 | drift at ceiling (1.84–1.99) in 5 of 6 cells |
| 6–8 | 0.45–1.8 | a second solution appears: g ≈ −0.85 with a larger a, same loss as the leaky one |

The a–g trade-off from the recovery study (#1) is visible again at k = 6–8: two very different parameter
sets describe the same RT distributions. Every good boxed fit has g at or near +1 (leaky) at all gains.

## 2. With the box opened at native time, the OU fits gains 1.0 and 1.2 essentially exactly — far outside the box

| gain (bound 1.5) | loss | a | drift at coh 0.15 | g (per s) | t (ms) |
|---|---|---|---|---|---|
| 0.8 | 1.78 | 0.39 | 4.0 | +5.9 | 44 |
| 1.0 | 0.13 | 0.36 | 4.0 | +4.4 | 3 |
| 1.2 | 0.14 | 0.28 | 4.2 | +9.3 | 6 |

LAN box: a ≥ 0.3, |drift| ≤ 2, |g| ≤ 1. All three main parameters are at or beyond an edge, and no time
stretch moves the point inside (g scales as 1/k, drift and a as √k: bringing g to 1 needs k ≈ 9, which puts
drift at ≈ 13). Gain 0.8 fits worse because the network's errors are as fast as its correct responses at high
coherence and a leaky OU makes errors slower. Relaxing only drift (to 8, k = 4) leaves g pinned at +1;
relaxing only g (to 3) sends it to +2.4 (≈ +9.5 native) at gains 0.8 and 1.0.

**So the OU model class describes the fixed-bound behaviour; the pretrained likelihood does not cover the
region where these networks live.** A LAN retrained over a ≥ 0.05, |drift| ≤ 6, |g| ≤ 10 (native time) would.

## 3. But the fitted g does not track the theoretical regime — it tracks the bound

Theory (landscape drift slope at the undecided state, `RNN_Gain_Mod/lab/2026-09-21_gain-leaky-to-bistable`):
leaky at 0.8 (≈ +4.3 in this sign convention), ≈ 0 at 1.0, repelling at 1.2 (≈ −6.2). The OU fit says
leaky everywhere and *most* leaky at gain 1.2. Raising the bound moves g toward zero at gain 1.2 but the
one-timescale OU stops fitting:

| bound (gain 1.2) | omissions | g (per s) | loss |
|---|---|---|---|
| 1.5 | 0.0 % | +9.3 | 0.14 |
| 2.0 | 0.2 % | +8.6 | 0.62 |
| 2.5 | 5.6 % | +5.2 | 2.2 |
| 2.94 (= ln 19, the original height) | 14 % | +0.5 | 10.3 |

At 2.94 the OU's leading edge is ~50 ms too late and its tail too light at every coherence; forcing g < 0
there is worse still (loss 24 at g = −10 vs 11 at g = 0). Gain 1.0 shows the same fall (4.4 → 3.2 → 1.9)
until the fit breaks at 2.94.

Reading: a bound inside the noise band (1.5) is crossed within 50–100 ms by the fast, strongly mean-reverting
fluctuations of the readout (10 ms unit time constant, non-integrating directions), and the OU's single leak
term reports *their* decay, which grows with gain. Only a bound near the committed attractors (≈ 3) is
governed by the slow choice mode the landscape measures — and there a single OU can no longer represent the
mixture of fast and slow crossings, and gain 0.8 loses two thirds of its trials to omissions.

## 4. Conclusions

1. The pretrained OU LAN is unusable for these networks at any time stretch (a, drift and g all outside its box).
2. A retrained LAN would fit the fixed-bound data well, but the g it returns would be leaky at every gain and
   would not reproduce the leaky → perfect → attractive transition; that transition is not in the marginal
   RT/choice distributions under a low threshold.
3. The theory's behavioural signature is the temporal weighting kernel (recency → primacy), which needs
   within-trial evidence fluctuations, not RT/choice marginals.
4. If an accumulator-model readout is still wanted: high or collapsing bound (so the slow mode governs
   crossings), a likelihood that handles omissions or the Weibull collapse, and very likely a two-timescale
   process (fast mean-reverting noise on a slow integrator) rather than a 1-D OU. Test each with
   `ou_match.py` before training anything.
5. Side result: the bimodal RT distributions in the original data are produced by the Weibull collapse; with a
   constant bound every RT distribution is unimodal.
