# Ornstein-Uhlenbeck fits — analysis notes (v1)

## What this is

The Ornstein-Uhlenbeck (OU) model is a DDM with a free leak/instability parameter `g`:
`g < 0` leaky integrator, `g = 0` standard DDM, `g > 0` unstable/attractor dynamics. `g` maps
directly onto the Brunton simulator's lambda parameter, which is the motivation for fitting it
here.

Fit with `model/fit_ornstein.py`: `v ~ 1 + coherence`, `g` free (prior `Uniform(-1, 1)`),
numpyro/NUTS, 4 chains, 1000 tune + 1000 draws, `target_accept=0.95`. Data: RNN behavioral
output for network nxx1, seed 42, at gain 0.8 / 1.0 / 1.2 (2000 trials each), with a flat +0.3s
RT offset applied before fitting.

## Parameter estimates

Posterior mean, sd, and 3-97% HDI.

**gain = 0.8**

| param | mean | sd | hdi3 | hdi97 |
|---|---|---|---|---|
| t | 0.3865 | 0.0036 | 0.3780 | 0.3916 |
| z | 0.5221 | 0.0133 | 0.4968 | 0.5423 |
| g | -0.7922 | 0.1098 | -0.9346 | -0.6039 |
| a | 0.6812 | 0.0125 | 0.6633 | 0.7072 |
| v_coherence | 10.1590 | 1.0082 | 7.6485 | 11.5928 |
| v_Intercept | 0.4677 | 0.1505 | 0.2558 | 0.8419 |

**gain = 1.0**

| param | mean | sd | hdi3 | hdi97 |
|---|---|---|---|---|
| a | 0.6009 | 0.0088 | 0.5826 | 0.6149 |
| v_Intercept | 0.6281 | 0.1236 | 0.4382 | 0.8741 |
| v_coherence | 9.0843 | 0.8240 | 7.4424 | 10.3692 |
| t | 0.3337 | 0.0021 | 0.3296 | 0.3379 |
| z | 0.5484 | 0.0107 | 0.5324 | 0.5703 |
| g | -0.9060 | 0.0870 | -0.9975 | -0.7621 |

**gain = 1.2**

| param | mean | sd | hdi3 | hdi97 |
|---|---|---|---|---|
| v_coherence | 8.4627 | 0.9725 | 6.7265 | 10.1470 |
| t | 0.3228 | 0.0012 | 0.3207 | 0.3253 |
| v_Intercept | 0.7213 | 0.1456 | 0.4692 | 0.9862 |
| z | 0.5356 | 0.0100 | 0.5150 | 0.5536 |
| a | 0.4963 | 0.0075 | 0.4789 | 0.5068 |
| g | -0.9101 | 0.0634 | -0.9828 | -0.7780 |

## Convergence diagnostics

R-hat (split-chain) and ESS (Geyer initial-positive-sequence), out of 4000 total draws
(4 chains x 1000). Target: R-hat < 1.01, ESS in the thousands. None of the fits below meet
either criterion.

| gain | param | R-hat | ESS |
|---|---|---|---|
| 0.8 | t | 1.78 | 27.9 |
| 0.8 | z | 2.64 | 21.0 |
| 0.8 | g | 4.08 | 12.6 |
| 0.8 | a | 2.27 | 20.4 |
| 0.8 | v_coherence | 2.40 | 27.7 |
| 0.8 | v_Intercept | 2.44 | 26.7 |
| 1.0 | a | 1.47 | 25.1 |
| 1.0 | v_Intercept | 1.27 | 14.5 |
| 1.0 | v_coherence | 1.26 | 14.8 |
| 1.0 | t | 1.26 | 42.0 |
| 1.0 | z | 1.18 | 15.6 |
| 1.0 | g | 5.67 | 28.4 |
| 1.2 | v_coherence | 2.30 | 30.9 |
| 1.2 | t | 1.08 | 37.8 |
| 1.2 | v_Intercept | 2.32 | 30.1 |
| 1.2 | z | 1.68 | 31.2 |
| 1.2 | a | 1.86 | 14.5 |
| 1.2 | g | 1.96 | 16.4 |

None of the point estimates above should be treated as reliable given these numbers — they
tell us where the sampler got stuck, not necessarily where the true posterior mass is.

## Diagnosis

### t is squeezed against min(RT)

Decision time (RT - t) must be >= 0 for every trial. Comparing fitted t to the minimum RT
after the +0.3s offset:

| gain | raw RT min | offset RT min | fitted t (mean) | margin |
|---|---|---|---|---|
| 0.8 | 0.0670 | 0.3670 | 0.3865 | t exceeds min RT by 0.0195s (impossible region) |
| 1.0 | 0.0350 | 0.3350 | 0.3337 | 0.0013s |
| 1.2 | 0.0260 | 0.3260 | 0.3228 | 0.0032s |

At gain 0.8, the posterior mean for t is already past the fastest observed offset RT — the
model is being asked to explain a trial with negative decision time, a zero-likelihood region.
At gain 1.0 and 1.2, t sits within 1-3ms of the boundary. This is a standard pathology in
DDM/OU fitting and is a much more direct explanation for R-hat in the 1.2-5.7 range than
anything to do with the g prior.

Root cause: the RNN's fastest decisions resolve in 1-3 simulation timesteps (raw RT floor
26-67ms), well below anything a fixed non-decision-time constant can accommodate once
RT = decision_time + t with a realistic t. A flat +0.3s offset shifts the whole distribution
up but doesn't change the *gap* between the fastest RT and a plausible t — it just relocates
the squeeze.

### g prior is truncating the posterior

Fraction of posterior mass within 5% of the -1.0 lower bound (0% mass approaches +1.0 at any
gain):

| gain | % near lower bound (-1.0) |
|---|---|
| 0.8 | 7.0% |
| 1.0 | 50.0% |
| 1.2 | 59.3% |

The truncation gets worse as gain increases, meaning the network is likely becoming *more*
leaky at higher gain than we can currently see — the prior is hiding it.

## Recommendations

1. **Put an informative prior on t** with an upper bound safely below the RT floor per gain
   (e.g. `Uniform(0, 0.2)` at gain 1.2, where the floor is 0.326), instead of letting t float
   freely against the data. This is the priority fix — it targets the dominant failure mode.
2. **Widen the g prior** to `Uniform(-2, 1)` (asymmetric, since no gain shows any mass toward
   +1) to stop truncating the leak estimate.
3. Re-check R-hat/ESS after both changes before trusting any point estimate. More
   draws/tuning on the current model won't help — the problem is posterior geometry
   (a hard boundary sitting right at the data), not sample size.

## Next: v2 fits

Re-running all three gains with the updated t prior and the widened g prior
(`Uniform(-2, 1)`), same sampler settings otherwise. Will report convergence diagnostics
before revisiting the parameter estimates.
