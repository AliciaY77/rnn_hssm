# Stimulus-conditioned OU fit reproduces the leaky → perfect → attractive transition

Branch `kernel-ou-fit`. The "cheapest Brunton": a one-dimensional leaky/unstable accumulator driven by the
networks' **own per-trial evidence streams**, fitted to the networks' **deadline choices** through the
psychophysical kernel (Fig 1F of gain-controller-rnn) and the psychometric curve. No RTs, no LAN.

Data: `kernel_fit/export_evidence.py` — nxx1 e100, seeds 42–61, gains 0.8 / 1.0 / 1.2, 2000 trials each
(same stimuli across gains, `make_dataset(seed=0)`), choice = `choices_from_dv` (sign of dv at T), kernel =
`regime_lib.psychophysical_kernel` (8 bins, logistic regression, L1-normalised). Fit: `fit_ou_kernel.py`
(20 networks pooled) and `fit_per_network.py` (one fit per network).

Model: a_{t+1} = a_t + (v·e_t − g·a_t)·dt + √dt·ξ_t, a_0 = 0, dt = 1 ms, unit diffusion, sticky bound ±B,
choice = sign(a_T). **g > 0 leaky (recency), g < 0 unstable (primacy).** Loss = 100·Σ(Δkernel)² + 20·Σ(Δpsychometric)².

## Pooled fit (40,000 trials per gain)

| gain | g (per s) | v | B | kernel slope, network | kernel slope, OU | accuracy network / OU |
|---|---|---|---|---|---|---|
| 0.8 | **+3.98** | 46.5 | 2.6 | +0.039 | +0.041 | 0.866 / 0.865 |
| 1.0 | **+0.24** | 40.6 | 2.1 | +0.002 | +0.002 | 0.887 / 0.885 |
| 1.2 | **−2.50** | 40.5 | 4.9 | −0.026 | −0.028 | 0.869 / 0.871 |

The OU kernels lie on top of the network kernels at all eight bins for all three gains, the psychometric
curves overlap, and v is the same at every gain (the evidence tilt is gain-independent, as the landscape
analysis found). The mapping g → kernel slope is identical across gains (`g_sweep.csv`; right panel of
`output/kernel_fit/ou_kernel_fits.png`), so the kernel is a clean, monotonic readout of g.

Comparison with the landscape drift slope at the undecided state (same sign convention; leaky positive):
landscape +4.3 / −1.6 / −6.2 vs OU **+4.0 / +0.2 / −2.5**. Same ordering, same zero crossing near the
trained gain, magnitudes within a factor of ~2 at the extremes (the OU is linear; the network's repelling
regime at gain 1.2 saturates into wells at |dv| ≈ 3, which a linear fit under-reports).

## Per-network fits (n = 20, `ou_kernel_fits_per_network.csv`)

| gain | mean g ± sem | median g | networks with g > 0 |
|---|---|---|---|
| 0.8 | +4.3 ± 0.2 | +4.4 | 20 / 20 |
| 1.0 | +1.6 ± 0.7 | +0.9 | 13 / 20 |
| 1.2 | −0.7 ± 0.9 | −1.8 | 5 / 20 |

Fitted g tracks each network's own kernel slope (right panel of `ou_kernel_fits_per_network.png`).

**Caveat — the bound / leak degeneracy.** Two of the five positive-g networks at gain 1.2 (and one at 1.0)
landed on g ≈ +12 with a low sticky bound: strong leak plus early commitment also yields a primacy kernel.
This is the known λ-vs-bound trade-off of the Brunton model; it appears in single-network fits (2000 trials)
and not in the pooled fit. Breaking it needs either the fraction of trials that hit the bound before T
(available from the network's dv), or RTs, or a prior that B ≥ the typical |a_T|. Median g is the robust
per-network summary until then.

## What this settles

1. The transition the theory predicts **is** recoverable from behaviour with an accumulator model — provided
   the fit is conditioned on the trial-by-trial evidence. It is not recoverable from RT/choice marginals
   (see `feasibility/RESULTS.md`), which read out the fast readout fluctuations instead.
2. A proper Brunton-style likelihood fit (choice given evidence stream, λ free) is the parametric confirmation
   to build next; this simulation-matching version is its proof of concept.
3. If HSSM is to stay in the loop, the LAN would have to take the evidence stream as an input, which is not
   what the pretrained models do.
