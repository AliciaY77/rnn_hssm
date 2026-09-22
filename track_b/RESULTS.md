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
