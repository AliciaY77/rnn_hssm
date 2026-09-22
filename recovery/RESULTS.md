# OU LAN parameter recovery — results (issue #1)

Oscar jobs 6604819 (HSSM defaults: 5% lapse mixture, 4.5 s deadline) and 6604891 (no lapse, with and
without the deadline). 60 fits, 20 grid cells × 3 variants, 4 chains × 1000/1000, `target_accept 0.95`.
**No fit diverged, all R-hat ≤ 1.01 (one 1.06, one 1.41 in the default variant), 2–10 min each.**
Tables: `output/recovery/recovery_table.csv`; figure: `output/recovery/recovery_g.png`; per-cell
`cell*_summary.csv` / `cell*_meta.json`.

Sign convention (verified against ssms 0.8.3): `dx = (v − g·x) dt + dW`, so **g > 0 = leaky, g < 0 = unstable**.
Boundaries at ±a; z is the relative start point.

## Verdict: the acceptance criteria are not met

| variant | sign of g correct (g ≠ 0) | 94% HDI covers true g (interior) |
|---|---|---|
| default (lapse + deadline) | 7 / 16 | 4 / 12 |
| no lapse, deadline | 5 / 16 | 3 / 12 |
| no lapse, no deadline | 12 / 16 | 7 / 12 |

Three separate problems, in order of importance:

**1. Deadline truncation flips the sign of g.** Dropping trials past a 4.5 s deadline and fitting with an
untruncated likelihood makes every leaky cell at a = 1.5 come out as *unstable*: true g = +0.5 / +1.0 →
recovered −0.72 / −0.65 (sd 0.02, n = 3700), with a inflated from 1.5 to 2.2 and t driven to ~0. The
fraction dropped in those cells is 27–51 %, but the flip is already present at 12 % (g = 0, a = 1.5 →
−0.79). Without the deadline the same cells recover the correct sign. The RNN data are truncated by
construction (750 ms horizon), so this failure mode applies directly to them.

**2. g is not identifiable when decision times are short relative to 1/|g|.** At a = 1.0 (median RT
0.6–0.8 s, so |g|·T ≈ 0.6) the posterior for g has sd 0.2–0.5 and its mean sits between −0.3 and +0.4
regardless of the true value, even with 3700 trials and no deadline. At a = 1.5 (median RT ~1–2 s,
|g|·T ≈ 1.5–2) the sign is recovered. For the RNN networks, |g|·RT ≈ 1.2–1.6 at gains 0.8 and 1.2 and
≈ 0.4 at gain 1.0 — between the two regimes for the extreme gains, hopeless for the trained gain.

**3. Even where the sign is right, the magnitude is not.** In the clean variant at a = 1.5: true
g = +0.5 → 0.99 (pinned at the ceiling, HDI excludes the truth); g = −1.0 → −0.33; g = −0.5 → −0.28.
Only g = 0 is recovered well (−0.08, −0.11). a is recovered to within ~0.15 without the deadline.

The lapse mixture is *not* the culprit: the default and no-lapse variants are indistinguishable.

## What this means

The pretrained constant-bound OU LAN cannot give a trustworthy g on the RNN data, with or without
the fixed-bound regeneration in #2: the zero-coherence route is under-identified at the RNN's RT scale
and the horizon truncation biases g toward "unstable" — the direction the three existing real-data fits
show. Per the issue's stop rule, the next step is the custom likelihood: an OU integrator with the
generating Weibull bound built in (`a = ln 19`, shape 4.0, scale 468.75 ms, horizon 750 ms), trained
over g ≈ ±10 s⁻¹ and drift up to ~4 in native time via the pipeline in `lan/`. Because that bound
collapses to zero by the horizon, every trial terminates and the truncation problem disappears
along with the confound.
