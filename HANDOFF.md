# HANDOFF — OU accumulator fits to the gain-modulated RNNs (state as of 2026-09-22)

This branch (`ou-round2-base`) is the consolidated result of one working session. It is the base for
issues #3 (Track A) and #4 (Track B). Read this file first, then the two issues, then the two result notes
`feasibility/RESULTS.md` and `kernel_fit/RESULTS.md`, then `docs/OU_fits_to_gain_RNNs_2026-09-22.pdf`.

## 1. The question

Theory (gain-controller-rnn manuscript + `RNN_Gain_Mod/lab/2026-09-21_gain-leaky-to-bistable/NOTE.md`): a
scalar gain f on the NXX1 units' transfer slope moves the trained network from a **leaky** integrator
(f = 0.8) through a **near-perfect** integrator / line attractor (f = 1.0) into a **bistable / attractive**
regime (f = 1.2). Direct measurement: the Langevin drift of the choice-axis position on noise-only trials
has slope at the undecided state of about -4.3 / +1.6 / +6.2 per second at f = 0.8 / 1.0 / 1.2
(negative = restoring = leaky; 3 networks pooled; the 20-network curvature crosses zero at f = 0.94 ± 0.07).
Behavioural signature already in the manuscript (Fig 1F): psychophysical kernel recency → flat → primacy,
slope zero-crossing at f = 1.016 (20 networks).

Goal of this project: show the same transition with an accumulator model fitted to behaviour, i.e. an
Ornstein–Uhlenbeck (OU) leak parameter that goes leaky → zero → unstable across gain.

## 2. Repositories and locations

| what | where |
|---|---|
| this repo (modelling) | GitHub `AliciaY77/rnn_hssm` (public; igrahek has push). Local: `~/Library/CloudStorage/Dropbox-Brown/Ivan Grahek/Ivan/Studies/rnn_hssm`. Oscar: `/users/igrahek/rnn_hssm` |
| RNN code (frozen networks, rollouts, kernel) | local `…/Studies/RNN_Gain_Mod/gain-controller-rnn` @ 1ca084b (package `gainrnn`; needs env `rnn_gain`: `/opt/homebrew/anaconda3/envs/rnn_gain/bin/python`, set `RNN_T=750 RNN_EPOCHS=100`) |
| per-trial cache with full dv traces | `…/Studies/RNN_Gain_Controller/notes/mscont_cal_pertrial.parquet` (1.36 M rows; cols incl. `seed, act, f_nm, rel_coh, label, choice, correct, crossed, bounded_rt, logit_diff (list of 750 floats)`) |
| lab notes on the dynamics | `…/Studies/RNN_Gain_Mod/lab/2026-09-21_gain-leaky-to-bistable/NOTE.md` (+ `out/`) |
| paper figures | `…/Studies/RNN_Gain_Mod/gain-controller-rnn/figures/fig1.py`, `figures/out/fig1.png` (panel F = kernel), `figures/build/fig1.stats.json` |

Branches in `rnn_hssm`: `main` (Alicia's original), `issue-1-ou-param-recovery` (recovery + feasibility),
`kernel-ou-fit` (evidence-conditioned fit + report), `ou-round2-base` (= kernel-ou-fit + this file; base for #3/#4).

## 3. Directory map of this branch

```
model/fit_ornstein.py            Alicia's original HSSM OU fit (0.3 s offset, t free, k=1). Has the sign-convention note.
model/fit_ddm_fixed.py, fit_ddm_weibull.py   DDM fits (Weibull one also adds 0.3 s offset; fixed one does not)
data/process/process_data.py     Alicia's processing of the ORIGINAL (Weibull-bound) data, seed 42 only
data/process/fixed_bound_from_traces.py      regenerates RT/choice from dv traces with a CONSTANT bound (issue #2)
data/processed/fixed_bound/      (gitignored) hssm_ready_nxx1_fixed_b{1.25,1.5,2.0,2.5,2.94}_g{0.8,1.0,1.2}.{parquet,csv}, 20 seeds pooled
data/processed/kernel/           (gitignored) seed{42..61}.npz: rel evidence (2000×750 float16), labels, goals, coh_signed, choice_g{f}, dvT_g{f}
recovery/                        issue #1 parameter-recovery study (simulate_and_fit_ou.py, SLURM array, RESULTS.md)
feasibility/                     exact-simulator OU matching to fixed-bound RT/choice (ou_match.py, collect_matches.py, RESULTS.md)
kernel_fit/                      evidence-conditioned OU fit (export_evidence.py, fit_ou_kernel.py, fit_per_network.py, RESULTS.md)
output/recovery, output/fixed_bound, output/feasibility, output/kernel_fit   result tables/figures (tracked); netcdfs not tracked
docs/OU_fits_to_gain_RNNs_2026-09-22.pdf (+ docs/report_src/build_report.py)   3-page memo for the advisor
cluster/log/                     (gitignored) SLURM logs on Oscar
```

Regenerating gitignored data: fixed-bound → `python data/process/fixed_bound_from_traces.py --bounds 1.25 1.5 2.0 2.5 2.94`
(needs pyarrow; anaconda base python has it). Kernel npz → `RNN_T=750 RNN_EPOCHS=100 <rnn_gain python> kernel_fit/export_evidence.py --n-trials 2000`
(≈ 40 s per network). Both also exist on Oscar under `~/rnn_hssm/data/processed/{fixed_bound (CSV), kernel}`.

## 4. Conventions (get these right or everything flips)

- **OU leak sign.** ssm-simulators / HSSM implement dx = (v − g·x) dt + dW. **g > 0 = leaky (recency), g < 0 = unstable (primacy).**
  Verified empirically (median FPT 0.61 / 0.79 / 1.14 s for g = −1 / 0 / +1). `kernel_fit/` uses the same sign.
  The landscape's "drift slope" has the opposite sign (negative = leaky). The old `fit_ornstein.py` docstring was wrong.
- **Boundaries** at −a and +a (a = half-separation); z = relative start point (0.5 = unbiased).
- **dv convention** (network): dv = logits[0] − logits[1]; dv < 0 at crossing ⇔ choice "+" ⇔ label 1. Deadline choice = `(dv[:, -1] < 0)`.
- **Time rescaling.** RT × k ⇔ g → g/k, a → a·√k, v → v/√k (verified by simulation; an earlier note had v backwards).
- **Pretrained OU LAN box** (ssm-simulators `ornstein`): v ∈ [−2, 2], a ∈ [0.3, 3], z ∈ [0.1, 0.9], g ∈ [−1, 1] (per s), t ∈ [1e-3, 2] s.
  Networks in native time need a ≈ 0.3, v ≈ 4, g ≈ +4…+9 → **stretch k ≈ 8–10 to get inside** (k = 8 fits were interior except g slightly clipped at f = 1.2).
- **Non-decision time rule** (agreed): add a fixed offset c to RTs and **fix** t = c + min RT of that dataset (same units). Never fit t.
  Min RT: constant bound b = 1.5 → 6 / 6 / 5 ms native at f = 0.8 / 1.0 / 1.2; original Weibull data → 38 / 18 / 14 ms.
  In HSSM fix a parameter by passing a float (`hssm.HSSM(..., t=0.336)`), not via `include`.
- **HSSM 0.2.4 (Oscar env) sampler name** is `nuts_numpyro`; ≥ 0.3 uses `numpyro`. Default priors Uniform within the box; default lapse p_outlier = 0.05 with Uniform(0, 20 s) — disable with `p_outlier=None, lapse=None` unless matched.
- **Deadline / omissions.** Fitting horizon-truncated data with an untruncated likelihood **flips the sign of g** toward unstable (recovery study). The RNN horizon is 750 ms. Report omission fractions; never force a choice at the deadline.

## 5. What was found (numbers in the RESULTS notes)

1. Original 3 HSSM OU fits (seed 42): ~100 % divergences, g at the −1 (unstable) edge, drift at the +2 edge, t absorbed the 0.3 s offset.
2. Recovery study (60 fits, pretrained LAN, k = 6, zero coherence): sampler fine; deadline truncation flips g's sign; g unidentifiable when |g|·T ≲ 0.6; lapse mixture irrelevant.
3. Constant-bound regeneration: at b = 1.5, omissions 3.4 % (f = 0.8), 0.3 %, 0 %; RTs unimodal (the bimodality in the original data comes from the Weibull collapse); median RT 145 / 99 / 76 ms.
4. Exact-simulator OU matched to fixed-bound RT/choice: fits f = 1.0 and 1.2 essentially exactly (loss 0.13 / 0.14) at a ≈ 0.3, v ≈ 4, g = +5.9 / +4.4 / +9.3 per s — **leaky at every gain, most at f = 1.2**, opposite to theory. Raising the bound lowers g (9.3 → 0.5 at b = 2.94 for f = 1.2) but the 1-D OU stops fitting. Interpretation: a low bound is crossed by fast readout fluctuations; the OU's single leak term reports their decay.
5. Evidence-conditioned OU (same model driven by each trial's actual evidence stream, matched on kernel + psychometric): **g = +3.98 / +0.24 / −2.50 per s** pooled over 20 networks; per network 20/20 leaky at 0.8, 15/20 unstable at 1.2 (median −1.8). Kernel reproduced bin-by-bin. Caveat: 3/60 single-network fits hit a leak-vs-bound degeneracy (g ≈ +12 with a low sticky bound gives primacy too).

## 6. Environments and gotchas

- Local: `/opt/homebrew/anaconda3/bin/python` (pyarrow, sklearn, scipy, matplotlib; NO hssm); `rnn_gain` env for torch + gainrnn; a scratch venv had arviz/reportlab/pypdfium2 (recreate with pip if needed).
- Oscar: ssh alias `oscar` (ControlMaster). Env `/users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python` (hssm 0.2.4, ssms 0.8.3, jax 0.5.3, numpyro 0.18; NO pyarrow → ship CSV). SLURM: `#!/bin/bash -l`, account `carney-frankmj-condo2`, partition `batch`, qos `carney-condo2`; templates in `recovery/bash/`, `feasibility/bash/`. 20 × 4-core jobs start immediately.
- ssm-simulators C code crashes (`free(): invalid next size`) for max_t ≈ 0.75 s: simulate to ≥ 2 s and truncate afterwards.
- macOS blocks shell access to `~/Downloads` for this tool; Finder via osascript works.
- Paths contain spaces (Dropbox); quote everything. zsh globbing: use `--include='*.py'` quoting or `setopt nullglob`.
- `output/` is gitignored except the listed subfolders; add `!output/<new>/` to `.gitignore` for new result folders.
