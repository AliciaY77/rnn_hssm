"""
Quantile-probability posterior predictive for the pooled Track A fits (round-3 report), following HSSM's
plot_quantile_probability defaults (as used in model/fit_ornstein.py with predictive_style="ellipse", ellipse_confidence=0.95):

  * n_samples = 20 draws per chain x 4 chains = 80 posterior-predictive datasets; here 80 draws chosen at random without
    replacement from the 400 saved (evenly thinned) posterior draws;
  * each predictive dataset = one simulated trial per observed data row, with that row's signed coherence (HSSM simulates the
    original data's covariates);
  * condition = |coherence|; is_correct = response > 0 in accuracy coding, i.e. choice matches the row's label (at coherence 0
    the label is the stimulus's nominal side, as in the accuracy-coded data);
  * q = 5  ->  quantiles np.linspace(0, 1, 5)[1:-1] = 0.25, 0.5, 0.75 of RT per (dataset, condition, is_correct), no minimum
    trial count; proportion = share of that dataset's trials in the condition with that is_correct value.
The one intended difference from HSSM: simulated trials that do not reach the bound by the 750 ms horizon are dropped, as they
are in the data (HSSM's predictive sampler would keep them with RTs up to its 20 s max_t). Same numpy engine as ppc_rt.py.
Output: output/track_a/ppc_qpp_g<gain>_k<k>_b<bound>.csv, long format (draw = -1 for the network data). Positive g is leaky.

  python track_a/ppc_qpp.py --gain 0.8 --bound 2.94 [--k 10 --n-draws 80]
"""
import argparse, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ppc_rt import simulate_ou_numpy, OFFSET, HORIZON_NATIVE

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "output" / "track_a"; DATA = ROOT / "data" / "processed" / "fixed_bound"
Q = np.linspace(0, 1, 5)[1:-1]


def summarise(acoh, rt_ms, correct, draw):
    df = pd.DataFrame(dict(acoh=acoh, rt=rt_ms, is_correct=correct))
    qs = df.groupby(["acoh", "is_correct"])["rt"].quantile(q=Q).reset_index().rename(columns={"level_2": "quantile"})
    pc = df.groupby("acoh")["is_correct"].value_counts(normalize=True).rename("proportion").reset_index()
    n = df.groupby(["acoh", "is_correct"]).size().rename("n").reset_index()
    return qs.merge(pc, on=["acoh", "is_correct"]).merge(n, on=["acoh", "is_correct"]).assign(draw=draw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True); ap.add_argument("--bound", type=float, required=True)
    ap.add_argument("--k", type=float, default=10.0); ap.add_argument("--n-draws", type=int, default=80); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(); t0 = time.time(); tag = f"g{a.gain}_k{a.k:g}_b{a.bound}"
    draws = pd.read_csv(OUT / f"{tag}_pooled_draws.csv")
    d = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{a.bound}_g{a.gain}.csv")
    cs = d.coherence_signed.values; acoh = np.round(np.abs(cs), 2); lab = np.where(d.label.values == 1, 1.0, -1.0)
    out = [summarise(acoh, d.rt.values * 1000, d.response.values > 0, -1)]
    horizon = OFFSET + a.k * HORIZON_NATIVE
    pick = np.sort(np.random.default_rng(a.seed).choice(len(draws), size=min(a.n_draws, len(draws)), replace=False))
    ucs = np.unique(cs)
    for j, ii in enumerate(pick):
        row = draws.iloc[ii]; rng = np.random.default_rng(a.seed + 7 + 104729 * j)
        rt = np.empty(len(d)); ch = np.empty(len(d))
        for c in ucs:                                   # one simulated trial per data row, grouped by that row's coherence
            m = cs == c
            r, q = simulate_ou_numpy(np.array([row.v_Intercept + row.v_coherence_signed * c]), np.array([row.a]), np.array([row.z]),
                                     np.array([row.g]), float(row.t), int(m.sum()), horizon, rng)
            rt[m] = r; ch[m] = q
        ok = np.isfinite(rt)                             # drop non-crossers, as in the data
        out.append(summarise(acoh[ok], (rt[ok] - OFFSET) / a.k * 1000, ch[ok] == lab[ok], int(ii)))
        if (j + 1) % 20 == 0: print(f"  {j+1}/{len(pick)} datasets ({(time.time()-t0)/60:.1f} min)", flush=True)
    res = pd.concat(out, ignore_index=True).assign(gain=a.gain, bound=a.bound, k=a.k)
    res.to_csv(OUT / f"ppc_qpp_{tag}.csv", index=False); print(f"{tag}: {len(res)} rows, {len(pick)} datasets ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
