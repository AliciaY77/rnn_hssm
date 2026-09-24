"""
Quantile-probability posterior predictive for the pooled Track A fits (round-3 report; same idea as HSSM's
plot_quantile_probability used in model/fit_ornstein.py).

For one (bound, gain) fit: take --n-draws evenly spaced posterior draws; for each draw simulate one replicate dataset of the
data's size and coherence mix before omissions (as in ppc_replicates.py, same numpy engine as ppc_rt.py). For every |coherence|
and response side (correct / error; at coherence 0, choice "+" / "-") record the proportion of crossing trials on that side and
the RT quantiles (native ms). The same summaries are computed for the network data (draw = -1).
Output: output/track_a/ppc_qpp_g<gain>_k<k>_b<bound>.csv (long format). Positive g is leaky.

  python track_a/ppc_qpp.py --gain 0.8 --bound 2.94 [--k 10 --n-draws 40]
"""
import argparse, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ppc_rt import simulate_ou_numpy, OFFSET, HORIZON_NATIVE

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "output" / "track_a"
DATA = ROOT / "data" / "processed" / "fixed_bound"; OMIT = ROOT / "output" / "fixed_bound" / "omissions.csv"
QS = [0.1, 0.3, 0.5, 0.7, 0.9]


def summarise(acoh, rt_ms, correct, draw):
    rows = []
    for c in np.unique(acoh):
        m = acoh == c; n_all = int(m.sum())
        for side, sel in [("correct", m & correct), ("error", m & ~correct)]:
            r = rt_ms[sel]; row = dict(draw=draw, acoh=float(c), side=side, n=int(sel.sum()), p=float(sel.sum() / n_all) if n_all else np.nan)
            row.update({f"q{int(q * 100)}": (float(np.quantile(r, q)) if len(r) >= 5 else np.nan) for q in QS}); rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True); ap.add_argument("--bound", type=float, required=True)
    ap.add_argument("--k", type=float, default=10.0); ap.add_argument("--n-draws", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); tag = f"g{a.gain}_k{a.k:g}_b{a.bound}"
    draws = pd.read_csv(OUT / f"{tag}_pooled_draws.csv")
    om = pd.read_csv(OMIT); om = om[(om.bound == a.bound) & (om.gain == a.gain) & (om.coherence != "all")].copy(); om["acoh"] = om.coherence.astype(float).round(2)
    cohs, counts = [], []
    for _, r in om.iterrows():
        if r.acoh == 0: cohs.append(0.0); counts.append(int(r.n))
        else: cohs += [r.acoh, -r.acoh]; counts += [int(r.n) // 2, int(r.n) - int(r.n) // 2]
    d = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{a.bound}_g{a.gain}.csv"); acoh = d.coherence.round(2).values
    corr_net = np.where(acoh == 0, d.response_choice.values > 0, d.response.values > 0)
    rows = summarise(acoh, d.rt.values * 1000, corr_net, -1)
    horizon = OFFSET + a.k * HORIZON_NATIVE; idx = np.linspace(0, len(draws) - 1, min(a.n_draws, len(draws))).astype(int)
    for j, ii in enumerate(idx):
        row = draws.iloc[ii]; rng = np.random.default_rng(a.seed + 104729 * j); rts, chs, sg = [], [], []
        for c, n in zip(cohs, counts):
            rt, ch = simulate_ou_numpy(np.array([row.v_Intercept + row.v_coherence_signed * c]), np.array([row.a]), np.array([row.z]), np.array([row.g]), float(row.t), n, horizon, rng)
            rts.append(rt); chs.append(ch); sg.append(np.full(n, c))
        rt = np.concatenate(rts); ch = np.concatenate(chs); cs = np.concatenate(sg); ok = np.isfinite(rt)
        rt, ch, cs = (rt[ok] - OFFSET) / a.k * 1000, ch[ok], cs[ok]
        corr = np.where(cs == 0, ch > 0, ch * np.sign(cs) > 0)
        rows += summarise(np.round(np.abs(cs), 2), rt, corr, int(ii))
        if (j + 1) % 10 == 0: print(f"  {j+1}/{len(idx)} draws ({(time.time()-t0)/60:.1f} min)", flush=True)
    out = pd.DataFrame(rows).assign(gain=a.gain, bound=a.bound, k=a.k); out.to_csv(OUT / f"ppc_qpp_{tag}.csv", index=False)
    print(f"{tag}: saved {len(out)} rows ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
