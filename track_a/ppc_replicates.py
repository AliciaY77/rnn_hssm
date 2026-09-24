"""
Posterior-predictive replicates for the pooled Track A fits (issue #3, round-3 figures).

For one (bound, gain) fit, take --n-draws evenly spaced posterior draws; for each draw simulate ONE replicate dataset of the
same size and coherence mix as the network data before omissions (n per |coherence| from output/fixed_bound/omissions.csv,
split equally between the two signs), with the same numpy Euler engine as ppc_rt.py (stretched frame, dt = 1 ms, bounds at
+/- a, start a(2z - 1), fixed t, horizon 0.3 + k * 0.75 s; non-crossers are omissions). Per replicate: omission fraction,
median RT of crossers (native ms), accuracy of crossers at non-zero coherence. The spread across replicates is the 95 %
posterior-predictive interval for each statistic, directly comparable with the network's value.
Positive g is leaky.

  python track_a/ppc_replicates.py --gain 0.8 --bound 2.94 [--k 10 --n-draws 40]
Output: output/track_a/ppc_replicates_g<gain>_k<k>_b<bound>.csv (one row per draw, plus the network values).
"""
import argparse, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ppc_rt import simulate_ou_numpy, OFFSET, HORIZON_NATIVE

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "output" / "track_a"
DATA = ROOT / "data" / "processed" / "fixed_bound"; OMIT = ROOT / "output" / "fixed_bound" / "omissions.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True); ap.add_argument("--bound", type=float, required=True)
    ap.add_argument("--k", type=float, default=10.0); ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    tag = f"g{a.gain}_k{a.k:g}_b{a.bound}_pooled"
    draws = pd.read_csv(OUT / f"{tag}_draws.csv")
    om = pd.read_csv(OMIT); om = om[(om.bound == a.bound) & (om.gain == a.gain) & (om.coherence != "all")].copy()
    om["acoh"] = om.coherence.astype(float).round(2)
    cohs, counts = [], []
    for _, r in om.iterrows():
        if r.acoh == 0: cohs.append(0.0); counts.append(int(r.n))
        else: cohs += [r.acoh, -r.acoh]; counts += [int(r.n) // 2, int(r.n) - int(r.n) // 2]
    d = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{a.bound}_g{a.gain}.csv")
    n_all = int(om.n.sum()); net = dict(omit=1 - len(d) / n_all, median_rt=float(np.median(d.rt) * 1000),
                                        acc=float((d.response[d.coherence.round(2) > 0] == 1).mean()))
    horizon = OFFSET + a.k * HORIZON_NATIVE
    idx = np.linspace(0, len(draws) - 1, min(a.n_draws, len(draws))).astype(int); rows = []
    for j, ii in enumerate(idx):
        row = draws.iloc[ii]; rng = np.random.default_rng(a.seed + 7919 * j)
        vs = np.array([row.v_Intercept + row.v_coherence_signed * c for c in cohs])
        # simulate_ou_numpy repeats each drift n_per times; run it once per signed coherence to honour unequal counts
        rts, chs, sg = [], [], []
        for v, c, n in zip(vs, cohs, counts):
            rt, ch = simulate_ou_numpy(np.array([v]), np.array([row.a]), np.array([row.z]), np.array([row.g]), float(row.t), n, horizon, rng)
            rts.append(rt); chs.append(ch); sg.append(np.full(n, np.sign(c)))
        rt = np.concatenate(rts); ch = np.concatenate(chs); sg = np.concatenate(sg); ok = np.isfinite(rt)
        rt_nat = (rt[ok] - OFFSET) / a.k * 1000; nz = ok & (sg != 0)
        rows.append(dict(draw=int(ii), g_nat=float(row.g * a.k), omit=float(1 - ok.mean()), median_rt=float(np.median(rt_nat)),
                         acc=float((ch[nz] * sg[nz] > 0).mean())))
        if (j + 1) % 10 == 0: print(f"  {j+1}/{len(idx)} draws ({(time.time()-t0)/60:.1f} min)", flush=True)
    out = pd.DataFrame(rows).assign(gain=a.gain, bound=a.bound, k=a.k, net_omit=net["omit"], net_median_rt=net["median_rt"], net_acc=net["acc"])
    out.to_csv(OUT / f"ppc_replicates_{tag.replace('_pooled', '')}.csv", index=False)
    q = out[["omit", "median_rt", "acc"]].quantile([0.025, 0.5, 0.975])
    print(f"{tag}: network omit {net['omit']:.3f} median RT {net['median_rt']:.0f} ms acc {net['acc']:.3f}\n{q.round(3).to_string()}\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
