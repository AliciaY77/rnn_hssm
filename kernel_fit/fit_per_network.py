"""Per-network version of fit_ou_kernel.py: one (v, g, B) fit per (seed, gain) on that network's 2000 trials,
so g gets a spread across the 20 networks (like the per-seed kernel slopes behind Fig 1F).
Usage: python kernel_fit/fit_per_network.py [--n-random 80]  ->  output/kernel_fit/ou_kernel_fits_per_network.csv/.png"""
import argparse, glob, pathlib, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_ou_kernel import kernel, psychometric, simulate, loss_fn, DATA, OUT


def fit_one(rel, coh, ch, rng, n_random):
    w_net, s_net = kernel(rel, ch); _, P_net = psychometric(coh, ch); noise = rng.standard_normal(rel.shape, dtype=np.float32)
    cands = []
    for _ in range(n_random):
        th = [float(np.exp(rng.uniform(np.log(1), np.log(300)))), float(rng.uniform(-15, 15)), float(np.exp(rng.uniform(np.log(0.3), np.log(30))))]
        cands.append((loss_fn(th, rel, noise, coh, w_net, P_net), th))
    cands.sort(key=lambda c: c[0]); best = []
    for _, th in cands[:2]:
        r = minimize(loss_fn, th, args=(rel, noise, coh, w_net, P_net), method="Nelder-Mead", options=dict(maxiter=120, xatol=1e-3, fatol=1e-5))
        best.append((r.fun, list(r.x)))
    best.sort(key=lambda c: c[0]); loss, (v, g, B) = best[0]
    ch_sim = simulate(rel, v, g, B, noise); _, s_sim = kernel(rel, ch_sim)
    return dict(v=v, g=g, B=B, loss=loss, slope_net=s_net, slope_sim=s_sim, acc_net=float((ch == (coh > 0)).mean()), acc_sim=float((ch_sim == (coh > 0)).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-random", type=int, default=80); ap.add_argument("--gains", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    a = ap.parse_args(); rng = np.random.default_rng(1); rows = []
    for f in sorted(glob.glob(str(DATA / "seed*.npz"))):
        d = np.load(f); seed = int(pathlib.Path(f).stem[4:]); rel = d["rel"].astype(np.float32); coh = d["coh_signed"]; t0 = time.time()
        for gn in a.gains:
            rows.append(dict(seed=seed, gain=gn, **fit_one(rel, coh, d[f"choice_g{gn}"], rng, a.n_random)))
        print(f"seed {seed} ({time.time()-t0:.0f}s): " + " | ".join(f"g{r['gain']}: g={r['g']:+.2f} slope net {r['slope_net']:+.3f} sim {r['slope_sim']:+.3f}" for r in rows[-len(a.gains):]), flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "ou_kernel_fits_per_network.csv", index=False)
    summ = tab.groupby("gain").agg(g_mean=("g", "mean"), g_sem=("g", "sem"), g_median=("g", "median"), frac_g_pos=("g", lambda x: (x > 0).mean()),
                                   v_mean=("v", "mean"), B_median=("B", "median"), slope_net=("slope_net", "mean"), slope_sim=("slope_sim", "mean"), loss=("loss", "mean"))
    print("\n" + summ.round(3).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for gn, dd in tab.groupby("gain"):
        x = np.full(len(dd), gn) + rng.uniform(-0.02, 0.02, len(dd)); axes[0].plot(x, dd.g, "o", alpha=.6, ms=4)
        axes[1].plot(dd.slope_net, dd.g, "o", alpha=.6, ms=4, label=f"gain {gn}")
    axes[0].errorbar(summ.index, summ.g_mean, yerr=summ.g_sem, fmt="k_", ms=14, capsize=4, lw=2); axes[0].axhline(0, color="gray", lw=.5)
    axes[0].set_xlabel("gain"); axes[0].set_ylabel("fitted OU leak g (per s; >0 leaky)"); axes[0].set_title("per-network fits (n=20)")
    axes[1].axhline(0, color="gray", lw=.5); axes[1].axvline(0, color="gray", lw=.5); axes[1].set_xlabel("network kernel slope"); axes[1].set_ylabel("fitted g"); axes[1].legend(fontsize=7); axes[1].set_title("g vs kernel slope")
    plt.tight_layout(); fig.savefig(OUT / "ou_kernel_fits_per_network.png", dpi=140); print("saved", OUT / "ou_kernel_fits_per_network.png")


if __name__ == "__main__":
    main()
