"""
Stimulus-conditioned OU fit ("cheapest Brunton"): drive a leaky/unstable accumulator with the networks'
OWN per-trial evidence streams and find (v, g, B) whose deadline choices reproduce the networks'
psychophysical kernel (Fig 1F) and psychometric curve.  One fit per gain, 20 networks pooled.

Model (time in seconds, dt = 1 ms, unit diffusion):
    a_{t+1} = a_t + (v * e_t - g * a_t) dt + sqrt(dt) * xi_t ,   a_0 = 0,
    sticky bound: once |a| >= B the accumulator freezes;   choice = 1 ("+") iff a_T > 0.
  g > 0 leaky (recency), g < 0 unstable (primacy) — same sign convention as ssm-simulators.
  e_t is the relevant stream (coherence + N(0,1) noise per step) exactly as the network received it.

Objective per gain: 100 * sum_bins (w_sim - w_net)^2  +  20 * sum_coh (P_sim - P_net)^2, kernels computed with
the same procedure as gainrnn.regime_lib.psychophysical_kernel (8 bins, logistic regression, L1-normalised).
Common random numbers make the objective deterministic for the optimiser.

Usage:  python kernel_fit/fit_ou_kernel.py [--gains 0.8 1.0 1.2] [--n-random 150] [--sub 20000]
Outputs: output/kernel_fit/ou_kernel_fits.csv, ou_kernel_fits.png, g_sweep.csv
"""
import argparse, glob, json, pathlib, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "kernel"; OUT = ROOT / "output" / "kernel_fit"
DT = 0.001; N_BINS = 8


def logistic_weights(X, y):
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000).fit(X, y).coef_[0]
    except ImportError:                                  # IRLS fallback, L2 ~ sklearn default C=1
        Xb = np.c_[np.ones(len(X)), X]; w = np.zeros(Xb.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-Xb @ w)); W = p * (1 - p)
            H = Xb.T @ (Xb * W[:, None]) + np.eye(len(w)) * 1.0; grad = Xb.T @ (y - p) - w
            w = w + np.linalg.solve(H, grad)
        return w[1:]


def kernel(rel, choice, n_bins=N_BINS):
    """Replicates gainrnn.regime_lib.psychophysical_kernel on given trials."""
    edges = np.linspace(0, rel.shape[1], n_bins + 1).astype(int)
    X = np.stack([rel[:, edges[k]:edges[k + 1]].mean(1) for k in range(n_bins)], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if len(np.unique(choice)) < 2: return np.full(n_bins, np.nan), np.nan
    W = logistic_weights(X, choice); Wn = W / (np.abs(W).sum() + 1e-9)
    return Wn, float(np.polyfit(np.arange(n_bins), Wn, 1)[0])


def psychometric(coh, choice):
    lv = np.unique(np.round(coh, 3)); return lv, np.array([choice[np.round(coh, 3) == c].mean() for c in lv])


def simulate(rel, v, g, B, noise):
    """rel (N,T) float32, noise (N,T) standard normals (common random numbers). Returns deadline choice."""
    N, T = rel.shape; a = np.zeros(N, np.float32); frozen = np.zeros(N, bool); sq = np.float32(np.sqrt(DT))
    for t in range(T):
        da = (v * rel[:, t] - g * a) * DT + sq * noise[:, t]
        a = np.where(frozen, a, a + da)
        frozen |= np.abs(a) >= B
    return (a > 0).astype(int)


def loss_fn(theta, rel, noise, coh, w_net, P_net):
    v, g, B = theta
    if not (0.1 <= v <= 500 and -30 <= g <= 30 and 0.2 <= B <= 50): return 1e6
    ch = simulate(rel, v, g, B, noise); w, _ = kernel(rel, ch)
    if np.isnan(w).any(): return 1e6
    _, P = psychometric(coh, ch)
    return float(100 * np.sum((w - w_net) ** 2) + 20 * np.sum((P - P_net) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--n-random", type=int, default=150); ap.add_argument("--sub", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(DATA / "seed*.npz"))); print(f"{len(files)} networks")
    rng = np.random.default_rng(a.seed); rows, sweeps, fits = [], [], {}
    for gn in a.gains:
        t0 = time.time()
        rel = np.concatenate([np.load(f)["rel"].astype(np.float32) for f in files])
        coh = np.concatenate([np.load(f)["coh_signed"] for f in files])
        ch_net = np.concatenate([np.load(f)[f"choice_g{gn}"] for f in files])
        idx = rng.choice(len(rel), min(a.sub, len(rel)), replace=False)          # search subsample
        rel_s, coh_s, ch_s = rel[idx], coh[idx], ch_net[idx]
        w_net, s_net = kernel(rel_s, ch_s); lv, P_net = psychometric(coh_s, ch_s)
        noise = rng.standard_normal(rel_s.shape, dtype=np.float32)
        print(f"\n=== gain {gn}: {len(rel)} trials (search on {len(idx)}), network kernel slope {s_net:+.4f}, acc {(ch_s == (coh_s > 0)).mean():.3f}", flush=True)
        cands = []
        for i in range(a.n_random):
            th = [float(np.exp(rng.uniform(np.log(1), np.log(300)))), float(rng.uniform(-15, 15)), float(np.exp(rng.uniform(np.log(0.3), np.log(30))))]
            cands.append((loss_fn(th, rel_s, noise, coh_s, w_net, P_net), th))
            if (i + 1) % 50 == 0: print(f"  random {i+1}: best {min(c[0] for c in cands):.4f}", flush=True)
        cands.sort(key=lambda c: c[0]); best = []
        for l0, th in cands[:3]:
            r = minimize(loss_fn, th, args=(rel_s, noise, coh_s, w_net, P_net), method="Nelder-Mead", options=dict(maxiter=150, xatol=1e-3, fatol=1e-5))
            best.append((r.fun, list(r.x))); print(f"  refine {l0:.4f} -> {r.fun:.4f}  v={r.x[0]:.2f} g={r.x[1]:+.2f} B={r.x[2]:.2f}", flush=True)
        best.sort(key=lambda c: c[0]); loss, (v, g, B) = best[0]
        # evaluate best on ALL trials with fresh noise; network kernel on all trials too
        noise_all = np.random.default_rng(a.seed + 99).standard_normal(rel.shape, dtype=np.float32)
        ch_sim = simulate(rel, v, g, B, noise_all); w_sim, s_sim = kernel(rel, ch_sim); w_all, s_all = kernel(rel, ch_net)
        lv, P_all = psychometric(coh, ch_net); _, P_sim = psychometric(coh, ch_sim)
        fits[gn] = dict(w_net=w_all, w_sim=w_sim, lv=lv, P_net=P_all, P_sim=P_sim)
        rows.append(dict(gain=gn, v=v, g=g, B=B, loss=loss, slope_net=s_all, slope_sim=s_sim, acc_net=float((ch_net == (coh > 0)).mean()),
                         acc_sim=float((ch_sim == (coh > 0)).mean()), frac_bounded=float(np.nan), minutes=(time.time() - t0) / 60,
                         **{f"wnet{i}": w_all[i] for i in range(N_BINS)}, **{f"wsim{i}": w_sim[i] for i in range(N_BINS)}))
        print(f"  BEST gain {gn}: v={v:.2f} g={g:+.2f}/s B={B:.2f} | slope net {s_all:+.4f} sim {s_sim:+.4f} | acc net {rows[-1]['acc_net']:.3f} sim {rows[-1]['acc_sim']:.3f}", flush=True)
        for gg in np.linspace(-15, 15, 13):                                          # how slope depends on g at best (v, B)
            chs = simulate(rel_s, v, gg, B, noise); _, sl = kernel(rel_s, chs); sweeps.append(dict(gain=gn, g=gg, slope=sl, acc=float((chs == (coh_s > 0)).mean())))
    tab = pd.DataFrame(rows); tab.to_csv(OUT / "ou_kernel_fits.csv", index=False); pd.DataFrame(sweeps).to_csv(OUT / "g_sweep.csv", index=False)
    print("\n" + tab[["gain", "v", "g", "B", "loss", "slope_net", "slope_sim", "acc_net", "acc_sim"]].round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4)); cols = {0.8: "C0", 1.0: "k", 1.2: "C3"}; xb = (np.arange(N_BINS) + 0.5) * 750 / N_BINS
    for gn, f in fits.items():
        axes[0].plot(xb, f["w_net"], "-o", color=cols.get(gn, "C2"), label=f"network g={gn}"); axes[0].plot(xb, f["w_sim"], "--s", color=cols.get(gn, "C2"), alpha=.7, label=f"OU fit g={gn}")
        axes[1].plot(f["lv"], f["P_net"], "-o", color=cols.get(gn, "C2")); axes[1].plot(f["lv"], f["P_sim"], "--s", color=cols.get(gn, "C2"), alpha=.7)
    axes[0].set_xlabel("time within trial (ms)"); axes[0].set_ylabel("influence on choice (L1-normalised)"); axes[0].legend(fontsize=7); axes[0].set_title("psychophysical kernel: network (solid) vs OU (dashed)")
    axes[1].set_xlabel("signed relevant coherence"); axes[1].set_ylabel("P(choice = +)"); axes[1].set_title("psychometric")
    sw = pd.DataFrame(sweeps)
    for gn, d in sw.groupby("gain"): axes[2].plot(d.g, d.slope, "-o", color=cols.get(gn, "C2"), label=f"gain {gn}")
    for _, r in tab.iterrows(): axes[2].axhline(r.slope_net, color=cols.get(r.gain, "C2"), ls=":", lw=1)
    axes[2].axhline(0, color="gray", lw=.5); axes[2].set_xlabel("OU leak g (per s; >0 leaky)"); axes[2].set_ylabel("kernel slope"); axes[2].set_title("kernel slope vs g at best (v, B); dotted = network"); axes[2].legend(fontsize=7)
    plt.tight_layout(); fig.savefig(OUT / "ou_kernel_fits.png", dpi=140); print("saved", OUT / "ou_kernel_fits.png")


if __name__ == "__main__":
    main()
