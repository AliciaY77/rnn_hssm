"""Posterior predictive checks for the Track B (evidence-conditioned, unbounded) fit, the way an experimentalist would look
at them: per gain, (a) P(choice = +) per signed coherence, network vs model, with per-network points; (b) accuracy per
|coherence|; (c) calibration: trials binned by the model's predicted P(choice = +), observed fraction of + choices.
Model predictions are the analytic per-trial probabilities Phi(mu/s) at each network's own route-1 MLE (g, v, a_bias)."""
import glob, pathlib, numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "output" / "report"; OUT.mkdir(exist_ok=True, parents=True)
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; DT = 1e-3; T = 750
fits = pd.read_csv(ROOT / "output/track_b/analytic_per_network.csv")
files = sorted(glob.glob(str(ROOT / "data/processed/kernel/seed*.npz")))
tt = np.arange(T)
def p_plus(rel, g, v, ab):
    rho = 1 - g * DT; w = rho ** (T - 1 - tt); mu = ab * rho ** T + v * (rel @ w) * DT; s = np.sqrt(np.sum(rho ** (2 * (T - 1 - tt))) * DT)
    return norm.cdf(mu / s)
rows = []; cal = {gn: ([], []) for gn in GAINS}
for f in files:
    d = np.load(f); seed = int(pathlib.Path(f).stem[4:]); rel = d["rel"].astype(np.float32); coh = np.round(d["coh_signed"], 3)
    for gn in GAINS:
        r = fits[(fits.seed == seed) & (fits.gain == gn)].iloc[0]; p = p_plus(rel, r.g, r.v, r.a_bias); ch = d[f"choice_g{gn}"]
        cal[gn][0].append(p); cal[gn][1].append(ch)
        for c in np.unique(coh):
            m = coh == c; rows.append(dict(seed=seed, gain=gn, coh=c, n=int(m.sum()), p_net=ch[m].mean(), p_mod=p[m].mean()))
tab = pd.DataFrame(rows); tab.to_csv(OUT / "track_b_ppc_psychometric.csv", index=False)
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
for i, gn in enumerate(GAINS):
    t = tab[tab.gain == gn]; g = t.groupby("coh").agg(p_net=("p_net", "mean"), p_mod=("p_mod", "mean"), sd_net=("p_net", "std"), sd_mod=("p_mod", "std")).reset_index()
    ax = axes[i, 0]
    for s, ts in t.groupby("seed"): ax.plot(ts.coh, ts.p_net, "-", color=COL[gn], alpha=0.12, lw=0.8); ax.plot(ts.coh, ts.p_mod, "--", color="gray", alpha=0.12, lw=0.8)
    ax.plot(g.coh, g.p_net, "-o", color=COL[gn], ms=5, label="network (20 nets, mean)"); ax.plot(g.coh, g.p_mod, "--s", color="k", ms=4, label="fitted OU (predicted)")
    ax.set_title(f"gain {gn}: psychometric"); ax.set_xlabel("signed coherence"); ax.set_ylabel("P(choice = +)"); ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[i, 1]; t2 = t.assign(acoh=t.coh.abs(), acc_net=np.where(t.coh >= 0, t.p_net, 1 - t.p_net), acc_mod=np.where(t.coh >= 0, t.p_mod, 1 - t.p_mod))
    t2 = t2[t2.acoh > 0]; a = t2.groupby("acoh").agg(acc_net=("acc_net", "mean"), acc_mod=("acc_mod", "mean"), lo_net=("acc_net", lambda x: x.quantile(.1)), hi_net=("acc_net", lambda x: x.quantile(.9))).reset_index()
    ax.fill_between(a.acoh, a.lo_net, a.hi_net, color=COL[gn], alpha=0.15, label="network, 10–90 % across nets"); ax.plot(a.acoh, a.acc_net, "-o", color=COL[gn], ms=5, label="network mean"); ax.plot(a.acoh, a.acc_mod, "--s", color="k", ms=4, label="fitted OU")
    ax.set_title(f"gain {gn}: accuracy by |coherence|"); ax.set_xlabel("|coherence|"); ax.set_ylabel("P(correct)"); ax.set_ylim(0.5, 1.02); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[i, 2]; p = np.concatenate(cal[gn][0]); ch = np.concatenate(cal[gn][1]); edges = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    xs = [p[idx == k].mean() for k in range(10)]; ys = [ch[idx == k].mean() for k in range(10)]; ns = [(idx == k).sum() for k in range(10)]
    se = [np.sqrt(y * (1 - y) / max(n, 1)) for y, n in zip(ys, ns)]
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8); ax.errorbar(xs, ys, yerr=se, fmt="o", color=COL[gn], ms=5, capsize=2)
    ax.set_title(f"gain {gn}: calibration ({len(p):,} trials)"); ax.set_xlabel("model P(choice = +) for the trial"); ax.set_ylabel("observed fraction of + choices"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.05, 0.9, f"mean |obs − pred| over bins: {np.mean(np.abs(np.array(xs) - np.array(ys))):.3f}", transform=ax.transAxes, fontsize=8)
plt.tight_layout(); fig.savefig(OUT / "fig_track_b_ppc.png", dpi=150); print("saved", OUT / "fig_track_b_ppc.png")
summ = tab.assign(acoh=tab.coh.abs()).groupby(["gain", "coh"]).agg(p_net=("p_net", "mean"), p_mod=("p_mod", "mean")).reset_index()
summ["diff"] = summ.p_mod - summ.p_net; print(summ.round(3).to_string(index=False)); print("max |diff| per gain:", summ.groupby("gain")["diff"].apply(lambda x: x.abs().max()).round(3).to_dict())
