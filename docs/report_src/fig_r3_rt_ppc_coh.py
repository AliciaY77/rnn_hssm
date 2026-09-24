"""Round-3 Figure 1: RT posterior predictive of the pooled HSSM OU fits by threshold (blocks), |coherence| (rows) and gain
(columns). Each panel: RT distribution at that |coherence|, correct trials up and errors down, network (filled) vs the fitted
OU at its posterior mean (line), each normalised over its crossing trials so that up + down = 1. Model simulated with the same
numpy engine as track_a/ppc_rt.py (stretched frame, k = 10), 10 000 trials per sign of coherence. Block panel titles: native g
with its 95 % posterior interval (2.5 / 97.5 % quantiles of the 400 saved posterior draws). Positive g is leaky."""
import pathlib, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "track_a"))
from ppc_rt import simulate_ou_numpy, OFFSET, HORIZON_NATIVE
TA = ROOT / "output/track_a"; OUT = ROOT / "output/report"; DATA = ROOT / "data/processed/fixed_bound"
GAINS = [0.8, 1.0, 1.2]; BOUNDS = [1.5, 2.5, 2.94]; COHS = [0.03, 0.09, 0.15]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; K = 10.0; N = 10000
H = OFFSET + K * HORIZON_NATIVE; rng = np.random.default_rng(11); bins = np.linspace(0, 750, 39)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
fig = plt.figure(figsize=(7.4, 9.6)); sfs = fig.subfigures(len(BOUNDS), 1, hspace=0.015)
rows = []
for sf, b in zip(sfs, BOUNDS):
    sf.suptitle(f"threshold {b} on dv", fontsize=9.5, fontweight="bold", y=0.985)
    axes = sf.subplots(len(COHS), 3, sharex=True); sf.subplots_adjust(top=0.80, bottom=0.09, left=0.06, right=0.99, hspace=0.22, wspace=0.12)
    for j, gn in enumerate(GAINS):
        dr = pd.read_csv(TA / f"g{gn}_k10_b{b}_pooled_draws.csv"); p = dr.mean(numeric_only=True)
        gq = dr.g * K; lo, hi = np.quantile(gq, [0.025, 0.975])
        d = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{b}_g{gn}.csv")
        for i, c in enumerate(COHS):
            ax = axes[i, j]; dd = d[d.coherence.round(2) == c]; cn = (dd.response == 1).values
            rt_s, cor_s = [], []
            for sgn in (1, -1):
                rt, ch = simulate_ou_numpy(np.array([p.v_Intercept + p.v_coherence_signed * sgn * c]), np.array([p.a]), np.array([p.z]), np.array([p.g]), float(p.t), N, H, rng)
                ok = np.isfinite(rt); rt_s.append((rt[ok] - OFFSET) / K * 1000); cor_s.append(ch[ok] * sgn > 0)
            rt = np.concatenate(rt_s); cm = np.concatenate(cor_s)
            ax.hist(dd.rt[cn] * 1000, bins=bins, weights=np.ones(cn.sum()) / len(dd), color=COL[gn], alpha=0.35)
            ax.hist(dd.rt[~cn] * 1000, bins=bins, weights=-np.ones((~cn).sum()) / len(dd), color=COL[gn], alpha=0.35)
            ax.hist(rt[cm], bins=bins, weights=np.ones(cm.sum()) / len(rt), histtype="step", color="k", lw=0.9)
            ax.hist(rt[~cm], bins=bins, weights=-np.ones((~cm).sum()) / len(rt), histtype="step", color="k", lw=0.9)
            ax.axhline(0, color="k", lw=0.4); ax.set_yticks([]); ax.set_xlim(0, 750)
            ax.text(0.97, 0.9, f"P(correct) {cn.mean():.2f} / {cm.mean():.2f}\nmedian {np.median(dd.rt)*1000:.0f} / {np.median(rt):.0f} ms", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.8))
            if i == 0: ax.set_title(f"gain {gn}\ng = {gq.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] /s", fontsize=7.8)
            if j == 0: ax.set_ylabel(f"|coh| {c:.2f}")
            if b == BOUNDS[-1] and i == len(COHS) - 1: ax.set_xlabel("RT (ms)")
            rows.append(dict(bound=b, gain=gn, coh=c, n_net=len(dd), pc_net=cn.mean(), pc_model=cm.mean(), med_net=np.median(dd.rt) * 1000, med_model=np.median(rt), g=gq.mean(), g_lo=lo, g_hi=hi))
fig.legend([Patch(color="gray", alpha=0.35), Line2D([], [], color="k", lw=0.9)], ["network (correct up, error down)", "fitted OU, posterior mean (correct up, error down)"],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.025), fontsize=8)
fig.savefig(OUT / "fig_r3_rt_ppc_coh.png", dpi=160, bbox_inches="tight"); pd.DataFrame(rows).to_csv(OUT / "r3_rt_ppc_coh.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
